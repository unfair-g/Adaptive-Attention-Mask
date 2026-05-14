"""Utils function for data loading and data processing.
"""
import torch
import numpy as np
import logging as lg
import random as r
import json
import ast

from kornia.color.ycbcr import rgb_to_ycbcr, ycbcr_to_rgb
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
# from kornia.augmentation import Resize
from torch import nn
from sklearn.cluster import KMeans

from src.utils.utils import get_device
from src.datasets import MNIST, Number, FashionMNIST, SplitFashion, ImageNet
from src.datasets import CIFAR10, SplitCIFAR10, CIFAR100, SplitCIFAR100, SplitImageNet
from src.datasets import BlurryCIFAR10, BlurryCIFAR100, BlurryTiny
from src.datasets.tinyImageNet import TinyImageNet
from src.datasets.split_tiny import SplitTiny
from src.datasets.ImageNet import ImageNet
from src.datasets.split_ImageNet import SplitImageNet


device = get_device()


def parse_task_classes(task_classes_arg):
    """Parse task_classes argument from string or list format.
    
    Args:
        task_classes_arg: Can be a string like '[[0,1,2],[3,4],[5,6,7,8,9]]' or already a list of lists
        
    Returns:
        List of lists containing class indices for each task
    """
    if task_classes_arg is None:
        return None
    
    # If already a list (from yaml config), return directly
    if isinstance(task_classes_arg, list):
        return task_classes_arg
    
    # If string, try to parse it
    if isinstance(task_classes_arg, str):
        try:
            # Try JSON format first
            return json.loads(task_classes_arg)
        except json.JSONDecodeError:
            try:
                # Try Python literal eval as fallback
                return ast.literal_eval(task_classes_arg)
            except (ValueError, SyntaxError):
                raise ValueError(f"Cannot parse task_classes: {task_classes_arg}. "
                               f"Expected format: '[[0,1,2],[3,4],[5,6,7,8,9]]'")
    
    raise ValueError(f"task_classes must be a string or list, got {type(task_classes_arg)}")


def parse_increment_config(increment_arg, n_classes, labels_order=None):
    """Parse increment configuration in 'X-Y' format and generate task classes.
    
    Args:
        increment_arg: String in format 'X-Y' where:
            - X (Base Classes): Number of classes in the first task (Step 1)
            - Y (Incremental Classes): Number of new classes added in each subsequent step
        n_classes: Total number of classes in the dataset
        labels_order: Optional list specifying the order of class labels
        
    Returns:
        List of lists containing class indices for each task, or None if increment_arg is None
        
    Example:
        parse_increment_config('50-10', 100) with default order returns:
        [[0-49], [50-59], [60-69], [70-79], [80-89], [90-99]]
        
        This means:
        - Task 0 (Base): 50 classes
        - Task 1-5 (Incremental): 10 classes each
    """
    if increment_arg is None:
        return None
    
    if isinstance(increment_arg, str):
        # Parse 'X-Y' format
        parts = increment_arg.strip().split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid increment format: '{increment_arg}'. "
                           f"Expected format: 'X-Y' (e.g., '50-10')")
        
        try:
            base_classes = int(parts[0].strip())
            inc_classes = int(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid increment format: '{increment_arg}'. "
                           f"X and Y must be integers. Expected format: 'X-Y' (e.g., '50-10')")
    else:
        raise ValueError(f"increment must be a string in 'X-Y' format, got {type(increment_arg)}")
    
    # Validate the configuration
    if base_classes <= 0:
        raise ValueError(f"Base classes (X={base_classes}) must be positive")
    if inc_classes <= 0:
        raise ValueError(f"Incremental classes (Y={inc_classes}) must be positive")
    if base_classes > n_classes:
        raise ValueError(f"Base classes (X={base_classes}) cannot exceed total classes ({n_classes})")
    
    remaining_classes = n_classes - base_classes
    if remaining_classes > 0 and remaining_classes % inc_classes != 0:
        lg.warning(f"Warning: Remaining classes ({remaining_classes}) is not divisible by "
                   f"incremental classes ({inc_classes}). Last task may have fewer classes.")
    
    # Generate task classes
    task_classes = []
    
    # Use provided labels_order or default sequential order
    if labels_order is None:
        all_labels = list(range(n_classes))
    else:
        all_labels = list(labels_order)
    
    # Task 0: Base classes
    task_classes.append(all_labels[:base_classes])
    
    # Subsequent tasks: Incremental classes
    current_idx = base_classes
    while current_idx < n_classes:
        end_idx = min(current_idx + inc_classes, n_classes)
        task_classes.append(all_labels[current_idx:end_idx])
        current_idx = end_idx
    
    return task_classes

def get_loaders(args):
    tf = transforms.ToTensor()
    dataloaders = {}
    
    # First, generate random labels_order if not specified
    if args.labels_order is None:
        l = np.arange(args.n_classes)
        np.random.shuffle(l)
        args.labels_order = l.tolist()
    
    # Priority: increment > task_classes > default (equal split)
    # Parse increment configuration if provided (format: 'X-Y')
    increment_arg = getattr(args, 'increment', None)
    task_classes = None
    
    if increment_arg is not None:
        # Use increment configuration to generate task classes
        task_classes = parse_increment_config(
            increment_arg, 
            args.n_classes, 
            args.labels_order
        )
        lg.info(f"Using increment configuration: '{increment_arg}'")
        lg.info(f"  Base classes (X): {len(task_classes[0])}")
        lg.info(f"  Incremental classes (Y): {len(task_classes[1]) if len(task_classes) > 1 else 0}")
        lg.info(f"  Total tasks: {len(task_classes)}")
    else:
        # Parse and validate task_classes if provided
        task_classes = parse_task_classes(getattr(args, 'task_classes', None))
    
    if task_classes is not None:
        # Custom task classes specified - validate and update args
        all_classes = []
        for task_idx, classes in enumerate(task_classes):
            if not isinstance(classes, list):
                raise ValueError(f"Task {task_idx} classes must be a list, got {type(classes)}")
            all_classes.extend(classes)
        
        # Update n_tasks based on task_classes
        args.n_tasks = len(task_classes)
        
        # Create labels_order from flattened task_classes for compatibility
        args.labels_order = all_classes
        
        # Store parsed task_classes for use in add_incremental_splits
        args._parsed_task_classes = task_classes
        
        lg.info(f"Using custom task classes configuration:")
        for task_idx, classes in enumerate(task_classes):
            lg.info(f"  Task {task_idx}: classes {classes}")
    else:
        # Default behavior: equal split based on n_classes and n_tasks
        args._parsed_task_classes = None
    
    if args.dataset == 'mnist':
        dataset_train = MNIST(args.data_root_dir, train=True, download=True, transform=tf)
        dataset_test = MNIST(args.data_root_dir, train=False, download=True, transform=tf)
    elif args.dataset == 'fmnist':
        dataset_train = FashionMNIST(args.data_root_dir, train=True, download=True, transform=tf)
        dataset_test = FashionMNIST(args.data_root_dir, train=False, download=True, transform=tf)
    elif args.dataset == 'cifar10':
        if args.training_type == 'blurry':
            dataset_train = BlurryCIFAR10(root=args.data_root_dir, labels_order=args.labels_order,
                train=True, download=True, transform=tf, n_tasks=args.n_tasks, scale=args.blurry_scale)
            dataset_test = CIFAR10(args.data_root_dir, train=False, download=True, transform=tf)
        else:
            dataset_train = CIFAR10(args.data_root_dir, train=True, download=True, transform=tf)
            dataset_test = CIFAR10(args.data_root_dir, train=False, download=True, transform=tf)
    elif args.dataset == 'cifar100':
        if args.training_type == 'blurry':
            dataset_train = BlurryCIFAR100(root=args.data_root_dir, labels_order=args.labels_order,
                train=True, download=True, transform=tf, n_tasks=args.n_tasks, scale=args.blurry_scale)
            dataset_test = CIFAR100(args.data_root_dir, train=False, download=True, transform=tf)
        else:
            dataset_train = CIFAR100(args.data_root_dir, train=True, download=True, transform=tf)
            dataset_test = CIFAR100(args.data_root_dir, train=False, download=True, transform=tf)
    elif args.dataset == 'tiny':
        if args.training_type == 'blurry':
            dataset_train = BlurryTiny(root=args.data_root_dir, labels_order=args.labels_order,
                train=True, download=True, transform=tf, n_tasks=args.n_tasks, scale=args.blurry_scale)
            dataset_test = TinyImageNet(args.data_root_dir, train=False, download=True, transform=tf)
        else:
            dataset_train = TinyImageNet(args.data_root_dir, train=True, download=True, transform=tf)
            dataset_test = TinyImageNet(args.data_root_dir, train=False, download=True, transform=tf) 
    elif args.dataset == 'imagenet100':
        # Loading only the first 100 labels
        dataset_train = SplitImageNet(root=args.data_root_dir, train=True,
                                        selected_labels=np.arange(args.n_classes), transform=tf)
        dataset_test = SplitImageNet(root=args.data_root_dir, train=False,
                                        selected_labels=np.arange(args.n_classes), transform=tf)
    elif args.dataset == 'yt':
        tf = transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                Resize(size=(256,256)),
        ])
        dataset_train = ImageFolder('/storage8To/datasets/deepsponsorblock/images/old/train_old', transform=tf)
        dataset_test = ImageFolder('/storage8To/datasets/deepsponsorblock/images/old/test_old', transform=tf)
        
    if args.training_type == 'inc':
        dataloaders = add_incremental_splits(args, dataloaders, tf, tag="train")
        dataloaders = add_incremental_splits(args, dataloaders, tf, tag="test")
    
    dataloaders['train'] = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=args.training_type != 'blurry', num_workers=args.num_workers)
    dataloaders['test'] = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return dataloaders


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def add_incremental_splits(args, dataloaders, tf, tag="train"):
    is_train = tag == "train"
    
    # Check if custom task_classes is provided
    task_classes = getattr(args, '_parsed_task_classes', None)
    
    if task_classes is not None:
        # Use custom task classes configuration
        lg.info(f"Loading incremental splits with custom task classes ({tag}):")
        for task_idx, classes in enumerate(task_classes):
            lg.info(f"  Task {task_idx}: classes {classes}")
        
        for task_idx, selected_labels in enumerate(task_classes):
            dataset = _create_dataset_for_labels(args, is_train, tf, selected_labels)
            dataloaders[f"{tag}{task_idx}"] = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            )
    else:
        # Default behavior: equal split based on n_classes and n_tasks
        step_size = int(args.n_classes / args.n_tasks)
        lg.info("Loading incremental splits with labels :")
        for i in range(0, args.n_classes, step_size):
            lg.info([args.labels_order[j] for j in range(i, i+step_size)])
        for i in range(0, args.n_classes, step_size):
            selected_labels = [args.labels_order[j] for j in range(i, i+step_size)]
            dataset = _create_dataset_for_labels(args, is_train, tf, selected_labels)
            dataloaders[f"{tag}{int(i/step_size)}"] = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            )
 
    return dataloaders


def _create_dataset_for_labels(args, is_train, tf, selected_labels):
    """Create a dataset containing only the specified labels.
    
    Args:
        args: Arguments namespace
        is_train: Whether this is training data
        tf: Transform to apply
        selected_labels: List of class labels to include
        
    Returns:
        Dataset object
    """
    if args.dataset == 'mnist':
        return Number(
            args.data_root_dir,
            train=is_train,
            transform=tf,
            download=True,
            selected_labels=selected_labels,
            permute=False
        )
    elif args.dataset == 'fmnist':
        return SplitFashion(
            args.data_root_dir,
            train=is_train,
            transform=tf,
            download=True,
            selected_labels=selected_labels
        )
    elif args.dataset == 'cifar10':
        return SplitCIFAR10(
            args.data_root_dir,
            train=is_train,
            transform=tf,
            download=True,
            selected_labels=selected_labels
        )
    elif args.dataset == 'cifar100':
        return SplitCIFAR100(
            args.data_root_dir,
            train=is_train,
            transform=tf,
            download=True,
            selected_labels=selected_labels
        )
    elif args.dataset == 'tiny':
        return SplitTiny(
            args.data_root_dir,
            train=is_train,
            transform=tf,
            download=True,
            selected_labels=selected_labels
        )
    elif args.dataset == "imagenet100":
        return SplitImageNet(
            root=args.data_root_dir,
            train=is_train,
            selected_labels=selected_labels,
            transform=tf
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported for incremental learning")
