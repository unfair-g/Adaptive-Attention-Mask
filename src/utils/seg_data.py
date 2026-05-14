"""Data loading utilities for semantic segmentation continual learning.

This module provides functions to create dataloaders for Cityscapes/BDD100K
and other segmentation datasets in continual learning settings.
"""
import torch
import numpy as np
import logging as lg

from torchvision import transforms
from torch.utils.data import DataLoader

from src.datasets.cityscapes import Cityscapes, SplitCityscapes, BlurryCityscapes
from src.datasets.bdd100k import BDD100K, SplitBDD100K, BlurryBDD100K
from src.datasets.mapillary import (
    MapillaryVistas,
    SplitMapillaryVistas,
    BlurryMapillaryVistas,
)


# ImageNet normalization (required for SegFormer pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_seg_increment_config(increment_arg, n_classes, labels_order=None):
    """Parse increment configuration in 'X-Y' format for segmentation.
    
    Args:
        increment_arg: String in format 'X-Y' where:
            - X (Base Classes): Number of classes in Task 0
            - Y (Incremental Classes): Number of new classes per subsequent task
        n_classes: Total number of classes (e.g., 19 for Cityscapes)
        labels_order: Optional list specifying class order
        
    Returns:
        List of lists containing class indices for each task
        
    Example:
        parse_seg_increment_config('5-5', 19) returns:
        [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18]]
        This creates 4 tasks with the last task having fewer classes.
    """
    if increment_arg is None:
        return None
    
    if isinstance(increment_arg, str):
        parts = increment_arg.strip().split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid increment format: '{increment_arg}'. "
                           f"Expected format: 'X-Y' (e.g., '5-5')")
        try:
            base_classes = int(parts[0].strip())
            inc_classes = int(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid increment format: '{increment_arg}'. "
                           f"X and Y must be integers.")
    else:
        raise ValueError(f"increment must be a string in 'X-Y' format, got {type(increment_arg)}")
    
    # Validate
    if base_classes <= 0 or inc_classes <= 0:
        raise ValueError(f"Base classes ({base_classes}) and incremental classes ({inc_classes}) must be positive")
    if base_classes > n_classes:
        raise ValueError(f"Base classes ({base_classes}) exceeds total classes ({n_classes})")
    
    # Use labels_order or default sequential
    if labels_order is None:
        all_labels = list(range(n_classes))
    else:
        all_labels = list(labels_order)
    
    # Generate task classes
    task_classes = []
    
    # Task 0: Base classes
    task_classes.append(all_labels[:base_classes])
    
    # Subsequent tasks: Incremental classes
    current_idx = base_classes
    while current_idx < n_classes:
        end_idx = min(current_idx + inc_classes, n_classes)
        task_classes.append(all_labels[current_idx:end_idx])
        current_idx = end_idx
    
    return task_classes


def denormalize_image(img, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Denormalize image tensor for visualization.
    
    Args:
        img: Image tensor (C, H, W) or (B, C, H, W) normalized
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        denormalized: Image tensor in [0, 1] range
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if img.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    if img.device != mean.device:
        mean = mean.to(img.device)
        std = std.to(img.device)
    
    denormalized = img * std + mean
    return torch.clamp(denormalized, 0, 1)


def get_seg_loaders(args):
    """Create dataloaders for semantic segmentation.
    
    Args:
        args: Configuration arguments
        
    Returns:
        dataloaders: Dictionary of dataloaders
    """
    if isinstance(getattr(args, "dataset", None), str):
        args.dataset = args.dataset.strip().lower()

    # Apply ImageNet normalization for pretrained SegFormer
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    dataloaders = {}
    
    # Initialize label order if not specified
    if args.labels_order is None:
        l = np.arange(args.n_classes)
        np.random.shuffle(l)
        args.labels_order = l.tolist()
    
    if args.dataset == 'cityscapes':
        img_size = tuple(args.img_size) if isinstance(args.img_size, list) else args.img_size
        
        if args.training_type == 'blurry':
            dataset_train = BlurryCityscapes(
                root=args.data_root_dir,
                labels_order=args.labels_order,
                split='train',
                transform=tf,
                n_tasks=args.n_tasks,
                scale=args.blurry_scale,
                img_size=img_size
            )
        else:
            dataset_train = Cityscapes(
                root=args.data_root_dir,
                split='train',
                transform=tf,
                img_size=img_size
            )
        
        dataset_test = Cityscapes(
            root=args.data_root_dir,
            split='val',
            transform=tf,
            img_size=img_size
        )
    elif args.dataset == 'bdd100k':
        img_size = tuple(args.img_size) if isinstance(args.img_size, list) else args.img_size

        if args.training_type == 'blurry':
            dataset_train = BlurryBDD100K(
                root=args.data_root_dir,
                labels_order=args.labels_order,
                split='train',
                transform=tf,
                n_tasks=args.n_tasks,
                scale=args.blurry_scale,
                img_size=img_size
            )
        else:
            dataset_train = BDD100K(
                root=args.data_root_dir,
                split='train',
                transform=tf,
                img_size=img_size
            )

        dataset_test = BDD100K(
            root=args.data_root_dir,
            split='val',
            transform=tf,
            img_size=img_size
        )
    elif args.dataset == 'mapillary':
        img_size = tuple(args.img_size) if isinstance(args.img_size, list) else args.img_size

        if args.training_type == 'blurry':
            dataset_train = BlurryMapillaryVistas(
                root=args.data_root_dir,
                labels_order=args.labels_order,
                split='train',
                transform=tf,
                n_tasks=args.n_tasks,
                scale=args.blurry_scale,
                img_size=img_size
            )
        else:
            dataset_train = MapillaryVistas(
                root=args.data_root_dir,
                split='train',
                transform=tf,
                img_size=img_size
            )

        dataset_test = MapillaryVistas(
            root=args.data_root_dir,
            split='val',
            transform=tf,
            img_size=img_size
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported for segmentation")
    
    # Create incremental splits if needed
    if args.training_type == 'inc':
        dataloaders = add_seg_incremental_splits(args, dataloaders, tf, tag="train")
        dataloaders = add_seg_incremental_splits(args, dataloaders, tf, tag="test")
    
    # Full dataset loaders
    dataloaders['train'] = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=args.training_type != 'blurry',
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    dataloaders['test'] = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return dataloaders


def get_default_transform():
    """Get default transform with ImageNet normalization."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def add_seg_incremental_splits(args, dataloaders, tf=None, tag="train"):
    """Add incremental task splits to dataloaders.
    
    For each task, creates a dataset where:
    - ALL images are used (no filtering)
    - Labels are processed based on label_mode:
      - current_only: only current task labels, others=255(ignore)
      - current_and_old: current+old visible, future=255
      - unknown_as_background: current+old visible, unknown=background_class(19)
      - all_unknown_as_background: only current visible, others=background_class
    
    IMPORTANT: For TEST splits, we always use 'current_and_old' mode to properly
    evaluate all seen classes. The training mode only affects training data.
    
    Supports two task configuration methods:
    1. increment: "X-Y" format (e.g., "5-5" for 5 base + 5 per task)
    2. n_tasks: Equal split (e.g., n_tasks=4 for 19/4 classes per task)
    
    Args:
        args: Configuration arguments
        dataloaders: Existing dataloaders dictionary
        tf: Transform to apply (if None, uses ImageNet normalization)
        tag: 'train' or 'test'
        
    Returns:
        dataloaders: Updated dataloaders dictionary
    """
    if isinstance(getattr(args, "dataset", None), str):
        args.dataset = args.dataset.strip().lower()

    # Use default transform with ImageNet normalization if not provided
    if tf is None:
        tf = get_default_transform()
    
    is_train = tag == "train"
    split = 'train' if is_train else 'val'
    
    # Note: based on original foreground classes, not including background.
    if args.dataset in ['cityscapes', 'bdd100k', 'mapillary']:
        original_n_classes = 19
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")
    
    img_size = tuple(args.img_size) if isinstance(args.img_size, list) else args.img_size
    
    # Get label mode and background class
    # IMPORTANT: For test splits, always use 'current_and_old' to properly evaluate all seen classes
    if is_train:
        mode = getattr(args, 'label_mode', 'unknown_as_background')
    else:
        # Test mode: keep all seen class labels visible for proper evaluation
        mode = 'current_and_old'
    background_class = getattr(args, 'background_class', 19)
    
    # =========================================================
    # Parse task configuration: Priority increment > n_tasks
    # =========================================================
    increment_arg = getattr(args, 'increment', None)
    
    if increment_arg is not None:
        # Use increment configuration
        task_classes_list = parse_seg_increment_config(
            increment_arg,
            original_n_classes,
            args.labels_order
        )
        n_tasks = len(task_classes_list)
        
        # Update args.n_tasks for consistency
        args.n_tasks = n_tasks
        
        lg.info(f"Creating {tag} incremental splits using increment='{increment_arg}':")
        lg.info(f"  Label mode: {mode} (training), Background class: {background_class}")
        lg.info(f"  Total tasks: {n_tasks}")
        for i, tc in enumerate(task_classes_list):
            lg.info(f"    Task {i}: {len(tc)} classes -> {tc}")
    else:
        # Fallback to n_tasks equal split
        task_classes_list = None
        n_tasks = args.n_tasks
        base_size = original_n_classes // n_tasks
        remainder = original_n_classes % n_tasks
        
        lg.info(f"Creating {tag} incremental splits using n_tasks={n_tasks}:")
        lg.info(f"  Label mode: {mode}, Background class: {background_class}")
        lg.info(f"  Classes per task: base={base_size}, remainder={remainder}")
    
    old_classes = []  # Accumulate old classes
    current_idx = 0
    
    for task_id in range(n_tasks):
        if task_classes_list is not None:
            # Use pre-computed task classes from increment config
            task_classes = task_classes_list[task_id]
        else:
            # Calculate classes for this task (equal split with remainder)
            task_size = base_size
            if task_id >= n_tasks - remainder:
                task_size += 1
            task_classes = [args.labels_order[j] for j in range(current_idx, current_idx + task_size)]
            current_idx += task_size
        
        if task_classes_list is None:
            lg.info(f"  Task {task_id}: current={task_classes}, old={old_classes}")
        
        if args.dataset == 'cityscapes':
            dataset = SplitCityscapes(
                root=args.data_root_dir,
                split=split,
                transform=tf,
                selected_labels=task_classes,      # Current task labels
                img_size=img_size,
                old_labels=list(old_classes),      # Previous task labels
                mode=mode,
                background_class=background_class
            )
        elif args.dataset == 'bdd100k':
            dataset = SplitBDD100K(
                root=args.data_root_dir,
                split=split,
                transform=tf,
                selected_labels=task_classes,      # Current task labels
                img_size=img_size,
                old_labels=list(old_classes),      # Previous task labels
                mode=mode,
                background_class=background_class
            )
        elif args.dataset == 'mapillary':
            dataset = SplitMapillaryVistas(
                root=args.data_root_dir,
                split=split,
                transform=tf,
                selected_labels=task_classes,      # Current task labels
                img_size=img_size,
                old_labels=list(old_classes),      # Previous task labels
                mode=mode,
                background_class=background_class
            )
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not supported")
        
        dataloaders[f"{tag}{task_id}"] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=is_train,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=is_train
        )
        
        # Add current classes to old classes for next task
        old_classes.extend(task_classes)
    
    return dataloaders


def get_class_order_cityscapes(order_type='sequential', n_classes=19, seed=0):
    """Get class order for Cityscapes incremental learning.
    
    Args:
        order_type: 'sequential', 'random', or 'frequency'
        n_classes: Number of classes
        seed: Random seed for reproducibility
        
    Returns:
        order: List of class indices in order
    """
    if order_type == 'sequential':
        return list(range(n_classes))
    
    elif order_type == 'random':
        np.random.seed(seed)
        order = list(range(n_classes))
        np.random.shuffle(order)
        return order
    
    elif order_type == 'frequency':
        # Order by typical frequency in Cityscapes
        # (road and building are most common, train and bicycle are rare)
        frequency_order = [
            0,   # road (most common)
            2,   # building
            8,   # vegetation
            13,  # car
            1,   # sidewalk
            10,  # sky
            11,  # person
            7,   # traffic sign
            5,   # pole
            9,   # terrain
            6,   # traffic light
            4,   # fence
            3,   # wall
            12,  # rider
            14,  # truck
            15,  # bus
            17,  # motorcycle
            18,  # bicycle
            16,  # train (rarest)
        ]
        return frequency_order[:n_classes]
    
    elif order_type == 'disjoint':
        # Group semantically similar classes together
        # Background stuff -> Things
        disjoint_order = [
            # Task 1: Road/ground
            0, 1, 9,  # road, sidewalk, terrain
            # Task 2: Buildings/structures
            2, 3, 4,  # building, wall, fence
            # Task 3: Nature/sky
            8, 10,  # vegetation, sky
            # Task 4: Small objects
            5, 6, 7,  # pole, traffic light, traffic sign
            # Task 5: Vehicles
            13, 14, 15, 16, 17, 18,  # car, truck, bus, train, motorcycle, bicycle
            # Task 6: People
            11, 12  # person, rider
        ]
        return disjoint_order[:n_classes]
    
    else:
        raise ValueError(f"Unknown order type: {order_type}")


def compute_class_weights(dataset, n_classes, ignore_index=255):
    """Compute class weights for imbalanced dataset.
    
    Uses inverse frequency weighting.
    
    Args:
        dataset: Dataset to compute weights from
        n_classes: Number of classes
        ignore_index: Label to ignore
        
    Returns:
        weights: Tensor of class weights
    """
    counts = torch.zeros(n_classes)
    
    for i in range(len(dataset)):
        _, mask, _ = dataset[i]
        for c in range(n_classes):
            counts[c] += (mask == c).sum()
    
    # Inverse frequency
    total = counts.sum()
    weights = total / (n_classes * counts.clamp(min=1))
    
    # Normalize
    weights = weights / weights.sum() * n_classes
    
    return weights


class SegmentationCollator:
    """Custom collator for segmentation that handles variable-size inputs.
    
    Args:
        img_size: Target image size (H, W)
    """
    
    def __init__(self, img_size=(512, 1024)):
        self.img_size = img_size
        
    def __call__(self, batch):
        """Collate batch of (image, mask, index) tuples."""
        images = []
        masks = []
        indices = []
        
        for img, mask, idx in batch:
            # Resize if necessary
            if img.shape[-2:] != self.img_size:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    size=self.img_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=self.img_size,
                    mode='nearest'
                ).squeeze().long()
            
            images.append(img)
            masks.append(mask)
            indices.append(idx)
        
        return torch.stack(images), torch.stack(masks), torch.tensor(indices)

