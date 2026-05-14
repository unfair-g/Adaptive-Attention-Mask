"""Base learner for semantic segmentation continual learning.

This module provides the base class for all semantic segmentation
continual learning methods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging as lg
import os
import pickle
import time
import numpy as np
import pandas as pd
import json
import wandb

from datetime import datetime
from copy import deepcopy
from torchvision import transforms
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter

from src.utils.utils import save_model, get_device
from src.utils.seg_metrics import SegmentationMetrics, ContinualSegmentationMetrics
from src.models.segformer import create_segformer


device = get_device()


class BaseSegmentationLearner(nn.Module):
    """Base class for semantic segmentation continual learners.
    
    This class provides the foundation for implementing various
    continual learning methods for semantic segmentation.
    
    Args:
        args: Configuration arguments
    """
    
    def __init__(self, args):
        super().__init__()
        self.params = args
        self.device = get_device()
        self.init_tag()
        self.model = self.load_model()
        self.optim = self.load_optim()
        self.buffer = None
        self.start = time.time()
        self.criterion = self.load_criterion()
        
        self.loss = 0
        self.stream_idx = 0
        self.init_results()
        
        # Data augmentation
        self.transform_train = nn.Sequential(
            RandomHorizontalFlip(p=0.5),
            ColorJitter(0.3, 0.3, 0.3, 0.1, p=0.5),
        ).to(device)
        
        self.transform_test = nn.Identity()
        
        # Track seen classes
        self.seen_classes = set()
        self.task_classes = {}  # task_id -> list of classes
        
    def init_results(self):
        """Initialize results storage."""
        self.results = []  # List of per-task metrics
        self.results_forgetting = []
        self.continual_metrics = ContinualSegmentationMetrics(
            n_classes=self.params.n_classes,
            n_tasks=self.params.n_tasks,
            ignore_index=self.params.ignore_index
        )
        
    def init_tag(self):
        """Initialize experiment tag."""
        if self.params.training_type == 'inc':
            self.params.tag = (
                f"{self.params.learner},{self.params.dataset},"
                f"m{self.params.mem_size}mbs{self.params.mem_batch_size}"
                f"sbs{self.params.batch_size}{self.params.tag}"
            )
        elif self.params.training_type == 'blurry':
            self.params.tag = (
                f"{self.params.learner},{self.params.dataset},"
                f"m{self.params.mem_size}mbs{self.params.mem_batch_size}"
                f"sbs{self.params.batch_size}blurry{self.params.blurry_scale}"
                f"{self.params.tag}"
            )
        else:
            self.params.tag = (
                f"{self.params.learner},{self.params.dataset},"
                f"{self.params.epochs}b{self.params.batch_size},uni"
                f"{self.params.tag}"
            )
        print(f"Experiment tag: {self.params.tag}")
        
    def load_model(self):
        """Load the segmentation model.
        
        Returns:
            model: SegFormer model
        """
        model = create_segformer(
            n_classes=self.params.n_classes,
            pretrained=self.params.segformer_variant,
            img_size=(self.params.img_size[0], self.params.img_size[1]),
            model_type='standard'
        )
        model.to(self.device)
        return model
    
    def load_optim(self):
        """Load optimizer.
        
        Returns:
            optimizer: Torch optimizer
        """
        if self.params.optim == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay
            )
        elif self.params.optim == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay
            )
        elif self.params.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.params.optim}")
        return optimizer
    
    def load_criterion(self):
        """Load loss criterion.
        
        Returns:
            criterion: Loss function
        """
        from src.utils.seg_losses import CrossEntropyLoss2d
        return CrossEntropyLoss2d(ignore_index=self.params.ignore_index)
    
    def save(self, model_name):
        """Save model and buffer state.
        
        Args:
            model_name: Name for the model checkpoint
        """
        if self.params.save_ckpt:
            save_dir = os.path.join(
                self.params.ckpt_root,
                self.params.tag,
                str(self.params.run_id)
            )
            save_model(self.model.state_dict(), model_name, dir=save_dir)
                    
    def resume(self, model_path=None, buffer_path=None):
        """Resume from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            buffer_path: Path to buffer checkpoint
        """
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        if buffer_path is not None:
            with open(buffer_path, 'rb') as f:
                self.buffer = pickle.load(f)
        torch.cuda.empty_cache()
        
    def train(self, dataloader, **kwargs):
        """Training loop - to be implemented by subclasses."""
        raise NotImplementedError
    
    def forward(self, x):
        """Forward pass through model."""
        return self.model(x)
    
    def evaluate(self, dataloaders, task_id, **kwargs):
        """Evaluate model on test sets.
        
        Args:
            dataloaders: Dictionary of dataloaders
            task_id: Current task ID
            
        Returns:
            avg_miou: Average mIoU across seen tasks
            avg_fgt: Average forgetting
        """
        self.model.eval()
        
        all_mious = []
        
        # Confidence threshold for prediction filtering
        # If max probability < threshold, mark as ignored
        pred_confidence_threshold = getattr(self.params, 'pred_confidence_threshold', None)
        # YAML/CLI may parse numeric fields as strings (e.g. "0.15"),
        # so normalize type before comparisons.
        if pred_confidence_threshold is not None:
            try:
                pred_confidence_threshold = float(pred_confidence_threshold)
            except (TypeError, ValueError):
                pred_confidence_threshold = None
        
        with torch.no_grad():
            for j in range(task_id + 1):
                test_loader = dataloaders.get(f"test{j}")
                if test_loader is None:
                    continue
                    
                metrics = SegmentationMetrics(
                    n_classes=self.params.n_classes,
                    ignore_index=self.params.ignore_index
                )
                
                for batch in test_loader:
                    imgs, masks = batch[0].to(self.device), batch[1].to(self.device)
                    
                    logits = self.model(imgs)
                    probs = F.softmax(logits, dim=1)
                    max_probs, preds = probs.max(dim=1)
                    
                    # Apply confidence filtering: mark low-confidence predictions as ignored
                    if pred_confidence_threshold is not None and pred_confidence_threshold > 0:
                        low_conf_mask = max_probs < pred_confidence_threshold
                        preds[low_conf_mask] = self.params.ignore_index
                    
                    metrics.update(preds, masks)
                
                # Get mIoU for this task's classes
                task_classes = self.task_classes.get(j, None)
                
                # Debug: print evaluation info
                per_class_iou = metrics.get_iou()
                gt_per_class = metrics.confusion_matrix.sum(axis=1)  # GT pixels per class
                pred_per_class = metrics.confusion_matrix.sum(axis=0)  # Pred pixels per class
                
                lg.info(f"  [DEBUG] Eval task {j}: task_classes={task_classes}")
                if task_classes:
                    for c in task_classes:
                        lg.info(f"    Class {c}: IoU={per_class_iou[c]*100:.2f}%, GT pixels={gt_per_class[c]}, Pred pixels={pred_per_class[c]}")
                
                miou = metrics.get_miou(task_classes)
                lg.info(f"    Task {j} mIoU: {miou*100:.2f}%")
                
                all_mious.append(miou)
                
                # Record metrics
                self.continual_metrics.record_metrics(
                    task_id, j,
                    {'miou': miou, **metrics.get_results(task_classes)}
                )
                
                # Wandb logging
                if not self.params.no_wandb:
                    wandb.log({
                        f"miou_task{j}": miou,
                        "task_id": task_id
                    })
        
        # Compute average metrics
        for _ in range(self.params.n_tasks - task_id - 1):
            all_mious.append(np.nan)
        
        self.results.append(all_mious)
        
        # Compute forgetting
        if len(self.results) > 1:
            from src.utils.seg_metrics import segmentation_forgetting_line
            fgt_line = segmentation_forgetting_line(
                pd.DataFrame(self.results),
                task_id,
                self.params.n_tasks
            )
            self.results_forgetting.append(fgt_line.tolist())
        else:
            self.results_forgetting.append([np.nan] * self.params.n_tasks)
        
        avg_miou = np.nanmean(all_mious)
        avg_fgt = np.nanmean(self.results_forgetting[-1])
        
        self.print_results(task_id)
        
        return avg_miou, avg_fgt
    
    def evaluate_offline(self, dataloaders, epoch):
        """Evaluate in offline setting.
        
        Args:
            dataloaders: Dictionary with 'test' dataloader
            epoch: Current epoch
            
        Returns:
            miou: Mean IoU on test set
        """
        self.model.eval()
        
        metrics = SegmentationMetrics(
            n_classes=self.params.n_classes,
            ignore_index=self.params.ignore_index
        )
        
        with torch.no_grad():
            for batch in dataloaders['test']:
                imgs, masks = batch[0].to(self.device), batch[1].to(self.device)
                
                logits = self.model(imgs)
                preds = logits.argmax(dim=1)
                
                metrics.update(preds, masks)
        
        miou = metrics.get_miou()
        self.results.append(miou)
        
        lg.info(f"Epoch {epoch}: mIoU = {miou*100:.2f}%")
        
        return miou
    
    def print_results(self, task_id):
        """Print evaluation results (in percentage).
        
        Args:
            task_id: Current task ID
        """
        n_dashes = 20
        pad_size = 8
        
        lg.info('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        lg.info('-' * n_dashes + "FORGETTING (%)" + '-' * n_dashes)
        for line in self.results_forgetting:
            lg.info(' '.join(f'{v*100:5.2f}'.ljust(pad_size) if not np.isnan(v) else 'nan'.ljust(pad_size) 
                           for v in line) + f" {np.nanmean(line)*100:5.2f}%")
        
        lg.info('-' * n_dashes + "mIoU (%)" + '-' * n_dashes)
        for line in self.results:
            lg.info(' '.join(f'{v*100:5.2f}'.ljust(pad_size) if not np.isnan(v) else 'nan'.ljust(pad_size)
                           for v in line) + f" {np.nanmean(line)*100:5.2f}%")
            
    def save_results(self):
        """Save results to files."""
        results_dir = os.path.join(
            self.params.results_root,
            self.params.tag,
            f"run{self.params.seed}"
        )
        lg.info(f"Saving results in: {results_dir}")
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        # Save mIoU results
        cols = [f'task {i}' for i in range(self.params.n_tasks)]
        df = pd.DataFrame(self.results, columns=cols)
        df['avg'] = df.mean(axis=1)
        df.to_csv(os.path.join(results_dir, 'miou.csv'), index=False)
        
        # Save forgetting results
        df_fgt = pd.DataFrame(self.results_forgetting, columns=cols)
        df_fgt['avg'] = df_fgt.mean(axis=1)
        df_fgt.to_csv(os.path.join(results_dir, 'forgetting.csv'), index=False)
        
        # Save parameters
        self.save_parameters()
        
        # Print final summary
        self.continual_metrics.print_results('miou')
        
    def save_results_offline(self):
        """Save results for offline training."""
        results_dir = os.path.join(
            self.params.results_root,
            self.params.tag,
            f"run{self.params.seed}" if self.params.run_id is None else f"run{self.params.run_id}"
        )
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            
        pd.DataFrame(self.results).to_csv(
            os.path.join(results_dir, 'miou.csv'),
            index=False
        )
        self.save_parameters()
        
    def save_parameters(self):
        """Save training parameters."""
        filename = os.path.join(
            self.params.results_root,
            self.params.tag,
            f"run{self.params.seed}/params.json"
        )
        with open(filename, 'w') as f:
            json.dump(self.params.__dict__, f, indent=2, default=str)
            
    def before_eval(self, **kwargs):
        """Hook called before evaluation."""
        pass
    
    def after_eval(self, **kwargs):
        """Hook called after evaluation."""
        pass
    
    def before_task(self, task_id, **kwargs):
        """Hook called before training on a new task.
        
        Args:
            task_id: ID of the upcoming task
        """
        pass
    
    def after_task(self, task_id, **kwargs):
        """Hook called after training on a task.
        
        Args:
            task_id: ID of the completed task
        """
        pass
    
    def combine(self, batch_x, batch_y, mem_x, mem_y):
        """Combine stream and memory data.
        
        Order is ``[stream, replay]`` so callers that slice with
        ``stream_size = batch_x.size(0)`` (e.g. ER+EMA+Attention KD) align
        stream logits/labels with ``batch_y_original`` from the dataloader.
        
        Args:
            batch_x: Stream images
            batch_y: Stream masks
            mem_x: Memory images
            mem_y: Memory masks
            
        Returns:
            combined_x: Combined images
            combined_y: Combined masks
        """
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        
        combined_x = torch.cat([batch_x, mem_x])
        combined_y = torch.cat([batch_y, mem_y])
        
        if self.params.memory_only:
            return mem_x, mem_y
        return combined_x, combined_y
    
    def update_seen_classes(self, masks):
        """Update set of seen classes from masks.
        
        Args:
            masks: Batch of segmentation masks
        """
        unique = torch.unique(masks)
        for c in unique.cpu().numpy():
            if c != self.params.ignore_index and c < self.params.n_classes:
                self.seen_classes.add(int(c))

