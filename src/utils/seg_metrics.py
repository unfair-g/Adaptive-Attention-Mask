"""Evaluation metrics for semantic segmentation.

This module provides metrics for evaluating semantic segmentation
performance in continual learning scenarios, including per-class
and aggregate metrics.
"""
import torch
import numpy as np
import pandas as pd
from collections import defaultdict


class SegmentationMetrics:
    """Accumulator for segmentation metrics.
    
    Computes IoU, accuracy, and related metrics for semantic segmentation.
    
    Args:
        n_classes: Number of segmentation classes
        ignore_index: Label to ignore in metric computation
        class_names: Optional list of class names for reporting
    """
    
    def __init__(self, n_classes, ignore_index=255, class_names=None):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f'class_{i}' for i in range(n_classes)]
        self.reset()
        
    def reset(self):
        """Reset accumulated metrics."""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        self.n_samples = 0
        
    def update(self, preds, targets):
        """Update metrics with new predictions.
        
        Args:
            preds: (B, H, W) predicted labels or (B, C, H, W) logits
            targets: (B, H, W) ground truth labels
        """
        # Handle logits input
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
            
        preds = preds.cpu().numpy().astype(np.int64)
        targets = targets.cpu().numpy().astype(np.int64)
        
        # Flatten arrays
        preds = preds.flatten()
        targets = targets.flatten()
        
        # Filter out ignored labels
        mask = targets != self.ignore_index
        preds = preds[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        valid_mask = (targets >= 0) & (targets < self.n_classes) & \
                    (preds >= 0) & (preds < self.n_classes)
        
        targets = targets[valid_mask]
        preds = preds[valid_mask]
        
        self.confusion_matrix += np.bincount(
            self.n_classes * targets + preds,
            minlength=self.n_classes ** 2
        ).reshape(self.n_classes, self.n_classes)
        
        self.n_samples += 1
        
    def get_iou(self):
        """Compute per-class IoU.
        
        Returns:
            iou: Array of per-class IoU values
        """
        # True positives
        tp = np.diag(self.confusion_matrix)
        # False positives
        fp = self.confusion_matrix.sum(axis=0) - tp
        # False negatives
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        # IoU per class
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = tp / (tp + fp + fn)
            iou = np.where(np.isnan(iou), 0, iou)
        
        return iou
    
    def get_miou(self, classes=None):
        """Compute mean IoU.
        
        Args:
            classes: Optional list of class indices to include
            
        Returns:
            miou: Mean IoU over specified classes
        """
        iou = self.get_iou()
        
        if classes is not None:
            iou = iou[classes]
        
        # Only average over classes present in ground truth
        valid = (self.confusion_matrix.sum(axis=1) > 0)
        if classes is not None:
            valid = valid[classes]
            iou = iou[valid[:len(iou)]]
        else:
            iou = iou[valid]
            
        if len(iou) == 0:
            return 0.0
            
        return iou.mean()
    
    def get_accuracy(self):
        """Compute pixel accuracy.
        
        Returns:
            accuracy: Overall pixel accuracy
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        
        if total == 0:
            return 0.0
        return correct / total
    
    def get_class_accuracy(self):
        """Compute per-class accuracy.
        
        Returns:
            acc: Array of per-class accuracy values
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
            acc = np.where(np.isnan(acc), 0, acc)
        return acc
    
    def get_mean_accuracy(self, classes=None):
        """Compute mean class accuracy.
        
        Args:
            classes: Optional list of class indices to include
            
        Returns:
            macc: Mean class accuracy
        """
        acc = self.get_class_accuracy()
        
        if classes is not None:
            acc = acc[classes]
            
        valid = acc > 0
        if valid.sum() == 0:
            return 0.0
            
        return acc[valid].mean()
    
    def get_frequency_weighted_iou(self):
        """Compute frequency-weighted IoU.
        
        Returns:
            fwiou: Frequency weighted IoU
        """
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        iou = self.get_iou()
        
        valid = freq > 0
        return (freq[valid] * iou[valid]).sum()
    
    def get_results(self, classes=None):
        """Get all metrics as a dictionary.
        
        Args:
            classes: Optional list of class indices to include
            
        Returns:
            results: Dictionary of metric values
        """
        iou = self.get_iou()
        
        results = {
            'miou': self.get_miou(classes),
            'pixel_acc': self.get_accuracy(),
            'mean_acc': self.get_mean_accuracy(classes),
            'fwiou': self.get_frequency_weighted_iou(),
        }
        
        # Add per-class IoU
        for i, name in enumerate(self.class_names):
            results[f'iou_{name}'] = iou[i]
            
        return results
    
    def get_results_df(self):
        """Get per-class metrics as DataFrame.
        
        Returns:
            df: DataFrame with per-class metrics
        """
        iou = self.get_iou()
        acc = self.get_class_accuracy()
        
        data = {
            'class': self.class_names,
            'iou': iou,
            'accuracy': acc,
            'n_pixels': self.confusion_matrix.sum(axis=1)
        }
        
        return pd.DataFrame(data)


def compute_miou(preds, targets, n_classes, ignore_index=255, classes=None):
    """Compute mean IoU for a batch.
    
    Args:
        preds: (B, H, W) predicted labels or (B, C, H, W) logits
        targets: (B, H, W) ground truth labels
        n_classes: Number of classes
        ignore_index: Label to ignore
        classes: Optional list of classes to include
        
    Returns:
        miou: Mean IoU value
    """
    metrics = SegmentationMetrics(n_classes, ignore_index)
    metrics.update(preds, targets)
    return metrics.get_miou(classes)


def compute_per_class_iou(preds, targets, n_classes, ignore_index=255):
    """Compute per-class IoU for a batch.
    
    Args:
        preds: (B, H, W) predicted labels or (B, C, H, W) logits
        targets: (B, H, W) ground truth labels
        n_classes: Number of classes
        ignore_index: Label to ignore
        
    Returns:
        iou: Array of per-class IoU values
    """
    metrics = SegmentationMetrics(n_classes, ignore_index)
    metrics.update(preds, targets)
    return metrics.get_iou()


class ContinualSegmentationMetrics:
    """Metrics tracker for continual learning scenarios.
    
    Tracks per-task metrics and computes forgetting measures.
    
    Args:
        n_classes: Number of segmentation classes
        n_tasks: Number of tasks
        ignore_index: Label to ignore
        class_names: Optional list of class names
    """
    
    def __init__(self, n_classes, n_tasks, ignore_index=255, class_names=None):
        self.n_classes = n_classes
        self.n_tasks = n_tasks
        self.ignore_index = ignore_index
        self.class_names = class_names or [f'class_{i}' for i in range(n_classes)]
        
        # Store metrics after each task
        self.task_metrics = {}  # task_id -> {old_task_id -> metrics}
        self.classes_per_task = {}  # task_id -> list of classes
        
    def set_task_classes(self, task_id, classes):
        """Set the classes for a task.
        
        Args:
            task_id: Task identifier
            classes: List of class indices for this task
        """
        self.classes_per_task[task_id] = classes
        
    def record_metrics(self, task_id, eval_task_id, metrics_dict):
        """Record metrics after evaluating on a task.
        
        Args:
            task_id: Current training task
            eval_task_id: Task being evaluated
            metrics_dict: Dictionary of metric values
        """
        if task_id not in self.task_metrics:
            self.task_metrics[task_id] = {}
        self.task_metrics[task_id][eval_task_id] = metrics_dict
        
    def get_forgetting(self, metric='miou'):
        """Compute forgetting for each task.
        
        Forgetting is defined as the difference between peak performance
        and current performance on old tasks.
        
        Args:
            metric: Which metric to use for forgetting computation
            
        Returns:
            forgetting: Array of forgetting values per task
        """
        forgetting = []
        
        for eval_task in range(self.n_tasks):
            # Find peak performance on this task
            peak = 0
            current = 0
            
            for train_task in sorted(self.task_metrics.keys()):
                if eval_task in self.task_metrics[train_task]:
                    value = self.task_metrics[train_task][eval_task].get(metric, 0)
                    if train_task >= eval_task:
                        peak = max(peak, value)
                    if train_task == max(self.task_metrics.keys()):
                        current = value
            
            if peak > 0:
                forgetting.append(peak - current)
            else:
                forgetting.append(0)
                
        return np.array(forgetting)
    
    def get_average_forgetting(self, metric='miou'):
        """Get average forgetting across all old tasks.
        
        Args:
            metric: Which metric to use
            
        Returns:
            avg_forgetting: Average forgetting value
        """
        forgetting = self.get_forgetting(metric)
        # Exclude last task (no forgetting possible)
        if len(forgetting) > 1:
            return forgetting[:-1].mean()
        return 0.0
    
    def get_backward_transfer(self, metric='miou'):
        """Compute backward transfer.
        
        BWT measures how learning new tasks affects old task performance.
        
        Args:
            metric: Which metric to use
            
        Returns:
            bwt: Backward transfer value
        """
        if len(self.task_metrics) < 2:
            return 0.0
            
        bwt = 0
        count = 0
        
        for i in range(self.n_tasks - 1):
            # Performance on task i after learning task i
            if i in self.task_metrics and i in self.task_metrics[i]:
                r_ii = self.task_metrics[i][i].get(metric, 0)
                
                # Performance on task i after learning all tasks
                final_task = max(self.task_metrics.keys())
                if i in self.task_metrics[final_task]:
                    r_Ti = self.task_metrics[final_task][i].get(metric, 0)
                    bwt += (r_Ti - r_ii)
                    count += 1
        
        if count > 0:
            return bwt / count
        return 0.0
    
    def get_forward_transfer(self, metric='miou'):
        """Compute forward transfer.
        
        FWT measures how learning old tasks helps with new tasks.
        
        Args:
            metric: Which metric to use
            
        Returns:
            fwt: Forward transfer value
        """
        # Would need zero-shot baseline to compute properly
        # Returning 0 as placeholder
        return 0.0
    
    def get_results_matrix(self, metric='miou'):
        """Get results as a matrix.
        
        Returns:
            matrix: n_tasks x n_tasks matrix where entry (i,j) is
                   performance on task j after training on task i
        """
        matrix = np.zeros((self.n_tasks, self.n_tasks))
        matrix.fill(np.nan)
        
        for train_task in self.task_metrics:
            for eval_task in self.task_metrics[train_task]:
                if train_task < self.n_tasks and eval_task < self.n_tasks:
                    matrix[train_task, eval_task] = \
                        self.task_metrics[train_task][eval_task].get(metric, np.nan)
        
        return matrix
    
    def print_results(self, metric='miou', scale=100, log_file=None):
        """Print formatted results and optionally save to log file.
        
        Args:
            metric: Which metric to print
            scale: Scale factor for display (default 100 for percentage)
            log_file: Optional path to log file to save results
        """
        import logging as lg
        
        matrix = self.get_results_matrix(metric)
        
        # Build result string
        result_lines = []
        result_lines.append(f"\n{'='*70}")
        result_lines.append(f"Results Matrix ({metric}, %):")
        result_lines.append(f"{'='*70}")
        
        # Header
        header = "Train\\Eval"
        for j in range(self.n_tasks):
            header += f"  Task{j}"
        header += "   Mean"
        result_lines.append(header)
        result_lines.append("-" * len(header))
        
        # Rows
        for i in range(self.n_tasks):
            row = f"Task{i}    "
            values = []
            for j in range(self.n_tasks):
                if j <= i and not np.isnan(matrix[i, j]):
                    val = matrix[i, j] * scale
                    row += f"  {val:5.2f}"
                    values.append(matrix[i, j])
                else:
                    row += "     -  "
            if values:
                row += f"   {np.mean(values) * scale:5.2f}"
            result_lines.append(row)
        
        forgetting = self.get_average_forgetting(metric) * scale
        bwt = self.get_backward_transfer(metric) * scale
        result_lines.append(f"\nForgetting: {forgetting:.2f}%")
        result_lines.append(f"Backward Transfer: {bwt:+.2f}%")
        result_lines.append(f"{'='*70}\n")
        
        # Print to console
        result_str = '\n'.join(result_lines)
        print(result_str)
        
        # Save to log file if provided
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(result_str)
        else:
            # Try to get log file from logger
            logger = lg.getLogger()
            for handler in logger.handlers:
                if isinstance(handler, lg.FileHandler):
                    handler.stream.write(result_str)
                    handler.stream.flush()
                    break


def segmentation_forgetting_line(results_df, task_id, n_tasks):
    """Compute forgetting line for segmentation (adapted from classification).
    
    Args:
        results_df: DataFrame with mIoU results per task
        task_id: Current task ID
        n_tasks: Total number of tasks
        
    Returns:
        forgetting: Series of forgetting values
    """
    if task_id == 0:
        return pd.Series([np.nan] * n_tasks)
    
    forgettings = []
    for p in range(task_id):
        # Find max difference between any previous step and current
        max_diff = 0
        for k in range(task_id + 1):
            diff = results_df.iloc[task_id - k, p] - results_df.iloc[task_id, p]
            if k < task_id:
                prev_diff = results_df.iloc[task_id - k - 1, p] - results_df.iloc[task_id, p]
                max_diff = max(max_diff, diff, prev_diff)
            else:
                max_diff = max(max_diff, diff)
        forgettings.append(max(0, max_diff))
    
    forgettings.extend([np.nan] * (n_tasks - task_id))
    return pd.Series(forgettings)

