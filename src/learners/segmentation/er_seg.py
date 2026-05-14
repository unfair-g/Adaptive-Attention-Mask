"""Experience Replay learner for semantic segmentation.

This module implements ER-based continual learning for semantic
segmentation with reservoir sampling memory buffer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import logging as lg

from copy import deepcopy

from src.learners.segmentation.base_seg import BaseSegmentationLearner
from src.buffers.seg_reservoir import SegmentationReservoir, SegmentationLogitsReservoir
from src.utils.utils import get_device


device = get_device()


class ERSegmentationLearner(BaseSegmentationLearner):
    """Experience Replay learner for semantic segmentation.
    
    Implements the basic ER approach with reservoir sampling
    for memory buffer management.
    
    Args:
        args: Configuration arguments
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # Initialize memory buffer
        self.buffer = SegmentationReservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method
        )
        self._data_validated = False
        
    def _validate_data(self, batch_x, batch_y):
        """Validate that data is properly normalized (for debugging)."""
        if self._data_validated:
            return
        
        # Check image statistics
        mean = batch_x.mean(dim=[0, 2, 3])
        std = batch_x.std(dim=[0, 2, 3])
        
        # ImageNet normalized images should have mean close to 0 and std close to 1
        # (after normalization with ImageNet mean/std)
        expected_mean_range = (-0.5, 0.5)
        expected_std_range = (0.5, 2.0)
        
        is_normalized = (
            all(expected_mean_range[0] < m < expected_mean_range[1] for m in mean) and
            all(expected_std_range[0] < s < expected_std_range[1] for s in std)
        )
        
        print(f"\n[Data Validation]")
        print(f"  Image shape: {batch_x.shape}")
        print(f"  Image mean per channel: {mean.tolist()}")
        print(f"  Image std per channel: {std.tolist()}")
        print(f"  Image min/max: {batch_x.min():.3f} / {batch_x.max():.3f}")
        print(f"  Properly normalized: {is_normalized}")
        
        # Check labels
        unique_labels = batch_y.unique()
        print(f"  Unique labels: {unique_labels.tolist()}")
        
        if not is_normalized:
            print("  WARNING: Data appears NOT normalized for ImageNet pretrained model!")
            print("  Expected: mean close to 0, std close to 1 (after ImageNet normalization)")
            print("  If images are in [0,1] range, they need Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])")
        
        self._data_validated = True
        print()
    
    def train(self, dataloader, **kwargs):
        """Training loop with experience replay.
        
        Args:
            dataloader: Training data loader
            **kwargs: Additional arguments (task_name, task_id, etc.)
        """
        task_name = kwargs.get('task_name', 'unknown')
        task_id = kwargs.get('task_id', 0)
        
        self.model.train()
        
        for j, batch in enumerate(dataloader):
            # Validate data normalization on first batch
            if j == 0:
                self._validate_data(batch[0], batch[1])
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            # Update seen classes
            self.update_seen_classes(batch_y)
            
            for _ in range(self.params.mem_iters):
                # Retrieve from memory
                mem_x, mem_y = self.buffer.random_retrieve(
                    n_imgs=self.params.mem_batch_size
                )
                
                if mem_x.size(0) > 0:
                    # Combine stream and memory data
                    combined_x, combined_y = self.combine(
                        batch_x, batch_y, mem_x, mem_y
                    )
                    
                    # Apply augmentations (for images only)
                    # Note: For segmentation, we need to apply same transform to both
                    # For now, skip augmentation or use aligned transforms
                    
                    # Forward pass
                    logits = self.model(combined_x)
                    
                    # Loss
                    loss = self.criterion(logits, combined_y)
                    self.loss = loss.item()
                    
                    # Backward pass
                    self.optim.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    if self.params.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.params.grad_clip
                        )
                    
                    self.optim.step()
                else:
                    # No memory samples yet - train on stream only
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)
                    self.loss = loss.item()
                    
                    self.optim.zero_grad()
                    loss.backward()
                    
                    if self.params.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.params.grad_clip
                        )
                    
                    self.optim.step()
            
            # Update memory buffer
            self.buffer.update(
                imgs=batch_x.cpu(),
                masks=batch_y.cpu()
            )
            
            # Logging
            if j % 10 == 0 or j == len(dataloader) - 1:
                print(
                    f"Task: {task_name}  batch {j}/{len(dataloader)}  "
                    f"Loss: {self.loss:.4f}  time: {time.time() - self.start:.1f}s",
                    end="\r"
                )
        
        print()  # New line after training
        
    def before_task(self, task_id, **kwargs):
        """Called before training on a new task."""
        task_classes = kwargs.get('task_classes', [])
        self.task_classes[task_id] = task_classes
        self.continual_metrics.set_task_classes(task_id, task_classes)


class ERSegmentationDERppLearner(BaseSegmentationLearner):
    """Dark Experience Replay++ for semantic segmentation.
    
    Extends ER with knowledge distillation from stored logits.
    
    Args:
        args: Configuration arguments
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # Use buffer that stores logits
        self.buffer = SegmentationLogitsReservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method
        )
        
        self.alpha = self.params.derpp_alpha
        self.beta = self.params.derpp_beta
        
    def train(self, dataloader, **kwargs):
        """Training with DER++ style knowledge distillation.
        
        Args:
            dataloader: Training data loader
            **kwargs: Additional arguments
        """
        task_name = kwargs.get('task_name', 'unknown')
        task_id = kwargs.get('task_id', 0)
        
        self.model.train()
        
        for j, batch in enumerate(dataloader):
            batch_x, batch_y = batch[0], batch[1]
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.stream_idx += len(batch_x)
            self.update_seen_classes(batch_y)
            
            for _ in range(self.params.mem_iters):
                # Get current predictions for stream data
                stream_logits = self.model(batch_x)
                
                # CE loss on stream
                loss_ce = self.criterion(stream_logits, batch_y)
                
                # Retrieve from memory with logits
                mem_x, mem_y, mem_logits = self.buffer.random_retrieve_with_logits(
                    n_imgs=self.params.mem_batch_size
                )
                
                loss_kd = 0
                loss_mem = 0
                
                if mem_x.size(0) > 0:
                    mem_x = mem_x.to(self.device)
                    mem_y = mem_y.to(self.device)
                    mem_logits = mem_logits.to(self.device)
                    
                    # Get current predictions on memory
                    current_logits = self.model(mem_x)
                    
                    # CE loss on memory
                    loss_mem = self.criterion(current_logits, mem_y)
                    
                    # KD loss (MSE on logits)
                    # Upsample stored logits to match current
                    mem_logits_up = F.interpolate(
                        mem_logits,
                        size=current_logits.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    loss_kd = F.mse_loss(current_logits, mem_logits_up)
                
                # Combined loss
                loss = loss_ce + self.alpha * loss_mem + self.beta * loss_kd
                self.loss = loss.item()
                
                # Backward pass
                self.optim.zero_grad()
                loss.backward()
                
                if self.params.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.params.grad_clip
                    )
                
                self.optim.step()
            
            # Update buffer with current logits
            with torch.no_grad():
                store_logits = self.model(batch_x).detach()
            
            self.buffer.update(
                imgs=batch_x.cpu(),
                masks=batch_y.cpu(),
                logits=store_logits.cpu()
            )
            
            if j % 10 == 0 or j == len(dataloader) - 1:
                print(
                    f"Task: {task_name}  batch {j}/{len(dataloader)}  "
                    f"Loss: {self.loss:.4f}  time: {time.time() - self.start:.1f}s",
                    end="\r"
                )
        
        print()


class ERSegmentationEMALearner(BaseSegmentationLearner):
    """ER with Exponential Moving Average teacher for segmentation.
    
    Uses an EMA model as teacher for knowledge distillation.
    Supports proper handling of old classes in all_unknown_as_background mode.
    
    Args:
        args: Configuration arguments
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        self.buffer = SegmentationReservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method
        )
        
        # Initialize EMA model
        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
        self.ema_alpha = self.params.ema_alpha
        self.kd_weight = self.params.alpha_kd
        self._data_validated = False
        
        # Track old classes for KD masking / old-class-only supervision
        self.old_classes = []
        self.current_classes = []
        self.background_class = getattr(self.params, 'background_class', None)
        self.label_mode = getattr(self.params, 'label_mode', 'current_only')
        
        # === Class-Balanced Buffer Retrieval ===
        self.use_balanced_sampling = getattr(self.params, 'use_balanced_sampling', True)
        self.balanced_sampling_mode = getattr(self.params, 'balanced_sampling_mode', 'old_classes')
    
    def _validate_data(self, batch_x, batch_y):
        """Validate that data is properly normalized (for debugging)."""
        if self._data_validated:
            return
        
        # Check image statistics
        mean = batch_x.mean(dim=[0, 2, 3])
        std = batch_x.std(dim=[0, 2, 3])
        
        print(f"\n[Data Validation]")
        print(f"  Image shape: {batch_x.shape}")
        print(f"  Image mean per channel: {mean.tolist()}")
        print(f"  Image std per channel: {std.tolist()}")
        print(f"  Image min/max: {batch_x.min():.3f} / {batch_x.max():.3f}")
        
        # Check if in ImageNet normalized range (-2.5 to 2.7 approx)
        is_imagenet_range = batch_x.min() < -1.5 and batch_x.max() > 1.5
        print(f"  In ImageNet normalized range: {is_imagenet_range}")
        
        # Check labels
        unique_labels = batch_y.unique()
        print(f"  Unique labels: {unique_labels.tolist()}")
        print(f"  Label shape: {batch_y.shape}")
        print(f"  Label dtype: {batch_y.dtype}")
        
        # Test forward pass and loss
        self.model.eval()
        with torch.no_grad():
            batch_x_dev = batch_x.to(self.device)
            batch_y_dev = batch_y.to(self.device)
            
            logits = self.model(batch_x_dev)
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits min/max: {logits.min():.3f} / {logits.max():.3f}")
            print(f"  Logits mean: {logits.mean():.3f}")
            
            # Check if shapes match
            if logits.shape[2:] != batch_y_dev.shape[1:]:
                print(f"  WARNING: Shape mismatch! Logits: {logits.shape[2:]}, Labels: {batch_y_dev.shape[1:]}")
            
            # Calculate loss components
            loss = self.criterion(logits, batch_y_dev)
            print(f"  Test loss: {loss.item():.4f}")
            
            # Count valid pixels (not ignore_index)
            valid_pixels = (batch_y_dev != 255).sum().item()
            total_pixels = batch_y_dev.numel()
            print(f"  Valid pixels: {valid_pixels} / {total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
            
            # Check predictions
            preds = logits.argmax(dim=1)
            print(f"  Predicted labels: {preds.unique().tolist()}")
            
            # Check if model outputs are reasonable
            probs = F.softmax(logits, dim=1)
            max_probs = probs.max(dim=1)[0]
            print(f"  Max probability mean: {max_probs.mean():.4f}")
        
        self.model.train()
        self._data_validated = True
        print()
    
    def before_task(self, task_id, **kwargs):
        """Called before training on a new task. Track old/current classes."""
        task_classes = kwargs.get('task_classes', [])
        self.task_classes[task_id] = task_classes
        self.continual_metrics.set_task_classes(task_id, task_classes)
        
        # Update old and current classes
        self.current_classes = list(task_classes)
        if task_id > 0:
            # Old classes = all previously seen classes, EXCLUDING background class
            # Background (class 19) should not be treated as an "old class"
            self.old_classes = [c for c in self.seen_classes 
                               if c != self.background_class and c != self.params.ignore_index]
        else:
            self.old_classes = []
        
        lg.info(f"Task {task_id}: current_classes={self.current_classes}, old_classes={self.old_classes}")
        
    def update_ema(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data = (
                    self.ema_alpha * ema_param.data +
                    (1 - self.ema_alpha) * param.data
                )
    
    def _create_teacher_annotated_labels(self, labels, teacher_preds, teacher_probs):
        """Pseudo-label recovery is disabled.

        We keep the original labels as-is (ignored old pixels stay `ignore_index`),
        so teacher is used only for logit KD (not for hard label recovery).
        """
        return labels.clone()
    
    def _minority_balanced_retrieve(self, n_imgs: int = 4):
        """Retrieve samples with priority given to minority classes.
        
        优先采样小样本类别，基于逆频率权重采样。
        """
        import random as r
        
        if self.buffer.n_added_so_far == 0:
            return torch.Tensor(), torch.Tensor()
        
        n_available = min(self.buffer.n_added_so_far, self.buffer.max_size)
        class_counts = self.buffer.buffer_class_counts[:n_available]
        total_per_class = class_counts.sum(dim=0).float()
        total_pixels = total_per_class.sum()
        
        if total_pixels > 0:
            class_weights = total_pixels / (total_per_class + 1.0)
            class_weights = class_weights / class_weights.sum()
        else:
            return self.buffer.random_retrieve(n_imgs)
        
        sample_weights = torch.zeros(n_available)
        for i in range(n_available):
            sample_class_counts = class_counts[i].float()
            if sample_class_counts.sum() > 0:
                sample_weights[i] = (sample_class_counts * class_weights).sum()
        
        if sample_weights.sum() > 0:
            sample_weights = sample_weights / sample_weights.sum()
        else:
            sample_weights = torch.ones(n_available) / n_available
        
        sample_weights_np = sample_weights.cpu().numpy()
        try:
            selected_indices = np.random.choice(
                n_available, size=min(n_imgs, n_available),
                replace=False, p=sample_weights_np
            )
        except ValueError:
            selected_indices = r.sample(range(n_available), min(n_imgs, n_available))
        
        return self.buffer.buffer_imgs[selected_indices], self.buffer.buffer_masks[selected_indices]
    
    def _compute_loss_with_teacher_annotations(self, student_logits, teacher_logits,
                                                annotated_labels, teacher_probs=None,
                                                original_labels=None):
        """Compute loss with teacher used only for soft distillation.
        
        Pseudo-label / hard label recovery is disabled, so `annotated_labels`
        should match the original labels (ignored old pixels remain ignored).
        
        Training strategy for current_only mode:
        1. Focal Loss on current class pixels (focus on hard examples for new classes)
           - Formula: L_CE^cur = -α(1-p_t)^γ log(p_t)
        2. CE loss on old class pixels becomes inactive when old-pixel recovery is disabled
        3. KD loss on ALL non-current-class pixels (including ignored pixels!)
           - Teacher provides soft targets, not hard recovered labels
        
        Args:
            student_logits: Student model logits (B, C, H, W)
            teacher_logits: Teacher model logits (B, C, H, W)
            annotated_labels: Labels with old classes annotated by teacher
            teacher_probs: Teacher softmax probabilities (B, C, H, W), optional
            original_labels: Original labels before teacher annotation (for KD mask)
            
        Returns:
            loss_ce: Cross-entropy loss on annotated labels (Focal for current, CE for old)
            loss_kd: Knowledge distillation loss
        """
        T = self.params.kd_temperature
        ignore_index = self.params.ignore_index
        
        # Focal Loss parameters for current classes
        focal_alpha = getattr(self.params, 'focal_alpha', 1.0)  # Class weighting factor
        focal_gamma = getattr(self.params, 'focal_gamma', 2.0)  # Focusing parameter
        
        # Compute teacher probs if not provided
        if teacher_probs is None and teacher_logits is not None:
            teacher_probs = F.softmax(teacher_logits, dim=1)
        
        # Ignore mask (after annotation)
        ignore_mask = (annotated_labels == ignore_index)
        
        # Mask for current task classes
        current_class_mask = torch.zeros_like(annotated_labels, dtype=torch.bool)
        for c in self.current_classes:
            current_class_mask = current_class_mask | (annotated_labels == c)
        
        # Mask for old classes (will be empty if hard label recovery is disabled)
        old_class_mask = torch.zeros_like(annotated_labels, dtype=torch.bool)
        for c in self.old_classes:
            old_class_mask = old_class_mask | (annotated_labels == c)
        
        # === FOCAL LOSS for current classes, CE for old classes ===
        # Focal Loss: L = -α(1-p_t)^γ log(p_t)
        # This focuses on hard examples and down-weights easy ones for new classes
        
        # Compute standard CE loss (used for old classes)
        ce_loss_raw = F.cross_entropy(
            student_logits, annotated_labels,
            ignore_index=ignore_index,
            reduction='none'
        )
        
        # CE loss on current class pixels using FOCAL LOSS with CLASS BALANCING
        # Formula: L_CE^cur = -α_c(1-p_t)^γ log(p_t)
        # α_c = inverse frequency weight for class c (minority classes get higher weight)
        current_ce_mask = current_class_mask & (~ignore_mask)
        if current_ce_mask.any():
            # Get predicted probabilities
            student_probs = F.softmax(student_logits, dim=1)
            
            # Get p_t (probability of the true class) for each pixel
            # Clone labels and set ignored pixels to 0 for gather operation
            labels_for_gather = annotated_labels.clone()
            labels_for_gather[~current_ce_mask] = 0
            
            # Gather probabilities for ground truth classes: p_t
            pt = student_probs.gather(1, labels_for_gather.unsqueeze(1)).squeeze(1)  # (B, H, W)
            
            # Compute focal weight: (1 - p_t)^γ
            focal_weight = (1 - pt) ** focal_gamma
            
            # === CLASS BALANCING for current task classes ===
            # Compute per-class inverse frequency weights to handle class imbalance
            # Minority classes (small objects) get higher weight
            use_class_balance = getattr(self.params, 'use_current_class_balance', True)
            class_balance_power = getattr(self.params, 'class_balance_power', 1.0)  # 1.0=linear, 0.5=sqrt
            class_balance_max = getattr(self.params, 'class_balance_max', 20.0)  # Max weight for minority class
            
            if use_class_balance and len(self.current_classes) > 1:
                # Compute class frequencies in this batch
                class_weights = torch.ones_like(annotated_labels, dtype=torch.float32)
                total_current_pixels = current_ce_mask.sum().float()
                
                # Debug: track weights for logging
                weight_info = {}
                
                for c in self.current_classes:
                    class_mask = (annotated_labels == c) & current_ce_mask
                    class_count = class_mask.sum().float()
                    
                    if class_count > 0:
                        # Inverse frequency weight: w_c = (total / (num_classes * count_c))^power
                        # power=1.0: linear (strong balancing)
                        # power=0.5: sqrt (moderate balancing)
                        inv_freq = total_current_pixels / (len(self.current_classes) * class_count)
                        # Apply power scaling
                        inv_freq = inv_freq ** class_balance_power
                        # Clamp to reasonable range to avoid extreme values
                        inv_freq = torch.clamp(inv_freq, min=0.5, max=class_balance_max)
                        class_weights[class_mask] = inv_freq
                        weight_info[c] = (class_count.item(), inv_freq.item())
                
                # Log class weights periodically (every 100 batches)
                if hasattr(self, '_balance_log_counter'):
                    self._balance_log_counter += 1
                else:
                    self._balance_log_counter = 0
                
                if self._balance_log_counter % 100 == 0 and weight_info:
                    weight_str = ", ".join([f"c{c}: {cnt:.0f}px->w{w:.2f}" 
                                           for c, (cnt, w) in sorted(weight_info.items())])
                    lg.debug(f"[ClassBalance] {weight_str}")
                
                # Apply class-balanced focal loss
                focal_loss_raw = focal_alpha * class_weights * focal_weight * ce_loss_raw
            else:
                # Standard focal loss without class balancing
                focal_loss_raw = focal_alpha * focal_weight * ce_loss_raw
            
            # Average only over current class pixels
            loss_ce_current = (focal_loss_raw * current_ce_mask.float()).sum() / current_ce_mask.sum().clamp(min=1)
        else:
            loss_ce_current = torch.tensor(0.0, device=student_logits.device)
        
        # CE loss on old class pixels (teacher-annotated)
        old_ce_mask = old_class_mask & (~ignore_mask)
        if old_ce_mask.any():
            loss_ce_old = (ce_loss_raw * old_ce_mask.float()).sum() / old_ce_mask.sum().clamp(min=1)
        else:
            loss_ce_old = torch.tensor(0.0, device=student_logits.device)
        
        # Combined CE loss with higher weight for current classes
        # This ensures new classes get enough learning signal
        current_class_weight = 2.0  # Give current classes 2x weight
        loss_ce = current_class_weight * loss_ce_current + loss_ce_old
        
        # === KD Loss on ALL non-current-class pixels ===
        # KEY CHANGE: Apply KD on BOTH:
        #   1. Non-current-class pixels (ignored pixels included)
        #   2. Still-ignored pixels (where teacher wasn't confident enough)
        # Teacher provides soft distillation targets on masked regions
        # IMPORTANT: Still exclude current class pixels to allow learning new classes
        if self.label_mode == 'current_only' and original_labels is not None:
            # In current_only: KD on all originally-ignored pixels (old + future classes)
            # This includes both recovered and still-ignored pixels
            originally_ignored = (original_labels == ignore_index)
            kd_mask = originally_ignored  # All non-current-class pixels
        else:
            # Fallback: old-class-only KD mask derived from `annotated_labels`
            # (hard label recovery is disabled, so in `current_only` this typically
            # reduces to KD on masked/ignored regions).
            kd_mask = old_class_mask & (~ignore_mask)
        
        if kd_mask.any() and teacher_probs is not None:
            student_log_probs = F.log_softmax(student_logits / T, dim=1)
            teacher_soft = teacher_probs
            if T != 1.0 and teacher_logits is not None:
                teacher_soft = F.softmax(teacher_logits / T, dim=1)
            
            # KL divergence per pixel
            kd_loss_raw = F.kl_div(student_log_probs, teacher_soft, reduction='none')
            kd_loss_raw = kd_loss_raw.sum(dim=1)  # Sum over classes
            
            # Apply mask - ONLY on KD-selected pixels (no hard label recovery)
            loss_kd = (kd_loss_raw * kd_mask.float()).sum() / kd_mask.sum().clamp(min=1)
            loss_kd = loss_kd * (T ** 2)
        else:
            loss_kd = torch.tensor(0.0, device=student_logits.device)
        
        return loss_ce, loss_kd
                
    def train(self, dataloader, **kwargs):
        """Training with EMA teacher.
        
        Args:
            dataloader: Training data loader
            **kwargs: Additional arguments
        """
        task_name = kwargs.get('task_name', 'unknown')
        task_id = kwargs.get('task_id', 0)
        
        self.model.train()
        self.ema_model.eval()
        
        for j, batch in enumerate(dataloader):
            batch_x, batch_y = batch[0], batch[1]
            
            # Validate data normalization on first batch
            if j == 0:
                self._validate_data(batch_x, batch_y)
            
            self.stream_idx += len(batch_x)
            self.update_seen_classes(batch_y)
            
            for _ in range(self.params.mem_iters):
                # === Class-Balanced Buffer Retrieval ===
                if self.use_balanced_sampling and self.buffer.n_added_so_far > 0:
                    if self.balanced_sampling_mode == 'old_classes' and len(self.old_classes) > 0:
                        mem_x, mem_y = self.buffer.class_balanced_retrieve(
                            n_imgs=self.params.mem_batch_size,
                            target_classes=self.old_classes
                        )
                    elif self.balanced_sampling_mode == 'all_seen' and len(self.seen_classes) > 0:
                        mem_x, mem_y = self.buffer.class_balanced_retrieve(
                            n_imgs=self.params.mem_batch_size,
                            target_classes=list(self.seen_classes)
                        )
                    elif self.balanced_sampling_mode == 'minority':
                        mem_x, mem_y = self._minority_balanced_retrieve(
                            n_imgs=self.params.mem_batch_size
                        )
                    else:
                        mem_x, mem_y = self.buffer.random_retrieve(
                            n_imgs=self.params.mem_batch_size
                        )
                else:
                    mem_x, mem_y = self.buffer.random_retrieve(
                        n_imgs=self.params.mem_batch_size
                    )
                
                if mem_x.size(0) > 0:
                    combined_x, combined_y = self.combine(
                        batch_x, batch_y, mem_x, mem_y
                    )
                    
                    # Get student predictions
                    student_logits = self.model(combined_x)
                    
                    # Get EMA teacher predictions (slow-updating teacher preserves old knowledge)
                    with torch.no_grad():
                        teacher_logits = self.ema_model(combined_x)
                        teacher_probs = F.softmax(teacher_logits, dim=1)
                    
                    # === No pseudo-label recovery ===
                    # Teacher is used only for KD loss; labels remain unchanged.
                    if len(self.old_classes) > 0:
                        # Store original labels before "annotation" (used by KD mask in current_only mode)
                        original_labels = combined_y.clone()
                        annotated_labels = combined_y

                        loss_ce, loss_kd = self._compute_loss_with_teacher_annotations(
                            student_logits,
                            teacher_logits,
                            annotated_labels,
                            teacher_probs,
                            original_labels=original_labels,
                        )
                    else:
                        # First task: standard CE + KD loss
                        loss_ce = self.criterion(student_logits, combined_y)
                        
                        # KD loss - use 'none' reduction and manually average over all dimensions
                        kd_loss_raw = F.kl_div(
                            F.log_softmax(student_logits / self.params.kd_temperature, dim=1),
                            F.softmax(teacher_logits / self.params.kd_temperature, dim=1),
                            reduction='none'
                        )
                        loss_kd = kd_loss_raw.mean() * (self.params.kd_temperature ** 2)
                    
                    # Combined loss
                    loss = loss_ce + self.kd_weight * loss_kd
                    self.loss = loss.item()
                    
                    self.optim.zero_grad()
                    loss.backward()
                    
                    if self.params.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.params.grad_clip
                        )
                    
                    self.optim.step()
                    
                    # Update EMA
                    self.update_ema()
                else:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)
                    self.loss = loss.item()
                    
                    self.optim.zero_grad()
                    loss.backward()
                    
                    if self.params.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.params.grad_clip
                        )
                    
                    self.optim.step()
                    self.update_ema()
            
            self.buffer.update(
                imgs=batch_x.cpu(),
                masks=batch_y.cpu()
            )
            
            if j % 10 == 0 or j == len(dataloader) - 1:
                print(
                    f"Task: {task_name}  batch {j}/{len(dataloader)}  "
                    f"Loss: {self.loss:.4f}  time: {time.time() - self.start:.1f}s",
                    end="\r"
                )
        
        print()
    
    def _create_averaged_model(self):
        """Create a model with averaged parameters from teacher and student."""
        avg_model = deepcopy(self.model)
        with torch.no_grad():
            for avg_param, student_param, teacher_param in zip(
                avg_model.parameters(),
                self.model.parameters(),
                self.ema_model.parameters()
            ):
                # Average of student and teacher parameters
                avg_param.data = (student_param.data + teacher_param.data) / 2.0
        return avg_model
    
    def evaluate(self, dataloaders, task_id, **kwargs):
        """Evaluate using specified model.
        
        eval_mode options:
            'student': use student model (default)
            'teacher': use EMA teacher model  
            'avg': use averaged model (student + teacher) / 2
        """
        eval_mode = getattr(self.params, 'eval_mode', 'student')
        
        # Backward compatibility with eval_teacher
        if getattr(self.params, 'eval_teacher', False) and eval_mode == 'student':
            eval_mode = 'teacher'
        
        if eval_mode == 'teacher':
            # Use teacher model for evaluation
            original_model = self.model
            self.model = self.ema_model
            result = super().evaluate(dataloaders, task_id, **kwargs)
            self.model = original_model
            return result
        elif eval_mode == 'avg':
            # Use averaged model for evaluation
            original_model = self.model
            self.model = self._create_averaged_model()
            self.model.eval()
            result = super().evaluate(dataloaders, task_id, **kwargs)
            self.model = original_model
            return result
        else:
            # Default: use student model
            return super().evaluate(dataloaders, task_id, **kwargs)


class ACESegmentationLearner(BaseSegmentationLearner):
    """Asymmetric Cross-Entropy learner for segmentation.
    
    Uses modified cross-entropy that handles old and new classes
    asymmetrically for better stability.
    
    Args:
        args: Configuration arguments
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        self.buffer = SegmentationReservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method
        )
        
        self.old_classes = []
        self.new_classes = []
        
    def load_criterion(self):
        """Load ACE loss."""
        from src.utils.seg_losses import ACELoss
        return ACELoss(ignore_index=self.params.ignore_index)
    
    def before_task(self, task_id, **kwargs):
        """Update old/new class lists."""
        task_classes = kwargs.get('task_classes', [])
        self.task_classes[task_id] = task_classes
        
        # Update old/new classes
        self.new_classes = task_classes
        if task_id > 0:
            self.old_classes = list(self.seen_classes - set(task_classes))
        else:
            self.old_classes = []
            
    def train(self, dataloader, **kwargs):
        """Training with ACE loss."""
        task_name = kwargs.get('task_name', 'unknown')
        task_id = kwargs.get('task_id', 0)
        
        self.model.train()
        
        for j, batch in enumerate(dataloader):
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            self.update_seen_classes(batch_y)
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(
                    n_imgs=self.params.mem_batch_size
                )
                
                if mem_x.size(0) > 0:
                    combined_x, combined_y = self.combine(
                        batch_x, batch_y, mem_x, mem_y
                    )
                    
                    logits = self.model(combined_x)
                    
                    # ACE loss with class info
                    loss = self.criterion(
                        logits, combined_y,
                        old_classes=self.old_classes,
                        new_classes=self.new_classes
                    )
                    self.loss = loss.item()
                    
                    self.optim.zero_grad()
                    loss.backward()
                    
                    if self.params.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.params.grad_clip
                        )
                    
                    self.optim.step()
                else:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    logits = self.model(batch_x)
                    loss = self.criterion(
                        logits, batch_y,
                        old_classes=self.old_classes,
                        new_classes=self.new_classes
                    )
                    self.loss = loss.item()
                    
                    self.optim.zero_grad()
                    loss.backward()
                    
                    if self.params.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.params.grad_clip
                        )
                    
                    self.optim.step()
            
            self.buffer.update(
                imgs=batch_x.cpu(),
                masks=batch_y.cpu()
            )
            
            if j % 10 == 0 or j == len(dataloader) - 1:
                print(
                    f"Task: {task_name}  batch {j}/{len(dataloader)}  "
                    f"Loss: {self.loss:.4f}  time: {time.time() - self.start:.1f}s",
                    end="\r"
                )
        
        print()

