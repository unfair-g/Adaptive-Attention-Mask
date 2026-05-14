"""ER with Attention Map Knowledge Distillation for Semantic Segmentation.

This module extends the EMA-based continual learning with attention map
distillation, specifically designed to preserve "where to look" knowledge
from the teacher model during online continual learning.

Version: v3 (Attention Map Distillation with Cosine Loss)

Key Design Principle:
- SegFormer is based on Transformer architecture with Self-Attention mechanism
- Teacher model trained on old tasks knows "where to look" to recognize objects
- Teacher's Attention Map (Query × Key) contains this knowledge
- By aligning Student and Teacher attention, student learns to focus on same regions

Formula:
    L_Attn = Σ_l weight_l × (1 - cosine_similarity(Attn_S^l, Attn_T^l))

Benefits:
- Forces student to "mimic teacher's gaze pattern"
- Scale-invariant (cosine loss)
- Particularly effective for classes that require contextual understanding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging as lg
import numpy as np

from copy import deepcopy
from typing import Optional, List, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.learners.segmentation.base_seg import BaseSegmentationLearner
from src.buffers.seg_reservoir import SegmentationReservoir
from src.utils.utils import get_device

from src_attention.models.segformer_attention import SegFormerAttention
from src_attention.utils.attention_distillation import AttentionMapDistillationLoss


device = get_device()


def _parse_mixed_replay_ratios(value) -> Tuple[int, int]:
    """Parse [w_tpr, w_minority] weights for mixed replay; default (1, 1)."""
    default = (1, 1)
    if value is None:
        return default
    if isinstance(value, str):
        for sep in ("-", ":", ","):
            if sep in value:
                parts = [p.strip() for p in value.replace(",", sep).split(sep) if p.strip()]
                if len(parts) == 2:
                    try:
                        value = [int(parts[0]), int(parts[1])]
                    except ValueError:
                        return default
                    break
        else:
            return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            a, b = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return default
        if a < 0 or b < 0 or (a == 0 and b == 0):
            return default
        return (a, b)
    return default


class ERSegmentationEMAAttentionLearner(BaseSegmentationLearner):
    """ER with Attention Map EMA Teacher for semantic segmentation.
    
    This learner uses attention map distillation to preserve the teacher's
    knowledge about "where to look" for recognizing objects.
    
    Architecture:
        Student Model ─────────────────────────────────────────────> Logits
            │                                                          │
            ├─ Stage 2 Attention ─> Attn Distill ─┐                   │
            └─ Stage 3 Attention ─> Attn Distill ─┼─> Attention KD    │
                      ↑                            │                   │
        Teacher Model (EMA) ──────────────────────┴────────────> Logit KD
    
    Key Features:
    1. Attention map distillation from last 2 stages (memory efficient)
    2. Uses COSINE SIMILARITY LOSS (not MSE) - scale invariant, meaningful gradients
    3. Logit-level KD for output alignment (on old classes only)
    4. EMA teacher for smooth knowledge preservation
    
    Args:
        args: Configuration arguments
    """
    
    def __init__(self, args):
        # Set model type to use attention SegFormer
        args.model_type = 'attention'
        super().__init__(args)
        
        # Memory buffer
        self.buffer = SegmentationReservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method
        )
        
        # Replace model with attention version
        self._init_attention_model()
        
        # Initialize EMA teacher (also attention model)
        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False
        
        # EMA and KD parameters
        self.ema_alpha = self.params.ema_alpha
        self.kd_weight = self.params.alpha_kd
        
        # Attention distillation parameters
        self.attention_kd_weight = getattr(self.params, 'attention_kd_weight', 1.0)
        
        # Active attention stages (default: last 2 stages for efficiency)
        self.active_attention_stages = getattr(
            self.params, 'active_attention_stages', [2, 3]
        )
        
        # Stage weights for attention distillation
        self.attention_stage_weights = getattr(
            self.params, 'attention_stage_weights', 
            [1.0, 1.5]  # Higher weight for later (more semantic) stage
        )
        
        # Attention loss type (cosine recommended)
        self.attention_loss_type = getattr(self.params, 'attention_loss_type', 'cosine')
        
        # When to start attention KD (default: task 1, after teacher has learned)
        self.attention_start_task = getattr(self.params, 'attention_start_task', 1)
        
        # Current task ID
        self.current_task_id = 0
        
        # Masked Attention Distillation (MAD) - New Class Protection
        # 核心思想: 不让教师模型指导它不懂的区域 (当前新类别)
        self.use_masked_distillation = getattr(self.params, 'use_masked_distillation', True)
        
        # Initialize attention distillation loss with COSINE (not MSE!)
        # STU-Mask hyperparameters from config
        stu_enabled = getattr(self.params, 'stu_enabled', True)
        stu_eps = getattr(self.params, 'stu_eps', 1e-8)
        stu_pool_size = getattr(self.params, 'stu_pool_size', 32)

        # Get STU mask parameters from config
        stu_min_mask_value = getattr(self.params, 'stu_min_mask_value', 0.0)
        stu_entropy_threshold = getattr(self.params, 'stu_entropy_threshold', 0.5)
        stu_margin_threshold = getattr(self.params, 'stu_margin_threshold', 1.5)
        stu_entropy_gamma = getattr(self.params, 'stu_entropy_gamma', 1.5)
        stu_entropy_use_power = getattr(self.params, 'stu_entropy_use_power', True)

        debug_mask_stats = getattr(self.params, 'debug_mask_stats', False)
        self.debug_mask_stats = bool(debug_mask_stats)

        attention_loss_kwargs = dict(
            num_active_stages=len(self.active_attention_stages),
            stage_weights=self.attention_stage_weights,
            loss_type=self.attention_loss_type,  # 'cosine' by default
            pool_size=stu_pool_size,
            head_aggregation=getattr(self.params, 'attention_head_aggregation', 'mean'),
            use_masked_distillation=self.use_masked_distillation,
            stu_enabled=stu_enabled,
            eps=stu_eps,
            min_mask_value=stu_min_mask_value,
            entropy_threshold=stu_entropy_threshold,
            margin_threshold=stu_margin_threshold,
            entropy_gamma=stu_entropy_gamma,
            stu_entropy_use_power=stu_entropy_use_power,
            mad_relaxed_threshold=getattr(self.params, 'mad_relaxed_threshold', 0.001),
            debug_mask_stats=debug_mask_stats,
            stu_fallback_enabled=getattr(self.params, 'stu_fallback_enabled', True),
            stu_fallback_ratio=getattr(self.params, 'stu_fallback_ratio', 0.1),
            stu_weight_mode=getattr(self.params, 'stu_weight_mode', 'global'),
            stu_margin_space=getattr(self.params, 'stu_margin_space', 'pooled'),
            stu_disable_margin_gate=getattr(self.params, 'stu_disable_margin_gate', False),
        )
        self.attention_distill_loss = AttentionMapDistillationLoss(**attention_loss_kwargs)

        lg.info(f"  - STU-Mask: enabled={stu_enabled}, eps={stu_eps}, pool_size={stu_pool_size}")
        lg.info(
            f"    → entropy_gamma={stu_entropy_gamma}, stu_entropy_use_power={stu_entropy_use_power}, "
            f"min_mask_value={stu_min_mask_value}"
        )
        lg.info(f"    → entropy_threshold={stu_entropy_threshold}")
        lg.info(f"    → margin_threshold={stu_margin_threshold}")
        lg.info(f"    → stu_margin_space={getattr(self.params, 'stu_margin_space', 'pooled')}")
        lg.info(f"    → stu_disable_margin_gate={getattr(self.params, 'stu_disable_margin_gate', False)}")
        lg.info(f"    → mad_relaxed_threshold={getattr(self.params, 'mad_relaxed_threshold', 0.001)}")
        lg.info(f"    → debug_mask_stats={self.debug_mask_stats}")
        
        # Class tracking
        self.old_classes = []
        self.current_classes = []
        
        self._data_validated = False
        
        # Class-balanced settings
        self.use_class_weights = getattr(self.params, 'use_class_weights', False)
        if self.use_class_weights:
            self._init_class_weights()
        
        # === Class-Balanced Buffer Retrieval ===
        self.use_balanced_sampling = getattr(self.params, 'use_balanced_sampling', True)
        self.balanced_sampling_mode = getattr(self.params, 'balanced_sampling_mode', 'old_classes')
        self.mixed_replay_ratios = _parse_mixed_replay_ratios(
            getattr(self.params, "mixed_replay_ratios", None)
        )
        
        # === Loss Type Selection (from offline version) ===
        # Options: 'ce', 'focal', 'ohem', 'focal_ohem'
        self.seg_loss_type = getattr(self.params, 'seg_loss_type', 'ce')
        self.focal_gamma = getattr(self.params, 'focal_gamma', 2.0)
        self.focal_alpha = getattr(self.params, 'focal_alpha', 0.25)
        self.ohem_ratio = getattr(self.params, 'ohem_ratio', 0.25)
        
        # === Adaptive Replay (from offline version) ===
        self.use_adaptive_replay = getattr(self.params, 'use_adaptive_replay', False)
        self.max_replay_multiplier = getattr(self.params, 'max_replay_multiplier', 3.0)
        
        # EMA update strategy is fixed to standard step-wise EMA.
        self.ema_update_strategy = 'step'
        
        # Prediction confidence threshold for evaluation
        self.pred_confidence_threshold = getattr(self.params, 'pred_confidence_threshold', None)
        
        lg.info(f"Initialized ERSegmentationEMAAttentionLearner (v3) with:")
        lg.info(f"  - Model variant: {self._get_model_variant()}")
        lg.info(f"  - Attention KD weight: {self.attention_kd_weight}")
        lg.info(f"  - Attention loss type: {self.attention_loss_type}")
        lg.info(f"  - Attention start task: {self.attention_start_task}")
        lg.info(f"  - Logit KD weight: {self.kd_weight}")
        lg.info(f"  - Logit KD: OLD CLASSES ONLY (prevents teacher misleading on new classes)")
        lg.info(f"  - Active attention stages: {self.active_attention_stages}")
        lg.info(f"  - Attention stage weights: {self.attention_stage_weights}")
        lg.info(f"  - Native attention supported: {self.model.attention_supported()}")
        lg.info(f"  - Masked Attention Distillation (MAD): {'enabled' if self.use_masked_distillation else 'disabled'}")
        if self.use_masked_distillation:
            lg.info(f"    → MAD excludes current class pixels from attention distillation")
        lg.info(f"  - Class-Balanced Buffer Retrieval: {'enabled' if self.use_balanced_sampling else 'disabled'}")
        if self.use_balanced_sampling:
            lg.info(f"    → Mode: {self.balanced_sampling_mode}")
            if self.balanced_sampling_mode == "mixed":
                w0, w1 = self.mixed_replay_ratios
                lg.info(
                    f"    → mixed_replay_ratios (target_pixel_ratio : minority) = [{w0}, {w1}]"
                )
        if self.pred_confidence_threshold:
            lg.info(f"  - Eval pred_confidence_threshold: {self.pred_confidence_threshold}")
        
        # Log loss type settings
        lg.info(f"  - Segmentation loss type: {self.seg_loss_type}")
        if self.seg_loss_type == 'focal':
            lg.info(f"    → Focal Loss: gamma={self.focal_gamma}, alpha={self.focal_alpha}")
        elif self.seg_loss_type == 'ohem':
            lg.info(f"    → OHEM: top_k={self.ohem_ratio*100}%")
        elif self.seg_loss_type == 'focal_ohem':
            lg.info(f"    → Focal + OHEM combined")
        
        # Log adaptive replay settings
        if self.use_adaptive_replay:
            lg.info(f"  - Adaptive replay: enabled (max_multiplier={self.max_replay_multiplier})")
        
        # Log EMA strategy
        lg.info(f"  - EMA update strategy: {self.ema_update_strategy}")
    
    def _init_class_weights(self):
        """Initialize class weights for balanced loss."""
        cityscapes_frequencies = torch.tensor([
            0.3685, 0.0529, 0.2149, 0.0072, 0.0080, 0.0145, 0.0019,
            0.0065, 0.1700, 0.0082, 0.0330, 0.0128, 0.0021, 0.0639,
            0.0030, 0.0011, 0.0014, 0.0008, 0.0070,
        ], dtype=torch.float32)
        
        epsilon = 1e-6
        inv_freq = 1.0 / (cityscapes_frequencies + epsilon)
        weights = torch.sqrt(inv_freq)
        weights = weights / weights.mean()
        weights = torch.clamp(weights, min=0.7, max=3.0)
        
        self.class_weights = weights.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=self.params.ignore_index
        )
        
        lg.info(f"Class weights initialized: min={weights.min():.2f}, max={weights.max():.2f}")
    
    def _get_model_variant(self):
        """Get the SegFormer model variant from config."""
        variant = getattr(self.params, 'segformer_variant', None)
        if variant and isinstance(variant, str):
            return variant
        
        pretrained = getattr(self.params, 'pretrained', None)
        if pretrained and isinstance(pretrained, str):
            return pretrained
        
        return 'mit_b0'
    
    def _init_attention_model(self):
        """Initialize SegFormer model with attention extraction."""
        variant = self._get_model_variant()
        lg.info(f"Initializing SegFormerAttention with variant: {variant}")
        
        self.model = SegFormerAttention(
            n_classes=self.params.n_classes,
            pretrained=variant,
            img_size=self.params.img_size
        ).to(self.device)
        
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params.learning_rate,
            weight_decay=getattr(self.params, 'weight_decay', 0.01)
        )
    
    def load_model(self):
        """Override to load SegFormer with attention."""
        variant = self._get_model_variant()
        return SegFormerAttention(
            n_classes=self.params.n_classes,
            pretrained=variant,
            img_size=self.params.img_size
        )
    
    def _validate_data(self, batch_x, batch_y):
        """Validate data and attention extraction."""
        if self._data_validated:
            return
        
        mean = batch_x.mean(dim=[0, 2, 3])
        std = batch_x.std(dim=[0, 2, 3])
        
        print(f"\n[Data Validation - Attention Distillation v3]")
        print(f"  Image shape: {batch_x.shape}")
        print(f"  Image mean: {mean.tolist()}")
        print(f"  Image std: {std.tolist()}")
        print(f"  Model variant: {self._get_model_variant()}")
        print(f"  Native attention supported: {self.model.attention_supported()}")
        print(f"  Attention loss type: {self.attention_loss_type}")
        
        self.model.eval()
        with torch.no_grad():
            batch_x_dev = batch_x[:1].to(self.device)
            try:
                logits, attentions, _attn_hw = self.model(
                    batch_x_dev, 
                    return_attention=True,
                    active_attention_stages=self.active_attention_stages
                )
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                print(f"  Number of attention stages extracted: {len(attentions)}")
                for i, attn in enumerate(attentions):
                    if attn is not None:
                        print(f"    Stage {self.active_attention_stages[i]}: {attn.shape}, "
                              f"range=[{attn.min().item():.4f}, {attn.max().item():.4f}]")
                    else:
                        print(f"    Stage {self.active_attention_stages[i]}: None")
                
                # Test EMA model
                ema_logits, ema_attentions, _ema_hw = self.ema_model(
                    batch_x_dev,
                    return_attention=True,
                    active_attention_stages=self.active_attention_stages
                )
                print(f"  EMA model attention stages: {len(ema_attentions)}")
                
                # Test attention distillation loss
                attn_loss, stage_losses = self.attention_distill_loss(
                    attentions, ema_attentions, attention_query_hw=_attn_hw
                )
                print(f"  Attention loss (identical models): {attn_loss.item():.6f}")
                print(f"  Stage losses: {stage_losses}")
                
            except Exception as e:
                print(f"  ERROR during validation: {e}")
                import traceback
                traceback.print_exc()
        
        self.model.train()
        self._data_validated = True
        print()
    
    def before_task(self, task_id, **kwargs):
        """Called before training on a new task."""
        task_classes = kwargs.get('task_classes', [])
        self.task_classes[task_id] = task_classes
        self.continual_metrics.set_task_classes(task_id, task_classes)
        
        # Store current task ID
        self.current_task_id = task_id
        
        self.current_classes = list(task_classes)
        background_class = getattr(self.params, 'background_class', None)
        if task_id > 0:
            self.old_classes = [c for c in self.seen_classes 
                               if c != background_class and c != self.params.ignore_index]
        else:
            self.old_classes = []
        
        # Log status
        attn_enabled = task_id >= self.attention_start_task
        lg.info(f"Task {task_id}: current_classes={self.current_classes}")
        lg.info(f"  - old_classes: {self.old_classes}")
        lg.info(f"  - Attention KD: {'enabled' if attn_enabled else 'disabled (starts at task ' + str(self.attention_start_task) + ')'}")
        
        # Log teacher-student distance at task start
        if task_id > 0:
            dist = self.compute_teacher_student_distance()
            lg.info(f"  - Teacher-Student L2 distance: {dist['l2_distance']:.6f}")
    
    def after_task(self, task_id, **kwargs):
        """Called after training on a task completes."""
        # Log distance before potential update
        dist = self.compute_teacher_student_distance()
        lg.info(f"Task {task_id} complete: Teacher-Student distance = {dist['l2_distance']:.6f}")
    
    def update_ema(self):
        """Update EMA teacher model parameters (standard step-wise EMA)."""
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data = (
                    self.ema_alpha * ema_param.data +
                    (1 - self.ema_alpha) * param.data
                )
    
    def _compute_attention_kd_loss(
        self,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        original_labels: Optional[torch.Tensor] = None,
        teacher_predictions: Optional[torch.Tensor] = None,
        teacher_probs: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.5,
        is_replay: bool = False,
        attention_query_hw: Optional[List[Optional[Tuple[int, int]]]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute attention map distillation loss using cosine similarity.
        
        With Masked Attention Distillation (MAD): excludes current class pixels
        from distillation to prevent teacher misleading student on new classes.
        
        IMPORTANT: This function is called ONLY on current task images (not replay samples).
        The attention distillation is applied exclusively to current task images to ensure
        that the STU mask strategy works correctly on images containing new classes.
        
        In current_only mode, teacher_predictions are used to identify old class pixels
        that are labeled as 255 in the ground truth.
        
        Args:
            student_attentions: List of student attention tensors (from current task images only)
            teacher_attentions: List of teacher attention tensors (from current task images only)
            labels: Ground truth labels for mask creation (from current task images only)
            teacher_predictions: Optional teacher predictions (B, H, W) for identifying old classes
            
        Returns:
            attention_loss: Total attention distillation loss
            stage_losses: Dict with per-stage losses
        """
        attention_loss, stage_losses = self.attention_distill_loss(
            student_attentions, 
            teacher_attentions,
            labels=labels,
            original_labels=original_labels,
            current_classes=self.current_classes if self.use_masked_distillation else None,
            old_classes=self.old_classes if self.use_masked_distillation else None,
            is_replay=is_replay,
            teacher_predictions=teacher_predictions,
            teacher_probs=teacher_probs,
            confidence_threshold=confidence_threshold,
            attention_query_hw=attention_query_hw,
        )
        return attention_loss, stage_losses

    def _get_original_labels_for_batch(
        self,
        dataloader,
        batch_indices: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Fetch pre-mask converted labels (train IDs) for stream samples.

        Returns None when dataset does not expose this capability.
        """
        if batch_indices is None:
            return None
        dataset = getattr(dataloader, "dataset", None)
        get_mask_fn = getattr(dataset, "get_original_converted_mask", None)
        if get_mask_fn is None:
            return None

        if torch.is_tensor(batch_indices):
            indices = [int(i) for i in batch_indices.detach().cpu().tolist()]
        else:
            indices = [int(i) for i in batch_indices]

        masks = []
        for idx in indices:
            masks.append(get_mask_fn(idx))
        if not masks:
            return None
        return torch.stack(masks, dim=0).to(self.device)
    
    def _minority_balanced_retrieve(
        self,
        n_imgs: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve samples with priority given to minority classes.
        
        优先采样小样本类别，基于逆频率权重采样。
        """
        import random as r
        import numpy as np
        
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

    def _target_pixel_ratio_retrieve(
        self,
        n_imgs: int = 4,
        target_classes: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replay sampling weighted by target-class pixel fraction per stored mask.

        score_i = (sum_{c in targets} pixel_count_i[c]) / (sum_c pixel_count_i[c] + eps)

        Favors images that contain more target-class pixels (e.g. old_classes),
        without using dominant-class bucketing.
        """
        import random as r

        if self.buffer.n_added_so_far == 0:
            return torch.Tensor(), torch.Tensor()

        targets = target_classes if target_classes is not None else self.old_classes
        if not targets:
            return self.buffer.random_retrieve(n_imgs)

        n_available = min(self.buffer.n_added_so_far, self.buffer.max_size)
        class_counts = self.buffer.buffer_class_counts[:n_available].float()

        idx = torch.tensor(
            [int(c) for c in targets if c is not None],
            device=class_counts.device,
            dtype=torch.long,
        )
        idx = idx[(idx >= 0) & (idx < class_counts.shape[1])]
        if idx.numel() == 0:
            return self.buffer.random_retrieve(n_imgs)

        total_px = class_counts.sum(dim=1).clamp_min(1.0)
        target_px = class_counts[:, idx].sum(dim=1)
        scores = (target_px / total_px).clamp_min(0.0)
        scores_np = scores.cpu().numpy() + 1e-6
        denom = scores_np.sum()
        if denom <= 0 or not np.isfinite(denom):
            return self.buffer.random_retrieve(n_imgs)
        p = scores_np / denom

        try:
            selected = np.random.choice(
                n_available,
                size=min(n_imgs, n_available),
                replace=False,
                p=p,
            )
        except ValueError:
            selected = r.sample(range(n_available), min(n_imgs, n_available))

        return (
            self.buffer.buffer_imgs[selected],
            self.buffer.buffer_masks[selected],
        )

    def _compute_focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25
    ) -> torch.Tensor:
        """Compute Focal Loss for handling class imbalance.
        
        Focal Loss down-weights easy examples and focuses on hard ones:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        
        Args:
            logits: Model predictions [B, C, H, W]
            labels: Ground truth [B, H, W]
            gamma: Focusing parameter (default 2.0)
            alpha: Balance parameter (default 0.25)
            
        Returns:
            Focal loss value
        """
        n_classes = logits.shape[1]
        ignore_index = self.params.ignore_index
        
        # Reshape for computation
        logits = logits.permute(0, 2, 3, 1).reshape(-1, n_classes)  # [N, C]
        labels = labels.reshape(-1)  # [N]
        
        # Create mask for valid pixels
        valid_mask = labels != ignore_index
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        logits = logits[valid_mask]
        labels = labels[valid_mask]
        
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probability of true class
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** gamma
        
        # Apply focal loss
        focal_loss = alpha * focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def _compute_ohem_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        top_k_percent: float = 0.25
    ) -> torch.Tensor:
        """Online Hard Example Mining - focus on hardest pixels.
        
        Args:
            logits: Model predictions [B, C, H, W]
            labels: Ground truth [B, H, W]
            top_k_percent: Percentage of hardest pixels to use
            
        Returns:
            OHEM loss value
        """
        ignore_index = self.params.ignore_index
        
        # Compute per-pixel CE loss
        ce_loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction='none')
        
        # Flatten
        ce_loss_flat = ce_loss.reshape(-1)
        
        # Get valid pixels
        labels_flat = labels.reshape(-1)
        valid_mask = labels_flat != ignore_index
        valid_losses = ce_loss_flat[valid_mask]
        
        if valid_losses.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Select top-k hardest examples
        k = max(1, int(valid_losses.numel() * top_k_percent))
        top_k_losses, _ = torch.topk(valid_losses, k)
        
        return top_k_losses.mean()
    
    def _get_adaptive_replay_size(self, epoch: int, base_size: int) -> int:
        """Get adaptive replay batch size based on epoch.
        
        Later epochs use more replay to reinforce old knowledge.
        
        Args:
            epoch: Current epoch number
            base_size: Base replay batch size
            
        Returns:
            Adaptive replay size
        """
        use_adaptive = getattr(self.params, 'use_adaptive_replay', False)
        if not use_adaptive:
            return base_size
        
        # Increase replay size in later epochs
        # epoch 0: base_size, epoch 1: base_size*1.5, epoch 2: base_size*2, etc.
        multiplier = 1.0 + epoch * 0.5
        max_multiplier = getattr(self.params, 'max_replay_multiplier', 3.0)
        multiplier = min(multiplier, max_multiplier)
        
        return int(base_size * multiplier)
    
    @staticmethod
    def _split_mixed_slot_counts(
        n_imgs: int, w_tpr: int, w_minority: int
    ) -> Tuple[int, int]:
        """Split n_imgs into (n_target_pixel_ratio, n_minority) by integer weights."""
        if n_imgs <= 0:
            return 0, 0
        if w_tpr <= 0 and w_minority <= 0:
            w_tpr, w_minority = 1, 1
        if w_tpr <= 0:
            return 0, n_imgs
        if w_minority <= 0:
            return n_imgs, 0
        s = w_tpr + w_minority
        n_tpr = (n_imgs * w_tpr) // s
        n_min = n_imgs - n_tpr
        return n_tpr, n_min
    
    def _mixed_retrieve(self, n_imgs: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixed retrieval: configurable shares of target_pixel_ratio + minority.
        
        Proportions from ``mixed_replay_ratios`` = [w_tpr, w_minority] (e.g. [1, 1] ≈ 50/50).
        Order in batch: TPR branch first, then minority.
        """
        if self.buffer.n_added_so_far == 0:
            return torch.Tensor(), torch.Tensor()
        
        w_tpr, w_min = self.mixed_replay_ratios
        n_tpr, n_minority = self._split_mixed_slot_counts(n_imgs, w_tpr, w_min)
        
        if n_tpr > 0 and len(self.old_classes) > 0:
            imgs_tpr, masks_tpr = self._target_pixel_ratio_retrieve(
                n_imgs=n_tpr, target_classes=self.old_classes
            )
        elif n_tpr > 0:
            imgs_tpr, masks_tpr = self.buffer.random_retrieve(n_tpr)
        else:
            imgs_tpr, masks_tpr = torch.Tensor(), torch.Tensor()
        
        if n_minority > 0:
            imgs_minority, masks_minority = self._minority_balanced_retrieve(n_minority)
        else:
            imgs_minority, masks_minority = torch.Tensor(), torch.Tensor()
        
        if imgs_tpr.size(0) > 0 and imgs_minority.size(0) > 0:
            return torch.cat([imgs_tpr, imgs_minority]), torch.cat([masks_tpr, masks_minority])
        if imgs_tpr.size(0) > 0:
            return imgs_tpr, masks_tpr
        return imgs_minority, masks_minority
    
    def _compute_segmentation_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute segmentation loss based on configured loss type.
        
        Supports: 'ce', 'focal', 'ohem', 'focal_ohem'
        
        Args:
            logits: Model predictions [B, C, H, W]
            labels: Ground truth [B, H, W]
            
        Returns:
            Loss value
        """
        if self.seg_loss_type == 'focal':
            return self._compute_focal_loss(
                logits, labels,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha
            )
        elif self.seg_loss_type == 'ohem':
            return self._compute_ohem_loss(
                logits, labels,
                top_k_percent=self.ohem_ratio
            )
        elif self.seg_loss_type == 'focal_ohem':
            # Combine both: Focal on all + OHEM boost
            loss_focal = self._compute_focal_loss(
                logits, labels,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha
            )
            loss_ohem = self._compute_ohem_loss(
                logits, labels,
                top_k_percent=self.ohem_ratio
            )
            return 0.5 * loss_focal + 0.5 * loss_ohem
        else:  # 'ce'
            return self.criterion(logits, labels)
    
    def _compute_logit_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        original_labels: Optional[torch.Tensor] = None,
        temperature: float = 3.0,
        old_classes_only: bool = True,
        confidence_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Compute logit-level knowledge distillation loss.
        
        Args:
            student_logits: (B, C, H, W)
            teacher_logits: (B, C, H, W)
            labels: Incremental labels (post-mask)
            original_labels: Labels before incremental masking
            temperature: Temperature for softmax
            old_classes_only: If True, only apply KD on old classes
            confidence_threshold: Teacher confidence threshold for KD pixels
        """
        T = temperature
        _, C, _, _ = student_logits.shape
        old_class_indices = [c for c in self.old_classes if 0 <= c < C]
        if old_classes_only and len(old_class_indices) == 0:
            return torch.tensor(0.0, device=student_logits.device)

        kd_mask = None
        if old_classes_only and labels is not None and original_labels is not None:
            ignore_index = self.params.ignore_index
            became_ignore_mask = (labels == ignore_index) & (original_labels != ignore_index)

            teacher_probs_all = F.softmax(teacher_logits, dim=1)
            teacher_pred = teacher_probs_all.argmax(dim=1)

            teacher_pred_is_old = torch.zeros_like(teacher_pred, dtype=torch.bool)
            for c in old_class_indices:
                teacher_pred_is_old = teacher_pred_is_old | (teacher_pred == int(c))

            old_idx_tensor = torch.tensor(
                old_class_indices, device=teacher_probs_all.device, dtype=torch.long
            )
            old_probs = teacher_probs_all[:, old_idx_tensor, :, :]
            max_old_prob, _ = old_probs.max(dim=1)

            kd_mask = became_ignore_mask & teacher_pred_is_old & (
                max_old_prob > float(confidence_threshold)
            )
            if kd_mask.sum().item() == 0:
                return torch.tensor(0.0, device=student_logits.device)

        if old_classes_only:
            s_sel = student_logits[:, old_class_indices, :, :]
            t_sel = teacher_logits[:, old_class_indices, :, :]
        else:
            s_sel = student_logits
            t_sel = teacher_logits

        s_logits = s_sel.permute(0, 2, 3, 1).reshape(-1, s_sel.shape[1])
        t_logits = t_sel.permute(0, 2, 3, 1).reshape(-1, t_sel.shape[1])

        if kd_mask is not None:
            mask_flat = kd_mask.reshape(-1)
            s_logits = s_logits[mask_flat]
            t_logits = t_logits[mask_flat]
            if s_logits.numel() == 0:
                return torch.tensor(0.0, device=student_logits.device)

        s_log_probs = F.log_softmax(s_logits / T, dim=1)
        t_probs = F.softmax(t_logits / T, dim=1)
        kd_loss = F.kl_div(
            s_log_probs, t_probs, reduction='batchmean'
        ) * (T ** 2)
        return kd_loss
    
    def train(self, dataloader, **kwargs):
        """Training with attention map EMA teacher distillation."""
        task_name = kwargs.get('task_name', 'unknown')
        task_id = kwargs.get('task_id', 0)
        
        # Check if attention KD should be enabled
        attn_enabled = task_id >= self.attention_start_task
        if not attn_enabled:
            lg.info(f"Task {task_id}: Attention KD disabled (starts from task {self.attention_start_task})")
        else:
            lg.info(f"Task {task_id}: Attention KD enabled with weight {self.attention_kd_weight}")
        
        self.model.train()
        self.ema_model.eval()
        
        # Logging accumulators
        total_ce_loss = 0.0
        total_logit_kd_loss = 0.0
        total_attention_kd_loss = 0.0
        total_batches = 0
        
        for j, batch in enumerate(dataloader):
            batch_x, batch_y = batch[0], batch[1]
            batch_indices = batch[2] if len(batch) > 2 else None
            batch_y_original = self._get_original_labels_for_batch(dataloader, batch_indices)
            
            if j == 0:
                self._validate_data(batch_x, batch_y)
            
            self.stream_idx += len(batch_x)
            self.update_seen_classes(batch_y)
            
            for _ in range(self.params.mem_iters):
                # === Class-Balanced Buffer Retrieval ===
                if self.use_balanced_sampling and self.buffer.n_added_so_far > 0:
                    if self.balanced_sampling_mode == 'mixed':
                        # mixed_replay_ratios: [w_tpr, w_minority] → target_pixel_ratio + minority
                        mem_x, mem_y = self._mixed_retrieve(n_imgs=self.params.mem_batch_size)
                    elif (
                        self.balanced_sampling_mode == 'target_pixel_ratio'
                        and len(self.old_classes) > 0
                    ):
                        mem_x, mem_y = self._target_pixel_ratio_retrieve(
                            n_imgs=self.params.mem_batch_size,
                            target_classes=self.old_classes,
                        )
                    elif self.balanced_sampling_mode == 'old_classes' and len(self.old_classes) > 0:
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
                    # Keep stream slice metadata available for both attention/non-attention paths.
                    stream_size = batch_x.size(0)
                    stream_y = combined_y[:stream_size]
                    
                    if attn_enabled:
                        # === Single-pass attention forward on combined batch ===
                        # This avoids extra stream/replay-only forwards and reduces memory pressure.
                        student_logits, student_attentions_all, student_attn_hw_all = self.model(
                            combined_x,
                            return_attention=True,
                            active_attention_stages=self.active_attention_stages
                        )

                        with torch.no_grad():
                            teacher_logits, teacher_attentions_all, teacher_attn_hw_all = self.ema_model(
                                combined_x,
                                return_attention=True,
                                active_attention_stages=self.active_attention_stages
                            )

                        # Extract current task images and replay images from combined data
                        replay_y = combined_y[stream_size:]

                        # Split attentions/logits by batch dim for stream vs replay.
                        student_attentions_stream = []
                        teacher_attentions_stream = []
                        student_attentions_replay = []
                        teacher_attentions_replay = []
                        student_attn_hw_stream: List[Optional[Tuple[int, int]]] = []
                        teacher_attn_hw_stream: List[Optional[Tuple[int, int]]] = []
                        student_attn_hw_replay: List[Optional[Tuple[int, int]]] = []
                        teacher_attn_hw_replay: List[Optional[Tuple[int, int]]] = []
                        for s_attn, t_attn, s_hw, t_hw in zip(
                            student_attentions_all,
                            teacher_attentions_all,
                            student_attn_hw_all,
                            teacher_attn_hw_all,
                        ):
                            if s_attn is None or t_attn is None:
                                student_attentions_stream.append(None)
                                teacher_attentions_stream.append(None)
                                student_attentions_replay.append(None)
                                teacher_attentions_replay.append(None)
                                student_attn_hw_stream.append(None)
                                teacher_attn_hw_stream.append(None)
                                student_attn_hw_replay.append(None)
                                teacher_attn_hw_replay.append(None)
                            else:
                                student_attentions_stream.append(s_attn[:stream_size])
                                teacher_attentions_stream.append(t_attn[:stream_size])
                                student_attentions_replay.append(s_attn[stream_size:])
                                teacher_attentions_replay.append(t_attn[stream_size:])
                                student_attn_hw_stream.append(s_hw)
                                teacher_attn_hw_stream.append(t_hw)
                                student_attn_hw_replay.append(s_hw)
                                teacher_attn_hw_replay.append(t_hw)

                        teacher_logits_stream = teacher_logits[:stream_size]
                        teacher_probs_stream = F.softmax(teacher_logits_stream, dim=1)  # (B, C, H, W)
                        teacher_pred_stream = teacher_logits_stream.argmax(dim=1)  # (B, H, W)

                        # === Attention map distillation loss with MAD (stream) ===
                        confidence_threshold = getattr(self.params, 'mad_relaxed_threshold', 0.001)
                        loss_attention_stream, attn_stage_losses = self._compute_attention_kd_loss(
                            student_attentions_stream,
                            teacher_attentions_stream,
                            labels=stream_y,
                            original_labels=batch_y_original,
                            teacher_predictions=teacher_pred_stream,
                            teacher_probs=teacher_probs_stream,
                            confidence_threshold=confidence_threshold,
                            is_replay=False,
                            attention_query_hw=student_attn_hw_stream,
                        )

                        # === Attention map distillation loss with MAD (replay) ===
                        # Replay samples distill on old/ignore regions via is_replay=True.
                        if replay_y.size(0) > 0:
                            teacher_logits_replay = teacher_logits[stream_size:]
                            teacher_probs_replay = F.softmax(teacher_logits_replay, dim=1)
                            teacher_pred_replay = teacher_logits_replay.argmax(dim=1)

                            loss_attention_replay, _ = self._compute_attention_kd_loss(
                                student_attentions_replay,
                                teacher_attentions_replay,
                                labels=replay_y,
                                original_labels=None,
                                teacher_predictions=teacher_pred_replay,
                                teacher_probs=teacher_probs_replay,
                                confidence_threshold=confidence_threshold,
                                is_replay=True,
                                attention_query_hw=student_attn_hw_replay,
                            )
                            n_stream = max(stream_size, 1)
                            n_replay = replay_y.size(0)
                            loss_attention_kd = (
                                loss_attention_stream * n_stream + loss_attention_replay * n_replay
                            ) / float(n_stream + n_replay)
                        else:
                            loss_attention_kd = loss_attention_stream
                    else:
                        # === Student forward without attention ===
                        student_logits = self.model(combined_x)
                        
                        # === Teacher forward ===
                        with torch.no_grad():
                            teacher_logits = self.ema_model(combined_x)
                        
                        loss_attention_kd = torch.tensor(0.0, device=self.device)
                    
                    # === Segmentation loss (CE/Focal/OHEM based on config) ===
                    loss_ce = self._compute_segmentation_loss(student_logits, combined_y)
                    
                    # === Logit-level KD loss ===
                    kd_conf_threshold = getattr(
                        self.params,
                        'logit_kd_confidence_threshold',
                        0.5,
                    )
                    loss_logit_kd = self._compute_logit_kd_loss(
                        student_logits[:stream_size], teacher_logits[:stream_size],
                        labels=stream_y,
                        original_labels=batch_y_original,
                        temperature=self.params.kd_temperature,
                        old_classes_only=True,
                        confidence_threshold=kd_conf_threshold,
                    )
                    
                    # === Combined loss ===
                    if attn_enabled:
                        loss = (
                            loss_ce + 
                            self.kd_weight * loss_logit_kd +
                            self.attention_kd_weight * loss_attention_kd
                        )
                    else:
                        loss = loss_ce + self.kd_weight * loss_logit_kd
                    
                    self.loss = loss.item()
                    
                    # Accumulate for logging
                    total_ce_loss += loss_ce.item()
                    total_logit_kd_loss += loss_logit_kd.item()
                    total_attention_kd_loss += loss_attention_kd.item() if isinstance(loss_attention_kd, torch.Tensor) else loss_attention_kd
                    total_batches += 1
                    
                    self.optim.zero_grad()
                    if isinstance(loss, torch.Tensor) and loss.requires_grad:
                        loss.backward()
                        if self.params.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.params.grad_clip
                            )
                        self.optim.step()
                        self.update_ema()
                    else:
                        lg.debug(
                            "Skip backward: no differentiable loss "
                            "(e.g. all pixels ignored and empty KD)."
                        )
                    
                else:
                    # No memory samples - train on stream only
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    if attn_enabled:
                        # === Student forward with attention maps ===
                        # Forward pass on stream data for logits (used for CE and Logit KD)
                        student_logits = self.model(
                            batch_x,
                            return_attention=False
                        )
                        
                        # === Teacher forward ===
                        with torch.no_grad():
                            teacher_logits = self.ema_model(batch_x)
                        
                        # === Attention distillation: ONLY on current task images ===
                        # Forward pass on current task images to get attention maps
                        _, student_attentions, student_attn_hw = self.model(
                            batch_x,
                            return_attention=True,
                            active_attention_stages=self.active_attention_stages
                        )
                        
                        with torch.no_grad():
                            teacher_logits_no_replay, teacher_attentions, _teacher_attn_hw = self.ema_model(
                                batch_x,
                                return_attention=True,
                                active_attention_stages=self.active_attention_stages
                            )
                            # Get teacher predictions and probabilities for identifying old classes
                            teacher_probs_no_replay = F.softmax(teacher_logits_no_replay, dim=1)  # (B, C, H, W)
                            teacher_pred_no_replay = teacher_logits_no_replay.argmax(dim=1)  # (B, H, W)
                        
                        # === Attention map distillation loss with MAD ===
                        # Masked Attention Distillation: excludes current class pixels
                        # Only distill on current task images, using batch_y for mask creation
                        # Note: confidence_threshold here does not create hard pseudo-labels;
                        # MAD mask uses the internal relaxed threshold.
                        confidence_threshold = getattr(self.params, 'mad_relaxed_threshold', 0.001)
                        loss_attention_kd, attn_stage_losses = self._compute_attention_kd_loss(
                            student_attentions, teacher_attentions, labels=batch_y,
                            original_labels=batch_y_original,
                            teacher_predictions=teacher_pred_no_replay,
                            teacher_probs=teacher_probs_no_replay,
                            confidence_threshold=confidence_threshold,
                            attention_query_hw=student_attn_hw,
                        )
                        
                        # === Segmentation loss (CE/Focal/OHEM based on config) ===
                        loss_ce = self._compute_segmentation_loss(student_logits, batch_y)
                        
                        # === Logit-level KD loss ===
                        kd_conf_threshold = getattr(
                            self.params,
                            'logit_kd_confidence_threshold',
                            0.5,
                        )
                        loss_logit_kd = self._compute_logit_kd_loss(
                            student_logits, teacher_logits,
                            labels=batch_y,
                            original_labels=batch_y_original,
                            temperature=self.params.kd_temperature,
                            old_classes_only=True,
                            confidence_threshold=kd_conf_threshold,
                        )
                        
                        # === Combined loss ===
                        loss = (
                            loss_ce + 
                            self.kd_weight * loss_logit_kd +
                            self.attention_kd_weight * loss_attention_kd
                        )
                        
                        # Accumulate for logging
                        total_ce_loss += loss_ce.item()
                        total_logit_kd_loss += loss_logit_kd.item()
                        total_attention_kd_loss += loss_attention_kd.item() if isinstance(loss_attention_kd, torch.Tensor) else loss_attention_kd
                        total_batches += 1
                    else:
                        # === Student forward without attention ===
                        student_logits = self.model(batch_x)
                        
                        # === Teacher forward ===
                        with torch.no_grad():
                            teacher_logits = self.ema_model(batch_x)
                        
                        # === Segmentation loss (CE/Focal/OHEM based on config) ===
                        loss_ce = self._compute_segmentation_loss(student_logits, batch_y)
                        
                        # === Logit-level KD loss ===
                        kd_conf_threshold = getattr(
                            self.params,
                            'logit_kd_confidence_threshold',
                            0.5,
                        )
                        loss_logit_kd = self._compute_logit_kd_loss(
                            student_logits, teacher_logits,
                            labels=batch_y,
                            original_labels=batch_y_original,
                            temperature=self.params.kd_temperature,
                            old_classes_only=True,
                            confidence_threshold=kd_conf_threshold,
                        )
                        
                        # === Combined loss ===
                        loss = loss_ce + self.kd_weight * loss_logit_kd
                        
                        # Accumulate for logging
                        total_ce_loss += loss_ce.item()
                        total_logit_kd_loss += loss_logit_kd.item()
                        total_batches += 1
                    
                    self.loss = loss.item()
                    
                    self.optim.zero_grad()
                    if isinstance(loss, torch.Tensor) and loss.requires_grad:
                        loss.backward()
                        if self.params.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.params.grad_clip
                            )
                        self.optim.step()
                        self.update_ema()
                    else:
                        lg.debug(
                            "Skip backward: no differentiable loss "
                            "(e.g. all pixels ignored and empty KD)."
                        )
            
            # Update buffer
            self.buffer.update(
                imgs=batch_x.cpu(),
                masks=batch_y.cpu()
            )
            
            # Logging
            if j % 10 == 0 or j == len(dataloader) - 1:
                avg_ce = total_ce_loss / max(total_batches, 1)
                avg_logit_kd = total_logit_kd_loss / max(total_batches, 1)
                avg_attn_kd = total_attention_kd_loss / max(total_batches, 1)
                
                print(
                    f"Task: {task_name}  batch {j}/{len(dataloader)}  "
                    f"Loss: {self.loss:.4f} (CE:{avg_ce:.3f} LogitKD:{avg_logit_kd:.3f} "
                    f"AttnKD:{avg_attn_kd:.4f}/{avg_attn_kd:.3e})  "
                    f"time: {time.time() - self.start:.1f}s",
                    end="\r"
                )
                if self.debug_mask_stats and attn_enabled and isinstance(attn_stage_losses, dict):
                    dbg_items = []
                    for k in sorted(attn_stage_losses.keys()):
                        if any(tag in k for tag in ("w_entropy_mean", "w_diff_mean", "w_adaptive_mean", "As_sum_lastdim_mean", "Ps_sum_lastdim_mean")):
                            v = attn_stage_losses[k]
                            if isinstance(v, (int, float)):
                                dbg_items.append(f"{k.split('attn_stage_')[-1]}={v:.4f}")
                    if dbg_items:
                        print("    [mask-debug] " + " | ".join(dbg_items[:8]))
        
        print()
        lg.info(f"Task {task_name} completed. Avg losses - CE: {avg_ce:.4f}, "
                f"LogitKD: {avg_logit_kd:.4f}, AttnKD: {avg_attn_kd:.4f}")
    
    def _create_averaged_model(self):
        """Create a model with averaged parameters from teacher and student."""
        avg_model = deepcopy(self.model)
        with torch.no_grad():
            for avg_param, student_param, teacher_param in zip(
                avg_model.parameters(),
                self.model.parameters(),
                self.ema_model.parameters()
            ):
                avg_param.data = (student_param.data + teacher_param.data) / 2.0
        return avg_model
    
    def evaluate(self, dataloaders, task_id, **kwargs):
        """Evaluate using specified model."""
        eval_mode = getattr(self.params, 'eval_mode', 'student')
        
        if getattr(self.params, 'eval_teacher', False) and eval_mode == 'student':
            eval_mode = 'teacher'
        
        if eval_mode == 'teacher':
            original_model = self.model
            self.model = self.ema_model
            result = super().evaluate(dataloaders, task_id, **kwargs)
            self.model = original_model
            return result
        elif eval_mode == 'avg':
            original_model = self.model
            self.model = self._create_averaged_model()
            self.model.eval()
            result = super().evaluate(dataloaders, task_id, **kwargs)
            self.model = original_model
            return result
        else:
            return super().evaluate(dataloaders, task_id, **kwargs)
    
    def save(self, model_name):
        """Save both student and teacher models.
        
        新版本同时保存teacher和student，便于分析距离和恢复训练。
        
        Args:
            model_name: Name for the model checkpoint
        """
        import os
        from src.utils.utils import save_model
        
        if self.params.save_ckpt:
            save_dir = os.path.join(
                self.params.ckpt_root,
                self.params.tag,
                str(self.params.run_id)
            )
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 保存包含student和teacher的完整checkpoint
            checkpoint = {
                'student': self.model.state_dict(),
                'teacher': self.ema_model.state_dict(),
                'task_id': self.current_task_id,
                'old_classes': list(self.old_classes),
                'seen_classes': list(self.seen_classes),
            }
            
            ckpt_path = os.path.join(save_dir, model_name)
            torch.save(checkpoint, ckpt_path)
            lg.info(f"Saved checkpoint (student + teacher) to: {ckpt_path}")
    
    def resume(self, model_path=None, buffer_path=None):
        """Resume from checkpoint with teacher model support.
        
        支持新旧两种checkpoint格式:
        - 新格式: {'student': ..., 'teacher': ...}
        - 旧格式: 直接的state_dict
        
        Args:
            model_path: Path to model checkpoint
            buffer_path: Path to buffer checkpoint
        """
        import pickle
        
        if model_path is not None:
            ckpt = torch.load(model_path, map_location=self.device)
            
            if isinstance(ckpt, dict) and 'student' in ckpt:
                # 新格式: 包含student和teacher
                self.model.load_state_dict(ckpt['student'])
                if 'teacher' in ckpt:
                    self.ema_model.load_state_dict(ckpt['teacher'])
                    lg.info("Loaded both student and teacher from checkpoint")
                else:
                    # 没有teacher，从student复制
                    self.ema_model.load_state_dict(ckpt['student'])
                    lg.info("Loaded student, copied to teacher (no teacher in ckpt)")
                
                # 恢复训练状态
                if 'task_id' in ckpt:
                    self.current_task_id = ckpt['task_id']
                if 'old_classes' in ckpt:
                    self.old_classes = list(ckpt['old_classes'])
                if 'seen_classes' in ckpt:
                    self.seen_classes = set(ckpt['seen_classes'])
            else:
                # 旧格式: 直接的state_dict
                self.model.load_state_dict(ckpt)
                self.ema_model.load_state_dict(ckpt)
                lg.info("Loaded old format checkpoint (student only, copied to teacher)")
        
        if buffer_path is not None:
            with open(buffer_path, 'rb') as f:
                self.buffer = pickle.load(f)
        
        torch.cuda.empty_cache()
    
    def compute_teacher_student_distance(self) -> dict:
        """计算教师和学生参数之间的距离.
        
        用于监控训练过程中的收敛情况。
        
        Returns:
            dict: 包含各种距离指标
        """
        total_l2 = 0.0
        total_l1 = 0.0
        total_params = 0
        max_diff = 0.0
        
        with torch.no_grad():
            for t_param, s_param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                diff = t_param.data - s_param.data
                l2 = torch.norm(diff).item()
                l1 = torch.abs(diff).sum().item()
                
                total_l2 += l2 ** 2
                total_l1 += l1
                total_params += t_param.numel()
                max_diff = max(max_diff, torch.abs(diff).max().item())
        
        total_l2 = total_l2 ** 0.5
        
        return {
            'l2_distance': total_l2,
            'l1_distance': total_l1,
            'l2_normalized': total_l2 / total_params if total_params > 0 else 0,
            'max_diff': max_diff,
            'total_params': total_params
        }


