"""Attention Map Distillation losses for continual semantic segmentation.

This module provides losses for knowledge distillation at the attention level,
enabling the student model to learn "where to look" from the teacher's attention
patterns, which helps preserve old knowledge about recognizing objects.

Version: v3 (Attention Map Distillation with Cosine Loss + Masked Attention Distillation)

Key Design:
- SegFormer attention has shape (B, num_heads, seq_len_q, seq_len_k)
- We process attention by: averaging heads -> resize to 32x32 -> flatten
- Use COSINE SIMILARITY LOSS (not MSE) because:
  - MSE gives ~1e-8 (too small for meaningful gradients)
  - Cosine loss is scale-invariant, range [0, 2]
  - Focuses on pattern matching, not magnitude
Masked Attention Distillation (MAD):
- Prevents teacher from guiding student on regions it doesn't understand (new classes)
- Masks out positions where current task classes appear in ground truth
- Only distills attention for old class regions where teacher is reliable

Formula:
    L_Attn = Σ_l weight_l × mask × (1 - cosine_similarity(Attn_S^l, Attn_T^l))
    
where mask[i,j] = 0 if pixel i or j belongs to current new class, else 1
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class AttentionMapDistillationLoss(nn.Module):
    """Attention map distillation loss for SegFormer with Masked Attention Distillation.
    
    SegFormer attention has shape (B, num_heads, seq_len_q, seq_len_k).
    
    We compare attention patterns by:
    1. Pooling across heads (mean)
    2. Downsampling to manageable size (32x32)
    3. Computing cosine similarity loss (pattern matching, scale invariant)
    
    Why Cosine Loss (not MSE)?
    - After pooling, attention values are small (~0.001)
    - MSE of small differences → ~1e-8 (no gradient)
    - Cosine loss is scale-invariant, range [0, 2]
    - Focuses on "pattern" not "magnitude"
    
    Masked Attention Distillation (MAD):
    - When use_masked_distillation=True, masks out current class pixels
    - Prevents teacher from misleading student on new classes
    - Teacher doesn't know new classes, so its attention for those regions is unreliable
    
    Args:
        num_active_stages: Number of stages to distill
        stage_weights: Weights for each active stage's distillation loss
        loss_type: 'cosine' (default), 'mse', 'l1', or 'kl'
        pool_size: Downsample attention to this size
        use_masked_distillation: Whether to mask out current class pixels
    """
    
    def __init__(
        self,
        num_active_stages: int = 2,
        stage_weights: Optional[List[float]] = None,
        loss_type: str = 'cosine',  # Cosine is default and recommended
        pool_size: int = 32,
        head_aggregation: str = 'mean',
        use_masked_distillation: bool = True,
        stu_enabled: bool = True,
        eps: float = 1e-8,
        min_mask_value: float = 0.0,
        entropy_threshold: float = 0.5,
        margin_threshold: float = 1.5,
        entropy_gamma: float = 1.5,
        # If False, normalized entropy weights are used without raising to ``entropy_gamma``.
        stu_entropy_use_power: bool = True,
        mad_relaxed_threshold: float = 0.001,
        debug_mask_stats: bool = False,
        # If STU mask becomes too aggressive (too small), we fall back to the base 0-1 mask.
        # Lowering this ratio makes STU less likely to be replaced.
        stu_fallback_enabled: bool = True,
        stu_fallback_ratio: float = 0.1,
        # "global": one entropy scalar + one diff scalar per image (broadcast to all positions)
        # "local_pixel": per pooled spatial index (B, N) for both entropy-based and difference weights
        # "label_pixel": build entropy/diff maps on label HxW, then downsample to pool
        stu_weight_mode: str = "global",
        # Teacher margin top1/top2 ratio: "pooled" = on pool_size grid; "label" = on label H×W then downsample to pool
        stu_margin_space: str = "pooled",
        # Alpha for cosine adaptive weight:
        # W_adaptive = w_difference + alpha * w_entropy * 1(margin < threshold)
        cosine_entropy_alpha: float = 1.0,
        # If True (cosine branch), disable margin gate and use:
        # W_adaptive = w_difference + alpha * w_entropy on all positions.
        stu_disable_margin_gate: bool = False,
    ):
        super().__init__()
        
        self.num_active_stages = num_active_stages
        self.loss_type = loss_type
        self.pool_size = pool_size
        self.head_aggregation = head_aggregation
        self.use_masked_distillation = use_masked_distillation
        # STU-Mask (Spatial Uncertainty + Temporal Stability) settings
        self.stu_enabled = stu_enabled
        self.stu_fallback_enabled = bool(stu_fallback_enabled)
        try:
            self.stu_fallback_ratio = float(stu_fallback_ratio)
        except (ValueError, TypeError):
            self.stu_fallback_ratio = 0.1
        # Ensure eps is a float
        try:
            self.eps = float(eps)
        except (ValueError, TypeError):
            self.eps = 1e-8
        try:
            self.min_mask_value = float(min_mask_value)
        except (ValueError, TypeError):
            self.min_mask_value = 0.0
        # Entropy threshold for high/low entropy region classification
        try:
            self.entropy_threshold = float(entropy_threshold)
        except (ValueError, TypeError):
            self.entropy_threshold = 0.5
        # Confidence threshold for teacher-guided adaptive strategy
        # Margin-based boundary detection
        try:
            self.margin_threshold = float(margin_threshold)
        except (ValueError, TypeError):
            self.margin_threshold = 1.5
        # Entropy gamma parameter for normalized entropy weight
        try:
            self.entropy_gamma = float(entropy_gamma)
        except (ValueError, TypeError):
            self.entropy_gamma = 1.5
        self.stu_entropy_use_power = bool(stu_entropy_use_power)
        try:
            self.mad_relaxed_threshold = float(mad_relaxed_threshold)
        except (ValueError, TypeError):
            self.mad_relaxed_threshold = 0.001
        self.debug_mask_stats = bool(debug_mask_stats)

        wm = str(stu_weight_mode).lower().strip()
        if wm in {"global", "image", "batch"}:
            self.stu_weight_mode = "global"
        elif wm in {"local", "local_pixel", "pixel", "per_pixel"}:
            self.stu_weight_mode = "local_pixel"
        elif wm in {"label_pixel", "label", "pixel_hw", "hw_pixel"}:
            self.stu_weight_mode = "label_pixel"
        else:
            raise ValueError(
                f"Unknown stu_weight_mode={stu_weight_mode!r}. "
                "Expected 'global', 'local_pixel', or 'label_pixel'."
            )

        ms = str(stu_margin_space).lower().strip()
        if ms in {"pooled", "pool", "attention"}:
            self.stu_margin_space = "pooled"
        elif ms in {"label", "label_hw", "pixel_hw", "hw", "full"}:
            self.stu_margin_space = "label"
        else:
            raise ValueError(
                f"Unknown stu_margin_space={stu_margin_space!r}. "
                "Expected 'pooled' or 'label' (margin on label resolution, then resize to pool)."
            )
        try:
            self.cosine_entropy_alpha = float(cosine_entropy_alpha)
        except (ValueError, TypeError):
            self.cosine_entropy_alpha = 1.0
        self.stu_disable_margin_gate = bool(stu_disable_margin_gate)
        
        if stage_weights is None:
            # Default: higher weight for later (more semantic) stages
            self.stage_weights = [1.0, 1.5][:num_active_stages]
        else:
            # Ensure stage_weights are floats
            try:
                self.stage_weights = [float(w) for w in stage_weights]
            except (ValueError, TypeError):
                self.stage_weights = [1.0, 1.5][:num_active_stages]

    def _apply_entropy_gamma_power(self, w: torch.Tensor) -> torch.Tensor:
        """Apply ``entropy_gamma`` as exponent to normalized entropy weights when enabled."""
        if not self.stu_entropy_use_power:
            return w
        return torch.pow(w, self.entropy_gamma)

    def _hw_mask_to_pool_flat(
        self,
        mask_hw: torch.Tensor,
        target_size: int,
        mode: str = "nearest",
    ) -> torch.Tensor:
        """MAD-style pipeline: build mask on (B, H, W), then downsample to attention pool.

        Attention KD compares flattened maps of length ``target_size ** 2``. Anything that
        should align with semantics on the label map should be computed at ``H × W`` first,
        then resized to ``(target_size, target_size)`` and flattened.

        Args:
            mask_hw: (B, H, W) float or bool (cast to float).
            target_size: Usually ``self.pool_size`` (same as ``_process_attention``).
            mode: ``nearest`` for hard 0/1 masks (MAD) so grid cells stay in ``{0, 1}``;
                  ``bilinear`` for soft fields (e.g. teacher margin on label resolution).

        Returns:
            (B, target_size * target_size)
        """
        if mask_hw.dim() != 3:
            raise ValueError(
                f"_hw_mask_to_pool_flat expects (B, H, W), got {tuple(mask_hw.shape)}"
            )
        B = mask_hw.shape[0]
        m = mask_hw.unsqueeze(1).float()
        if mode == "nearest":
            m_down = F.interpolate(
                m, size=(target_size, target_size), mode="nearest"
            )
        elif mode == "bilinear":
            m_down = F.interpolate(
                m,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            raise ValueError(
                f"_hw_mask_to_pool_flat: mode must be 'nearest' or 'bilinear', got {mode!r}"
            )
        return m_down.view(B, -1)
    
    def _create_distillation_mask(
        self,
        labels: torch.Tensor,
        current_classes: List[int],
        target_size: int,
        original_labels: Optional[torch.Tensor] = None,
        old_classes: Optional[List[int]] = None,
        is_replay: bool = False,
        ignore_index: int = 255,
        teacher_predictions: Optional[torch.Tensor] = None,
        teacher_probs: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.5
    ) -> torch.Tensor:
        """Create mask for Masked Attention Distillation (MAD).
        
        Strategy:
        - For current task images (non-replay): 
          * Distill on old classes (identified via teacher predictions in current_only mode)
          * Distill on ignore pixels (255)
        - For replay samples: Distill only on non-ignore pixels (labels != 255)
        
        In current_only mode, old class pixels are labeled as 255. We use teacher
        predictions to identify which 255 pixels are actually old classes.
        
        Args:
            labels: Ground truth labels (B, H, W)
            current_classes: List of current task class IDs
            target_size: Target size to resize mask to (pool_size)
            old_classes: Optional list of old task class IDs (for replay samples)
            is_replay: Whether these are replay samples
            ignore_index: Index for ignore pixels (default: 255)
            teacher_predictions: Optional teacher model predictions (B, H, W) for identifying old classes
            
        Returns:
            mask: (B, target_size * target_size) spatial mask
                  1 for pixels to distill, 0 for pixels to skip
        """
        B, H, W = labels.shape
        device = labels.device
        
        # Ensure current_classes and old_classes are lists of integers
        if current_classes is not None:
            try:
                current_classes = [int(c) for c in current_classes if c is not None]
            except (ValueError, TypeError):
                current_classes = []
        else:
            current_classes = []
            
        if old_classes is not None:
            try:
                old_classes = [int(c) for c in old_classes if c is not None]
            except (ValueError, TypeError):
                old_classes = []
        else:
            old_classes = []
        
        if is_replay:
            # Replay samples: distill ONLY on valid (non-ignore) regions.
            # This avoids forcing attention alignment on unlabeled/ambiguous pixels.
            distill_mask = (labels != ignore_index).float()
        else:
            # Current task images: Conservatively distill on ignore pixels (255)
            # In current_only mode, old classes are labeled as 255, but we need to be careful
            # Strategy: Only distill on 255 pixels where teacher predicts old class with high confidence
            # This avoids distilling on regions where teacher is uncertain or wrong
            distill_mask = torch.zeros_like(labels, dtype=torch.float32)
            
            # Only consider pixels labeled as ignore_index (255)
            # In current_only mode, these include old classes and true ignore pixels
            ignore_mask = (labels == ignore_index)
            # Distill only on pixels that became ignore due to task masking.
            # Exclude dataset-native ignore regions (already ignore in original labels).
            if original_labels is not None:
                became_ignore_mask = ignore_mask & (original_labels != ignore_index)
            else:
                became_ignore_mask = ignore_mask
            
            if became_ignore_mask.any() and teacher_predictions is not None and teacher_probs is not None and len(old_classes) > 0:
                # Get max probability among old classes for each pixel
                old_class_indices = torch.tensor(old_classes, device=teacher_probs.device, dtype=torch.long)
                old_class_probs = teacher_probs[:, old_class_indices, :, :]  # (B, num_old, H, W)
                max_old_prob, max_old_idx = old_class_probs.max(dim=1)  # (B, H, W)
                
                # Map back to actual class indices
                old_class_tensor = torch.tensor(old_classes, device=teacher_probs.device, dtype=torch.long)
                predicted_old_class = old_class_tensor[max_old_idx]  # (B, H, W)
                
                # Check if teacher prediction matches one of the old classes
                teacher_pred_is_old = torch.zeros_like(labels, dtype=torch.bool)
                for c in old_classes:
                    try:
                        c_val = int(c)
                        teacher_pred_is_old = teacher_pred_is_old | (teacher_predictions == c_val)
                    except (ValueError, TypeError):
                        continue
                
                # Debug: Log detailed statistics before filtering
                import logging as lg
                total_ignore = became_ignore_mask.sum().item()
                teacher_pred_old_count = (became_ignore_mask & teacher_pred_is_old).sum().item()
                
                # Get max probability across ALL classes for comparison
                max_all_prob, _ = teacher_probs.max(dim=1)  # (B, H, W)
                
                # Strategy: Use teacher prediction as primary signal
                # Since teacher predictions are already filtered (argmax), we trust them more
                # Only use a very low confidence threshold to filter out completely uncertain predictions
                # This is much more lenient than the original threshold
                relaxed_threshold = self.mad_relaxed_threshold
                high_conf_mask = max_all_prob > relaxed_threshold
                
                # Only distill on ignore pixels where:
                # 1. Teacher predicts an old class (primary signal - most important)
                # 2. Overall confidence is above a very low threshold (just to filter noise)
                high_conf_old_mask = high_conf_mask & teacher_pred_is_old
                distill_on_old = became_ignore_mask & high_conf_old_mask
                
                high_conf_count = high_conf_old_mask.sum().item()
                
                distill_mask = distill_mask + distill_on_old.float()
                
                # Fallback: If still no pixels selected, trust teacher prediction completely
                # This happens when teacher predicts old class but confidence is very low
                # In this case, we rely on teacher's prediction as the signal
                high_conf_old_count = high_conf_old_mask.sum().item()
                if high_conf_old_count == 0 and teacher_pred_old_count > 0:
                    distill_mask = distill_mask + (became_ignore_mask & teacher_pred_is_old).float()
            else:
                # Fallback: if no teacher info or no old classes, just use all ignore pixels
                # This is safer but less targeted
                distill_mask = distill_mask + became_ignore_mask.float()
            
            # Clamp to [0, 1]
            distill_mask = torch.clamp(distill_mask, 0.0, 1.0)
        
        # H×W → pool (same as any MAD-style mask): nearest keeps cell values in {0, 1}
        return self._hw_mask_to_pool_flat(distill_mask, target_size, mode="nearest")

    def _create_distillation_mask_hw(
        self,
        labels: torch.Tensor,
        current_classes: List[int],
        original_labels: Optional[torch.Tensor] = None,
        old_classes: Optional[List[int]] = None,
        is_replay: bool = False,
        ignore_index: int = 255,
        teacher_predictions: Optional[torch.Tensor] = None,
        teacher_probs: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Return MAD mask directly in label resolution (B, H, W)."""
        if current_classes is not None:
            try:
                current_classes = [int(c) for c in current_classes if c is not None]
            except (ValueError, TypeError):
                current_classes = []
        else:
            current_classes = []
        if old_classes is not None:
            try:
                old_classes = [int(c) for c in old_classes if c is not None]
            except (ValueError, TypeError):
                old_classes = []
        else:
            old_classes = []

        if is_replay:
            # Replay samples: distill only on non-ignore pixels.
            return (labels != ignore_index).float()

        ignore_mask = labels == ignore_index
        if original_labels is not None:
            became_ignore_mask = ignore_mask & (original_labels != ignore_index)
        else:
            became_ignore_mask = ignore_mask
        distill_mask = torch.zeros_like(labels, dtype=torch.float32)
        if (
            became_ignore_mask.any()
            and teacher_predictions is not None
            and teacher_probs is not None
            and len(old_classes) > 0
        ):
            teacher_pred_is_old = torch.zeros_like(labels, dtype=torch.bool)
            for c in old_classes:
                teacher_pred_is_old = teacher_pred_is_old | (teacher_predictions == int(c))
            max_all_prob, _ = teacher_probs.max(dim=1)
            high_conf_mask = max_all_prob > self.mad_relaxed_threshold
            distill_on_old = became_ignore_mask & high_conf_mask & teacher_pred_is_old
            distill_mask = distill_mask + distill_on_old.float()
            if distill_on_old.sum().item() == 0:
                distill_mask = distill_mask + (became_ignore_mask & teacher_pred_is_old).float()
        else:
            distill_mask = distill_mask + became_ignore_mask.float()
        return torch.clamp(distill_mask, 0.0, 1.0)

    def _hw_mask_to_query_flat(
        self,
        mask_hw: torch.Tensor,
        q_len: int,
        mode: str = "nearest",
        query_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Resize (B,H,W) mask to stage-query length (B,Q) via 2D resize (no 1D sequence interpolate)."""
        B = mask_hw.shape[0]
        if query_hw is not None:
            hq, wq = int(query_hw[0]), int(query_hw[1])
            if hq * wq != int(q_len):
                raise ValueError(
                    f"query_hw {hq}x{wq} must multiply to q_len={q_len}"
                )
            m4 = mask_hw.unsqueeze(1).float()
            if mode == "nearest":
                m4 = F.interpolate(m4, size=(hq, wq), mode="nearest")
                m = (m4.squeeze(1) >= 0.5).float()
            else:
                m4 = F.interpolate(m4, size=(hq, wq), mode="bilinear", align_corners=False)
                m = m4.squeeze(1)
            return m.reshape(B, -1)
        side = math.isqrt(int(q_len))
        if side * side == int(q_len):
            m = self._hw_mask_to_pool_flat(mask_hw, side, mode=mode)
            return m[:, :q_len]
        raise ValueError(
            f"Non-square q_len={q_len} requires attention_query_hw (H,W) for 2D mask resize."
        )
    
    def _process_attention(self, attn: torch.Tensor) -> torch.Tensor:
        """Process attention tensor for comparison.
        
        Args:
            attn: Attention tensor (B, num_heads, Q, K) or (B, 1, H, W)
            
        Returns:
            processed: (B, pool_size * pool_size) flattened attention map
        """
        if attn.dim() == 4:
            B, H, Q, K = attn.shape
            
            if H == 1:
                # Spatial attention from features: (B, 1, H, W)
                attn_2d = attn
            else:
                # Native attention: (B, num_heads, seq_q, seq_k)
                if self.head_aggregation == 'mean':
                    attn_2d = attn.mean(dim=1)  # (B, Q, K)
                else:
                    attn_2d = attn.max(dim=1)[0]
                attn_2d = attn_2d.unsqueeze(1)  # (B, 1, Q, K)
            
            # Resize to fixed size for consistent comparison
            attn_2d = F.interpolate(
                attn_2d,
                size=(self.pool_size, self.pool_size),
                mode='bilinear',
                align_corners=False
            )  # (B, 1, pool_size, pool_size)
            
            # Flatten to vector for cosine similarity
            attn_flat = attn_2d.view(B, -1)  # (B, pool_size * pool_size)
            
            return attn_flat
            
        elif attn.dim() == 3:
            B = attn.shape[0]
            attn_2d = attn.unsqueeze(1)
            attn_2d = F.interpolate(
                attn_2d,
                size=(self.pool_size, self.pool_size),
                mode='bilinear',
                align_corners=False
            )
            return attn_2d.view(B, -1)
        else:
            return attn.flatten(start_dim=1)

    def _stu_spatial_entropy_weights(
        self,
        s_attn: Optional[torch.Tensor],
        s_flat: torch.Tensor,
        query_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Per-pooled-pixel entropy-based weights (B, N) with values in [0, 1] before gamma.

        If native attention (B, num_heads, Q, K) with Q == K is available, uses row-wise
        softmax entropy H_q = -sum_k p(k|q) log p(k|q), normalized by log(K), then resized
        to (pool_size, pool_size).

        Otherwise falls back to marginal terms (-p_i log p_i) / log(N) on the pooled flat
        vector (one scalar per spatial index).
        """
        eps_val = float(self.eps) if isinstance(self.eps, (int, float)) else 1e-8
        B, N = s_flat.shape[0], s_flat.shape[-1]
        device = s_flat.device
        dtype = s_flat.dtype

        if (
            s_attn is not None
            and s_attn.dim() == 4
            and s_attn.shape[2] == s_attn.shape[3]
            and int(s_attn.shape[2]) >= 2
        ):
            if self.head_aggregation == "mean":
                a = s_attn.mean(dim=1)
            else:
                a = s_attn.max(dim=1)[0]
            # (B, Q, K), Q == K
            p = self._normalize_attention(a, dim=-1)
            H_q = -(p * torch.log(p + eps_val)).sum(dim=-1)
            Kk = int(a.shape[-1])
            max_h = torch.log(torch.tensor(float(Kk), device=device, dtype=dtype)) + eps_val
            H_norm = torch.clamp(H_q / max_h, 0.0, 1.0)
            Qlen = int(H_norm.shape[1])
            qh, qw = self._resolve_query_layout(s_attn, query_hw)
            if qh * qw != Qlen:
                qh, qw = self._pair_hw_for_len(Qlen, prefer=query_hw)
            h_map = H_norm.view(B, 1, qh, qw)
            h_map = F.interpolate(
                h_map,
                size=(self.pool_size, self.pool_size),
                mode="bilinear",
                align_corners=False,
            )
            w = h_map.view(B, -1)
            return self._apply_entropy_gamma_power(w)

        p = self._normalize_attention(s_flat, dim=-1)
        marg = -p * torch.log(p + eps_val)
        denom = torch.log(torch.tensor(float(N), device=device, dtype=dtype)) + eps_val
        w = torch.clamp(marg / denom, 0.0, 1.0)
        return self._apply_entropy_gamma_power(w)

    def _stu_per_pixel_difference_weights(self, s_flat: torch.Tensor, t_flat: torch.Tensor) -> torch.Tensor:
        """Per-pooled-pixel JSD contribution weight in [0, 1].

        We compute contribution of each position to JSD(s || t), then normalize by ln(2):
            w_i = 0.5 * [ p_i * log(p_i / m_i) + q_i * log(q_i / m_i) ] / ln(2)
        where p and q are directly normalized from input tensors (no softmax).
        """
        eps_val = float(self.eps) if isinstance(self.eps, (int, float)) else 1e-8
        p = self._normalize_attention(s_flat, dim=-1)
        q = self._normalize_attention(t_flat, dim=-1)
        m = 0.5 * (p + q)
        jsd_contrib = 0.5 * (
            p * (torch.log(p + eps_val) - torch.log(m + eps_val))
            + q * (torch.log(q + eps_val) - torch.log(m + eps_val))
        )
        ln2 = torch.log(torch.tensor(2.0, device=s_flat.device, dtype=s_flat.dtype)).clamp_min(eps_val)
        w = jsd_contrib / ln2
        return torch.clamp(w, min=0.0, max=1.0)

    def _margin_flat_from_teacher_probs(
        self,
        teacher_probs: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Top1/top2 probability ratio margin flattened to (B, pool_size**2).

        ``pooled``: resize teacher probs directly to (pool, pool) (legacy).
        ``label``: resize to label map (H, W), compute margin per pixel, then bilinear to (pool, pool).
        Attention loss still uses length N = pool_size**2; this only changes how clear/boundary is defined.
        """
        eps_val = float(self.eps) if isinstance(self.eps, (int, float)) else 1e-8
        p = self.pool_size

        if self.stu_margin_space == "label" and labels is not None and labels.dim() >= 2:
            H, W = int(labels.shape[-2]), int(labels.shape[-1])
            tp = F.interpolate(
                teacher_probs,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            top2_probs, _ = torch.topk(tp, k=2, dim=1)
            top1_prob = top2_probs[:, 0, :, :]
            top2_prob = top2_probs[:, 1, :, :]
            margin = top1_prob / (top2_prob + eps_val)
            return self._hw_mask_to_pool_flat(margin, p, mode="bilinear")

        teacher_probs_resized = F.interpolate(
            teacher_probs,
            size=(p, p),
            mode="bilinear",
            align_corners=False,
        )
        top2_probs, _ = torch.topk(teacher_probs_resized, k=2, dim=1)
        top1_prob = top2_probs[:, 0, :, :]
        top2_prob = top2_probs[:, 1, :, :]
        margin = top1_prob / (top2_prob + eps_val)
        return margin.flatten(1)

    def _attention_to_spatial_map(self, attn: torch.Tensor) -> torch.Tensor:
        """Convert attention tensor to a spatial map tensor (B, 1, h, w)."""
        if attn.dim() != 4:
            return attn.flatten(start_dim=1).unsqueeze(1).unsqueeze(-1)
        B, H, Q, K = attn.shape
        if H == 1:
            return attn
        if self.head_aggregation == "mean":
            a = attn.mean(dim=1)  # (B, Q, K)
        else:
            a = attn.max(dim=1)[0]
        return a.unsqueeze(1)  # (B, 1, Q, K)

    def _normalize_attention(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Clamp attention values for numerical stability (no re-normalization)."""
        _ = dim
        eps_val = float(self.eps) if isinstance(self.eps, (int, float)) else 1e-8
        return torch.clamp(x, min=eps_val)

    def _flat_resize_bilinear(
        self,
        x: torch.Tensor,
        src_h: int,
        src_w: int,
        dst_h: int,
        dst_w: int,
    ) -> torch.Tensor:
        """Reshape flat (B, src_h*src_w) → 2D bilinear → flat (B, dst_h*dst_w). No 1D linear."""
        B = x.shape[0]
        n_src = int(src_h) * int(src_w)
        n_dst = int(dst_h) * int(dst_w)
        if x.shape[-1] < n_src:
            x = F.pad(x, (0, n_src - x.shape[-1]))
        else:
            x = x[..., :n_src]
        if n_src == n_dst:
            return x
        dist_2d = x.reshape(B, 1, int(src_h), int(src_w)).contiguous()
        dist_2d_aligned = F.interpolate(
            dist_2d,
            size=(int(dst_h), int(dst_w)),
            mode="bilinear",
            align_corners=False,
        )
        return dist_2d_aligned.reshape(B, -1)

    def _pair_hw_for_len(
        self, n: int, prefer: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """Pick (H, W) with H*W == n for explicit 2D grids (pool, encoder hint, or 1×n)."""
        n = int(n)
        if prefer is not None:
            h, w = int(prefer[0]), int(prefer[1])
            if h * w == n:
                return (h, w)
        p = int(self.pool_size)
        if p * p == n:
            return (p, p)
        r = math.isqrt(n)
        if r * r == n:
            return (r, r)
        return (1, n)

    def _resize_1d_weights_bilinear2d(
        self, x: torch.Tensor, src_len: int, dst_len: int
    ) -> torch.Tensor:
        """Backward-compatible name: resize using explicit 2D bilinear only (no 1D linear)."""
        if src_len == dst_len:
            if x.shape[-1] >= src_len:
                return x[..., :src_len]
            return F.pad(x, (0, src_len - x.shape[-1]))
        sh, sw = self._pair_hw_for_len(src_len)
        dh, dw = self._pair_hw_for_len(dst_len)
        return self._flat_resize_bilinear(x, sh, sw, dh, dw)

    def _resolve_query_layout(
        self,
        s_attn: torch.Tensor,
        hint_hw: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """(H, W) with H*W == query length for 2D bilinear on per-query weights."""
        if s_attn.dim() == 4 and s_attn.shape[1] == 1:
            # (B,1,H,W) → aggregated (B,H,W); downstream q_len uses dim=H (see shape[2] of 4D).
            hq = int(s_attn.shape[2])
            return (hq, 1)
        q_len = int(s_attn.shape[2])
        if hint_hw is not None:
            h, w = int(hint_hw[0]), int(hint_hw[1])
            if h * w == q_len:
                return (h, w)
        return self._pair_hw_for_len(q_len, prefer=hint_hw)

    def _aggregate_attention_qk(self, attn: torch.Tensor) -> torch.Tensor:
        """Aggregate heads and return (B, Q, K)."""
        if attn.dim() != 4:
            raise ValueError(f"Expected 4D attention, got shape={tuple(attn.shape)}")
        if attn.shape[1] == 1:
            return attn[:, 0, :, :]
        if self.head_aggregation == "mean":
            return attn.mean(dim=1)
        return attn.max(dim=1)[0]

    def _query_entropy_jsd_weights(
        self, s_attn: torch.Tensor, t_attn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute per-query entropy and JSD weights in native (B,Q)."""
        eps_val = float(self.eps) if isinstance(self.eps, (int, float)) else 1e-8
        s_qk = self._aggregate_attention_qk(s_attn)
        t_qk = self._aggregate_attention_qk(t_attn)
        q_len = min(s_qk.shape[1], t_qk.shape[1])
        k_len = min(s_qk.shape[2], t_qk.shape[2])
        s_qk = s_qk[:, :q_len, :k_len]
        t_qk = t_qk[:, :q_len, :k_len]

        p = self._normalize_attention(s_qk, dim=-1)
        q = self._normalize_attention(t_qk, dim=-1)

        entropy_q = -(p * torch.log(p + eps_val)).sum(dim=-1)
        max_h = torch.log(torch.tensor(float(k_len), device=p.device, dtype=p.dtype)) + eps_val
        w_entropy = self._apply_entropy_gamma_power(
            torch.clamp(entropy_q / max_h, 0.0, 1.0)
        )

        m = 0.5 * (p + q)
        jsd_q = 0.5 * (
            (p * (torch.log(p + eps_val) - torch.log(m + eps_val))).sum(dim=-1)
            + (q * (torch.log(q + eps_val) - torch.log(m + eps_val))).sum(dim=-1)
        )
        ln2 = torch.log(torch.tensor(2.0, device=p.device, dtype=p.dtype)).clamp_min(eps_val)
        w_diff = torch.clamp(jsd_q / ln2, 0.0, 1.0)
        return w_entropy, w_diff, q_len

    def _cosine_difference_weights(
        self,
        s_attn: torch.Tensor,
        t_attn: torch.Tensor,
        query_hw: Tuple[int, int],
        dst_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Per-query cosine distance weights, 2D bilinear to ``dst_hw`` grid then flattened."""
        s_qk = self._aggregate_attention_qk(s_attn)
        t_qk = self._aggregate_attention_qk(t_attn)
        q_len = min(s_qk.shape[1], t_qk.shape[1])
        k_len = min(s_qk.shape[2], t_qk.shape[2])
        s_qk = s_qk[:, :q_len, :k_len]
        t_qk = t_qk[:, :q_len, :k_len]
        cos_q = F.cosine_similarity(s_qk, t_qk, dim=-1)
        w_q = torch.clamp(1.0 - cos_q, min=0.0, max=1.0)
        qh, qw = int(query_hw[0]), int(query_hw[1])
        if qh * qw != q_len:
            qh, qw = self._pair_hw_for_len(q_len, prefer=query_hw)
        dh, dw = int(dst_hw[0]), int(dst_hw[1])
        if qh * qw == dh * dw:
            return w_q
        return self._flat_resize_bilinear(w_q, qh, qw, dh, dw)

    def _margin_query_from_teacher_probs(
        self,
        teacher_probs: torch.Tensor,
        labels: Optional[torch.Tensor],
        q_len: int,
        query_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Top1/top2 margin aligned to native query length (B,Q) via 2D resize to ``query_hw``."""
        eps_val = float(self.eps) if isinstance(self.eps, (int, float)) else 1e-8
        hq, wq = int(query_hw[0]), int(query_hw[1])
        if hq * wq != int(q_len):
            hq, wq = self._pair_hw_for_len(int(q_len), prefer=query_hw)
        if self.stu_margin_space == "label" and labels is not None and labels.dim() >= 2:
            H, W = int(labels.shape[-2]), int(labels.shape[-1])
            tp = F.interpolate(teacher_probs, size=(H, W), mode="bilinear", align_corners=False)
        else:
            tp = F.interpolate(
                teacher_probs, size=(hq, wq), mode="bilinear", align_corners=False
            )
        top2_probs, _ = torch.topk(tp, k=2, dim=1)
        margin_hw = top2_probs[:, 0, :, :] / (top2_probs[:, 1, :, :] + eps_val)
        return self._hw_mask_to_query_flat(
            margin_hw, q_len, mode="bilinear", query_hw=(hq, wq)
        )

    def _compute_multihead_query_weighted_kl_loss(
        self,
        student_attn: torch.Tensor,
        teacher_attn: torch.Tensor,
        base_mask_q: torch.Tensor,
        teacher_probs: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        query_hw: Tuple[int, int],
        return_stats: bool = False,
    ):
        """KL loss in (B, heads, Q, K) with per-head adaptive weights.

        For each head h and query i:
            w_adaptive(h, i) = w_difference(h, i) + w_entropy(h, i) * 1(margin_i < threshold)
            L_h = sum_i w(h,i) * KL(A_s(h,i,:), A_t(h,i,:)) / sum_i w(h,i)
        Final loss is mean over heads and batch.
        """
        eps_val = float(self.eps) if isinstance(self.eps, (int, float)) else 1e-8
        if student_attn.dim() != 4 or teacher_attn.dim() != 4:
            raise ValueError("Multi-head KL expects attention tensors with shape (B, heads, Q, K).")

        s = student_attn
        t = teacher_attn
        h_len = min(s.shape[1], t.shape[1])
        q_len = min(s.shape[2], t.shape[2])
        k_len = min(s.shape[3], t.shape[3])
        s = s[:, :h_len, :q_len, :k_len]
        t = t[:, :h_len, :q_len, :k_len]

        # (B, heads, Q, K)
        p = self._normalize_attention(s, dim=-1)
        q = self._normalize_attention(t, dim=-1)

        # Per-head query entropy weight: (B, heads, Q)
        entropy_q = -(p * torch.log(p + eps_val)).sum(dim=-1)
        max_h = torch.log(torch.tensor(float(k_len), device=p.device, dtype=p.dtype)) + eps_val
        w_entropy = self._apply_entropy_gamma_power(
            torch.clamp(entropy_q / max_h, 0.0, 1.0)
        )

        # Per-head query JSD weight: (B, heads, Q)
        m = 0.5 * (p + q)
        jsd_q = 0.5 * (
            (p * (torch.log(p + eps_val) - torch.log(m + eps_val))).sum(dim=-1)
            + (q * (torch.log(q + eps_val) - torch.log(m + eps_val))).sum(dim=-1)
        )
        ln2 = torch.log(torch.tensor(2.0, device=p.device, dtype=p.dtype)).clamp_min(eps_val)
        w_diff = torch.clamp(jsd_q / ln2, 0.0, 1.0)

        # Margin indicator on query axis (shared across heads): (B, Q)
        if teacher_probs is not None:
            margin_q = self._margin_query_from_teacher_probs(
                teacher_probs, labels, q_len, query_hw=query_hw
            )
            is_boundary = (margin_q < self.margin_threshold).to(w_entropy.dtype)
        else:
            is_boundary = torch.zeros(
                (w_entropy.shape[0], q_len), device=w_entropy.device, dtype=w_entropy.dtype
            )

        # Expand base mask and boundary mask to heads
        base = base_mask_q[:, :q_len].unsqueeze(1).expand(-1, h_len, -1)  # (B, heads, Q)
        boundary = is_boundary.unsqueeze(1).expand(-1, h_len, -1)  # (B, heads, Q)

        adaptive = torch.clamp(w_diff + w_entropy * boundary, min=0.0)
        w = base * adaptive
        if self.min_mask_value > 0:
            w = torch.clamp(w, min=self.min_mask_value)

        # Per-head per-query KL: (B, heads, Q)
        kl_q = F.kl_div(
            torch.log(p + eps_val),
            q,
            reduction="none",
        ).sum(dim=-1)
        kl_q = torch.clamp(kl_q, min=0.0)

        # Weighted average over query for each head, then mean over heads and batch
        denom = w.sum(dim=-1).clamp_min(eps_val)  # (B, heads)
        loss_bh = (w * kl_q).sum(dim=-1) / denom  # (B, heads)
        loss = torch.clamp(loss_bh.mean(), min=0.0)
        if not return_stats:
            return loss
        stats = {
            "base_mask_mean": float(base.mean().mean().item()),
            "w_entropy_mean": float(w_entropy.mean().item()),
            "w_diff_mean": float(w_diff.mean().item()),
            "w_adaptive_mean": float(adaptive.mean().item()),
            "w_adaptive_max": float(adaptive.max().item()),
            "w_adaptive_min": float(adaptive.min().item()),
            "As_sum_lastdim_mean": float(s.sum(dim=-1).mean().item()),
            "Ps_sum_lastdim_mean": float(p.sum(dim=-1).mean().item()),
            "T_pixel_mean": None,
        }
        return loss, stats

    def _stu_hw_entropy_diff_weights(
        self,
        s_attn: torch.Tensor,
        t_attn: torch.Tensor,
        labels: Optional[torch.Tensor],
        compute_pixel_jsd: bool = True,
        query_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute entropy (and optionally per-pixel JSD) on label HxW, then pool to (B, N).

        When ``loss_type == "cosine"``, call with ``compute_pixel_jsd=False`` so JSD is not
        evaluated; ``w_diff_pos`` is then ``None`` and the caller sets **W_diff** via
        `_cosine_difference_weights`.
        """
        if labels is None or labels.dim() < 3:
            s_flat = self._process_attention(s_attn)
            t_flat = self._process_attention(t_attn)
            spatial_weight = self._stu_spatial_entropy_weights(s_attn, s_flat, query_hw=query_hw)
            if compute_pixel_jsd:
                w_diff_pos = self._stu_per_pixel_difference_weights(s_flat, t_flat)
            else:
                w_diff_pos = None
            return spatial_weight, w_diff_pos

        eps_val = float(self.eps) if isinstance(self.eps, (int, float)) else 1e-8
        H, W = int(labels.shape[-2]), int(labels.shape[-1])
        s_map = self._attention_to_spatial_map(s_attn)
        s_map = F.interpolate(s_map, size=(H, W), mode="bilinear", align_corners=False)

        B = s_map.shape[0]
        N_hw = H * W
        s_flat_hw = s_map.view(B, -1)

        p_hw = self._normalize_attention(s_flat_hw, dim=-1)
        entropy_pos_hw = -p_hw * torch.log(p_hw + eps_val)
        denom_hw = torch.log(torch.tensor(float(N_hw), device=s_flat_hw.device, dtype=s_flat_hw.dtype)) + eps_val
        spatial_hw = self._apply_entropy_gamma_power(
            torch.clamp(entropy_pos_hw / denom_hw, 0.0, 1.0)
        )

        spatial_weight = self._hw_mask_to_pool_flat(spatial_hw.view(B, H, W), self.pool_size, mode="bilinear")

        if not compute_pixel_jsd:
            return spatial_weight, None

        t_map = self._attention_to_spatial_map(t_attn)
        t_map = F.interpolate(t_map, size=(H, W), mode="bilinear", align_corners=False)
        t_flat_hw = t_map.view(B, -1)
        p_s = self._normalize_attention(s_flat_hw, dim=-1)
        p_t = self._normalize_attention(t_flat_hw, dim=-1)
        m_hw = 0.5 * (p_s + p_t)
        jsd_contrib_hw = 0.5 * (
            p_s * (torch.log(p_s + eps_val) - torch.log(m_hw + eps_val))
            + p_t * (torch.log(p_t + eps_val) - torch.log(m_hw + eps_val))
        )
        ln2_hw = torch.log(torch.tensor(2.0, device=s_flat_hw.device, dtype=s_flat_hw.dtype)).clamp_min(eps_val)
        w_diff_hw = torch.clamp(jsd_contrib_hw / ln2_hw, min=0.0, max=1.0)
        w_diff_pos = self._hw_mask_to_pool_flat(w_diff_hw.view(B, H, W), self.pool_size, mode="bilinear")
        return spatial_weight, w_diff_pos
    
    def _compute_single_stage_loss(
        self,
        student_attn: torch.Tensor,
        teacher_attn: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        base_mask_denom: Optional[torch.Tensor] = None,
        query_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Compute distillation loss for a single stage with optional masking.
        
        Args:
            student_attn: Student attention (B, num_heads, Q, K)
            teacher_attn: Teacher attention (B, num_heads, Q, K)
            mask: Optional spatial mask (B, pool_size * pool_size), 1=distill, 0=skip
            base_mask_denom: Unused for cosine; cosine uses
                :math:`\\sum_q w_q \\ell_q / (\\sum_q w_q + \\varepsilon)` with the same ``mask`` as :math:`w_q`.
            
        Returns:
            loss: Scalar loss value
        """
        if query_hw is None:
            query_hw = self._resolve_query_layout(student_attn, None)

        # Process both to (B, pool_size * pool_size)
        s_flat = self._process_attention(student_attn)
        t_flat = self._process_attention(teacher_attn)
        
        # Ensure same size
        if s_flat.shape != t_flat.shape:
            min_len = min(s_flat.shape[-1], t_flat.shape[-1])
            s_flat = s_flat[..., :min_len]
            t_flat = t_flat[..., :min_len]
        
        # Apply mask if provided (Masked Attention Distillation)
        if mask is not None:
            # For cosine/mse/l1 losses we need mask aligned to flattened (B, N).
            # For KL (query-weighted), mask is used later as query weights and may be non-square (B, Q),
            # so do not force square reshape here.
            if self.loss_type == 'cosine':
                if mask.shape[-1] != s_flat.shape[-1]:
                    mlen, nlen = int(mask.shape[-1]), int(s_flat.shape[-1])
                    msh, msw = self._pair_hw_for_len(mlen)
                    nsh, nsw = self._pair_hw_for_len(nlen)
                    mask = self._flat_resize_bilinear(mask, msh, msw, nsh, nsw)
                s_flat = s_flat * mask
                t_flat = t_flat * mask
            
            # Count valid positions for proper averaging
            valid_ratio = mask.mean()
            if valid_ratio < 0.01:
                # Too few valid positions, skip this loss
                import logging as lg
                lg.debug(f"Attention KD: valid_ratio={valid_ratio.item():.6f} < 0.01, skipping loss")
                return torch.tensor(0.0, device=s_flat.device)
        
        # Query-weighted KL: mask only applies on query dimension weights, not on attention entries.
        # Formula: sum_i w_i * KL(A_s(i, :), A_t(i, :))
        if (
            self.loss_type == 'kl'
            and student_attn.dim() == 4
            and teacher_attn.dim() == 4
            and student_attn.shape[1] > 1
            and teacher_attn.shape[1] > 1
        ):
            if self.head_aggregation == 'mean':
                s_qk = student_attn.mean(dim=1)  # (B, Q, K)
                t_qk = teacher_attn.mean(dim=1)  # (B, Q, K)
            else:
                s_qk = student_attn.max(dim=1)[0]
                t_qk = teacher_attn.max(dim=1)[0]

            q_len = min(s_qk.shape[1], t_qk.shape[1])
            k_len = min(s_qk.shape[2], t_qk.shape[2])
            s_qk = s_qk[:, :q_len, :k_len]
            t_qk = t_qk[:, :q_len, :k_len]

            s_prob = self._normalize_attention(s_qk, dim=-1)
            t_prob = self._normalize_attention(t_qk, dim=-1)

            kl_per_query = F.kl_div(
                torch.log(s_prob + 1e-8),
                t_prob,
                reduction='none'
            ).sum(dim=-1)  # (B, Q)
            kl_per_query = torch.clamp(kl_per_query, min=0.0)

            if mask is not None:
                qh, qw = int(query_hw[0]), int(query_hw[1])
                if qh * qw != int(q_len):
                    qh, qw = self._pair_hw_for_len(int(q_len), prefer=query_hw)
                mlen = int(mask.shape[-1])
                msh, msw = self._pair_hw_for_len(mlen)
                w_q = self._flat_resize_bilinear(mask, msh, msw, qh, qw)
                w_q = torch.clamp(w_q, min=0.0)
                denom = w_q.sum(dim=-1).clamp_min(1e-8)
                loss = ((w_q * kl_per_query).sum(dim=-1) / denom).mean()
            else:
                loss = kl_per_query.mean()
            return torch.clamp(loss, min=0.0)

        # Compute loss based on type
        if self.loss_type == 'cosine':
            # Compute full cosine distance first, then scale per-position loss by mask.
            if student_attn.dim() == 4 and teacher_attn.dim() == 4:
                s_qk = self._aggregate_attention_qk(student_attn)  # (B,Q,K)
                t_qk = self._aggregate_attention_qk(teacher_attn)  # (B,Q,K)
                q_len = min(s_qk.shape[1], t_qk.shape[1])
                k_len = min(s_qk.shape[2], t_qk.shape[2])
                s_qk = s_qk[:, :q_len, :k_len]
                t_qk = t_qk[:, :q_len, :k_len]
                dist_q = torch.clamp(1.0 - F.cosine_similarity(s_qk, t_qk, dim=-1), min=0.0)  # (B,Q)
                if mask is not None:
                    qh, qw = int(query_hw[0]), int(query_hw[1])
                    if qh * qw != int(q_len):
                        qh, qw = self._pair_hw_for_len(int(q_len), prefer=query_hw)
                    mlen = int(mask.shape[-1])
                    msh, msw = self._pair_hw_for_len(mlen)
                    w_q = self._flat_resize_bilinear(mask, msh, msw, qh, qw)
                    w_q = torch.clamp(w_q, min=0.0)
                    eps_val = float(self.eps) if isinstance(self.eps, (int, float)) else 1e-8
                    denom = w_q.sum(dim=-1) + eps_val
                    loss = ((w_q * dist_q).sum(dim=-1) / denom).mean()
                else:
                    loss = dist_q.mean()
            else:
                s_norm = F.normalize(s_flat, p=2, dim=-1)
                t_norm = F.normalize(t_flat, p=2, dim=-1)
                dist_pos = torch.clamp(1.0 - s_norm * t_norm, min=0.0)
                if mask is not None:
                    w = torch.clamp(mask, min=0.0)
                    eps_val = float(self.eps) if isinstance(self.eps, (int, float)) else 1e-8
                    denom = w.sum(dim=-1) + eps_val
                    loss = ((w * dist_pos).sum(dim=-1) / denom).mean()
                else:
                    loss = dist_pos.mean()
            
        elif self.loss_type == 'mse':
            # Normalize to unit length first (for fair comparison)
            s_norm = F.normalize(s_flat, p=2, dim=-1)
            t_norm = F.normalize(t_flat, p=2, dim=-1)
            loss = F.mse_loss(s_norm, t_norm)
            
        elif self.loss_type == 'l1':
            s_norm = F.normalize(s_flat, p=2, dim=-1)
            t_norm = F.normalize(t_flat, p=2, dim=-1)
            loss = F.l1_loss(s_norm, t_norm)
            
        elif self.loss_type == 'kl':
            # KL divergence - treat as probability distributions
            s_prob = self._normalize_attention(s_flat, dim=-1)
            t_prob = self._normalize_attention(t_flat, dim=-1)
            loss = F.kl_div(
                torch.log(s_prob + 1e-8),
                t_prob,
                reduction='batchmean'
            )
            loss = torch.clamp(loss, min=0.0)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def forward(
        self,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        original_labels: Optional[torch.Tensor] = None,
        current_classes: Optional[List[int]] = None,
        old_classes: Optional[List[int]] = None,
        is_replay: bool = False,
        teacher_predictions: Optional[torch.Tensor] = None,
        teacher_probs: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.5,
        attention_query_hw: Optional[List[Optional[Tuple[int, int]]]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute attention map distillation loss with optional masking.
        
        Args:
            student_attentions: List of student attention tensors
            teacher_attentions: List of teacher attention tensors
            attention_query_hw: Per-stage (H, W) from the encoder so flat weights use
                explicit 2D bilinear (no 1D linear). Same length as attention lists.
            labels: Optional ground truth labels for mask creation (B, H, W)
            current_classes: Optional list of current task class IDs
            old_classes: Optional list of old task class IDs (for replay samples)
            is_replay: Whether these are replay samples
            
        Returns:
            total_loss: Weighted sum of stage losses
            stage_losses: Dict with individual stage losses
        """
        if len(student_attentions) == 0 or len(teacher_attentions) == 0:
            import logging as lg
            lg.debug(f"Attention KD: Empty attention lists - student={len(student_attentions)}, teacher={len(teacher_attentions)}")
            return torch.tensor(0.0), {'attn_loss': 0.0}
        
        device = student_attentions[0].device if student_attentions[0] is not None else 'cpu'
        total_loss = torch.tensor(0.0, device=device)
        stage_losses = {}
        
        # Create base mask for Masked Attention Distillation (MAD)
        base_mask = None
        base_mask_hw = None
        if self.use_masked_distillation and labels is not None and self.loss_type == "kl":
            base_mask_hw = self._create_distillation_mask_hw(
                labels,
                current_classes=current_classes if current_classes else [],
                original_labels=original_labels,
                old_classes=old_classes,
                is_replay=is_replay,
                teacher_predictions=teacher_predictions,
                teacher_probs=teacher_probs,
                confidence_threshold=confidence_threshold,
            )
            stage_losses['mad_mask_ratio'] = base_mask_hw.mean().item()
        elif self.use_masked_distillation and labels is not None:
            base_mask = self._create_distillation_mask(
                labels, 
                current_classes=current_classes if current_classes else [],
                target_size=self.pool_size,
                original_labels=original_labels,
                old_classes=old_classes,
                is_replay=is_replay,
                teacher_predictions=teacher_predictions,
                teacher_probs=teacher_probs,
                confidence_threshold=confidence_threshold
            )
            stage_losses['mad_mask_ratio'] = base_mask.mean().item()

        for i, (s_attn, t_attn) in enumerate(zip(student_attentions, teacher_attentions)):
            if s_attn is None or t_attn is None:
                import logging as lg
                lg.debug(f"Attention KD Stage {i}: None attention - student={s_attn is None}, teacher={t_attn is None}")
                stage_losses[f'attn_stage_{i}_loss'] = 0.0
                continue
            
            try:
                hint_hw = None
                if attention_query_hw is not None and i < len(attention_query_hw):
                    hint_hw = attention_query_hw[i]
                query_hw = self._resolve_query_layout(s_attn, hint_hw)

                # If STU is enabled and we have a base mask, construct the STU-Mask
                # NOTE: STU mask can be too aggressive and filter out too many pixels
                # If base_mask ratio is reasonable but stu_mask ratio is very low,
                # consider disabling STU or using base_mask directly
                if self.stu_enabled and (base_mask is not None or base_mask_hw is not None):
                    if self.loss_type == "kl" and s_attn.dim() == 4 and t_attn.dim() == 4:
                        q_len = int(min(s_attn.shape[2], t_attn.shape[2]))
                        if base_mask_hw is not None:
                            stage_base_mask = self._hw_mask_to_query_flat(
                                base_mask_hw, q_len, mode="nearest", query_hw=query_hw
                            )
                        else:
                            stage_base_mask = torch.ones(
                                s_attn.shape[0], q_len, device=s_attn.device, dtype=s_attn.dtype
                            )
                        if self.debug_mask_stats:
                            stage_loss, dbg_stats = self._compute_multihead_query_weighted_kl_loss(
                                s_attn,
                                t_attn,
                                base_mask_q=stage_base_mask,
                                teacher_probs=teacher_probs,
                                labels=labels,
                                query_hw=query_hw,
                                return_stats=True,
                            )
                            for k, v in dbg_stats.items():
                                stage_losses[f"attn_stage_{i}_{k}"] = v
                        else:
                            stage_loss = self._compute_multihead_query_weighted_kl_loss(
                                s_attn,
                                t_attn,
                                base_mask_q=stage_base_mask,
                                teacher_probs=teacher_probs,
                                labels=labels,
                                query_hw=query_hw,
                                return_stats=False,
                            )
                        weight = self.stage_weights[i] if i < len(self.stage_weights) else 1.0
                        try:
                            weight = float(weight)
                        except (ValueError, TypeError):
                            weight = 1.0
                        total_loss = total_loss + weight * stage_loss
                        stage_losses[f'attn_stage_{i}_loss'] = stage_loss.item()
                        continue

                    # Process attentions to flattened spatial vectors
                    s_flat = self._process_attention(s_attn)  # (B, N)
                    t_flat = self._process_attention(t_attn)  # (B, N)
                    if s_flat.shape != t_flat.shape:
                        min_len = min(s_flat.shape[-1], t_flat.shape[-1])
                        s_flat = s_flat[..., :min_len]
                        t_flat = t_flat[..., :min_len]

                    eps_val = float(self.eps) if isinstance(self.eps, (int, float)) else 1e-8

                    if self.stu_weight_mode == "local_pixel":
                        # Per pooled spatial index (B, N); N must match s_flat / margin_flat / base_mask
                        spatial_weight = self._stu_spatial_entropy_weights(
                            s_attn, s_flat, query_hw=query_hw
                        )
                        if self.loss_type != "cosine":
                            w_diff_pos = self._stu_per_pixel_difference_weights(s_flat, t_flat)
                        else:
                            w_diff_pos = None
                        N_loc = s_flat.shape[-1]
                        if spatial_weight.shape[-1] != N_loc:
                            sl = int(spatial_weight.shape[-1])
                            sh, sw = self._pair_hw_for_len(sl)
                            dh, dw = self._pair_hw_for_len(int(N_loc))
                            spatial_weight = self._flat_resize_bilinear(
                                spatial_weight, sh, sw, dh, dw
                            )
                    elif self.stu_weight_mode == "label_pixel":
                        spatial_weight, w_diff_pos = self._stu_hw_entropy_diff_weights(
                            s_attn=s_attn,
                            t_attn=t_attn,
                            labels=labels,
                            compute_pixel_jsd=(self.loss_type != "cosine"),
                            query_hw=query_hw,
                        )
                        N_loc = s_flat.shape[-1]
                        if spatial_weight.shape[-1] != N_loc:
                            sl = int(spatial_weight.shape[-1])
                            sh, sw = self._pair_hw_for_len(sl)
                            dh, dw = self._pair_hw_for_len(int(N_loc))
                            spatial_weight = self._flat_resize_bilinear(
                                spatial_weight, sh, sw, dh, dw
                            )
                        if w_diff_pos is not None and w_diff_pos.shape[-1] != N_loc:
                            wl = int(w_diff_pos.shape[-1])
                            wh, ww = self._pair_hw_for_len(wl)
                            dh, dw = self._pair_hw_for_len(int(N_loc))
                            w_diff_pos = self._flat_resize_bilinear(
                                w_diff_pos, wh, ww, dh, dw
                            )
                        temporal_comp = None
                    else:
                        # Global: one scalar per image, broadcast to (B, N)
                        p = self._normalize_attention(s_flat, dim=-1)
                        entropy_pos = -p * torch.log(p + eps_val)
                        entropy_sum = entropy_pos.sum(dim=-1, keepdim=True)
                        N = s_flat.shape[-1]
                        max_entropy = torch.log(
                            torch.tensor(N, dtype=torch.float32, device=s_flat.device)
                        ) + eps_val
                        normalized_entropy = entropy_sum / max_entropy
                        spatial_weight = self._apply_entropy_gamma_power(normalized_entropy)
                        spatial_weight = spatial_weight.expand_as(entropy_pos)

                        if self.loss_type != "cosine":
                            q = self._normalize_attention(t_flat, dim=-1)
                            m = 0.5 * (p + q)
                            jsd = 0.5 * (
                                (p * (torch.log(p + eps_val) - torch.log(m + eps_val))).sum(dim=-1)
                                + (q * (torch.log(q + eps_val) - torch.log(m + eps_val))).sum(dim=-1)
                            )
                            ln2 = torch.log(
                                torch.tensor(2.0, dtype=s_flat.dtype, device=s_flat.device)
                            ).clamp_min(eps_val)
                            temporal_comp = torch.clamp(jsd / ln2, min=0.0, max=1.0)
                        else:
                            temporal_comp = None
                        w_diff_pos = None

                    if self.loss_type == "cosine":
                        N_loc = s_flat.shape[-1]
                        ph = pw = int(self.pool_size)
                        w_diff_pos = self._cosine_difference_weights(
                            s_attn=s_attn,
                            t_attn=t_attn,
                            query_hw=query_hw,
                            dst_hw=(ph, pw),
                        )
                        temporal_comp = None

                    # Teacher confidence-guided adaptive strategy
                    if teacher_probs is not None:
                        margin_flat = self._margin_flat_from_teacher_probs(teacher_probs, labels)
                        N_loc = s_flat.shape[-1]
                        if margin_flat.shape[-1] != N_loc:
                            ml = int(margin_flat.shape[-1])
                            mh, mw = self._pair_hw_for_len(ml)
                            dh, dw = self._pair_hw_for_len(int(N_loc))
                            margin_flat = self._flat_resize_bilinear(
                                margin_flat, mh, mw, dh, dw
                            )

                        if self.loss_type == "cosine":
                            temporal_b = (
                                w_diff_pos
                                if w_diff_pos is not None
                                else temporal_comp.view(-1, 1).expand_as(spatial_weight)
                            )
                            if self.stu_disable_margin_gate:
                                adaptive_weight = torch.clamp(
                                    temporal_b
                                    + self.cosine_entropy_alpha * spatial_weight,
                                    min=0.0,
                                )
                            else:
                                is_boundary = margin_flat < self.margin_threshold  # (B, N)
                                adaptive_weight = torch.clamp(
                                    temporal_b
                                    + self.cosine_entropy_alpha
                                    * spatial_weight
                                    * is_boundary.to(spatial_weight.dtype),
                                    min=0.0,
                                )
                        else:
                            is_clear = margin_flat >= self.margin_threshold
                            clear_weight = spatial_weight
                            if w_diff_pos is not None:
                                boundary_weight = torch.clamp(
                                    w_diff_pos + spatial_weight,
                                    min=0.0,
                                )
                            else:
                                boundary_weight = torch.clamp(
                                    temporal_comp.view(-1, 1) + spatial_weight,
                                    min=0.0,
                                )
                            adaptive_weight = torch.where(
                                is_clear,
                                clear_weight,
                                boundary_weight,
                            )

                        stu_mask = base_mask * adaptive_weight
                    else:
                        if w_diff_pos is not None:
                            stu_mask = base_mask * spatial_weight * w_diff_pos
                        else:
                            stu_mask = base_mask * spatial_weight * temporal_comp.view(-1, 1)
                    
                    # Apply minimum mask value to ensure some gradient flow
                    # This prevents mask from being too small even when entropy is high or l2_diff is small
                    if self.min_mask_value > 0:
                        stu_mask = torch.clamp(stu_mask, min=self.min_mask_value)
                    
                    # Check if STU mask is too aggressive (filters out >90% of base_mask)
                    # If so, use base_mask directly to ensure sufficient distillation
                    if (
                        self.stu_fallback_enabled
                        and base_mask.mean().item() > 0.01
                        and stu_mask.mean().item() / base_mask.mean().item() < self.stu_fallback_ratio
                    ):
                        stage_loss = self._compute_single_stage_loss(
                            s_attn,
                            t_attn,
                            base_mask,
                            base_mask_denom=base_mask,
                            query_hw=query_hw,
                        )
                    else:
                        # Use STU mask for this stage
                        stage_loss = self._compute_single_stage_loss(
                            s_attn,
                            t_attn,
                            stu_mask,
                            base_mask_denom=base_mask,
                            query_hw=query_hw,
                        )
                else:
                    # Fallback: use binary/distillation mask (0-1 mask) when STU is disabled
                    # Skip all AAE-Mask calculations (entropy, difference, margin) for faster training
                    if self.loss_type == "kl" and base_mask_hw is not None and s_attn.dim() == 4:
                        q_len = int(min(s_attn.shape[2], t_attn.shape[2]))
                        stage_mask = self._hw_mask_to_query_flat(
                            base_mask_hw, q_len, mode="nearest", query_hw=query_hw
                        )
                        stage_loss = self._compute_single_stage_loss(
                            s_attn, t_attn, stage_mask, query_hw=query_hw
                        )
                    elif base_mask is None:
                        # Create a full mask (all 1s) for 0-1 masking fallback
                        # This ensures distillation happens on all positions when STU is disabled
                        # Use pool_size directly to avoid processing attention tensor
                        B = s_attn.shape[0]
                        N = self.pool_size * self.pool_size  # Standard flattened size
                        base_mask = torch.ones(B, N, device=s_attn.device, dtype=torch.float32)
                        stage_loss = self._compute_single_stage_loss(
                            s_attn,
                            t_attn,
                            base_mask,
                            base_mask_denom=base_mask,
                            query_hw=query_hw,
                        )
                    else:
                        stage_loss = self._compute_single_stage_loss(
                            s_attn,
                            t_attn,
                            base_mask,
                            base_mask_denom=base_mask,
                            query_hw=query_hw,
                        )
                
                weight = self.stage_weights[i] if i < len(self.stage_weights) else 1.0
                # Ensure weight is a float
                try:
                    weight = float(weight)
                except (ValueError, TypeError):
                    weight = 1.0
                weighted_loss = weight * stage_loss
                total_loss = total_loss + weighted_loss
                
                stage_losses[f'attn_stage_{i}_loss'] = stage_loss.item()
            except Exception as e:
                import logging as lg
                import traceback
                error_msg = f"Attention KD Stage {i} error: {type(e).__name__}: {e}"
                lg.warning(error_msg)
                lg.debug(f"Traceback: {traceback.format_exc()}")
                stage_losses[f'attn_stage_{i}_loss'] = 0.0
                stage_losses[f'attn_stage_{i}_error'] = str(e)

        return total_loss, stage_losses


class CombinedAttentionDistillationLoss(nn.Module):
    """Combined attention and logit distillation loss.
    
    Combines:
    - Attention map distillation (pattern matching)
    - Logit-level KD (output alignment)
    """
    
    def __init__(
        self,
        attention_weight: float = 1.0,
        logit_weight: float = 1.0,
        temperature: float = 3.0,
        num_active_stages: int = 2,
        stage_weights: Optional[List[float]] = None,
        attention_loss_type: str = 'cosine'
    ):
        super().__init__()
        
        self.attention_weight = attention_weight
        self.logit_weight = logit_weight
        self.temperature = temperature
        
        self.attention_distill = AttentionMapDistillationLoss(
            num_active_stages=num_active_stages,
            stage_weights=stage_weights,
            loss_type=attention_loss_type,
            pool_size=32
        )
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute combined distillation loss.
        
        Args:
            student_logits: Student output logits (B, C, H, W)
            teacher_logits: Teacher output logits (B, C, H, W)
            student_attentions: List of student attention tensors
            teacher_attentions: List of teacher attention tensors
            
        Returns:
            total_loss: Combined loss
            logit_loss: Logit-level KD loss
            stage_losses: Dict with attention stage losses
        """
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        
        logit_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (T ** 2)
        
        attention_loss, stage_losses = self.attention_distill(
            student_attentions,
            teacher_attentions,
            attention_query_hw=None,
        )
        
        total_loss = (
            self.logit_weight * logit_loss +
            self.attention_weight * attention_loss
        )
        
        return total_loss, logit_loss, stage_losses


