"""Loss functions for semantic segmentation.

This module provides various loss functions commonly used in
semantic segmentation, including specialized losses for
continual learning scenarios.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):
    """Cross entropy loss for 2D segmentation masks.
    
    Args:
        weight: Class weights for imbalanced datasets
        ignore_index: Label to ignore in loss computation
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction
        )
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) predicted logits
            targets: (B, H, W) ground truth labels
            
        Returns:
            loss: Scalar loss value
        """
        return self.ce_loss(logits, targets)


class DiceLoss(nn.Module):
    """Dice loss for semantic segmentation.
    
    Dice loss is useful for handling class imbalance.
    
    Args:
        smooth: Smoothing factor to prevent division by zero
        ignore_index: Label to ignore
        reduction: 'mean' or 'none' for per-class losses
    """
    
    def __init__(self, smooth=1.0, ignore_index=255, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) predicted logits
            targets: (B, H, W) ground truth labels
            
        Returns:
            loss: Dice loss value
        """
        n_classes = logits.shape[1]
        
        # Create one-hot encoding of targets
        mask = targets != self.ignore_index
        targets_valid = targets.clone()
        targets_valid[~mask] = 0
        
        # Convert to one-hot
        targets_one_hot = F.one_hot(targets_valid, n_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Apply mask
        mask = mask.unsqueeze(1).float()  # (B, 1, H, W)
        targets_one_hot = targets_one_hot * mask
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        probs = probs * mask
        
        # Compute Dice per class
        dims = (0, 2, 3)  # Sum over batch and spatial dims
        intersection = (probs * targets_one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        if self.reduction == 'mean':
            return 1 - dice.mean()
        else:
            return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for semantic segmentation.
    
    Focal loss down-weights well-classified examples and focuses
    on hard examples.
    
    Args:
        alpha: Class weighting factor
        gamma: Focusing parameter (higher = more focus on hard examples)
        ignore_index: Label to ignore
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) predicted logits
            targets: (B, H, W) ground truth labels
            
        Returns:
            loss: Focal loss value
        """
        n_classes = logits.shape[1]
        
        # Create mask for valid pixels
        mask = targets != self.ignore_index
        targets_valid = targets.clone()
        targets_valid[~mask] = 0
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            logits, targets_valid,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Get probabilities of ground truth class
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha = self.alpha
            else:
                alpha = self.alpha.gather(0, targets_valid.view(-1)).view_as(targets_valid)
            focal_weight = alpha * focal_weight
        
        # Apply mask
        focal_loss = focal_weight * ce_loss * mask.float()
        
        if self.reduction == 'mean':
            return focal_loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined CE and Dice loss.
    
    Args:
        ce_weight: Weight for cross entropy loss
        dice_weight: Weight for dice loss
        ignore_index: Label to ignore
    """
    
    def __init__(self, ce_weight=1.0, dice_weight=1.0, ignore_index=255, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = CrossEntropyLoss2d(
            weight=class_weights,
            ignore_index=ignore_index
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) predicted logits
            targets: (B, H, W) ground truth labels
            
        Returns:
            loss: Combined loss value
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class UnbiasedCrossEntropy(nn.Module):
    """Unbiased cross entropy for class-incremental learning.
    
    This loss handles the background shift problem in incremental
    semantic segmentation by properly handling old classes.
    
    Args:
        old_classes: List of old class indices
        ignore_index: Label to ignore
    """
    
    def __init__(self, old_classes=None, ignore_index=255):
        super().__init__()
        self.old_classes = old_classes if old_classes is not None else []
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets, old_logits=None):
        """
        Args:
            logits: (B, C, H, W) current model logits
            targets: (B, H, W) ground truth labels
            old_logits: (B, C_old, H, W) old model logits (optional)
            
        Returns:
            loss: Unbiased CE loss value
        """
        n_classes = logits.shape[1]
        
        # Standard CE for new class pixels
        mask = targets != self.ignore_index
        
        # For pixels belonging to old classes, use soft labels from old model
        if old_logits is not None and len(self.old_classes) > 0:
            # Get old model predictions
            old_probs = F.softmax(old_logits, dim=1)
            
            # Create soft targets
            targets_soft = F.one_hot(
                targets.clamp(0, n_classes - 1),
                n_classes
            ).permute(0, 3, 1, 2).float()
            
            # Replace old class labels with soft labels
            for c in self.old_classes:
                if c < old_probs.shape[1]:
                    class_mask = (targets == c).unsqueeze(1)
                    targets_soft[:, c:c+1] = torch.where(
                        class_mask,
                        old_probs[:, c:c+1],
                        targets_soft[:, c:c+1]
                    )
            
            # Compute KL divergence loss
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(targets_soft * log_probs).sum(dim=1)
            loss = (loss * mask.float()).sum() / mask.sum().clamp(min=1)
        else:
            # Standard CE
            loss = F.cross_entropy(
                logits, targets,
                ignore_index=self.ignore_index
            )
        
        return loss


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for continual learning.
    
    Args:
        temperature: Softmax temperature for distillation
        alpha: Weight for distillation loss vs CE loss
    """
    
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, student_logits, teacher_logits, targets, ignore_index=255):
        """
        Args:
            student_logits: (B, C, H, W) student model logits
            teacher_logits: (B, C_old, H, W) teacher model logits
            targets: (B, H, W) ground truth labels
            ignore_index: Label to ignore
            
        Returns:
            loss: Combined CE + KD loss
        """
        # Standard CE loss
        ce_loss = F.cross_entropy(
            student_logits, targets,
            ignore_index=ignore_index
        )
        
        # KD loss on old class predictions
        n_old_classes = teacher_logits.shape[1]
        
        # Get soft labels from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Get student predictions for old classes only
        student_log_probs = F.log_softmax(
            student_logits[:, :n_old_classes] / self.temperature,
            dim=1
        )
        
        # KL divergence
        kd_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return (1 - self.alpha) * ce_loss + self.alpha * kd_loss


class PixelContrastiveLoss(nn.Module):
    """Pixel-wise contrastive loss for semantic segmentation.
    
    Pulls together features of same-class pixels and pushes apart
    features of different-class pixels.
    
    Args:
        temperature: Temperature for contrastive loss
        base_temperature: Base temperature for normalization
        ignore_index: Label to ignore
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07, ignore_index=255):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_index = ignore_index
        
    def forward(self, features, targets, n_samples_per_class=256):
        """
        Args:
            features: (B, D, H, W) feature maps
            targets: (B, H, W) ground truth labels
            n_samples_per_class: Number of pixels to sample per class
            
        Returns:
            loss: Pixel contrastive loss
        """
        batch_size, feat_dim, h, w = features.shape
        
        # Reshape features and targets
        features = features.permute(0, 2, 3, 1).reshape(-1, feat_dim)  # (B*H*W, D)
        targets = targets.reshape(-1)  # (B*H*W,)
        
        # Filter out ignored pixels
        valid_mask = targets != self.ignore_index
        features = features[valid_mask]
        targets = targets[valid_mask]
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Sample pixels per class
        unique_classes = targets.unique()
        sampled_features = []
        sampled_targets = []
        
        for c in unique_classes:
            class_mask = targets == c
            class_features = features[class_mask]
            
            if len(class_features) > n_samples_per_class:
                indices = torch.randperm(len(class_features))[:n_samples_per_class]
                class_features = class_features[indices]
            
            sampled_features.append(class_features)
            sampled_targets.append(torch.full((len(class_features),), c, device=targets.device))
        
        features = torch.cat(sampled_features, dim=0)
        targets = torch.cat(sampled_targets, dim=0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(features, features.t()) / self.temperature
        
        # Create positive mask
        targets = targets.unsqueeze(1)
        positive_mask = torch.eq(targets, targets.t()).float()
        
        # Mask out self-similarity
        identity_mask = torch.eye(len(features), device=features.device)
        positive_mask = positive_mask * (1 - identity_mask)
        
        # Compute log softmax
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits) * (1 - identity_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        
        # Compute mean log likelihood over positives
        positive_per_sample = positive_mask.sum(dim=1)
        positive_per_sample = positive_per_sample.clamp(min=1)
        
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_per_sample
        
        # Loss
        loss = -mean_log_prob.mean() * (self.temperature / self.base_temperature)
        
        return loss


class ACELoss(nn.Module):
    """Asymmetric Cross Entropy loss for continual learning.
    
    Modified CE that handles old and new classes asymmetrically.
    """
    
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets, old_classes=None, new_classes=None):
        """
        Args:
            logits: (B, C, H, W) predicted logits
            targets: (B, H, W) ground truth labels
            old_classes: List of old class indices
            new_classes: List of new class indices
            
        Returns:
            loss: ACE loss value
        """
        if old_classes is None or new_classes is None:
            # Standard CE if no class info
            return F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        
        n_classes = logits.shape[1]
        mask = targets != self.ignore_index
        
        # Separate loss for old and new class pixels
        old_mask = torch.zeros_like(targets, dtype=torch.bool)
        new_mask = torch.zeros_like(targets, dtype=torch.bool)
        
        for c in old_classes:
            old_mask = old_mask | (targets == c)
        for c in new_classes:
            new_mask = new_mask | (targets == c)
        
        loss = 0.0
        count = 0
        
        # Loss on old class pixels (use masked softmax)
        if old_mask.any():
            old_logits = logits.clone()
            # Mask new classes for old pixel predictions
            for c in new_classes:
                if c < n_classes:
                    old_logits[:, c] = -float('inf')
            
            old_loss = F.cross_entropy(
                old_logits, targets,
                ignore_index=self.ignore_index,
                reduction='none'
            )
            loss = loss + (old_loss * old_mask.float() * mask.float()).sum()
            count += (old_mask & mask).sum()
        
        # Loss on new class pixels (standard CE)
        if new_mask.any():
            new_loss = F.cross_entropy(
                logits, targets,
                ignore_index=self.ignore_index,
                reduction='none'
            )
            loss = loss + (new_loss * new_mask.float() * mask.float()).sum()
            count += (new_mask & mask).sum()
        
        return loss / count.clamp(min=1)

