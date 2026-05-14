"""SegFormer model wrapper for semantic segmentation.

This module provides a wrapper around the HuggingFace SegFormer implementation
for use in online continual learning experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class SegFormer(nn.Module):
    """SegFormer wrapper for semantic segmentation.
    
    Args:
        n_classes: Number of segmentation classes
        pretrained: Model variant ('mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5')
        img_size: Input image size (H, W)
        freeze_encoder: Whether to freeze the encoder (backbone)
    """
    
    MODEL_CONFIGS = {
        'mit_b0': 'nvidia/segformer-b0-finetuned-ade-512-512',
        'mit_b1': 'nvidia/segformer-b1-finetuned-ade-512-512',
        'mit_b2': 'nvidia/segformer-b2-finetuned-ade-512-512',
        'mit_b3': 'nvidia/segformer-b3-finetuned-ade-512-512',
        'mit_b4': 'nvidia/segformer-b4-finetuned-ade-512-512',
        'mit_b5': 'nvidia/segformer-b5-finetuned-ade-640-640',
    }
    
    def __init__(
        self,
        n_classes=19,
        pretrained='mit_b0',
        img_size=(512, 1024),
        freeze_encoder=False
    ):
        super().__init__()
        self.n_classes = n_classes
        self.img_size = img_size
        self.pretrained = pretrained
        
        # Load pretrained model and modify for Cityscapes classes
        if pretrained in self.MODEL_CONFIGS:
            model_name = self.MODEL_CONFIGS[pretrained]
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=n_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Create model from scratch with specified config
            config = SegformerConfig(
                num_labels=n_classes,
                num_encoder_blocks=4,
                depths=[2, 2, 2, 2] if 'b0' in pretrained else [3, 4, 6, 3],
                hidden_sizes=[32, 64, 160, 256] if 'b0' in pretrained else [64, 128, 320, 512],
                decoder_hidden_size=256,
            )
            self.model = SegformerForSemanticSegmentation(config)
        
        if freeze_encoder:
            self._freeze_encoder()
            
    def _freeze_encoder(self):
        """Freeze encoder weights."""
        for param in self.model.segformer.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.model.segformer.encoder.parameters():
            param.requires_grad = True
    
    def forward(self, x, return_features=False):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            logits: Segmentation logits of shape (B, n_classes, H, W)
            features: (optional) Dictionary of intermediate features
        """
        outputs = self.model(pixel_values=x, output_hidden_states=return_features)
        logits = outputs.logits
        
        # Upsample logits to input size
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        if return_features:
            return logits, outputs.hidden_states
        return logits
    
    def features(self, x):
        """Extract encoder features.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            features: Last hidden state from encoder
        """
        outputs = self.model.segformer(pixel_values=x, output_hidden_states=True)
        return outputs.hidden_states[-1]
    
    def logits(self, x):
        """Get segmentation logits.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            logits: Segmentation logits upsampled to input size
        """
        return self.forward(x)


class SegFormerWithProjection(nn.Module):
    """SegFormer with additional projection head for contrastive learning.
    
    This model adds a projection head that maps pixel features to an embedding space,
    useful for pixel-wise contrastive learning approaches.
    
    Args:
        n_classes: Number of segmentation classes
        pretrained: Model variant
        proj_dim: Dimension of projection output
        img_size: Input image size (H, W)
    """
    
    def __init__(
        self,
        n_classes=19,
        pretrained='mit_b0',
        proj_dim=256,
        img_size=(512, 1024)
    ):
        super().__init__()
        self.segformer = SegFormer(
            n_classes=n_classes,
            pretrained=pretrained,
            img_size=img_size
        )
        
        # Get feature dimension from model config
        if 'b0' in pretrained:
            feat_dim = 256
        else:
            feat_dim = 512
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, proj_dim, kernel_size=1)
        )
        
    def forward(self, x, return_proj=False):
        """Forward pass.
        
        Args:
            x: Input tensor
            return_proj: Whether to return projection features
            
        Returns:
            logits: Segmentation logits
            proj: (optional) Projection features
        """
        logits, hidden_states = self.segformer(x, return_features=True)
        
        if return_proj:
            # Use last hidden state for projection
            feat = hidden_states[-1]
            proj = self.projection(feat)
            # Upsample projection to input size
            proj = F.interpolate(
                proj,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            proj = F.normalize(proj, dim=1)
            return logits, proj
            
        return logits
    
    def features(self, x):
        """Extract encoder features."""
        return self.segformer.features(x)
    
    def logits(self, x):
        """Get segmentation logits."""
        return self.forward(x)


class IncrementalSegFormer(nn.Module):
    """SegFormer for class-incremental semantic segmentation.
    
    This model supports dynamic expansion of the classification head
    as new classes are learned incrementally.
    
    Args:
        initial_classes: Initial number of classes
        pretrained: Model variant
        img_size: Input image size (H, W)
    """
    
    def __init__(
        self,
        initial_classes=19,
        pretrained='mit_b0',
        img_size=(512, 1024)
    ):
        super().__init__()
        self.n_classes = initial_classes
        self.pretrained = pretrained
        self.img_size = img_size
        
        self.model = SegFormer(
            n_classes=initial_classes,
            pretrained=pretrained,
            img_size=img_size
        )
        
        # Store old classifier for knowledge distillation
        self.old_classifier = None
        
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def expand_classes(self, new_n_classes):
        """Expand the classification head for new classes.
        
        Args:
            new_n_classes: New total number of classes
        """
        if new_n_classes <= self.n_classes:
            return
            
        # Store old classifier
        old_weight = self.model.model.decode_head.classifier.weight.data.clone()
        old_bias = self.model.model.decode_head.classifier.bias.data.clone()
        
        # Create new classifier
        in_channels = old_weight.shape[1]
        self.model.model.decode_head.classifier = nn.Conv2d(
            in_channels, new_n_classes, kernel_size=1
        )
        
        # Copy old weights
        self.model.model.decode_head.classifier.weight.data[:self.n_classes] = old_weight
        self.model.model.decode_head.classifier.bias.data[:self.n_classes] = old_bias
        
        # Initialize new class weights
        nn.init.kaiming_normal_(
            self.model.model.decode_head.classifier.weight.data[self.n_classes:]
        )
        nn.init.zeros_(
            self.model.model.decode_head.classifier.bias.data[self.n_classes:]
        )
        
        self.n_classes = new_n_classes
        
    def features(self, x):
        """Extract encoder features."""
        return self.model.features(x)
    
    def logits(self, x):
        """Get segmentation logits."""
        return self.forward(x)


def create_segformer(
    n_classes=19,
    pretrained='mit_b0',
    img_size=(512, 1024),
    model_type='standard',
    **kwargs
):
    """Factory function to create SegFormer models.
    
    Args:
        n_classes: Number of classes
        pretrained: Model variant
        img_size: Input image size
        model_type: 'standard', 'projection', or 'incremental'
        **kwargs: Additional arguments for specific model types
        
    Returns:
        model: SegFormer model instance
    """
    if model_type == 'standard':
        return SegFormer(
            n_classes=n_classes,
            pretrained=pretrained,
            img_size=img_size
        )
    elif model_type == 'projection':
        return SegFormerWithProjection(
            n_classes=n_classes,
            pretrained=pretrained,
            img_size=img_size,
            proj_dim=kwargs.get('proj_dim', 256)
        )
    elif model_type == 'incremental':
        return IncrementalSegFormer(
            initial_classes=n_classes,
            pretrained=pretrained,
            img_size=img_size
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

