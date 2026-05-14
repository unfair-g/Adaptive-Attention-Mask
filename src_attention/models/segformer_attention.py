"""SegFormer model with attention map extraction for knowledge distillation.

This module extends the standard SegFormer to support attention map extraction,
enabling attention-level knowledge distillation to help preserve "where to look"
knowledge for recognizing objects.

Version: v3 (Attention Map Distillation)

Core Idea:
- SegFormer is based on Transformer architecture with Self-Attention mechanism
- Teacher model trained on old tasks knows "where to look" to recognize objects
- For difficult classes (e.g., small objects), model needs to attend to specific context regions
- Teacher's Attention Map (Query × Key) contains this "attention focus" knowledge
- Aligning Student and Teacher attention maps forces student to "mimic teacher's gaze"

Note: SegFormer uses "Efficient Self-Attention" with spatial reduction.
We extract attention from the Mix-FFN blocks' self-attention layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import logging as lg


class SegFormerAttention(nn.Module):
    """SegFormer with attention map extraction for knowledge distillation.
    
    This model extends the standard SegFormer to return attention maps
    from each encoder block, enabling attention-level knowledge distillation.
    
    SegFormer encoder produces 4 stages of features and attention maps:
        - Stage 0: 1/4 resolution (fine-grained, good for small objects)
        - Stage 1: 1/8 resolution
        - Stage 2: 1/16 resolution  
        - Stage 3: 1/32 resolution (semantic-level features)
    
    Args:
        n_classes: Number of segmentation classes
        pretrained: Model variant ('mit_b0', 'mit_b1', ..., 'mit_b5')
        img_size: Input image size (H, W)
        freeze_encoder: Whether to freeze the encoder
    """
    
    MODEL_CONFIGS = {
        'mit_b0': 'nvidia/segformer-b0-finetuned-ade-512-512',
        'mit_b1': 'nvidia/segformer-b1-finetuned-ade-512-512',
        'mit_b2': 'nvidia/segformer-b2-finetuned-ade-512-512',
        'mit_b3': 'nvidia/segformer-b3-finetuned-ade-512-512',
        'mit_b4': 'nvidia/segformer-b4-finetuned-ade-512-512',
        'mit_b5': 'nvidia/segformer-b5-finetuned-ade-640-640',
    }
    
    # Feature dimensions for each stage
    FEATURE_DIMS = {
        'mit_b0': [32, 64, 160, 256],
        'mit_b1': [64, 128, 320, 512],
        'mit_b2': [64, 128, 320, 512],
        'mit_b3': [64, 128, 320, 512],
        'mit_b4': [64, 128, 320, 512],
        'mit_b5': [64, 128, 320, 512],
    }
    
    # Number of transformer blocks per stage (for attention extraction)
    STAGE_DEPTHS = {
        'mit_b0': [2, 2, 2, 2],
        'mit_b1': [2, 2, 2, 2],
        'mit_b2': [3, 4, 6, 3],
        'mit_b3': [3, 4, 18, 3],
        'mit_b4': [3, 8, 27, 3],
        'mit_b5': [3, 6, 40, 3],
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
        self.num_stages = 4  # SegFormer has 4 encoder stages
        
        # Get feature dimensions and depths for this variant
        self.feature_dims = self.FEATURE_DIMS.get(pretrained, [32, 64, 160, 256])
        self.stage_depths = self.STAGE_DEPTHS.get(pretrained, [2, 2, 2, 2])
        
        # Load pretrained model
        if pretrained in self.MODEL_CONFIGS:
            model_name = self.MODEL_CONFIGS[pretrained]
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=n_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = SegformerConfig(
                num_labels=n_classes,
                num_encoder_blocks=4,
                depths=[2, 2, 2, 2] if 'b0' in pretrained else [3, 4, 6, 3],
                hidden_sizes=self.feature_dims,
                decoder_hidden_size=256,
            )
            self.model = SegformerForSemanticSegmentation(config)
        
        # Check if attention output is supported
        self._attention_supported = self._check_attention_support()
        if not self._attention_supported:
            lg.warning("Attention output not supported by this SegFormer version. "
                      "Falling back to feature-based spatial attention.")
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _check_attention_support(self):
        """Check if the model supports attention output."""
        try:
            # Try a dummy forward pass
            dummy_input = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                outputs = self.model.segformer(
                    pixel_values=dummy_input,
                    output_attentions=True,
                    return_dict=True
                )
            return outputs.attentions is not None and len(outputs.attentions) > 0
        except Exception as e:
            lg.warning(f"Attention check failed: {e}")
            return False
    
    def _freeze_encoder(self):
        """Freeze encoder weights."""
        for param in self.model.segformer.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.model.segformer.encoder.parameters():
            param.requires_grad = True
    
    def get_encoder_features(self, x):
        """Extract multi-scale encoder features.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            features: List of 4 feature tensors from each encoder stage
                - features[0]: (B, C1, H/4, W/4)   - Stage 0 (fine-grained)
                - features[1]: (B, C2, H/8, W/8)   - Stage 1
                - features[2]: (B, C3, H/16, W/16) - Stage 2
                - features[3]: (B, C4, H/32, W/32) - Stage 3 (semantic)
        """
        outputs = self.model.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states
        
        if len(hidden_states) == 5:
            features = list(hidden_states[1:])
        elif len(hidden_states) == 4:
            features = list(hidden_states)
        else:
            if len(hidden_states) > 4:
                features = list(hidden_states[-4:])
            else:
                raise ValueError(
                    f"Expected 4 or 5 hidden states from SegFormer encoder, "
                    f"but got {len(hidden_states)}. "
                    f"Shapes: {[h.shape for h in hidden_states]}"
                )
        
        return features
    
    def _compute_spatial_attention_from_features(self, features, active_stages=None):
        """Compute spatial attention maps from feature maps.
        
        This is a fallback when native attention output is not available.
        We compute spatial attention as the channel-wise average of feature activations,
        which indicates which spatial locations are "active".
        
        Args:
            features: List of feature tensors from each stage
            active_stages: Which stages to compute attention for
            
        Returns:
            attention_maps: List of spatial attention tensors (B, 1, H, W)
            attention_query_hw: Parallel list of (H, W) for each map (native grid size).
        """
        if active_stages is None:
            active_stages = [2, 3]
        
        attention_maps = []
        attention_query_hw = []
        for stage_idx in active_stages:
            if stage_idx < len(features):
                feat = features[stage_idx]  # (B, C, H, W)
                # Compute spatial attention as mean activation
                spatial_attn = feat.mean(dim=1, keepdim=True)  # (B, 1, H, W)
                # Normalize to [0, 1]
                spatial_attn = F.sigmoid(spatial_attn)
                attention_maps.append(spatial_attn)
                # Loss treats dim2 as query length (see _aggregate_attention_qk on (B,1,H,W) → (B,H,W)).
                attention_query_hw.append((int(spatial_attn.shape[2]), 1))
            else:
                attention_maps.append(None)
                attention_query_hw.append(None)
        
        return attention_maps, attention_query_hw

    def _aggregate_native_stage_attentions(
        self, all_attentions, encoder_features, active_stages
    ):
        """Average per-block attentions per stage; attach (H, W) from encoder feature maps."""
        attention_maps = []
        attention_query_hw = []
        block_idx = 0
        for stage_idx in range(self.num_stages):
            stage_depth = self.stage_depths[stage_idx]
            stage_attentions = []
            for _ in range(stage_depth):
                if block_idx < len(all_attentions):
                    stage_attentions.append(all_attentions[block_idx])
                block_idx += 1
            if stage_idx in active_stages:
                if stage_idx < len(encoder_features):
                    feat = encoder_features[stage_idx]
                    hq, wq = int(feat.shape[-2]), int(feat.shape[-1])
                else:
                    hq, wq = None, None
                if len(stage_attentions) > 0:
                    stacked = torch.stack(stage_attentions, dim=0)
                    avg_attention = stacked.mean(dim=0)
                    attention_maps.append(avg_attention)
                    if hq is not None and wq is not None:
                        attention_query_hw.append((hq, wq))
                    else:
                        attention_query_hw.append(None)
                else:
                    attention_maps.append(None)
                    attention_query_hw.append(None)
        return attention_maps, attention_query_hw
    
    def get_attention_maps(self, x, active_stages=None):
        """Extract attention maps from specified encoder stages.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            active_stages: List of stage indices to extract attention from
                          Default: [2, 3] (last two stages - most semantic)
        
        Returns:
            attention_maps: List of attention tensors for each active stage
                           Each tensor: (B, num_heads, seq_len_q, seq_len_k)
            attention_query_hw: Same length; each entry is (H, W) for query tokens
                (SegFormer stage feature map height/width), or None if map is missing.
        """
        if active_stages is None:
            active_stages = [2, 3]
        
        outputs = self.model.segformer(
            pixel_values=x,
            output_hidden_states=True,
            output_attentions=self._attention_supported,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states
        if len(hidden_states) == 5:
            encoder_features = list(hidden_states[1:])
        else:
            encoder_features = list(hidden_states[-4:])

        if self._attention_supported and outputs.attentions is not None:
            return self._aggregate_native_stage_attentions(
                outputs.attentions, encoder_features, active_stages
            )
        return self._compute_spatial_attention_from_features(
            encoder_features, active_stages
        )
    
    def forward(self, x, return_attention=False, active_attention_stages=None):
        """Forward pass with optional attention map extraction.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            return_attention: Whether to return attention maps for distillation
            active_attention_stages: Which stages to extract attention from
                                    Default: [2, 3] (last two stages)
            
        Returns:
            logits: Segmentation logits of shape (B, n_classes, H, W)
            attention_maps: List of attention maps for active stages
            attention_query_hw: Parallel list of (H, W) query grid sizes from the encoder
                (no sqrt inference in the loss — use these for 2D reshape + bilinear).
        """
        if return_attention:
            if active_attention_stages is None:
                active_attention_stages = [2, 3]
            
            # Get features and attention
            outputs = self.model.segformer(
                pixel_values=x,
                output_hidden_states=True,
                output_attentions=self._attention_supported,
                return_dict=True
            )
            
            # Get encoder features
            hidden_states = outputs.hidden_states
            if len(hidden_states) == 5:
                encoder_features = list(hidden_states[1:])
            else:
                encoder_features = list(hidden_states[-4:])
            
            # Get logits
            logits = self.model.decode_head(encoder_features)
            logits = F.interpolate(
                logits,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            if self._attention_supported and outputs.attentions is not None:
                attention_maps, attention_query_hw = self._aggregate_native_stage_attentions(
                    outputs.attentions, encoder_features, active_attention_stages
                )
            else:
                attention_maps, attention_query_hw = self._compute_spatial_attention_from_features(
                    encoder_features, active_attention_stages
                )
            
            return logits, attention_maps, attention_query_hw
        
        else:
            outputs = self.model(pixel_values=x)
            logits = outputs.logits
            
            logits = F.interpolate(
                logits,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            return logits
    
    def features(self, x):
        """Extract last encoder feature (for compatibility)."""
        outputs = self.model.segformer(pixel_values=x, output_hidden_states=True)
        return outputs.hidden_states[-1]
    
    def logits(self, x):
        """Get segmentation logits."""
        return self.forward(x)
    
    def get_feature_dims(self):
        """Get feature dimensions for each stage."""
        return self.feature_dims
    
    def get_stage_depths(self):
        """Get number of transformer blocks per stage."""
        return self.stage_depths
    
    def attention_supported(self):
        """Check if native attention output is supported."""
        return self._attention_supported


def create_segformer_attention(
    n_classes=19,
    pretrained='mit_b0',
    img_size=(512, 1024),
    **kwargs
):
    """Factory function to create SegFormer with attention extraction."""
    return SegFormerAttention(
        n_classes=n_classes,
        pretrained=pretrained,
        img_size=img_size
    )


