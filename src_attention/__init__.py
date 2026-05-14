"""src_attention - Attention Map Distillation for Continual Semantic Segmentation.

This module provides attention-level knowledge distillation for SegFormer,
enabling the student model to learn "where to look" from the teacher's
attention patterns.

Core Idea:
- SegFormer is based on Transformer with Self-Attention mechanism
- Teacher model trained on old tasks knows "where to look" to recognize objects
- Teacher's Attention Map (Query × Key) contains this knowledge
- Aligning Student and Teacher attention forces student to "mimic teacher's gaze"

Formula:
    L_Attn = Σ_l weight_l × (1 - cosine_similarity(Attn_S^l, Attn_T^l))

Version: v3 (Attention Map Distillation with Cosine Loss)
"""

__version__ = '3.0.0'












