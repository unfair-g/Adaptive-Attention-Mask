"""Name matching for segmentation learners and buffers."""

from src.learners.segmentation.er_seg import (
    ERSegmentationLearner,
    ERSegmentationDERppLearner,
    ERSegmentationEMALearner,
    ACESegmentationLearner,
)
from src.buffers.seg_reservoir import (
    SegmentationReservoir,
    SegmentationLogitsReservoir,
)

from src_attention.learners.segmentation.er_seg_attention import (
    ERSegmentationEMAAttentionLearner as ERSegmentationEMAAttentionLearner_amd,
)

ERSegmentationEMAAttentionLearner_val = None
try:
    from src_val.learners.segmentation import er_seg_attention_val as _val_seg_module

    _cls = getattr(
        _val_seg_module,
        "ERSegmentationEMAAttentionLearnerVal",
        getattr(_val_seg_module, "ERSegmentationEMAAttentionLearner", None),
    )
    if _cls is not None:
        ERSegmentationEMAAttentionLearner_val = _cls
except ImportError:
    pass

seg_learners = {
    "ER_Seg": ERSegmentationLearner,
    "DERpp_Seg": ERSegmentationDERppLearner,
    "ER_EMA_Seg": ERSegmentationEMALearner,
    "ACE_Seg": ACESegmentationLearner,
    "ERSegmentationEMAAttention": ERSegmentationEMAAttentionLearner_amd,
    "ER_EMA_Attention_Seg": ERSegmentationEMAAttentionLearner_amd,
    "ER_EMA_AMD_Seg": ERSegmentationEMAAttentionLearner_amd,
    "ERSegmentationEMAAttention_v3": ERSegmentationEMAAttentionLearner_amd,
    "ER_EMA_Attention_Seg_v3": ERSegmentationEMAAttentionLearner_amd,
    "ERSegmentationEMARKD": ERSegmentationEMAAttentionLearner_amd,
    "ER_EMA_RKD_Seg": ERSegmentationEMAAttentionLearner_amd,
    "ERSegmentationEMAMultiScale": ERSegmentationEMAAttentionLearner_amd,
    "ER_EMA_MultiScale_Seg": ERSegmentationEMAAttentionLearner_amd,
}

if ERSegmentationEMAAttentionLearner_val is not None:
    seg_learners["ER_EMA_Attention_Seg_Val"] = ERSegmentationEMAAttentionLearner_val

seg_buffers = {
    "seg_reservoir": SegmentationReservoir,
    "seg_logits_reservoir": SegmentationLogitsReservoir,
}


def get_seg_learner(name):
    """Return the segmentation learner class for the given registry name."""
    if name not in seg_learners:
        raise ValueError(
            f"Unknown segmentation learner: {name}. "
            f"Available: {list(seg_learners.keys())}"
        )
    return seg_learners[name]


def get_seg_buffer(name):
    """Return the segmentation buffer class for the given registry name."""
    if name not in seg_buffers:
        raise ValueError(
            f"Unknown segmentation buffer: {name}. "
            f"Available: {list(seg_buffers.keys())}"
        )
    return seg_buffers[name]
