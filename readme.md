# Online Continual Semantic Segmentation · Adaptive Attention Masking (AAM)


面向道路场景的**在线类增量语义分割**：在单次遍历、标签受限设定下缓解**背景偏移**与**灾难性遗忘**；在 Transformer **Query 维度**用**连续自适应掩码（AAM）**调节注意力蒸馏强度，并结合 **EMA 双分支**与**多级余弦注意力蒸馏（MAD）**。

---

## Highlights

- **AAM** — soft weights in $[0,1]$ on attention distillation, preserving long-range context vs. hard 0–1 masks.
- **MAD** — cosine alignment of teacher/student self-attention maps at selected SegFormer stages.
- **EMA + replay** — momentum teacher and reservoir buffer compatible with strict online updates.
- **Configs** — YAML-driven experiments under `config/seg/`.

---

## Method overview

| Component | Role |
|-----------|------|
| **EMA dual models** | Student is optimized each step; teacher is an EMA of student weights — a stable anchor for old knowledge. |
| **MAD** | Align deep self-attention maps (cosine similarity), not only final logits. |
| **AAM** | **Base mask** suppresses teacher guidance on new-class GT regions; relaxes into old-class confident background. **Adaptive weights** fuse **teacher–student attention discrepancy** and **attention entropy** at each Query, then multiply the cosine distillation term for smooth, state-dependent strength. |

Hard binary masks cut global attention links; **AAM** keeps a **smooth transition** at class boundaries and ambiguous regions.

---

## Quick start

### Requirements

```bash
pip install -r requirements.txt
pip install "transformers>=4.30.0" "timm>=0.9.0"
```

### Data

Prepare **Cityscapes** (or your dataset) as in [`README_SEGMENTATION.md`](README_SEGMENTATION.md).

### Train (example)

```bash
python main_seg.py --config config/seg/cityscapes_ema_attention.yaml
```

Use `--learner ER_EMA_Attention_Seg` (or the name set in your YAML). Parser and flags: `config/seg_parser.py`.

---

## Code map

| Topic | Location |
|-------|----------|
| Training entry | [`main_seg.py`](main_seg.py) |
| AAM / attention KD loss | [`src_attention/utils/attention_distillation.py`](src_attention/utils/attention_distillation.py) |
| Learner (ER + EMA + attention) | [`src_attention/learners/segmentation/er_seg_attention.py`](src_attention/learners/segmentation/er_seg_attention.py) |
| SegFormer + attention | [`src_attention/models/segformer_attention.py`](src_attention/models/segformer_attention.py) |
| Learner registry | [`src/utils/seg_name_match.py`](src/utils/seg_name_match.py) |
| Baseline registry (no attention package) | [`src_baseline/utils/seg_name_match.py`](src_baseline/utils/seg_name_match.py) |
| Segmentation datasets & loaders | [`src/datasets/`](src/datasets/), [`src/utils/seg_data.py`](src/utils/seg_data.py) |

Hyperparameters (stage weights, pooling size, fusion coefficient, thresholds, …) are **YAML-first** — align with your paper tables by checking the config you actually ran.

---

## Documentation

| File | Content |
|------|---------|
| [`README_SEGMENTATION.md`](README_SEGMENTATION.md) | Segmentation extension: datasets, losses, metrics, learners table |
| [`README_ICML24.md`](README_ICML24.md) | Original **image-classification** OCL + MKD (ICML 2024) — `main.py`, `config/icml24/` |

---

## Citation

**MKD / ICML codebase** (this repo’s classification branch):

```bibtex
@InProceedings{pmlr-v235-michel24a,
  title     = {Rethinking Momentum Knowledge Distillation in Online Continual Learning},
  author    = {Michel, Nicolas and Wang, Maorong and Xiao, Ling and Yamasaki, Toshihiko},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages     = {35607--35622},
  year      = {2024},
  volume    = {235},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v235/michel24a.html}
}
```
