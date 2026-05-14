# Online Continual Semantic Segmentation with SegFormer

This extension adds support for class-incremental online continual semantic segmentation using SegFormer on the Cityscapes dataset.

## Overview

The code extends the existing online continual learning framework to support:
- **SegFormer** as the backbone model for semantic segmentation
- **Cityscapes** dataset with class-incremental splits
- Segmentation-specific **memory buffers** for image-mask pairs
- Segmentation **loss functions** (CE, Dice, Focal, etc.)
- Segmentation **evaluation metrics** (mIoU, per-class IoU, forgetting)

## Installation

Install additional dependencies:

```bash
pip install transformers>=4.30.0 timm>=0.9.0
```

## Dataset Setup

Download Cityscapes dataset from https://www.cityscapes-dataset.com/

Expected directory structure:
```
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   │   ├── aachen/
│   │   └── ...
│   └── val/
│       └── ...
└── gtFine/
    ├── train/
    │   ├── aachen/
    │   └── ...
    └── val/
        └── ...
```

## Quick Start

### Run ER-based segmentation:

```bash
python main_seg.py --config config/seg/cityscapes_er.yaml
```

### Run DER++ segmentation:

```bash
python main_seg.py --config config/seg/cityscapes_derpp.yaml
```

### Run with EMA teacher:

```bash
python main_seg.py --config config/seg/cityscapes_ema.yaml
```

## Available Learners

| Learner | Description |
|---------|-------------|
| `ER_Seg` | Experience Replay with reservoir sampling |
| `DERpp_Seg` | Dark Experience Replay++ with logit distillation |
| `ER_EMA_Seg` | ER with Exponential Moving Average teacher |
| `ACE_Seg` | Asymmetric Cross-Entropy for class-incremental learning |

## Key Components

### Datasets (`src/datasets/cityscapes.py`)
- `Cityscapes`: Base dataset class
- `SplitCityscapes`: Class-incremental splits
- `BlurryCityscapes`: Blurry task boundaries

### Models (`src/models/segformer.py`)
- `SegFormer`: Standard SegFormer wrapper
- `SegFormerWithProjection`: With contrastive projection head
- `IncrementalSegFormer`: Dynamic head expansion

### Buffers (`src/buffers/seg_reservoir.py`)
- `SegmentationReservoir`: Stores image-mask pairs
- `SegmentationLogitsReservoir`: Also stores model logits

### Losses (`src/utils/seg_losses.py`)
- `CrossEntropyLoss2d`: Standard CE for segmentation
- `DiceLoss`: Dice loss for imbalanced classes
- `FocalLoss`: Focal loss for hard examples
- `UnbiasedCrossEntropy`: For class-incremental learning
- `KnowledgeDistillationLoss`: KD for continual learning
- `ACELoss`: Asymmetric CE

### Metrics (`src/utils/seg_metrics.py`)
- `SegmentationMetrics`: mIoU, accuracy, per-class IoU
- `ContinualSegmentationMetrics`: Forgetting, backward transfer

## Configuration Options

### Model Settings
- `segformer_variant`: `mit_b0` to `mit_b5`
- `freeze_encoder`: Freeze backbone weights
- `pretrained`: Use ImageNet pretrained weights

### Continual Learning Settings
- `n_tasks`: Number of incremental tasks
- `class_order`: `sequential`, `random`, `frequency`, `disjoint`
- `mask_old_classes`: Mask old classes as ignore

### Memory Settings
- `mem_size`: Number of samples to store
- `mem_batch_size`: Samples per replay
- `drop_method`: `random` or `class_balanced`

## Cityscapes Classes

The 19 evaluation classes in order:
0. road
1. sidewalk
2. building
3. wall
4. fence
5. pole
6. traffic light
7. traffic sign
8. vegetation
9. terrain
10. sky
11. person
12. rider
13. car
14. truck
15. bus
16. train
17. motorcycle
18. bicycle

## Example Results

Results will be saved to `results/<tag>/`:
- `miou.csv`: Per-task mIoU after each task
- `forgetting.csv`: Forgetting metrics
- `params.json`: Experiment configuration

## Customization

### Adding a New Learner

1. Create learner in `src/learners/segmentation/`
2. Inherit from `BaseSegmentationLearner`
3. Register in `src/utils/seg_name_match.py`

### Adding a New Dataset

1. Create dataset in `src/datasets/`
2. Add to `src/utils/seg_data.py`
3. Update `src/datasets/__init__.py`

## Citation

If you use this code, please cite the relevant papers:
- SegFormer: [Xie et al., NeurIPS 2021]
- Cityscapes: [Cordts et al., CVPR 2016]
- Online Continual Learning methods as appropriate

