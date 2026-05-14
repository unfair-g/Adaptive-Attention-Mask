# MKD-AMD 在线持续语义分割方法 — 技术方案介绍

## 基于动量知识蒸馏与注意力图对齐的道路场景在线持续语义分割系统

---

## 概述

本文档详细介绍 MKD-AMD（Momentum Knowledge Distillation with Attention Map Distillation）方法的技术方案实现，对应毕业设计任务书中的六项具体工作内容。本方法针对道路场景语义分割的在线持续学习问题，提出了一套完整的解决方案，包括数据处理、模型架构、经验回放、知识蒸馏、在线训练和性能评估六个核心模块。

---

## 一、道路场景数据集构建与预处理

### 1.1 数据集选择与任务划分

选用 **Cityscapes** 数据集作为道路场景分割的基准数据集，该数据集包含 19 个道路场景语义类别：

| 类别组 | 类别 | 说明 |
|--------|------|------|
| 平面 | road, sidewalk | 道路表面 |
| 建筑 | building, wall, fence | 建筑结构 |
| 物体 | pole, traffic light, traffic sign | 道路设施 |
| 自然 | vegetation, terrain, sky | 自然景观 |
| 交通参与者 | person, rider, car, truck, bus, train, motorcycle, bicycle | 动态目标 |

**任务划分策略**：采用 4-Task 增量学习设置，将 19 个类别按顺序划分为 4 个任务：

```
Task 0: Classes 0-4   (road, sidewalk, building, wall, fence)
Task 1: Classes 5-9   (pole, traffic light, traffic sign, vegetation, terrain)  
Task 2: Classes 10-13 (sky, person, rider, car)
Task 3: Classes 14-18 (truck, bus, train, motorcycle, bicycle)
```

### 1.2 数据预处理流程

**统一标注格式处理**：

```python
# src/datasets/cityscapes.py

# 原始标签ID到训练ID的映射 (0-18, 255为忽略)
CITYSCAPES_ID_TO_TRAINID = {
    -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
    16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18
}
```

**Current-Only 标签模式**：

为模拟真实场景中旧类标签不可见的严格条件，采用 Current-Only 标签模式：

```python
# Current-Only 模式下的标签处理
if self.mode == 'current_only':
    # 1. 首先将所有像素设为 ignore_index (255)
    processed_mask = np.full_like(mask, self.ignore_index)  # 255
    
    # 2. 只保留当前任务的类别标签
    for label in self.selected_labels:  # 当前任务类别
        processed_mask[mask == label] = label
```

| 像素类型 | 处理方式 | 说明 |
|----------|----------|------|
| 当前任务类别 | 保持原标签 | 作为训练目标 |
| 旧任务类别 | → 255 (ignore) | 不参与 CE 损失 |
| 未来任务类别 | → 255 (ignore) | 不参与 CE 损失 |

### 1.3 数据增强与连续序列生成

**图像预处理**：

```python
# 统一尺寸: 512×1024 (H×W)
img_size = (512, 1024)

# 标准化参数 (ImageNet)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# 数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(img_size),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    normalize
])
```

**在线数据流生成**：

每个任务的数据以小批量连续输入，模拟实际道路环境中的动态变化：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              在线数据流结构                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Task 0     Task 1          Task 2          Task 3                          │
│  ─────      ─────           ─────           ─────                           │
│  [B1][B2]...→[B1][B2][B3]...→[B1][B2][B3]...→[B1][B2][B3]...                │
│     ↓           ↓               ↓               ↓                           │
│  Classes     Classes         Classes         Classes                        │
│  0-4         5-9             10-13           14-18                          │
│                                                                             │
│  约束: 每个批次 (Batch) 仅访问一次，不可重复训练                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、Transformer-based 分割模型优化与模块化设计

### 2.1 骨干网络选择

选用 **SegFormer (MIT-B0)** 作为骨干网络，具有以下优势：

| 特性 | 说明 |
|------|------|
| **层级 Transformer 编码器** | 4 阶段结构，逐步降低分辨率（1/4→1/8→1/16→1/32） |
| **轻量级 MLP 解码器** | 高效融合多尺度特征 |
| **无位置编码** | 对输入尺寸更灵活，适合不同分辨率 |
| **预训练权重** | ImageNet-1K 预训练，迁移学习基础好 |

**模型配置**：

```yaml
# config/seg/cityscapes_ema_attention.yaml
segformer_variant: mit_b0
freeze_encoder: false
pretrained: true
n_classes: 19
```

### 2.2 注意力图提取模块

为支持注意力图蒸馏，对 SegFormer 进行模块化改造，增加注意力图提取功能：

```python
# src_v3/models/segformer_attention.py

class SegFormerAttention(nn.Module):
    """支持注意力图提取的 SegFormer 模型"""
    
    def __init__(self, num_classes=19, variant='mit_b0'):
        super().__init__()
        self.encoder = MixVisionTransformer(variant)
        self.decoder = SegFormerHead(num_classes)
        
    def forward(self, x, return_attention=False):
        # 编码器前向传播
        features, attention_maps = self.encoder(x, return_attention=True)
        
        # 解码器
        logits = self.decoder(features)
        
        if return_attention:
            return logits, attention_maps
        return logits
```

**注意力图结构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SegFormer 注意力图提取                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Image (512×1024)                                                     │
│       ↓                                                                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐                     │
│  │ Stage 1 │→  │ Stage 2 │→  │ Stage 3 │→  │ Stage 4 │                     │
│  │ 1/4 res │   │ 1/8 res │   │ 1/16 res│   │ 1/32 res│                     │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘                     │
│       │             │             │             │                          │
│       ↓             ↓             ↓             ↓                          │
│  [不提取]      [A_s^(2)]     [A_s^(3)]     [不提取]                         │
│               注意力图        注意力图                                       │
│               语义信息丰富     语义信息最丰富                                 │
│                                                                             │
│  形状: (B, num_heads, seq_len_q, seq_len_k)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 双模型架构设计

采用学生-教师双模型架构：

| 模型 | 架构 | 更新方式 | 功能 |
|------|------|----------|------|
| **学生模型** | SegFormer MIT-B0 | 梯度下降 | 学习新知识 + 保留旧知识 |
| **教师模型** | SegFormer MIT-B0 (共享结构) | EMA 更新 | 提供稳定的旧类知识 |

**EMA 更新公式**：

$$\theta' \leftarrow \alpha \cdot \theta' + (1 - \alpha) \cdot \theta$$

其中 $\alpha = 0.9999$，确保教师模型更新缓慢，保持历史知识稳定。

### 2.4 计算开销与显存优化

| 优化策略 | 实现方式 | 效果 |
|----------|----------|------|
| **轻量骨干** | MIT-B0 (3.7M 参数) | 显存占用低 |
| **混合精度训练** | torch.cuda.amp | 显存减半 |
| **梯度裁剪** | grad_clip=1.0 | 训练稳定 |
| **注意力下采样** | 32×32 统一尺寸 | 减少蒸馏计算量 |

---

## 三、经验回放机制设计与实现

### 3.1 蓄水池采样策略

采用 **蓄水池采样 (Reservoir Sampling)** 维护固定容量的记忆缓冲区：

```python
# src/buffers/seg_reservoir.py

class SegmentationReservoir(Buffer):
    """语义分割蓄水池采样缓冲区"""
    
    def __init__(self, max_size=500, img_size=(512, 1024), n_classes=19):
        self.max_size = max_size
        self.buffer_imgs = torch.zeros(max_size, 3, *img_size)
        self.buffer_masks = torch.zeros(max_size, *img_size).long()
        self.buffer_class_counts = torch.zeros(max_size, n_classes).long()
        
    def update(self, imgs, masks):
        """蓄水池采样更新"""
        for img, mask in zip(imgs, masks):
            # 随机索引
            reservoir_idx = int(random() * (self.n_seen_so_far + 1))
            
            if self.n_seen_so_far < self.max_size:
                reservoir_idx = self.n_added_so_far
                
            if reservoir_idx < self.max_size:
                self.buffer_imgs[reservoir_idx] = img
                self.buffer_masks[reservoir_idx] = mask
                self.buffer_class_counts[reservoir_idx] = self._compute_class_counts(mask)
                
            self.n_seen_so_far += 1
```

**蓄水池采样特性**：

| 特性 | 说明 |
|------|------|
| **无偏采样** | 每个样本被保留的概率相等 |
| **固定空间** | 缓冲区大小固定（500 样本） |
| **在线更新** | 每个样本仅访问一次时即决定是否保留 |

### 3.2 类别平衡采样策略

针对道路场景中类别严重不平衡的问题，设计类别平衡回放策略：

```python
def class_balanced_retrieve(self, n_imgs=100, target_classes=None):
    """平衡采样：确保各类别回放均衡"""
    
    # 按主导类别分组
    class_to_indices = {c: [] for c in target_classes}
    for i in range(self.n_added_so_far):
        dominant = self.buffer_class_counts[i].argmax()
        if dominant in class_to_indices:
            class_to_indices[dominant].append(i)
    
    # 每个类别等量采样
    selected_indices = []
    per_class = n_imgs // len(target_classes)
    
    for c in target_classes:
        if len(class_to_indices[c]) > 0:
            n_select = min(per_class, len(class_to_indices[c]))
            selected_indices.extend(random.sample(class_to_indices[c], n_select))
    
    return self.buffer_imgs[selected_indices], self.buffer_masks[selected_indices]
```

**采样模式**：

```yaml
# config/seg/cityscapes_ema_attention.yaml
use_balanced_sampling: true
balanced_sampling_mode: old_classes  # 优先平衡旧类别
# 可选: 'all_seen' (所有已见类), 'minority' (逆频率加权)
```

### 3.3 缓冲区管理与集成

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           经验回放工作流程                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  当前批次 (x, y)                                                            │
│       ↓                                                                     │
│  ┌─────────────────┐                                                        │
│  │  缓冲区更新      │ ← 蓄水池采样决定是否加入                               │
│  │  Buffer.update() │                                                       │
│  └────────┬────────┘                                                        │
│           ↓                                                                 │
│  ┌─────────────────┐                                                        │
│  │  类别平衡采样    │ → (x_mem, y_mem)                                       │
│  │  class_balanced │                                                        │
│  │  _retrieve()    │                                                        │
│  └────────┬────────┘                                                        │
│           ↓                                                                 │
│  ┌─────────────────┐                                                        │
│  │  批次合并        │ → (x_combined, y_combined)                             │
│  │  Concat         │                                                        │
│  └────────┬────────┘                                                        │
│           ↓                                                                 │
│       模型训练                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**配置参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mem_size` | 500 | 缓冲区容量（图像数） |
| `mem_batch_size` | 4 | 每次回放批次大小 |
| `mem_iters` | 2 | 每批次梯度更新次数 |
| `drop_method` | random | 替换策略 |

---

## 四、知识蒸馏与参数正则化策略开发

### 4.1 注意力图蒸馏 (Attention Map Distillation)

**核心思想**：让学生模型学习教师模型"在哪里看"的高层语义知识。

```python
# src_v3/utils/attention_distillation.py

class AttentionMapDistillationLoss(nn.Module):
    """注意力图蒸馏损失"""
    
    def __init__(self, active_stages=[2, 3], stage_weights=[1.0, 1.5], 
                 loss_type='cosine'):
        super().__init__()
        self.active_stages = active_stages
        self.stage_weights = stage_weights
        self.loss_type = loss_type
        
    def forward(self, student_attns, teacher_attns, mask=None):
        """
        Args:
            student_attns: 学生注意力图 {stage: (B, heads, seq_q, seq_k)}
            teacher_attns: 教师注意力图
            mask: MAD 掩码 (可选)
        """
        total_loss = 0.0
        
        for stage_idx, weight in zip(self.active_stages, self.stage_weights):
            attn_s = student_attns[stage_idx]  # 学生注意力
            attn_t = teacher_attns[stage_idx]  # 教师注意力
            
            # 头平均
            attn_s = attn_s.mean(dim=1)  # (B, seq_q, seq_k)
            attn_t = attn_t.mean(dim=1)
            
            # 展平
            attn_s = attn_s.flatten(1)  # (B, seq_q * seq_k)
            attn_t = attn_t.flatten(1)
            
            # 余弦相似度损失
            if self.loss_type == 'cosine':
                cos_sim = F.cosine_similarity(attn_s, attn_t, dim=1)
                stage_loss = (1 - cos_sim).mean()
            
            # 应用掩码
            if mask is not None:
                stage_loss = stage_loss * mask
                
            total_loss += weight * stage_loss
            
        return total_loss
```

**为什么选择余弦相似度损失**：

| 损失类型 | 公式 | 特点 | 适用性 |
|----------|------|------|--------|
| MSE | $\|A_s - A_t\|^2$ | 尺度敏感，梯度不稳定 | ❌ 不适合 |
| KL 散度 | $\sum A_t \log(A_t/A_s)$ | 需要归一化 | △ 可用 |
| **余弦相似度** | $1 - \cos(A_s, A_t)$ | **尺度不变，梯度稳定** | ✓ 推荐 |

### 4.2 掩码注意力蒸馏 (Masked Attention Distillation, MAD)

**问题**：教师模型不知道当前任务的新类别，其在新类区域的注意力是不可靠的。

**解决方案**：

```python
def create_mad_mask(labels, current_classes, attention_size):
    """创建 MAD 掩码"""
    B, H, W = labels.shape
    
    # 下采样标签到注意力尺寸
    labels_down = F.interpolate(
        labels.unsqueeze(1).float(), 
        size=attention_size, 
        mode='nearest'
    ).squeeze(1).long()
    
    # 创建掩码: 新类区域为 0, 其他区域为 1
    mask = torch.ones(B, attention_size[0] * attention_size[1])
    
    for b in range(B):
        for c in current_classes:
            new_class_positions = (labels_down[b] == c).flatten()
            mask[b, new_class_positions] = 0
            
    return mask
```

**MAD 工作原理**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        掩码注意力蒸馏 (MAD)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入标签:     [旧类] [旧类] [新类] [新类] [旧类] [ignore]                   │
│                  ↓      ↓      ↓      ↓      ↓       ↓                     │
│  MAD 掩码:    [  1  ] [  1  ] [  0  ] [  0  ] [  1  ] [  1  ]              │
│                                                                             │
│  效果:                                                                      │
│   - mask=1: 教师指导学生注意力模式 (旧类区域)                                │
│   - mask=0: 学生自由学习 (新类区域)                                          │
│                                                                             │
│  损失计算:                                                                  │
│   L_MAD = Σ mask × (1 - cos(A_s, A_t))                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Logit 级知识蒸馏

在输出层应用知识蒸馏，仅在旧类通道上进行：

```python
def logit_distillation_loss(student_logits, teacher_logits, old_classes, T=3.0):
    """Logit 级知识蒸馏 (仅旧类通道)"""
    
    # 仅取旧类通道
    s_old = student_logits[:, old_classes, :, :]
    t_old = teacher_logits[:, old_classes, :, :]
    
    # 软标签 (温度缩放)
    s_soft = F.softmax(s_old / T, dim=1)
    t_soft = F.softmax(t_old / T, dim=1)
    
    # KL 散度
    loss = F.kl_div(
        torch.log(s_soft + 1e-8),
        t_soft,
        reduction='batchmean'
    ) * (T ** 2)
    
    return loss
```

### 4.4 总损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{KD} \cdot \mathcal{L}_{KD} + \lambda_{AMD} \cdot \mathcal{L}_{AMD}$$

**各损失组件**：

| 损失 | 公式 | 权重 | 作用域 |
|------|------|------|--------|
| $\mathcal{L}_{CE}$ | Focal Loss | 1.0 | 当前类像素 |
| $\mathcal{L}_{KD}$ | KL 散度 (旧类通道) | 3.0 | 所有像素 |
| $\mathcal{L}_{AMD}$ | 余弦相似度 | 1.0 | 非新类区域 (MAD) |

---

## 五、在线增量训练策略与流式优化

### 5.1 训练管线设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MKD-AMD 在线训练管线                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  for each task t in [0, 1, 2, 3]:                                          │
│      # 更新类别集合                                                         │
│      old_classes = classes[:task_t_start]                                   │
│      current_classes = classes[task_t_start:task_t_end]                    │
│                                                                             │
│      for each batch (x, y) in task_t_data:  # 每个样本仅访问一次            │
│                                                                             │
│          ┌─────────────────────────────────────────────────────────────┐   │
│          │ Step 1: 经验回放                                            │   │
│          │   (x_mem, y_mem) ← Buffer.class_balanced_retrieve(old_classes)│   │
│          │   (x_combined, y_combined) ← Concat(x, y, x_mem, y_mem)    │   │
│          └────────────────────────────────────────────────────────────┘   │
│                                  ↓                                         │
│          ┌─────────────────────────────────────────────────────────────┐   │
│          │ Step 2: 前向传播 (带注意力提取)                              │   │
│          │   z_s, A_s ← StudentModel(x_combined, return_attention=True)│   │
│          │   z_t, A_t ← TeacherModel(x_combined, return_attention=True)│   │
│          └────────────────────────────────────────────────────────────┘   │
│                                  ↓                                         │
│          ┌─────────────────────────────────────────────────────────────┐   │
│          │ Step 3: 损失计算                                             │   │
│          │   L_CE  ← FocalCrossEntropy(z_s, y_combined)                │   │
│          │   L_KD  ← KLDivergence(z_s[old], z_t[old])                  │   │
│          │   mask  ← CreateMADMask(y_combined, current_classes)        │   │
│          │   L_AMD ← MaskedCosineLoss(A_s, A_t, mask)                  │   │
│          │   L_total ← L_CE + λ_KD × L_KD + λ_AMD × L_AMD              │   │
│          └────────────────────────────────────────────────────────────┘   │
│                                  ↓                                         │
│          ┌─────────────────────────────────────────────────────────────┐   │
│          │ Step 4: 参数更新                                             │   │
│          │   θ ← θ - η × ∇L_total              # 学生梯度下降          │   │
│          │   θ' ← α × θ' + (1-α) × θ           # 教师 EMA 更新         │   │
│          └────────────────────────────────────────────────────────────┘   │
│                                  ↓                                         │
│          ┌─────────────────────────────────────────────────────────────┐   │
│          │ Step 5: 缓冲区更新                                           │   │
│          │   Buffer.update(x, y)               # 蓄水池采样             │   │
│          └────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 训练优化策略

**梯度累积与多次更新**：

```yaml
mem_iters: 2  # 每批次梯度更新次数
# 效果: 增加对回放样本的学习强度
```

**学习率与优化器配置**：

```yaml
learning_rate: 0.00006  # 适合 SegFormer 微调
weight_decay: 0.01      # L2 正则化
optim: AdamW            # 自适应学习率
grad_clip: 1.0          # 梯度裁剪，稳定训练
```

**类别平衡策略**：

```yaml
# Focal Loss 处理类别不平衡
focal_alpha: 1.0
focal_gamma: 2.0

# 当前任务类别逆频率权重
use_current_class_balance: true
class_balance_power: 0.5   # 开方逆频率 (温和平衡)
class_balance_max: 10.0    # 最大权重上限
```

### 5.3 延迟与显存优化

| 优化措施 | 实现方式 | 效果 |
|----------|----------|------|
| **教师无梯度** | `with torch.no_grad()` | 减少 50% 显存 |
| **注意力下采样** | 统一到 32×32 | 减少蒸馏计算量 |
| **选择性蒸馏** | 仅 Stage 2, 3 | 减少注意力提取开销 |
| **批量合并** | 当前+回放一次前向 | 减少前向传播次数 |

---

## 六、性能评估与灾难性遗忘分析

### 6.1 评估指标体系

| 指标 | 公式 | 说明 |
|------|------|------|
| **mIoU** | $\frac{1}{C}\sum_{c=1}^{C}\frac{TP_c}{TP_c + FP_c + FN_c}$ | 平均交并比，分割精度 |
| **遗忘率 (F)** | $F = \frac{1}{T-1}\sum_{t=0}^{T-2}(mIoU_t^{best} - mIoU_t^{final})$ | 旧任务性能下降程度 |
| **后向迁移 (BWT)** | $BWT = \frac{1}{T-1}\sum_{t=0}^{T-2}(mIoU_t^{final} - mIoU_t^{first})$ | 新知识对旧知识的影响 |
| **前向迁移 (FWT)** | 新任务初始性能相比随机初始化的提升 | 旧知识对新任务的帮助 |

### 6.2 评估流程

```python
# 每个任务训练后评估所有已见类别
def evaluate_continual(model, test_loader, seen_classes):
    """持续学习评估"""
    
    per_class_iou = {}
    for class_id in seen_classes:
        per_class_iou[class_id] = compute_iou(
            predictions, 
            labels, 
            class_id
        )
    
    # 分组统计
    old_classes_iou = mean([per_class_iou[c] for c in old_classes])
    current_classes_iou = mean([per_class_iou[c] for c in current_classes])
    all_seen_iou = mean([per_class_iou[c] for c in seen_classes])
    
    return {
        'old_mIoU': old_classes_iou,
        'current_mIoU': current_classes_iou,
        'all_mIoU': all_seen_iou
    }
```

### 6.3 实验结果

**Cityscapes 4-Task 增量学习**：

| 方法 | 训练 mIoU | 测试 mIoU | 遗忘率 | 后向迁移 |
|------|-----------|-----------|--------|----------|
| 伪标签 (Baseline) | 44.78% | 27.65% | 15.27% | -15.27% |
| 结构关系蒸馏 (RKD) | 53.61% | 27.39% | 3.09% | -1.30% |
| **注意力图蒸馏 (AMD)** | **52.02%** | **33.82%** | **3.55%** | **-1.54%** |

### 6.4 消融实验分析

**各组件贡献**：

| 配置 | 测试 mIoU | ΔmIoU |
|------|-----------|-------|
| Baseline (伪标签 + Logit KD) | 27.65% | - |
| + 注意力蒸馏 (AMD) | 31.42% | +3.77% |
| + 掩码注意力蒸馏 (MAD) | 32.89% | +1.47% |
| + 类别平衡采样 | **33.82%** | +0.93% |

**关键发现**：

1. **注意力蒸馏显著提升泛化能力**：测试 mIoU 从 27.65% 提升至 33.82%（+6.17%）
2. **MAD 掩码有效**：避免教师在新类区域的误导，提升 1.47%
3. **类别平衡采样重要**：确保小样本类别回放充足，提升 0.93%

### 6.5 泛化能力分析

**为什么 AMD 测试表现优于 RKD？**

| 维度 | 结构关系蒸馏 (RKD) | 注意力图蒸馏 (AMD) |
|------|-------------------|-------------------|
| **约束数量** | O(N²) 像素对关系 | O(N) 注意力模式 |
| **知识抽象** | 中等 (空间几何) | 高 (语义关注模式) |
| **场景依赖** | 强 (特定几何结构) | 弱 (通用关注策略) |
| **过拟合风险** | 高 | 低 |
| **测试泛化** | 弱 | **强** |

---

## 七、系统配置与代码结构

### 7.1 配置文件

```yaml
# config/seg/cityscapes_ema_attention.yaml (关键配置)

# 模型
segformer_variant: mit_b0
pretrained: true

# 持续学习
n_tasks: 4
label_mode: current_only

# 经验回放
mem_size: 500
use_balanced_sampling: true
balanced_sampling_mode: old_classes

# EMA 教师
ema_alpha: 0.9999

# 知识蒸馏
alpha_kd: 3.0
kd_temperature: 3.0

# 注意力蒸馏
attention_kd_weight: 1.0
active_attention_stages: [2, 3]
attention_stage_weights: [1.0, 1.5]
attention_loss_type: cosine
use_masked_distillation: true

# 类别平衡
use_current_class_balance: true
focal_gamma: 2.0
```

### 7.2 代码结构

```
mkd_ocl/
├── src/                              # 基础框架
│   ├── datasets/
│   │   └── cityscapes.py             # Cityscapes 数据集
│   ├── buffers/
│   │   └── seg_reservoir.py          # 经验回放缓冲区
│   └── learners/segmentation/
│       ├── base_seg.py               # 分割学习器基类
│       └── er_seg.py                 # EMA 基线
│
├── src_v3/                           # 注意力蒸馏方法 (主方法)
│   ├── models/
│   │   └── segformer_attention.py    # 支持注意力提取的 SegFormer
│   ├── utils/
│   │   └── attention_distillation.py # 注意力蒸馏损失
│   └── learners/segmentation/
│       └── er_seg_attention.py       # MKD-AMD 学习器
│
├── config/seg/
│   └── cityscapes_ema_attention.yaml # 配置文件
│
└── docs/
    ├── research_background.md        # 课题背景
    ├── method_description.md         # 方法描述
    ├── method_comparison_analysis.md # 方法对比分析
    └── solution_introduction.md      # 技术方案介绍 (本文档)
```

### 7.3 运行命令

```bash
# 训练
python main_seg.py --config config/seg/cityscapes_ema_attention.yaml

# 评估
python evaluate_seg.py --config config/seg/cityscapes_ema_attention.yaml \
    --checkpoint checkpoints/ckpt_train3.pth

# 生成框架图
python docs/generate_attention_architecture.py
```

---

## 总结

本技术方案针对毕业设计任务书中的六项具体工作，提出了完整的 MKD-AMD 在线持续语义分割解决方案：

| 任务项 | 实现方案 | 核心技术 |
|--------|----------|----------|
| 1. 数据集构建 | Cityscapes 4-Task 划分 + Current-Only 模式 | 蓄水池采样、数据增强 |
| 2. 模型设计 | SegFormer + 注意力提取模块 | 双模型架构、模块化改造 |
| 3. 经验回放 | 类别平衡采样 + 动态缓冲区 | 逆频率加权、主导类分组 |
| 4. 知识蒸馏 | AMD + MAD + Logit KD | 余弦相似度、掩码蒸馏 |
| 5. 在线训练 | 流式管线 + 梯度优化 | 多次更新、延迟优化 |
| 6. 性能评估 | mIoU + 遗忘率 + BWT | 消融实验、泛化分析 |

**核心创新**：首次将 Transformer 自注意力图蒸馏应用于在线持续语义分割，通过传递"在哪里看"的高层语义知识，实现了 **33.82% 测试 mIoU**，相比基线提升 **6.17%**，遗忘率仅 **3.55%**。

