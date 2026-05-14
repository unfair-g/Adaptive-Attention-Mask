# 基于动量知识蒸馏与注意力图对齐的在线持续语义分割方法

## MKD-AMD: Momentum Knowledge Distillation with Attention Map Distillation for Online Continual Semantic Segmentation

---

## 1. 摘要

本方法提出了一种基于**动量知识蒸馏（MKD）**与**注意力图蒸馏（AMD）**的在线持续语义分割框架。针对在线持续学习中的灾难性遗忘问题，我们利用SegFormer的Transformer架构特性，通过对齐学生模型与EMA教师模型的**注意力模式**，传递"**在哪里看**"的高层次语义知识，实现对旧类别知识的有效保留。

**核心贡献**：
1. 提出基于余弦相似度的注意力图蒸馏损失，尺度不变且具有强泛化能力
2. 设计掩码注意力蒸馏（MAD）策略，排除教师不可靠区域的干扰
3. 结合EMA教师模型与类别平衡采样，实现稳定的在线持续学习

---

## 2. 问题定义

### 2.1 在线持续语义分割

给定一系列语义分割任务 $\{\mathcal{T}_0, \mathcal{T}_1, ..., \mathcal{T}_{T-1}\}$，每个任务 $\mathcal{T}_t$ 包含一组新类别 $C_t$。在**在线学习**约束下，每个样本仅能被访问一次。目标是训练一个模型 $f_\theta$，在学习新类别的同时保持对旧类别 $C_{old} = \bigcup_{i<t} C_i$ 的分割能力。

### 2.2 核心挑战

| 挑战 | 描述 |
|------|------|
| **灾难性遗忘** | 学习新类别时显著降低旧类别性能 |
| **在线约束** | 每个样本仅访问一次，无法重复训练 |
| **标签缺失** | Current-Only模式下，旧类和未来类标签不可见 |
| **类别不平衡** | 不同类别像素数量差异巨大（10倍以上） |

---

## 3. 方法框架

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     MKD-AMD 在线持续语义分割框架                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────┐    ┌──────────┐                                                     │
│  │ 数据流    │───>│  合并    │                                                     │
│  │ D_t      │    │         │                                                     │
│  └──────────┘    └────┬────┘                                                     │
│       │               │                                                          │
│       ↓               ↓                                                          │
│  ┌──────────┐    ┌─────────────────────────────────────────┐                     │
│  │ 经验回放  │───>│            学生模型 f_θ                  │                     │
│  │ Buffer M │    │  SegFormer (MIT-B0) + Attention        │──────┐              │
│  └──────────┘    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │      │              │
│                  │  │ S1  │→│ S2  │→│ S3  │→│ S4  │→Decoder│      │              │
│                  │  └─────┘ └──┬──┘ └──┬──┘ └─────┘       │      ↓              │
│                  └────────────┼──────┼───────────────────┘  ┌────────┐          │
│                               │      │                      │ Logits │          │
│                               ↓      ↓                      │  z_s   │          │
│                         ┌──────────────────┐                └───┬────┘          │
│                         │  注意力图 A_s    │                    │              │
│                         │ Stage 2 + Stage 3│                    ↓              │
│                         └────────┬─────────┘              ┌──────────┐          │
│         EMA ↑                    │                        │ L_CE     │          │
│       ┌─────┴─────┐              │    ┌───────────────┐   │ +L_KD    │          │
│       │           │              │    │               │   │ +L_AMD   │          │
│  ┌────┴───────────┴────┐         │    │  注意力蒸馏   │   └────┬─────┘          │
│  │     教师模型 f_θ'   │         │    │  (余弦损失)   │        │              │
│  │  SegFormer (EMA)    │         │    │               │        │              │
│  │  ┌─────┐ ┌─────┐    │         ↓    │  1 - cos(A_s, A_t) │   ↓              │
│  │  │ S2  │→│ S3  │→.. │    ┌────────┐│  × MAD_mask  │   反向传播            │
│  │  └──┬──┘ └──┬──┘    │    │ A_t    │└───────────────┘        │              │
│  └─────┼──────┼────────┘    └────────┘                         │              │
│        │      │                  ↑                             │              │
│        └──────┴──────────────────┘                             ↓              │
│                                                          θ ← θ - η∇L          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 双模型架构

#### 学生模型 (Student Model)

- **架构**：SegFormer MIT-B0，支持注意力图提取
- **更新方式**：梯度下降
- **功能**：学习当前任务 + 通过蒸馏保留旧知识

#### 教师模型 (Teacher Model)

- **架构**：与学生模型相同
- **更新方式**：指数移动平均 (EMA)
- **功能**：提供稳定的旧类别知识

**EMA更新公式**：
$$\theta' \leftarrow \alpha \cdot \theta' + (1 - \alpha) \cdot \theta$$

其中 $\alpha = 0.9999$ 为动量系数。

---

## 4. 注意力图蒸馏 (Attention Map Distillation)

### 4.1 核心思想

SegFormer基于Transformer架构，其自注意力机制 $A = \text{softmax}(QK^T/\sqrt{d})$ 编码了**模型关注的空间区域**。教师模型训练于旧任务，其注意力图包含"如何识别旧类物体"的知识。通过对齐学生与教师的注意力模式，学生学习**"在哪里看"**的识别策略。

### 4.2 注意力图提取

从SegFormer的Stage 2和Stage 3提取注意力图（更语义化的层级）：

```python
# 注意力图形状: (B, num_heads, seq_len_q, seq_len_k)
A_stage2 = model.get_attention(stage=2)  # 1/8 分辨率
A_stage3 = model.get_attention(stage=3)  # 1/16 分辨率

# 预处理: 头平均 + 下采样到统一尺寸
A_processed = pool(mean_heads(A), size=32×32)  # (B, 1024)
```

### 4.3 余弦相似度损失

采用**余弦相似度损失**而非MSE，具有以下优势：
- **尺度不变**：不受注意力值绝对大小影响
- **梯度稳定**：注意力值通常很小（~0.001），MSE梯度趋近于0
- **关注模式**：聚焦于"形状匹配"而非"数值匹配"

**损失函数**：
$$\mathcal{L}_{AMD} = \sum_{l \in \{2, 3\}} w_l \cdot (1 - \cos(A_s^{(l)}, A_t^{(l)}))$$

其中 $w_2 = 1.0$，$w_3 = 1.5$（高层注意力更重要）。

### 4.4 掩码注意力蒸馏 (Masked Attention Distillation, MAD)

**问题**：教师模型不知道当前任务的新类别，其在新类区域的注意力是不可靠的。

**解决方案**：在计算注意力蒸馏损失时，排除涉及当前新类像素的区域：

$$\text{mask}_{i,j} = \begin{cases} 
0 & \text{if } y_i \in C_{cur} \text{ or } y_j \in C_{cur} \\
1 & \text{otherwise}
\end{cases}$$

$$\mathcal{L}_{MAD} = \sum_l w_l \cdot \text{mask} \cdot (1 - \cos(A_s^{(l)}, A_t^{(l)}))$$

**效果**：
- 旧类区域：教师指导学生如何关注
- 新类区域：学生自由学习新的注意力模式

---

## 5. 损失函数

### 5.1 总损失

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{KD} \cdot \mathcal{L}_{KD} + \lambda_{AMD} \cdot \mathcal{L}_{AMD}$$

默认权重：$\lambda_{KD} = 3.0$，$\lambda_{AMD} = 1.0$

### 5.2 交叉熵损失

采用Focal Loss处理类别不平衡：

$$\mathcal{L}_{CE} = -\alpha_c (1 - p_t)^\gamma \log(p_t)$$

其中：
- $\gamma = 2.0$：聚焦参数
- $\alpha_c$：逆频率类别权重

### 5.3 Logit知识蒸馏

仅在旧类通道上应用KD，避免教师在新类上的误导：

$$\mathcal{L}_{KD} = T^2 \cdot \text{KL}(\text{softmax}(z_s^{old}/T) \| \text{softmax}(z_t^{old}/T))$$

其中 $T = 3.0$ 为温度参数，$z^{old}$ 表示仅取旧类通道。

---

## 6. 经验回放与类别平衡采样

### 6.1 蓄水池采样

使用蓄水池采样维护固定大小的记忆缓冲区 $\mathcal{M}$（默认500张图像）：

```python
class SegmentationReservoir:
    def update(self, image, mask):
        if buffer_full:
            # 随机替换
            idx = random.randint(0, n_seen)
            if idx < max_size:
                buffer[idx] = (image, mask)
        else:
            buffer.append((image, mask))
```

### 6.2 类别平衡采样

训练时优先采样包含旧类别的样本，防止遗忘：

```python
def class_balanced_retrieve(target_classes):
    # 计算每个样本的目标类别覆盖率
    scores = [sample.class_coverage(target_classes) for sample in buffer]
    # 按覆盖率采样
    return weighted_sample(buffer, weights=scores)
```

**采样模式**：
- `old_classes`：优先采样旧类别（默认）
- `all_seen`：平衡所有已见类别
- `minority`：逆频率加权采样

---

## 7. 训练流程

```
Algorithm: MKD-AMD Online Continual Learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: 数据流 D_t, 当前类别 C_cur, 旧类别 C_old
Output: 更新后的模型 f_θ

1: for each batch (x, y) in D_t do
2:    # 经验回放
3:    (x_mem, y_mem) ← ClassBalancedRetrieve(Buffer, C_old)
4:    (x_combined, y_combined) ← Concat((x, y), (x_mem, y_mem))
5:    
6:    # 学生前向传播（带注意力提取）
7:    z_s, A_s ← StudentModel(x_combined, return_attention=True)
8:    
9:    # 教师前向传播（带注意力提取）
10:   with no_grad():
11:       z_t, A_t ← TeacherModel(x_combined, return_attention=True)
12:   
13:   # 创建MAD掩码
14:   mask ← CreateMADMask(y_combined, C_cur)
15:   
16:   # 计算损失
17:   L_CE ← FocalCrossEntropy(z_s, y_combined)
18:   L_KD ← KLDivergence(z_s[C_old], z_t[C_old])
19:   L_AMD ← MaskedCosineLoss(A_s, A_t, mask)
20:   L_total ← L_CE + λ_KD × L_KD + λ_AMD × L_AMD
21:   
22:   # 更新学生模型
23:   θ ← θ - η × ∇L_total
24:   
25:   # EMA更新教师模型
26:   θ' ← α × θ' + (1 - α) × θ
27:   
28:   # 更新缓冲区
29:   Buffer.update(x, y)
30: end for

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 8. 实验结果

### 8.1 Cityscapes 4-Task 增量学习

| 方法 | 训练mIoU | 测试mIoU | 遗忘率 | 后向迁移 |
|------|----------|----------|--------|----------|
| 伪标签 (Baseline) | 44.78% | 27.65% | 15.27% | -15.27% |
| 结构关系蒸馏 (RKD) | 53.61% | 27.39% | 3.09% | -1.30% |
| **注意力图蒸馏 (AMD)** | **52.02%** | **33.82%** | **3.55%** | **-1.54%** |

### 8.2 方法优势分析

注意力图蒸馏在**测试集上表现最优**的原因：

1. **知识抽象层次高**：传递的是"在哪里看"的策略，而非具体数值
2. **约束稀疏**：O(N) 约束 vs RKD的O(N²)，避免过拟合
3. **场景无关性**：注意力模式对场景变化鲁棒
4. **架构契合**：充分利用SegFormer的Transformer注意力机制

---

## 9. 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ema_alpha` | 0.999 | EMA动量系数 |
| `alpha_kd` | 3.0 | Logit KD损失权重 |
| `attention_kd_weight` | 1.0 | 注意力蒸馏损失权重 |
| `kd_temperature` | 3.0 | 蒸馏温度 |
| `focal_gamma` | 2.0 | Focal Loss聚焦参数 |
| `active_attention_stages` | [2, 3] | 蒸馏的注意力层级 |
| `attention_stage_weights` | [1.0, 1.5] | 各层注意力权重 |
| `use_masked_distillation` | True | 启用MAD掩码 |
| `attention_loss_type` | cosine | 注意力损失类型 |
| `mem_size` | 500 | 记忆缓冲区大小 |
| `mem_iters` | 2 | 每批次梯度更新次数 |
| `use_balanced_sampling` | True | 启用类别平衡采样 |
| `balanced_sampling_mode` | old_classes | 采样模式 |

---

## 10. 代码结构

```
mkd_ocl/
├── src_v3/                           # 注意力蒸馏方法 (主方法)
│   ├── learners/segmentation/
│   │   └── er_seg_attention.py       # ERSegmentationEMAAttentionLearner
│   ├── models/
│   │   └── segformer_attention.py    # SegFormerAttention (注意力提取)
│   └── utils/
│       └── attention_distillation.py # AttentionMapDistillationLoss
│
├── src/                              # 基础框架
│   ├── learners/segmentation/
│   │   ├── base_seg.py               # 基类
│   │   └── er_seg.py                 # EMA基线
│   ├── models/
│   │   └── segformer.py              # SegFormer模型
│   ├── buffers/
│   │   └── seg_reservoir.py          # 经验回放缓冲区
│   └── datasets/
│       └── cityscapes.py             # Cityscapes数据集
│
├── config/seg/
│   └── cityscapes_ema_attention.yaml # 配置文件
│
└── docs/
    ├── method_description.md         # 方法说明 (本文件)
    ├── method_comparison_analysis.md # 方法对比分析
    └── generate_attention_architecture.py  # 框架图生成
```

---

## 11. 使用方法

### 11.1 训练

```bash
# 使用注意力图蒸馏方法训练
python main_seg.py --config config/seg/cityscapes_ema_attention.yaml
```

### 11.2 评估

```bash
# 评估模型
python evaluate_seg.py --config config/seg/cityscapes_ema_attention.yaml \
    --checkpoint checkpoints/ckpt_train3.pth
```

### 11.3 生成框架图

```bash
# 生成方法框架图
python docs/generate_attention_architecture.py
```

---

## 12. 引用

如果本方法对您的研究有帮助，请引用：

```bibtex
@article{mkd_amd_ocss,
  title={Momentum Knowledge Distillation with Attention Map Alignment 
         for Online Continual Semantic Segmentation},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 附录：方法对比

| 特性 | 伪标签 | 结构关系蒸馏 | **注意力图蒸馏** |
|------|--------|-------------|-----------------|
| 蒸馏层级 | 输出层 | 特征关系层 | **注意力层** |
| 知识类型 | 类别决策 | 空间结构 | **关注模式** |
| 约束数量 | O(C) | O(N²) | **O(N)** |
| 泛化能力 | 弱 | 中 | **强** |
| 计算开销 | 低 | 高 | **中** |
| 架构依赖 | 无 | 无 | Transformer |
| 测试mIoU | 27.65% | 27.39% | **33.82%** |
