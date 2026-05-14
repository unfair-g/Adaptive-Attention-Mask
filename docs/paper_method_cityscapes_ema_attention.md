# 在线流式类增量语义分割：双模型 EMA 与自适应掩码注意力蒸馏（方法框架）

本文档从**方法层面**概括 **`ER_EMA_Attention_Seg`** 所实现的在线持续学习语义分割框架：类增量设定、学生–教师双分支、经验回放、对数与注意力双层蒸馏，以及掩码化（MAD）与自适应（STU）的注意力加权机制。实现见 `src_v3/learners/segmentation/er_seg_attention.py`、`src_v3/utils/attention_distillation.py`。本文**仅保留 cosine 注意力蒸馏分支**；其余注意力损失分支不在本文展开。**具体数值超参**因数据集与实验而异，可通过 YAML（例如 `config/seg/cityscapes_ema_attention.yaml`）配置；文中在需要处用符号或“典型取值”说明，**不绑定某一种任务划分**（如固定的 base/increment 类数）。

---

## 1 在线流式持续学习框架

### 1.1 任务定义与在线持续学习设定

**类增量语义分割。** 考虑共 $C$ 个语义类别（如 Cityscapes 19 类、BDD100K 与 Cityscapes 对齐的 19 类等）。类别按任务 $t=0,1,\ldots,T-1$ **依次引入**：第 $t$ 个任务到达时，模型只知道**当前任务类别集合** $\mathcal{C}_t$，并仅在当前数据流上获得监督；已见类集合记为 $\mathcal{C}_{\le t}=\bigcup_{i\le t}\mathcal{C}_i$。目标是：在每一阶段学习 $\mathcal{C}_t$ 的同时，尽量保持对 $\mathcal{C}_{<t}$ 的分割性能（抑制灾难性遗忘）。

**任务划分（可配置）。** 工程上支持两类常见设定：（i）**等分**：给定 `n_tasks`，将 $C$ 类近似均分到各任务；（ii）**“基类–增量步长”格式** `increment: "X-Y"`：首任务含 $X$ 类，之后每任务至多增加 $Y$ 类，直至覆盖全部 $C$ 类（最后一任务可能不足 $Y$ 类）。类别在全数据上的**出现顺序**可由 `class_order`（如 sequential / random）与 `labels_order` 指定。**任意满足 $\sum_t |\mathcal{C}_t|=C$ 且各类仅属于一个任务的划分**均落在该框架内，不限于某一组 $(X,Y)$。

**训练标签可见性（严格类增量常用设定）。** **`label_mode: current_only`** 时：对当前任务样本，仅 $\mathcal{C}_t$ 内像素保留真实类别索引；**其余所有像素**（旧类、未来类、真实 ignore）在监督中统一映射为 **`ignore_index`**（如 255），**不参与**分割主损失中对“正确类”的拟合。因此训练阶段**不可见**旧类与未来类的像素级真值，符合严格类增量语义分割。**评估阶段**通常对已见类采用更宽可见协议（如 `current_and_old`）以报告 mIoU，与训练策略区分。

**在线流式学习。** 数据以**流或 epoch 扫描**形式到达；每个优化步在**小批量**上更新参数，并可多次混合**当前任务 batch** 与**缓冲区回放 batch**（由 `batch_size`、`mem_batch_size`、`mem_iters` 等控制）。**`epochs` 每任务**可取 1（单遍扫描）或更大，由实验设定。整体形态为：**步级更新 + 流式/准在线数据访问**，而非离线一次性混合全部数据。

**骨干与输入。** 框架以 **SegFormer** 族 **Transformer 编码器 + MLP 解码器** 为分割骨干；**`segformer_variant`** 选择容量（如 `mit_b0`…`mit_b5`），**`img_size`** 指定 $(H,W)$，**`pretrained`** 控制 ImageNet 初始化，**`freeze_encoder`** 控制是否冻结编码器。同一套方法可迁移到 **Cityscapes、BDD100K** 等已实现的数据管道（`dataset` + `data_root_dir`）。

---

### 1.2 基于双模型的在线流式架构设计

**学生–教师同构双分支。** **学生** $\theta_S$：可训练，接收当前监督、回放样本与蒸馏损失。**教师** $\theta_T$：与 $\theta_S$ 结构相同，**不反传**；由 $\theta_S$ 的 **EMA** 维护，在输出与中间注意力上提供**时间平滑**的蒸馏目标，减轻单步学生噪声对旧知识的破坏。

#### 1.2.1 学生模型的学习任务与目标

学生在同一前向中输出 logits（及中间注意力），损失端通常包含三项（系数见 §2.3）：

**（1）当前任务分割损失（本框架常用 Focal Loss）**

对非 `ignore_index` 像素，在交叉熵上加入难易调制：

$$
L_{\mathrm{focal}} = -\alpha \,(1 - p_t)^{\gamma} \,\log p_t ,
$$

$p_t$ 为模型对真类的预测概率；$\alpha,\gamma$ 由配置（如 `focal_alpha`、`focal_gamma`）给定，用于**强调难例、缓解类间像素量极不均衡**。（工程上亦可切换为 CE / OHEM 等，由 `seg_loss_type` 选择；**方法叙述以 Focal 为主**时与当前默认实验一致。）

**（2）对数层知识蒸馏（Logit KD，仅旧类）**

将教师 logits 经温度 $T_{\mathrm{kd}}$ 软化，与学生对齐；**蒸馏仅施加在旧类通道**上，使教师在未覆盖新类时**不会**通过新类 logit 误导学生。全局权重记为 $\lambda_{\mathrm{kd}}$（如 `alpha_kd`）。这是类增量中抑制“教师瞎指导新类区域”的标准技巧。

**（3）注意力层知识蒸馏（Attention KD，cosine 分支）**

从编码器 **若干 Transformer stage** 提取 **自注意力图**（由 `active_attention_stages` 指定 stage 索引，由 `attention_stage_weights` 对各 stage 损失加权）。**`attention_start_task`** 可设为 $k\ge 1$：仅从第 $k$ 个增量任务起启用注意力蒸馏，以便教师已具备最早一批类的知识后再对齐“看向何处”。本文固定采用 **cosine 相似度损失**，以约束学生与教师在注意力模式上的方向一致性；全局强度由 **`attention_kd_weight`** 控制。

**评估（非训练）。** **`eval_mode`** 可在学生、教师或二者平均等模式下报告指标；**`pred_confidence_threshold`** 可将低置信预测在指标中视为忽略类，与训练损失无关。

#### 1.2.2 教师模型的动量更新机制

每步在学生完成梯度更新后执行 **EMA**：

$$
\theta_T \leftarrow \alpha_{\mathrm{ema}} \, \theta_T + (1-\alpha_{\mathrm{ema}})\, \theta_S .
$$

$\alpha_{\mathrm{ema}}$ 接近 1（如 `0.999`–`0.9999`）时教师**惯性大**、目标更稳定；接近 0 则教师更紧跟学生。实现为**与优化步对齐的步级更新**；$\theta_T$ 固定 **`requires_grad=False`**。

---

### 1.3 经验回放（ER）

框架采用 **有界经验池**：新样本持续写入，容量满时按策略替换（**`drop_method: random`** 时为随机淘汰）。**`mem_size`** 控制池大小；每个训练步从池中读取 **`mem_batch_size`**（及 `mem_iters`）与当前流数据混合反传，使优化器仍接触**历史图像–标签对**，缓解旧类遗忘。

**说明。** 亦可配置**类别平衡采样、自适应回放批量**等扩展；若关闭则退化为**均匀随机回放**，方法核心仍为 **ER + 双模型蒸馏**，不因具体采样插件而改变框架定义。

---

## 2 基于自适应注意力掩码的动态知识蒸馏机制

### 2.1 注意力对齐（cosine 分支）

**表示。** 自注意力张量可写为 $(B, H_h, Q, K)$：batch、head、query、key。**`attention_head_aggregation`** 可在辅助路径上对 $H_h$ 做 mean/max；cosine 分支按 stage 对学生与教师注意力进行同尺度对齐后计算相似度损失。

**计算开销。** 常将 $(Q,K)$ 视作 2D 图并对齐到固定池化网格 $P\times P$（如 **`stu_pool_size` / `pool_size`**）。当前实现对所有长度不一致的权重对齐均采用**显式二维重塑 + 双线性插值**：先由编码器提供或已知网格给出 $(H_{\mathrm{old}},W_{\mathrm{old}})$，再映射到目标 $(H_{\mathrm{new}},W_{\mathrm{new}})$，最后展平。

**Cosine 项。** 设对齐后的学生与教师注意力表示分别为 $\mathbf{a}_S,\mathbf{a}_T$，定义

$$
L_{\mathrm{cos}} = 1 - \frac{\langle \mathbf{a}_S, \mathbf{a}_T \rangle}{\|\mathbf{a}_S\|_2\,\|\mathbf{a}_T\|_2 + \varepsilon}.
$$

该项度量学生与教师注意力模式的一致性。多 stage 的标量损失按 **`attention_stage_weights`** 加权求和得 $L_{\mathrm{attn}}$。

**与 `stu_weight_mode`。** cosine 分支中，可在**标签分辨率 $H\times W$** 上构造掩码/权重后再下采样到 $P$（`label_pixel`），并通过显式 $(H,W)$ 参数映射到 query 维用于加权。阅读代码时以 `loss_type == "cosine"` 分支为准。

---

### 2.2 掩码式与自适应加权的注意力蒸馏（MAD + STU）

**动机。** 教师对**当前新类**未充分学习时，其注意力模式不宜强行强加给学生；同时希望在**旧类与可信 ignore 区域**保留“向教师看齐”的约束。**MAD（Masked Attention Distillation）** 给出**基掩码** $M_{\mathrm{base}}\in[0,1]^{H\times W}$；**STU** 再乘上**连续自适应权重**，得到最终 query 级权重 $w$，对 cosine 注意力损失加权。

**（1）MAD 基掩码。** 在 **`current_only`** 下，对当前任务图像：有当前类真值标注的像素通常**不参与**注意力蒸馏（避免用教师误导新类）；对 **ignore** 区域，若教师 **argmax 属于旧类**且 **全类最大概率**高于松弛阈值 $\delta_{\mathrm{mad}}$（**`mad_relaxed_threshold`**），则允许蒸馏，以利用“旧类在 ignore 里的软一致性”；**回放**图像上可对旧类像素与 ignore **更宽松**地置 1。教师 softmax **不用于硬伪标签替换 GT**，仅用于 MAD 等掩码构造。

**（2）对齐到 query。** 将 $M_{\mathrm{base}}$ 在二维网格上（最近邻或双线性，依对象而定）映射到与各 stage 一致的 **query 网格 $(H_q,W_q)$**，再展平得到 `base_mask_q`。其中 $(H_q,W_q)$ 来自编码器 stage 的真实空间尺寸（`attention_query_hw`），不通过 $\sqrt{Q}$ 推断。

**（3）Query 级自适应（cosine 实现）。** 在池化 query 维上构造差分权重 $w_{\mathrm{diff}}$ 与熵权重 $w_{\mathrm{entropy}}$（实现依赖 **`stu_weight_mode`** 等），二者在**全体对齐位置**上**线性相加**后经非负截断得到自适应系数。记 $\alpha$ 为 **`cosine_entropy_alpha`**，则

$$
w_{\mathrm{ada}} = \mathrm{clamp}\bigl( w_{\mathrm{diff}} + \alpha\, w_{\mathrm{entropy}},\; \min 0 \bigr),
\qquad
w = M_{\mathrm{base}} \cdot w_{\mathrm{ada}} .
$$

**`stu_min_mask_value`** 可对 $w$ 设下界，避免蒸馏权重过小。**实现中若 STU 相对基掩码过稀，可回退为仅用基掩码**（fallback 比例由代码默认或配置指定）。

**（4）加权 cosine。** 对每个 head/查询位置，将权重 $w_i$ 作用于 cosine 损失并归一化聚合，再对 head 与 batch 聚合，乘 stage 权重并入 $L_{\mathrm{attn}}$。

---

### 2.3 总体损失与优化（符号化）

$$
L = L_{\mathrm{seg}}
+ \lambda_{\mathrm{kd}}\, L_{\mathrm{logit\text{-}KD}}^{\mathrm{(old)}}
+ \lambda_{\mathrm{attn}}\, L_{\mathrm{attn}} .
$$

- $L_{\mathrm{seg}}$：当前任务可见像素上的分割损失（如 Focal）。  
- $L_{\mathrm{logit\text{-}KD}}^{\mathrm{(old)}}$：带温度、**仅旧类** 的 logit 蒸馏。  
- $L_{\mathrm{attn}}$：§2.1–2.2 的 **多 stage、掩码加权** 注意力项。  

**优化。** 常用 **AdamW**：学习率 $\eta$、权重衰减 $\lambda_{wd}$、**梯度裁剪** $g_{\max}$。单步顺序：**前向（流 + 回放）→ 求 $L$ → 反传 → 更新 $\theta_S$ → EMA 更新 $\theta_T$**。数据增强（随机裁剪、翻转、颜色抖动等）由 `BaseSegmentationLearner` 管线提供，强度可由实现默认值或扩展配置控制。

---

## 附录 A：符号与配置项对应（与具体数值解耦）

| 符号 / 概念 | 典型配置键 |
|-------------|------------|
| 任务划分 | `increment` 或 `n_tasks`，`class_order`，`labels_order` |
| 训练标签 | `label_mode`（如 `current_only`），`ignore_index` |
| 骨干 | `segformer_variant`，`pretrained`，`freeze_encoder`，`img_size`，`n_classes` |
| 在线批量 | `batch_size`，`epochs`，`mem_batch_size`，`mem_iters` |
| 回放池 | `buffer`，`mem_size`，`drop_method` |
| EMA | `ema_alpha` |
| Logit KD | `alpha_kd`，`kd_temperature` |
| Attention KD (cosine) | `attention_kd_weight`，`active_attention_stages`，`attention_stage_weights`，`attention_start_task`，`attention_loss_type=cosine` |
| Head / 分辨率 | `attention_head_aggregation`，`stu_pool_size`（或等价 `pool_size`） |
| MAD / STU | `use_masked_distillation`，`stu_enabled`，`mad_relaxed_threshold`，`cosine_entropy_alpha`，`stu_entropy_gamma`，`stu_min_mask_value`，`stu_eps`，`stu_weight_mode` |
| 分割损失 | `seg_loss_type`，`focal_alpha`，`focal_gamma`（若用 Focal） |
| 评估 | `eval_mode`，`pred_confidence_threshold` |

---

## 附录 B：Cityscapes 实验超参数配置表（对应当前 YAML）

下表整理自 `config/seg/cityscapes_ema_attention.yaml`（当前版本）。若代码与文档更新，以 YAML 实际值为准。

| 模块 | 配置键 | 当前取值 |
|------|--------|----------|
| 训练与优化 | `epochs` | `1` |
| 训练与优化 | `batch_size` | `2` |
| 训练与优化 | `learning_rate` | `0.00006` |
| 训练与优化 | `optim` | `AdamW` |
| 训练与优化 | `weight_decay` | `0.01` |
| 训练与优化 | `grad_clip` | `1.0` |
| 模型 | `segformer_variant` | `mit_b0` |
| 模型 | `pretrained` | `true` |
| 模型 | `freeze_encoder` | `false` |
| 数据集 | `dataset` | `cityscapes` |
| 数据集 | `img_size` | `[512, 1024]` |
| 数据集 | `n_classes` | `19` |
| 数据集 | `ignore_index` | `255` |
| 数据集 | `label_mode` | `current_only` |
| 数据集 | `num_workers` | `4` |
| 任务划分 | `training_type` | `inc` |
| 任务划分 | `increment` | `"11-5"` |
| 任务划分 | `class_order` | `sequential` |
| 回放（ER） | `buffer` | `seg_reservoir` |
| 回放（ER） | `mem_size` | `500` |
| 回放（ER） | `mem_batch_size` | `4` |
| 回放（ER） | `mem_iters` | `2` |
| 回放（ER） | `drop_method` | `random` |
| EMA 教师 | `ema_alpha` | `0.9999` |
| EMA 教师 | `ema_update_strategy` | `step` |
| Logit KD | `alpha_kd` | `1.0` |
| Logit KD | `kd_temperature` | `4.0` |
| Attention KD | `attention_start_task` | `1` |
| Attention KD | `attention_kd_weight` | `5.0` |
| Attention KD | `attention_loss_type` | `cosine` |
| Attention KD | `active_attention_stages` | `[2, 3]` |
| Attention KD | `attention_stage_weights` | `[1.0, 1.5]` |
| Attention KD | `attention_head_aggregation` | `mean` |
| MAD/STU | `use_masked_distillation` | `true` |
| MAD/STU | `stu_enabled` | `true` |
| MAD/STU | `stu_weight_mode` | `label_pixel` |
| MAD/STU | `stu_pool_size` | `32` |
| MAD/STU | `stu_eps` | `1e-8` |
| MAD/STU | `stu_entropy_gamma` | `1.5` |
| MAD/STU | `cosine_entropy_alpha` | `0.5` |
| MAD/STU | `stu_min_mask_value` | `0.0` |
| MAD/STU | `mad_relaxed_threshold` | `0.1` |
| MAD/STU | `stu_disable_margin_gate` | `true` |
| 分割损失 | `seg_loss_type` | `focal` |
| 分割损失 | `focal_alpha` | `0.25` |
| 分割损失 | `focal_gamma` | `2.0` |
| 评估 | `eval_mode` | `avg` |
| 评估 | `pred_confidence_threshold` | `0.01` |
| 实验记录 | `learner` | `ER_EMA_Attention_Seg` |
| 实验记录 | `tag` | `cityscapes_ema_attention_seg_11-5` |
| 实验记录 | `seed` | `0` |
| 实验记录 | `n_runs` | `1` |

---

*未在文中单独展开的可选模块（如缓冲区类别平衡、自适应回放批量等）在配置关闭或未选用时，不改变上述核心框架表述。*
