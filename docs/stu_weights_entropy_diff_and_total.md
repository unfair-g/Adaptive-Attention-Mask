# STU 自适应权重：**W_entropy**、**W_diff** 与总权重

本文档在 **余弦注意力蒸馏**（`loss_type == "cosine"`）、**STU 开启**（`stu_enabled=True`），且 **`stu_weight_mode = label_pixel`**（有有效 `labels`）的前提下，给出两个分权与合成后 **stu_mask** 的数学定义。实现对应 `src_v3/utils/attention_distillation.py`。

当前实现口径（与近期代码一致）：
- 不使用 teacher 的 hard pseudo-label 恢复；`W_entropy` / `W_diff` 仅用于注意力蒸馏掩码加权。
- EMA 更新策略在训练实现中固定为 step（teacher 每步 EMA 更新）。

---

## 0. 核心约定：在 **Query 维** 上定义与对齐

注意力张量形状为 $(B,H_h,Q,K)$ 时，**第二维 $Q$ 为 query 维**：第 $q$ 个 query 对应一行 $\mathbf{A}_{:,h,q,:}$，即在 **该 query 位置**上对全体 **key**（第三维 $K$）的权重分布。

- **$\mathbf{W}_{\mathrm{diff}}$（差分权）**：**直接在 query 维上**逐下标 $q=1,\ldots,Q'$ 计算——每个 $q$ 取师生在该行的 $K'$ 维向量做余弦比较；再按显式 query 网格 $(H_q,W_q)$ 做二维双线性对齐到目标网格 $(H_t,W_t)$ 并展平，得到与损失张量同长的 $W_{\mathrm{diff},n}$。这是本方法中最严格的「query 维」对象。
- **$\mathbf{W}_{\mathrm{entropy}}$（熵权）**：先对学生 $(Q,K)$ 图做 head 聚合，其中 **第一维仍是 query、第二维是 key**；经双线性到标签栅格再池化并展平后，得到 $W_{\mathrm{entropy},n}$。展平索引 $n$ 与同一 stage 中 **`_process_attention` 得到的 `s_flat`、余弦损失的展平顺序一致**，因此在 SegFormer 类 **patch 序列与 query 一一对应**的设定下，**第 $n$ 个位置即第 $n$ 个 query 单元在池化格上的对齐点**。
- **总权重 / `stu_mask`**：各量均在 **与上述 query 展平索引 $n$ 对齐**后逐点相乘或相加；加权余弦损失是在 **每个 query 对齐位置 $n$** 上对 $(1-\cos)$ 类项应用 $\mathrm{stu\_mask}_n$。

下文凡出现序列下标 $n$，均指与 **query 轴展平后的损失索引**一致（除非另有所述）。

---

## 1. 符号与形状

| 符号 | 含义 |
|------|------|
| $b$ | batch 下标 |
| $B$ | batch 大小 |
| $H_h$ | 注意力 head 数 |
| $\mathbf{A}_S,\mathbf{A}_T$ | 学生 / 教师注意力，形状 $(B,H_h,Q,K)$；**$Q$=query 长度，$K$=key 长度** |
| $H_0,W_0$ | 标签图高宽；$N_0=H_0W_0$ |
| $P$ | `pool_size`；展平长度 $N=P^2$ |
| $n\in\{1,\ldots,N_{\mathrm{loc}}\}$ | 与当前 stage 的 **`s_flat` / 余弦项** 对齐的 **query 展平**下标（通常 $N_{\mathrm{loc}}=N$） |
| $\varepsilon$ | `eps` |
| $\gamma$ | `entropy_gamma` |
| $\alpha$ | `cosine_entropy_alpha` |
| $M_{\mathrm{base}}$ | MAD 等得到的 **base_mask**，与 STU 同形状 $(B,N_{\mathrm{loc}})$ |

**Agg** 表示对 head 维按配置做 **mean** 或 **max**（`attention_head_aggregation`）。

---

## 2. 注意力熵权重 $\mathbf{W}_{\mathrm{entropy}}$

仅使用**学生**注意力；在标签分辨率上构造标量场，再双线性到池化网格。**实现**：`_stu_hw_entropy_diff_weights`（`compute_pixel_jsd=False` 时）→ 张量 `spatial_weight`，下文记为 $W_{\mathrm{entropy}}$。

**与 query 维的关系**：$\mathbf{M}_S$ 的前两维是 **$(Q,K)$ = (query, key)** 网格；熵场经图像级插值与池化后得到的 $W_{\mathrm{entropy},n}$，按实现与 **`s_flat` 的第 $n$ 个分量**对齐，即与 **第 $n$ 个 query 对齐位置**上的蒸馏强度一致。

### 2.1 从多 head 到单张 $(Q,K)$ 图（query × key）

$$
\mathbf{M}_S = \mathrm{Agg}_{H_h}(\mathbf{A}_S)\in\mathbb{R}^{B\times 1\times Q\times K}.
$$

其中平面上的行索引对应 **query 维** $q$，列索引对应 **key 维** $k$。将 $\mathbf{M}_S$ 视作单通道 **2D 图**双线性插值到 $(H_0,W_0)$，展平为向量 $\mathbf{v}^{(b)}\in\mathbb{R}^{N_0}$（第 $b$ 个样本一行）。

### 2.2 归一化代理（非 softmax）

逐分量下界截断（与 `_normalize_attention` 一致）：

$$
p_t^{(b)} = \max\!\left(v_t^{(b)},\,\varepsilon\right),\quad t=1,\ldots,N_0.
$$

### 2.3 边际熵项、缩放与幂次

定义逐像素辅助量（对 $\mathbb{R}^{N_0}$ 的每个分量 $t$，**不是**整条分布的 Shannon 熵）：

$$
h_t^{(b)} = -\,p_t^{(b)}\,\ln\!\left(p_t^{(b)}+\varepsilon\right).
$$

缩放到 $[0,1]$ 再取幂：

$$
u_t^{(b)} = \mathrm{clip}\!\left(\frac{h_t^{(b)}}{\ln N_0 + \varepsilon},\,0,\,1\right),\qquad
\widetilde W_{t}^{(b)} = \bigl(u_t^{(b)}\bigr)^{\gamma}.
$$

将 $\widetilde W^{(b)}$ reshape 为 $H_0\times W_0$，**双线性**缩放到 $P\times P$，按行展平，得到与池化格对齐的 $\mathbf{w}_{\mathrm{entropy}}^{(b)}\in\mathbb{R}^{N}$。

若后续与当前 stage 的序列长度 $N_{\mathrm{loc}}$ 不一致，再对 $\mathbf{w}_{\mathrm{entropy}}^{(b)}$ 做显式二维对齐：按已知 $(H_{\mathrm{old}},W_{\mathrm{old}})$ 重塑后，双线性插值到 $(H_{\mathrm{new}},W_{\mathrm{new}})$，再展平得到最终

$$
\boxed{W_{\mathrm{entropy},\,n}^{(b)} \in [0,1],\quad n=1,\ldots,N_{\mathrm{loc}}.}
$$

---

## 3. 注意力差值权重 $\mathbf{W}_{\mathrm{diff}}$（**Query 维逐行**）

**在 query 维上**：对每个 query 下标 $q$，用师生在该行的两条 **key 维**向量衡量方向差异；**不在 key 维上再压缩成一个标量之前**，每一行只属于一个 query。**实现**：`_cosine_difference_weights` → `w_diff_pos`（Cosine STU 下即 **W_diff**）。

### 3.1 Head 聚合与对齐

$$
\widetilde{\mathbf{S}}_S = \mathrm{Agg}_{H_h}(\mathbf{A}_S),\quad
\widetilde{\mathbf{S}}_T = \mathrm{Agg}_{H_h}(\mathbf{A}_T)\in\mathbb{R}^{B\times Q\times K}.
$$

取公共截断长度 $Q'=\min(Q_S,Q_T)$、$K'=\min(K_S,K_T)$，只保留前 $Q'$ 行、前 $K'$ 列。

### 3.2 沿 query 维：逐行余弦距离

对每个 **query 索引** $(b,q)$，记 $\mathbf{s}_{S}^{(b,q)},\mathbf{s}_{T}^{(b,q)}\in\mathbb{R}^{K'}$ 为 $\widetilde{\mathbf{S}}_S,\widetilde{\mathbf{S}}_T$ 的第 $q$ 行（该 query 在全体 key 上的注意力向量），余弦相似度为

$$
\cos_{b,q}=
\frac{\mathbf{s}_{S}^{(b,q)\top}\mathbf{s}_{T}^{(b,q)}}
{\left\lVert \mathbf{s}_{S}^{(b,q)}\right\rVert_2\,\left\lVert \mathbf{s}_{T}^{(b,q)}\right\rVert_2}.
$$

差分权重（裁剪到 $[0,1]$）：

$$
\widetilde W_{\mathrm{diff},\,q}^{(b)} = \mathrm{clip}\!\left(1-\cos_{b,q},\,0,\,1\right),\quad q=1,\ldots,Q'.
$$

若 $Q'\neq N_{\mathrm{loc}}$，对 **query 维上**序列先按 query 网格 $(H_q,W_q)$ 重塑，再双线性对齐到目标网格 $(H_t,W_t)$ 并展平到 $N_{\mathrm{loc}}$（与 `s_flat` 的 query 展平轴对齐），记该算子为 $\mathrm{BilinearResize}_{(H_t,W_t)}$，则

$$
\boxed{W_{\mathrm{diff},\,n}^{(b)} =
\mathrm{vec}\!\left(
\mathrm{BilinearResize}_{(H_t,W_t)}
\left[\mathrm{reshape}_{(H_q,W_q)}(\widetilde W_{\mathrm{diff}}^{(b)})\right]
\right)_n,\quad n=1,\ldots,N_{\mathrm{loc}}.}
$$

---

## 4. 总自适应系数 $\mathbf{W}_{\mathrm{ada}}$ 与最终 **stu_mask**

STU 将 **MAD 门控** 与上述权重组合，得到送入加权余弦损失的软掩码。**Cosine + 有 `teacher_probs`**（管线中传入教师 softmax，用于 MAD 等）时，`adaptive_weight` 取差分权与熵权在**每个对齐位置**上的加权和再截断：

$$
\boxed{W_{\mathrm{ada},\,n}^{(b)} = \mathrm{clamp}\!\left(
W_{\mathrm{diff},\,n}^{(b)} + \alpha\,W_{\mathrm{entropy},\,n}^{(b)}
,\;\min=0\right).}
$$

MAD 在当前实现中还包含 `mad_relaxed_threshold` 等过滤（作用于可蒸馏位置选择），与上式 STU 系数相乘进入最终掩码。

**Cosine + 无 `teacher_probs`** 时，两权直接逐点相乘：

$$
\boxed{W_{\mathrm{ada},\,n}^{(b)} = W_{\mathrm{entropy},\,n}^{(b)}\,W_{\mathrm{diff},\,n}^{(b)}.}
$$

（实现里写作 `base_mask * spatial_weight * w_diff_pos`。）

### 4.1 与 base_mask 合成及下限

语义门控与 STU 合成：

$$
\mathrm{stu\_mask}_{n}^{(b)} = M_{\mathrm{base},\,n}^{(b)}\,W_{\mathrm{ada},\,n}^{(b)}.
$$

若 `min_mask_value` $=c>0$，则

$$
\mathrm{stu\_mask} \leftarrow \max(\mathrm{stu\_mask},\,c)
$$

（逐元素）。训练时若 STU 相对 `base_mask` 过稀，可追溯 `stu_fallback` 逻辑退回仅用 `base_mask`（见源码）。

---

## 5. 与损失的关系（摘要）

单 stage 的余弦蒸馏在 **与 query 展平一致的索引 $n$** 上使用 $\mathrm{stu\_mask}_n$ 对 $(1-\cos)$ 类项加权求平均：**总权重**在门控意义上即为 $\mathrm{stu\_mask}$；$\mathbf{W}_{\mathrm{entropy}}$ 与 $\mathbf{W}_{\mathrm{diff}}$ 仅通过 $W_{\mathrm{ada}}$ 进入该掩码。换言之，**自适应强度是按 query 位置（展平后下标 $n$）逐点施加的**。

---

## 6. 实现对照

| 量 | 变量 / 函数 |
|----|-------------|
| $W_{\mathrm{entropy}}$ | `spatial_weight`；`_stu_hw_entropy_diff_weights` |
| $W_{\mathrm{diff}}$ | `w_diff_pos`；`_cosine_difference_weights` |
| $W_{\mathrm{ada}}$ | `adaptive_weight`（有 `teacher_probs`）或 `spatial_weight * w_diff_pos` |
| $\mathrm{stu\_mask}$ | `stu_mask`；乘 `base_mask` 与可选 `clamp` |
