# ggml_tensor 与模型的对应关系

`ggml_tensor` 是承载数据的"容器"，而模型中的权重、偏置、归一化参数等都有对应的 `ggml_tensor` 实例。本文档说明模型组件如何映射到 `ggml_tensor`，以及从 GGUF 文件到计算图的三层对应关系。

---

## 目录

- [1. 模型结构的张量组织](#1-模型结构的张量组织)
- [2. GGUF 张量名称 → 模型结构字段的映射](#2-gguf-张量名称--模型结构字段的映射)
- [3. 权重张量的字段状态](#3-权重张量的字段状态)
- [4. 从权重到计算图：用具体数字走一遍](#4-从权重到计算图用具体数字走一遍)
  - [4.1 假设一个小模型](#41-假设一个小模型)
  - [4.2 步骤 1：输入 — 从 token 到向量](#42-步骤-1输入--从-token-到向量)
  - [4.3 步骤 2：注意力前的归一化（RMS Norm）](#43-步骤-2注意力前的归一化rms-norm)
  - [4.4 步骤 3：Q/K/V 投影（矩阵乘法）](#44-步骤-3qkv-投影矩阵乘法)
  - [4.5 步骤 4：RoPE 旋转位置编码](#45-步骤-4rope-旋转位置编码)
  - [4.6 步骤 5：注意力计算](#46-步骤-5注意力计算)
  - [4.7 步骤 6：输出投影 + 残差连接](#47-步骤-6输出投影--残差连接)
  - [4.8 步骤 7：FFN（前馈网络）](#48-步骤-7ffn前馈网络)
  - [4.9 步骤 8：再次残差连接](#49-步骤-8再次残差连接)
  - [4.10 全流程总结](#410-全流程总结一图看懂)
- [5. 模型权重的两种角色](#5-模型权重的两种角色)
- [6. 叶子节点与计算节点的区别](#6-叶子节点与计算节点的区别)
- [7. 输入张量的特殊处理](#7-输入张量的特殊处理)
- [8. 输出张量的对应](#8-输出张量的对应)
- [9. 总结：三层对应关系](#9-总结三层对应关系)
- [10. 模型加载的完整逻辑](#10-模型加载的完整逻辑)
  - [10.1 总览：加载的五步流程](#101-总览加载的五步流程)
  - [10.2 第 1 步：读取 GGUF 文件](#102-第-1-步读取-gguf-文件)
  - [10.3 第 2 步：检测模型架构](#103-第-2-步检测模型架构)
  - [10.4 第 3 步：加载超参数](#104-第-3-步加载超参数)
  - [10.5 第 4 步：加载词表](#105-第-4-步加载词表)
  - [10.6 第 5 步：加载张量（核心）](#106-第-5-步加载张量核心)
  - [10.7 完整加载流程图](#107-完整加载流程图从文件到可推理状态)
  - [10.8 分片模型的加载](#108-分片模型split-model的加载)
  - [10.9 错误处理](#109-错误处理)
- [11. 权重数据加载的详细代码逻辑](#11-权重数据加载的详细代码逻辑)
  - [11.1 三条加载路径总览](#111-三条加载路径总览)
  - [11.2 前置知识：llama_model_loader 的关键字段](#112-前置知识llama_model_loader-的关键字段)
  - [11.3 阶段 A：create_tensor — 只记录位置，不读数据](#113-阶段-acreate_tensor--只记录位置不读数据)
  - [11.4 阶段 B：init_mappings — 建立 mmap 映射](#114-阶段-binit_mappings--建立-mmap-映射)
  - [11.5 阶段 C：load_all_data — 把数据关联到张量](#115-阶段-cload_all_data--把数据关联到张量)
  - [11.6 用具体例子走一遍](#116-用具体例子走一遍)
  - [11.7 加载后的张量状态对比](#117-加载后的张量状态对比)
  - [11.8 完整时间线](#118-完整时间线)
- [12. 推理全流程：从用户输入到模型输出](#12-推理全流程从用户输入到模型输出)
  - [12.1 一句话总结](#121-一句话总结)
  - [12.2 用具体例子走一遍](#122-用具体例子走一遍)
  - [12.3 自回归生成循环](#123-自回归生成循环)
  - [12.4 完整推理时间线](#124-完整推理时间线)
  - [12.5 数据流转总结](#125-数据流转总结)

---

## 1. 模型结构的张量组织

`llama_model` 类（`src/llama-model.h`）将模型权重组织为两级结构：全局张量 + 逐层张量。

```
llama_model
  ├── tok_embd          (ggml_tensor*) 全局：token 嵌入矩阵
  ├── output_norm       (ggml_tensor*) 全局：输出层归一化权重
  ├── output            (ggml_tensor*) 全局：输出投影矩阵
  ├── pos_embd          (ggml_tensor*) 全局：位置嵌入（可选）
  │
  └── layers[0..N-1]    (vector<llama_layer>) 每层各自的权重
        │
        ├── 注意力部分
        │     ├── attn_norm       归一化权重
        │     ├── wq / wk / wv    Q/K/V 投影矩阵
        │     ├── wo              输出投影矩阵
        │     ├── wq_b / wk_b / wv_b / wo_b  偏置（可选）
        │     ├── wqkv            融合 QKV 投影（可选，部分模型使用）
        │     └── rope_freqs / rope_long / rope_short  RoPE 频率（可选）
        │
        ├── FFN 部分
        │     ├── ffn_norm        归一化权重
        │     ├── ffn_up          上投影矩阵
        │     ├── ffn_gate        门控矩阵
        │     ├── ffn_down        下投影矩阵
        │     └── ffn_up_b / ffn_gate_b / ffn_down_b  偏置（可选）
        │
        └── MoE 部分（可选）
              ├── ffn_gate_inp    路由器权重
              ├── ffn_gate_exps   专家门控权重
              ├── ffn_up_exps     专家上投影
              └── ffn_down_exps   专家下投影
```

每个字段都是 `struct ggml_tensor *` 指针，指向从 GGUF 文件加载的张量数据。

---

## 2. GGUF 张量名称 → 模型结构字段的映射

模型加载时，GGUF 文件中的每个张量通过名称匹配到对应的 `ggml_tensor*` 字段：

```
GGUF 张量名称                           →  模型结构字段
────────────────────────────────────────────────────────────
全局张量（不属于任何层）：

  "token_embd.weight"                   →  model.tok_embd
  "output_norm.weight"                  →  model.output_norm
  "output_norm.bias"                    →  model.output_norm_b
  "output.weight"                       →  model.output
  "rope_freqs.weight"                   →  model.rope_freqs

逐层张量（以第 i 层为例）：

  "blk.{i}.attn_norm.weight"            →  model.layers[i].attn_norm
  "blk.{i}.attn_q.weight"              →  model.layers[i].wq
  "blk.{i}.attn_k.weight"              →  model.layers[i].wk
  "blk.{i}.attn_v.weight"              →  model.layers[i].wv
  "blk.{i}.attn_output.weight"         →  model.layers[i].wo
  "blk.{i}.attn_q.bias"                →  model.layers[i].wq_b
  "blk.{i}.attn_k.bias"                →  model.layers[i].wk_b
  "blk.{i}.attn_v.bias"                →  model.layers[i].wv_b
  "blk.{i}.attn_output.bias"           →  model.layers[i].wo_b

  "blk.{i}.ffn_norm.weight"            →  model.layers[i].ffn_norm
  "blk.{i}.ffn_gate.weight"            →  model.layers[i].ffn_gate
  "blk.{i}.ffn_up.weight"              →  model.layers[i].ffn_up
  "blk.{i}.ffn_down.weight"            →  model.layers[i].ffn_down
  "blk.{i}.ffn_gate.bias"              →  model.layers[i].ffn_gate_b
  "blk.{i}.ffn_up.bias"                →  model.layers[i].ffn_up_b
  "blk.{i}.ffn_down.bias"              →  model.layers[i].ffn_down_b

MoE 扩展：

  "blk.{i}.ffn_gate_inp.weight"        →  model.layers[i].ffn_gate_inp
  "blk.{i}.ffn_gate_exps.weight"       →  model.layers[i].ffn_gate_exps
  "blk.{i}.ffn_up_exps.weight"         →  model.layers[i].ffn_up_exps
  "blk.{i}.ffn_down_exps.weight"       →  model.layers[i].ffn_down_exps
```

加载流程：`GGUF 文件 → 读取张量名称 → tn() 函数生成名称 → 匹配到对应枚举 → 赋值到结构体字段`。

---

## 3. 权重张量的字段状态

模型权重加载完成后，每个权重张量（如 `model.layers[0].wq`）的状态：

```
struct ggml_tensor（权重张量，如 wq）{
    type   = GGML_TYPE_Q4_0       ← 量化类型（来自 GGUF 文件）
    buffer = 非 NULL               ← 已分配到后端缓冲区（可能是 GPU 显存）
    ne     = {4096, 4096, 1, 1}    ← 形状：[输出维度, 输入维度, 1, 1]
    nb     = 自动计算               ← 基于形状和量化类型的字节步长
    op     = GGML_OP_NONE          ← 叶子节点：无算子（数据直接来自文件）
    src[]  = 全部 NULL              ← 叶子节点无输入依赖
    data   = 指向 mmap 或缓冲区    ← 权重数据的实际地址
    name   = "blk.0.attn_q.weight" ← GGUF 中的原始名称
}
```

**关键特征**：权重张量都是**叶子节点**（`op = GGML_OP_NONE`，`src[] = NULL`）。它们是计算图的"锚点"——图的起点。

---

## 4. 从权重到计算图：用具体数字走一遍

### 4.1 假设一个小模型

为了看懂计算逻辑，我们用一个极小的假设模型：

```
参数设置：
  hidden_dim = 8          隐藏维度（每个 token 用 8 个数字表示）
  n_heads    = 2          注意力头数
  head_dim   = 4          每个头的维度（= 8 / 2）
  ffn_dim    = 16         FFN 中间层维度
  n_tokens   = 3          输入 3 个 token
  eps        = 0.001      RMS Norm 的 epsilon

权重矩阵（每层）：
  attn_norm   [8]         归一化缩放系数，8 个数
  wq          [8 x 8]     Q 投影：输入 8 维 → 输出 8 维
  wk          [8 x 8]     K 投影
  wv          [8 x 8]     V 投影
  wo          [8 x 8]     输出投影
  ffn_norm    [8]         FFN 归一化缩放系数
  ffn_gate    [16 x 8]    门控：输入 8 维 → 输出 16 维
  ffn_up      [16 x 8]    上投影：输入 8 维 → 输出 16 维
  ffn_down    [8 x 16]    下投影：输入 16 维 → 输出 8 维
```

### 4.2 步骤 1：输入 — 从 token 到向量

```
用户输入："你好世界"（3 个 token）

假设分词结果：token_id = [1024, 3847, 2015]

查嵌入表 tok_embd（形状 [vocab_size x 8]）：

  inpL = [
    [0.1, 0.3, -0.2, 0.5, 0.8, -0.1, 0.4, 0.2],   ← token 1024 的向量
    [0.7, -0.4, 0.1, 0.2, -0.3, 0.6, -0.5, 0.9],   ← token 3847 的向量
    [0.0, 0.8, -0.6, 0.3, 0.1, -0.2, 0.7, -0.1],   ← token 2015 的向量
  ]

  inpL 的形状：ne = [8, 3, 1, 1]    ← 8维 x 3个token

  对应 ggml_tensor：
    inpL.data → 指向这 24 个 float
    inpL.op   = GGML_OP_NONE（输入是已知数据，不需要计算）
    inpL.name = "inpL"
```

### 4.3 步骤 2：注意力前的归一化（RMS Norm）

**目的**：把输入向量归一化，防止数值太大或太小，让训练更稳定。

```
通俗理解：
  类似于考试分数标准化——不管原始分数范围多大，都调整到平均为 0、标准差为 1 的范围。
  然后乘以一个可学习的缩放系数（attn_norm），让模型自己决定每个维度多重要。

数学公式：
  rms = sqrt(mean(x^2) + eps)
  norm_out = (x / rms) * attn_norm

计算过程（以第 1 个 token 为例）：
  x = [0.1, 0.3, -0.2, 0.5, 0.8, -0.1, 0.4, 0.2]

  x^2 = [0.01, 0.09, 0.04, 0.25, 0.64, 0.01, 0.16, 0.04]
  mean(x^2) = 1.24 / 8 = 0.155
  rms = sqrt(0.155 + 0.001) = sqrt(0.156) ≈ 0.395

  x / rms = [0.253, 0.759, -0.506, 1.266, 2.025, -0.253, 1.013, 0.506]

  假设 attn_norm = [1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3]
  norm_out = x/rms * attn_norm
          = [0.253, 0.607, -0.607, 1.139, 2.228, -0.253, 0.709, 0.658]

  对应 ggml_tensor（norm_out 节点）：
    norm_out.op   = GGML_OP_RMS_NORM
    norm_out.src[0] = inpL          ← "我的输入是 inpL"
    norm_out.op_params = 0.001      ← eps 值
    norm_out.ne = [8, 3, 1, 1]      ← 输出形状与输入相同

  attn_norm 是权重张量（叶子节点），在图构建时被引用为乘法操作的参数。
```

### 4.4 步骤 3：Q/K/V 投影（矩阵乘法）

**目的**：把归一化后的向量分别投影成 Q（查询）、K（键）、V（值），为注意力计算做准备。

```
通俗理解：
  想象你在图书馆找书：
    Q（Query）= "我想找什么"（查询向量）
    K（Key）  = "每本书是什么"（键向量）
    V（Value）= "书的内容"（值向量）

  投影就是把同一个输入向量 x，通过三套不同的权重矩阵，变成三种不同用途的向量。

数学公式：
  Q = wq^T × norm_out    形状：[8x8] × [8x3] → [8x3]
  K = wk^T × norm_out    形状：[8x8] × [8x3] → [8x3]
  V = wv^T × norm_out    形状：[8x8] × [8x3] → [8x3]

对应当码中的写法：
  Qcur = ggml_mul_mat(ctx, wq, norm_out)
  // Qcur.op     = GGML_OP_MUL_MAT
  // Qcur.src[0] = wq          ← 权重矩阵（叶子）
  // Qcur.src[1] = norm_out    ← 上一步的输出
  // Qcur.ne     = [8, 3, 1, 1]

  Kcur = ggml_mul_mat(ctx, wk, norm_out)
  Vcur = ggml_mul_mat(ctx, wv, norm_out)

这里 wq、wk、wv 是已经加载好的权重张量（叶子节点），
它们的 data 指向模型文件中的权重数据，op = GGML_OP_NONE。

为什么是 wq^T × x 而不是 wq × x？
  因为 ggml 的 MUL_MAT 自动做转置，即计算 result = a^T × b。
  权重矩阵 wq 的形状是 [8, 8]（ne=[8, 8]），
  MUL_MAT 把它当作 a^T，所以实际计算的是 wq^T × norm_out。
```

### 4.5 步骤 4：RoPE 旋转位置编码

**目的**：让 Q 和 K 包含位置信息，这样模型能区分"猫追狗"和"狗追猫"。

```
通俗理解：
  如果没有位置编码，模型看到 [猫, 追, 狗] 和 [狗, 追, 猫] 会觉得一样
  （因为 Q/K 计算只看内容不看顺序）。
  RoPE 根据每个 token 的位置，对 Q 和 K 做旋转变换，
  让位置相近的 token 在注意力计算时有特殊的关系。

数学公式（简化）：
  对 Q/K 的每对相邻元素施加旋转矩阵：
  [cos(θ), -sin(θ)] × [x1]
  [sin(θ),  cos(θ)]   [x2]
  其中 θ 与位置和维度有关。

代码中的写法：
  Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, ...)
  // Qcur.op     = GGML_OP_ROPE
  // Qcur.src[0] = 原始 Qcur    ← 要旋转的向量
  // Qcur.src[1] = inp_pos      ← 位置信息 [0, 1, 2]（3 个 token 的位置）
  // Qcur.src[2] = freq_factors ← 频率因子（可选）

  Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, ...)
  // 对 K 做同样的旋转

V 不需要位置编码——因为 V 是"内容"，位置信息已经通过 Q 和 K 的注意力权重传达了。
```

### 4.6 步骤 5：注意力计算

**目的**：让每个 token 看到其他所有 token 的信息，决定"该关注谁"。

```
通俗理解：
  假设你在读句子"猫坐在垫子上"：
  - 处理"垫子"时，应该更关注"坐"和"猫"（因为它们提供语义上下文）
  - 注意力机制让模型自动学会"该看谁"

数学公式（标准注意力）：
  score = Q × K^T           先算 Q 和 K 的相似度
  weight = softmax(score)   相似度转成概率（和为 1）
  output = weight × V       用概率加权 V

以我们的 2 头模型为例：

  Q 形状 [8, 3] → 拆分为 2 个头：Q1 [4, 3], Q2 [4, 3]
  K 形状 [8, 3] → 拆分为 2 个头：K1 [4, 3], K2 [4, 3]
  V 形状 [8, 3] → 拆分为 2 个头：V1 [4, 3], V2 [4, 3]

  头 1 的计算：
    score1 = Q1^T × K1        形状：[3x4] × [4x3] → [3x3]
                             3个token对3个token的相似度矩阵

    score1 = [[ 2.1,  0.5, -0.3],    ← token 0 对各 token 的分数
              [ 0.8,  1.9,  0.2],    ← token 1
              [-0.1,  0.3,  1.5]]    ← token 2

    weight1 = softmax(score1)   每行归一化为概率
    weight1 = [[0.68, 0.14, 0.06],    ← token 0 最关注自己(0.68)
               [0.18, 0.62, 0.12],    ← token 1 也最关注自己
               [0.07, 0.13, 0.59]]    ← token 2 同理

    attn1 = weight1 × V1        用概率加权 V
    attn1 = [4, 3]              输出 3 个 token 各 4 维的表示

  头 2 同理计算 attn2 = weight2 × V2

  合并两个头：concat(attn1, attn2) → [8, 3]

实际代码使用 Flash Attention（一步完成上述所有操作，更快更省内存）：
  attn_out = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale)
  // attn_out.op     = GGML_OP_FLASH_ATTN_EXT
  // attn_out.src[0] = Q
  // attn_out.src[1] = K
  // attn_out.src[2] = V
  // attn_out.src[3] = mask       ← 注意力掩码（防止看到未来的 token）
```

### 4.7 步骤 6：输出投影 + 残差连接

```
输出投影：把注意力的多头输出映射回原始维度
  attn_proj = wo^T × attn_out     形状：[8x8] × [8x3] → [8x3]
  // ggml_mul_mat(ctx, wo, attn_out)

残差连接：把注意力的输出加回原始输入
  ffn_inp = attn_proj + inpL      形状：[8x3] + [8x3] → [8x3]

通俗理解：
  残差连接 = "保留原始信息，再叠加新理解"
  类似于：考试成绩 = 原始分数 + 加分项
  没有残差连接，多层网络叠加后原始信息容易丢失。
```

### 4.8 步骤 7：FFN（前馈网络）

**目的**：对每个 token 独立做非线性变换，增加模型的表达能力。

```
通俗理解：
  如果说注意力是"看别人"（token 之间的信息交换），
  那 FFN 就是"自我思考"（每个 token 自己消化信息）。

  SwiGLU FFN 的结构：
  先把 8 维扩展到 16 维（更多空间处理），再缩回 8 维。

数学公式：
  gate_out = ffn_gate^T × ffn_norm_out     [16x8] × [8x3] → [16x3]   门控路径
  up_out   = ffn_up^T   × ffn_norm_out     [16x8] × [8x3] → [16x3]   上投影路径
  mid      = SiLU(gate_out) * up_out        逐元素：激活 * 上投影     [16x3]
  ffn_out  = ffn_down^T × mid               [8x16] × [16x3] → [8x3]  下投影回原维度

SiLU 函数：SiLU(x) = x * sigmoid(x)
  当 x 很大时，SiLU(x) ≈ x（直通）
  当 x 很小时，SiLU(x) ≈ 0（关闭）
  所以 gate 路径像个"开关"，决定 up 路径的信息哪些该通过。

逐元素乘法 gate * up 的直观含义：
  gate_out = [1.2, -0.3, 2.1, ...]    ← 正值=允许通过，负值=阻止
  up_out   = [0.5,  0.8, 0.3, ...]    ← 实际内容
  SiLU(gate) * up = [0.5*0.85, 0.8*0.35, 0.3*0.96, ...]
                  ≈ [0.42, 0.28, 0.29, ...]
  负的 gate 值经 SiLU 后接近 0，对应位置的 up 信息被"关掉"了。

代码：
  gate_out = ggml_mul_mat(ctx, ffn_gate, cur)    ← ffn_gate 是叶子节点（权重）
  up_out   = ggml_mul_mat(ctx, ffn_up,   cur)    ← ffn_up   是叶子节点（权重）
  mid      = ggml_swiglu_split(ctx, gate_out, up_out)
  ffn_out  = ggml_mul_mat(ctx, ffn_down, mid)     ← ffn_down 是叶子节点（权重）
```

### 4.9 步骤 8：再次残差连接

```
  inpL = ffn_out + ffn_inp

  把 FFN 的输出加回注意力残差连接的结果上。
  现在 inpL 包含了：原始嵌入 + 注意力信息 + FFN 非线性变换。
  这个 inpL 会传入下一层，重复步骤 2-8。

  最后一层结束后：
    cur = rms_norm(inpL, output_norm)     最终归一化
    logits = output^T × cur               输出投影 → 词表上每个词的分数
    形状：[vocab_size x hidden_dim] × [hidden_dim x n_tokens] → [vocab_size x n_tokens]

  logits 的每一列是对应 token 位置上所有词的"得分"，
  采样器从中选得分最高的词作为下一个 token。
```

### 4.10 全流程总结（一图看懂）

```
输入 token_ids = [1024, 3847, 2015]
         │
         ▼
  ┌──────────────┐
  │ 查嵌入表     │  tok_embd：token_id → 向量
  │ inpL [8x3]   │
  └──────┬───────┘
         │ 保存原始输入 inpSA = inpL
         ▼
  ┌──────────────┐
  │ RMS Norm     │  × attn_norm：归一化 + 缩放
  │ [8x3]        │
  └──────┬───────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
  × wq  × wk  × wv          三个矩阵乘法（叶子：wq, wk, wv）
  [8x3] [8x3] [8x3]
    │    │    │
  RoPE RoPE  │              旋转位置编码（叶子：inp_pos）
    │    │    │
    ▼    ▼    ▼
  ┌─────────────┐
  │ 注意力       │  Q×K^T → softmax → ×V（叶子：mask）
  │ [8x3]       │
  └──────┬──────┘
         │
         ▼
  × wo [8x3]                输出投影（叶子：wo）
         │
         ▼
  + inpSA [8x3]              残差连接：注意力结果 + 原始输入
         │ 保存 ffn_inp = 当前值
         ▼
  ┌──────────────┐
  │ RMS Norm     │  × ffn_norm
  │ [8x3]        │
  └──────┬───────┘
         │
    ┌────┴────┐
    ▼         ▼
  × ffn_gate  × ffn_up      两个矩阵乘法（叶子：ffn_gate, ffn_up）
  [16x3]      [16x3]
    │         │
    ▼         │
  SiLU(gate)  │             门控激活
    │         │
    × up ─────┘             逐元素相乘
    │
    ▼
  × ffn_down [8x3]          下投影（叶子：ffn_down）
    │
    ▼
  + ffn_inp [8x3]            残差连接：FFN结果 + 注意力残差
    │
    ▼
  inpL → 传入下一层（N 层重复以上过程）
    │
    ▼  （最后一层之后）
  ┌──────────────┐
  │ RMS Norm     │  × output_norm
  └──────┬───────┘
         ▼
  × output [vocab_size x 3]  输出投影（叶子：output）
         │
         ▼
  logits → 采样器选下一个 token
```

图中的 **叶子节点**（wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down, attn_norm, ffn_norm, output_norm, output）全部来自模型权重，它们在加载时就存在于内存中，推理时只读不改。

**计算节点**（RMS Norm、MUL_MAT、RoPE、Flash Attention、SiLU、ADD）是每次推理时临时创建的，执行完毕后丢弃。

---

## 5. 模型权重的两种角色

同一个权重张量（如 `model.layers[0].wq`）在不同阶段扮演不同角色：

```
┌────────────────────────────────────────────────────────────┐
│  模型加载阶段                                               │
│                                                            │
│  GGUF 文件 ──读取──→ ggml_tensor（wq）                      │
│                      ├─ data 指向 mmap 的文件区域            │
│                      ├─ op = GGML_OP_NONE（叶子）           │
│                      ├─ name = "blk.0.attn_q.weight"        │
│                      └─ type = Q4_0（量化存储）              │
├────────────────────────────────────────────────────────────┤
│  计算图构建阶段                                             │
│                                                            │
│  wq 作为叶子节点被引用：                                    │
│    Qcur = ggml_mul_mat(ctx, wq, norm_out)                  │
│                            │                                │
│    Qcur 节点：  op = MUL_MAT                                │
│                 src[0] = wq（指向权重张量）                  │
│                 src[1] = norm_out                           │
│                                                            │
│  wq 本身不变——它只是被 Qcur 的 src[0] 指向                  │
├────────────────────────────────────────────────────────────┤
│  计算执行阶段                                               │
│                                                            │
│  后端执行 MUL_MAT 节点时：                                  │
│    1. 从 src[0]（wq）读取权重数据                           │
│    2. 从 src[1]（norm_out）读取输入                         │
│    3. 执行矩阵乘法（自动反量化 Q4_0 → F32）                 │
│    4. 将结果写入 Qcur 的 data                               │
│                                                            │
│  wq 的 data 仍然是原始权重数据，不会被修改                   │
└────────────────────────────────────────────────────────────┘
```

---

## 6. 叶子节点与计算节点的区别

### 一句话总结

**叶子节点** = 已知数据（直接拿来用），**计算节点** = 要算才知道（需要执行运算）。

### 用小模型举例

假设模型已经加载到内存，现在要推理一个 token。

#### 叶子节点：权重张量 wq

```
这是从模型文件直接读进来的数据，已经躺在内存里了：

wq {
    op   = GGML_OP_NONE       ← "我不需要计算，数据已经有了"
    src[] = 全部 NULL          ← "我不依赖任何其他张量"
    data  → [0.12, -0.34, 0.56, ...]  ← 直接指向模型文件 mmap 的内存区域
    name  = "blk.0.attn_q.weight"
    type  = Q4_0              ← 磁盘上就是量化存储的
}

生命周期：模型加载时创建，一直活到模型卸载。推理 100 次也不变。
在图中的位置：leafs[0] = wq（叶子数组的第一项）
```

#### 计算节点：Qcur = wq × norm_out

```
这个值推理之前根本不存在，必须现场算：

Qcur {
    op   = GGML_OP_MUL_MAT    ← "我需要做矩阵乘法"
    src[0] = wq               ← "我的第一个输入是 wq"（指向上面那个叶子）
    src[1] = norm_out          ← "我的第二个输入是 norm_out"（上一步算出来的）
    data  = NULL               ← 现在还没有数据！要等执行完才有
    name  = "node_5"           ← 图构建时自动起的名
    type  = F32               ← 计算结果一定是 F32
}

生命周期：本次推理时创建，推理完就丢弃。下次推理重新创建。
在图中的位置：nodes[5] = Qcur（计算节点数组的第 6 项）
```

#### 执行时的区别

```
图按拓扑序执行到 Qcur 这个节点时：

  1. 看到 op = GGML_OP_MUL_MAT → "哦，要做矩阵乘法"
  2. 从 src[0]（wq）拿到权重数据    ← 读叶子的 data
  3. 从 src[1]（norm_out）拿到输入   ← 读上一步计算节点的 data
  4. 执行矩阵乘法
  5. 把结果写入 Qcur.data            ← 现在它有值了

图执行到 wq 时：
  op = GGML_OP_NONE → "跳过，不需要计算"
  （实际上叶子节点根本不会被遍历执行，只有 nodes[] 里的才执行）
```

### 打个比方

```
做菜：

  叶子节点 = 食材（从冰箱/菜场拿来的，现成的）
    ├── 鸡蛋（wq）
    ├── 盐（wk）
    ├── 油（wv）
    └── ...

  计算节点 = 烹饪步骤（必须动手做才有结果）
    ├── 步骤1：打蛋（把鸡蛋作为输入 src[0]）       → 得到蛋液
    ├── 步骤2：加盐（把蛋液和盐作为输入）            → 得到调味蛋液
    ├── 步骤3：下锅（把调味蛋液和油作为输入）         → 得到煎蛋
    └── ...

  食材不会变（同一批食材可以做好几道菜）
  步骤每次重新做（每次做菜都要重新打蛋、加盐、下锅）
```

### 内存中的关系

```
             leafs[]                        nodes[]
        ┌──────────────┐            ┌──────────────────────┐
  [0]   │ wq           │◄───────────│ norm_out (RMS_NORM)   │
  [1]   │ wk           │◄─────┐     │        src[0] = inpL  │
  [2]   │ wv           │◄──┐  │     │                       │
  [3]   │ wo           │    │  │     │ Qcur (MUL_MAT)        │
  [4]   │ ffn_gate     │    │  ├─────│        src[0] = wq    │ ← 引用叶子
  [5]   │ ffn_up       │    │  │     │        src[1] = norm_out│ ← 引用上一步
  [6]   │ ffn_down     │    │  │     │                       │
  [7]   │ attn_norm    │    │  │     │ Kcur (MUL_MAT)        │
  ...   │ ...          │    ├───│─────│        src[0] = wk    │
        └──────────────┘    │  │     │        src[1] = norm_out│
                            │  │     │                       │
                            │  │     │ Vcur (MUL_MAT)        │
                            │─────────        src[0] = wv    │
                            │  │     │        src[1] = norm_out│
                            │  │     └──────────────────────┘
                            │  │
                            │  └─── 计算节点的 src[] 可以指向前面的计算节点
                            └────── 计算节点的 src[] 可以指向叶子节点

  叶子节点之间没有引用关系（src[] 全是 NULL）
  计算节点之间形成链条（src[] 指向其他节点）
  整个图 = 叶子（锚点）+ 计算节点（链条）
```

### 对比总结

| 特征 | 叶子节点（如 wq） | 计算节点（如 Qcur） |
|------|-------------------|---------------------|
| `op` | `GGML_OP_NONE` | `GGML_OP_MUL_MAT` |
| `src[]` | 全部 `NULL` | `src[0]=wq, src[1]=norm_out` |
| `data` | 指向 mmap/缓冲区的权重数据 | 执行前为 NULL，执行后指向计算结果 |
| `type` | 量化类型（如 Q4_0） | `GGML_TYPE_F32`（计算结果） |
| `name` | `"blk.0.attn_q.weight"` | 自动生成 `"node_N"` 或自定义 |
| 生命周期 | 模型加载到卸载 | 每次推理重新创建 |
| 在图中位置 | `leafs[]` | `nodes[]` |
| 是否可变 | 不变（除非 LoRA 注入） | 每次推理都重新计算 |
| 是否被执行 | 跳过（op=NONE） | 按拓扑序执行 |
| 类比 | 食材（现成的） | 烹饪步骤（要动手做） |

---

## 7. 输入张量的特殊处理

推理时的动态输入（token embedding、位置编码等）也通过 `ggml_tensor` 表达：

```
动态输入张量：
  inpL     (token embedding)   → flags 包含 GGML_TENSOR_FLAG_INPUT
  inp_pos  (位置 ID)           → flags 包含 GGML_TENSOR_FLAG_INPUT
  inp_attn (注意力掩码)        → flags 包含 GGML_TENSOR_FLAG_INPUT
```

这些张量在每次推理前通过 `set_inputs()` 填入实际数据。图结构本身不变（权重和算子链不变），只是输入张量的 `data` 指向新的数据。这就是**图复用**的基础。

---

## 8. 输出张量的对应

```
输出张量：
  logits → 最终的 MUL_MAT 节点（output 矩阵 × 最后一层输出）
         → flags 包含 GGML_TENSOR_FLAG_OUTPUT
         → 执行后 data 包含 logits 值
         → 采样器从中选择下一个 token
```

---

## 9. 总结：三层对应关系

```
GGUF 文件层                    模型结构层                      计算图层
═══════════                    ═══════════                    ══════════

"token_embd.weight"      →    model.tok_embd           →    图的输入叶子节点
"output_norm.weight"      →    model.output_norm        →    build_norm() 的权重参数
"output.weight"           →    model.output             →    最终 MUL_MAT 的 src[0]
"blk.{i}.attn_q.weight"  →    model.layers[i].wq       →    build_lora_mm() 的权重参数
"blk.{i}.ffn_gate.weight" →    model.layers[i].ffn_gate →    build_ffn() 的门控权重
...                        →    ...                      →    ...

名称 = GGUF 键名              指针 = 结构体字段             src[0] = 计算节点引用
data = mmap 到文件             data = 同一指针               后端从 data 读取权重
```

三层各司其职：
- **GGUF 文件层**：持久化存储，张量以量化格式保存在磁盘上
- **模型结构层**：运行时组织，将张量按模型架构分组到 `llama_model` / `llama_layer` 字段中
- **计算图层**：推理执行，权重作为叶子节点被计算节点的 `src[]` 引用，参与前向传播

---

## 10. 模型加载的完整逻辑

### 10.1 总览：加载的五步流程

```
用户调用：llama_model_load_from_file("model.gguf", params)
  │
  ├── 第 1 步：打开 GGUF 文件，读取元数据
  │     └─ 解析文件头、KV 元数据、张量描述表
  │
  ├── 第 2 步：检测模型架构（load_arch）
  │     └─ 从元数据读取 "general.architecture" → 确定是 LLaMA / Qwen2 / Gemma 等
  │
  ├── 第 3 步：加载超参数（load_hparams）
  │     └─ 读取层数、隐藏维度、注意力头数、上下文长度等
  │
  ├── 第 4 步：加载词表（load_vocab）
  │     └─ 读取 token 列表、分词规则、特殊 token
  │
  └── 第 5 步：加载张量（load_tensors）
        ├─ 按架构分配：创建 ggml_tensor 并关联到 model 字段
        ├─ 设备分配：决定每个张量放 CPU 还是 GPU
        └─ 数据加载：通过 mmap 或 read 将权重数据映射到内存
```

### 10.2 第 1 步：读取 GGUF 文件

**GGUF 文件结构**：

```
┌──────────────────────────┐
│ 文件头（Header）          │  magic number + version
├──────────────────────────┤
│ KV 元数据区               │  键值对：架构名、层数、词表大小...
│   "general.architecture" = "llama"
│   "llama.context_length" = 4096
│   "llama.block_count"    = 32
│   "llama.embedding_length" = 4096
│   ...（几百个键值对）
├──────────────────────────┤
│ 张量描述表                │  每个张量的：名称、类型、形状、偏移
│   "token_embd.weight"    F32  [4096, 32000]  offset=0x...
│   "blk.0.attn_q.weight"  Q4_0 [4096, 4096]   offset=0x...
│   ...
├──────────────────────────┤
│ 张量数据区                │  实际的权重数据（量化后的二进制）
│   ... 几 GB 到几百 GB
└──────────────────────────┘
```

**加载过程**：

```c
// 1. 打开文件
gguf_context * ctx = gguf_init_from_file("model.gguf", params);

// 2. 读取元数据（不读权重数据）
int n_kv = gguf_get_n_kv(ctx);          // KV 对数量
int n_tensors = gguf_get_n_tensors(ctx); // 张量数量

// 3. 遍历 KV，读取超参数
for (int i = 0; i < n_kv; i++) {
    const char * key = gguf_get_key(ctx, i);
    // 按需读取各种参数
}

// 4. 遍历张量描述表（只读名称/形状/类型，不读数据）
for (int i = 0; i < n_tensors; i++) {
    const char * name = gguf_get_tensor_name(ctx, i);
    ggml_type type    = gguf_get_tensor_type(ctx, i);
    size_t offset     = gguf_get_tensor_offset(ctx, i);
}
```

**关键设计**：元数据和张量描述表在文件头部，体积小，可以快速读取。张量数据区在文件尾部，体积大，通过 mmap 按需映射（不一次性读入内存）。

### 10.3 第 2 步：检测模型架构

```cpp
void llama_model::load_arch(llama_model_loader & ml) {
    // 从 GGUF 元数据读取架构名称
    arch_name = ml.get_key_string("general.architecture");
    // 例如："llama", "qwen2", "gemma", "mistral", ...

    // 映射到内部枚举
    arch = llm_arch_from_string(arch_name);
    // 例如："llama" → LLM_ARCH_LLAMA

    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + arch_name + "'");
    }
}
```

架构决定了后续加载哪些张量、使用什么计算图。llama.cpp 支持 140+ 种架构。

### 10.4 第 3 步：加载超参数

```cpp
void llama_model::load_hparams(llama_model_loader & ml) {
    // 读取核心超参数
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab);       // 词表大小（如 32000）
    ml.get_key(LLM_KV_CONTEXT_LENGTH,      hparams.n_ctx_train);   // 训练时的上下文长度
    ml.get_key(LLM_KV_BLOCK_COUNT,         hparams.n_layer);       // Transformer 层数
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,    hparams.n_embd);        // 隐藏维度

    // 注意力相关
    ml.get_key(LLM_KV_ATTENTION_HEAD_COUNT,        hparams.n_head);     // 注意力头数
    ml.get_key(LLM_KV_ATTENTION_HEAD_COUNT_KV,     hparams.n_head_kv);  // KV 头数（GQA）
    ml.get_key(LLM_KV_ATTENTION_HEAD_DIM,          hparams.n_embd_head);// 每头维度

    // FFN 相关
    ml.get_key(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff);         // FFN 中间层维度

    // RoPE 相关
    ml.get_key(LLM_KV_ROPE_FREQ_BASE,       hparams.rope_freq_base);  // RoPE 基础频率
    ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT,  hparams.n_rot);          // RoPE 维度数

    // 归一化
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // ... 根据架构还有更多参数
}
```

### 10.5 第 4 步：加载词表

```
词表加载内容：
  ├── token 字符串列表：["<s>", "</s>", "<unk>", "a", "b", "c", ...]
  ├── token 分数（用于分词优先级）
  ├── token 类型（普通/控制/未知/填充）
  ├── 特殊 token 定义（BOS/EOS/PAD/UNK）
  ├── 分词器类型（SPM/BPE/WPM/UGM）
  └── 预分词模式（定义如何拆分文本为子串）
```

### 10.6 第 5 步：加载张量（核心）

这是最复杂的步骤，分为三个阶段。

#### 阶段 A：按架构创建张量并关联到模型字段

```cpp
bool llama_model::load_tensors(llama_model_loader & ml) {
    // 创建张量命名工具
    const auto tn = LLM_TN(arch);  // 根据 architecture 生成正确的命名前缀

    // 根据架构走不同的 switch 分支
    switch (arch) {
    case LLM_ARCH_LLAMA: {
        // ── 全局张量 ──
        model.tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab});
        model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});

        // ── 逐层张量 ──
        for (int il = 0; il < n_layer; il++) {
            llama_layer & layer = layers[il];

            // 注意力权重
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), {n_embd});
            layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", il), {n_embd, n_embd});
            layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", il), {n_embd, n_embd_kv});
            layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", il), {n_embd, n_embd_kv});
            layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_O,    "weight", il), {n_embd, n_embd});

            // FFN 权重
            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), {n_embd});
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd, n_ff});
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd, n_ff});
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {n_ff, n_embd});
        }
        break;
    }

    case LLM_ARCH_QWEN2: {
        // Qwen2 的张量布局（与 LLaMA 类似但有差异）
        // ...
        break;
    }

    // ... 140+ 种架构各有自己的分支
    }
}
```

**`create_tensor()` 做了什么**：

```
create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", 0), {4096, 4096})
  │
  ├── 1. tn() 生成 GGUF 名称："blk.0.attn_q.weight"
  │
  ├── 2. 在 GGUF 张量描述表中查找这个名称
  │     └─ gguf_find_tensor(ctx, "blk.0.attn_q.weight")
  │     └─ 找到：类型=Q4_0，形状=[4096,4096]，偏移=0x1A3F00
  │
  ├── 3. 创建 ggml_tensor 结构体
  │     ├─ type = Q4_0（来自 GGUF）
  │     ├─ ne = {4096, 4096, 1, 1}
  │     ├─ op = GGML_OP_NONE（叶子节点）
  │     ├─ name = "blk.0.attn_q.weight"
  │     └─ data = NULL（此时还没映射数据）
  │
  └── 4. 记录映射关系
        └─ weights_map["blk.0.attn_q.weight"] = {文件偏移, 大小, 类型}
```

#### 阶段 B：设备分配（CPU / GPU）

```
加载完所有张量的描述后，决定每个张量放在哪个设备上：

split_mode = NONE（默认）：所有张量放 CPU
split_mode = LAYER：按层分配到不同 GPU
  ├── GPU 0：layers[0..15]
  ├── GPU 1：layers[16..31]
  └── CPU：  tok_embd, output_norm, output

split_mode = ROW：同一层内按行切分到不同 GPU
  ├── GPU 0：wq 的前半部分行
  └── GPU 1：wq 的后半部分行

分配逻辑考虑：
  ├── 各设备的可用显存
  ├── 用户指定的 tensor_split 参数
  └── 张量大小估算
```

#### 阶段 C：数据加载（mmap 或 read）

```
方式 1：mmap（内存映射，默认）
  ├── 将文件区域直接映射到进程地址空间
  ├── data 指针指向 mmap 返回的地址
  ├── 操作系统按需将数据从磁盘读入物理内存（缺页中断）
  └── 优点：启动快，内存按需使用

方式 2：read（直接读取）
  ├── 将权重数据从文件读入已分配的缓冲区
  ├── data 指针指向缓冲区地址
  └── 用于不支持 mmap 的场景

数据加载到 GPU：
  ├── CPU 上先 mmap 获取数据
  ├── 通过 ggml_backend_cuda 分配 GPU 显存
  ├── 将数据从 CPU 拷贝到 GPU
  └── 张量的 buffer 字段更新为 GPU buffer
```

### 10.7 完整加载流程图（从文件到可推理状态）

```
"model.gguf"
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ gguf_init_from_file()                                    │
│   读文件头 + KV 元数据 + 张量描述表（不读数据区）          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ load_arch()                                              │
│   "general.architecture" = "llama"                       │
│   → arch = LLM_ARCH_LLAMA                               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ load_hparams()                                           │
│   n_vocab=32000, n_embd=4096, n_layer=32, n_head=32, ...│
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ load_vocab()                                             │
│   32000 个 token + 分词规则 + 特殊 token                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ load_tensors()                                           │
│                                                          │
│  A. 创建张量 + 关联字段                                   │
│     for il in 0..31:                                     │
│       layers[il].wq   = create_tensor("blk.{il}.attn_q") │
│       layers[il].wk   = create_tensor("blk.{il}.attn_k") │
│       layers[il].wv   = create_tensor("blk.{il}.attn_v") │
│       layers[il].wo   = create_tensor("blk.{il}.attn_o") │
│       layers[il].ffn_gate = create_tensor(...)           │
│       layers[il].ffn_up   = create_tensor(...)           │
│       layers[il].ffn_down = create_tensor(...)           │
│       ...                                                │
│                                                          │
│  B. 设备分配                                              │
│     决定每个张量放 CPU 还是 GPU                            │
│                                                          │
│  C. 数据加载                                              │
│     mmap 权重数据 → 拷贝到 GPU 显存（如果需要）            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 模型就绪                                                 │
│                                                          │
│  model.tok_embd    → data 指向 mmap/GPU 中的嵌入权重      │
│  model.layers[0].wq → data 指向 mmap/GPU 中的 Q 权重      │
│  model.layers[0].wk → data 指向 mmap/GPU 中的 K 权重      │
│  ...                                                     │
│  所有权重张量 op=GGML_OP_NONE，src[]=NULL                  │
│  → 可以开始推理了                                         │
└─────────────────────────────────────────────────────────┘
```

### 10.8 分片模型（Split Model）的加载

大模型可能被拆分为多个文件（如 `model-00001-of-00008.gguf`）：

```
加载分片模型：

1. llama_model_load_from_splits({"model-00001.gguf", ..., "model-00008.gguf"})

2. 从第一个文件读取完整元数据（所有分片的元数据相同）

3. 每个分片只包含部分张量数据：
   分片 1：token_embd, blk.0.*, blk.1.*, ...
   分片 2：blk.5.*, blk.6.*, ...
   ...

4. create_tensor() 时在所有分片中查找张量名称
   └─ 找到后记录 (文件索引, 偏移)

5. 数据加载时从正确的分片文件 mmap
```

### 10.9 错误处理

```
加载过程中可能遇到的错误及处理方式：

  文件不存在/格式错误 → gguf_init_from_file 返回 NULL → 抛异常
  未知架构名称         → arch = LLM_ARCH_UNKNOWN → throw "unknown model architecture"
  缺少必需的超参数     → ml.get_key() 找不到 → throw "key not found"
  张量名称不匹配       → create_tensor() 在 GGUF 中找不到 → throw "tensor not found"
  张量形状不匹配       → 期望 [4096, 4096] 实际 [2048, 4096] → throw "wrong tensor shape"
  显存不足             → GPU 分配失败 → fallback 到 CPU 或报错
  文件损坏             → mmap 后校验失败 → throw "invalid tensor data"
```

---

## 11. 权重数据加载的详细代码逻辑

### 11.1 三条加载路径总览

```
                        GGUF 文件中的权重数据
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
            路径 A          路径 B         路径 C
           mmap            read           read
          (CPU)           (CPU)       (CPU → GPU)
            │               │              │
            ▼               ▼              ▼
     tensor->data      tensor->data    GPU 显存
     直接指向           指向 CPU       tensor->data
     mmap 区域          缓冲区         指向 GPU 缓冲区
```

### 11.2 前置知识：llama_model_loader 的关键字段

```cpp
struct llama_model_loader {
    bool use_mmap;             // 是否使用 mmap（默认 true）

    llama_files files;         // FILE* 句柄列表（每个分片一个）
    llama_mmaps mappings;      // mmap 映射对象列表

    // 核心数据结构：记录每个张量在文件中的位置
    std::map<std::string, llama_tensor_weight> weights_map;
};

struct llama_tensor_weight {
    uint16_t idx;              // 张量在第几个分片文件中
    size_t   offs;             // 张量数据在文件中的字节偏移
    ggml_tensor * tensor;      // 指向对应的 ggml_tensor
};
```

### 11.3 阶段 A：create_tensor — 只记录位置，不读数据

```cpp
// llama-model-loader.cpp
ggml_tensor * llama_model_loader::create_tensor(tn, ne, flags) {

    // 1. tn 生成 GGUF 名称，如 "blk.0.attn_q.weight"
    std::string name = tn.str();

    // 2. 在 GGUF 张量描述表中查找
    int tensor_idx = gguf_find_tensor(metadata, name.c_str());

    // 3. 创建 ggml_tensor（此时 data = NULL）
    ggml_tensor * tensor = ggml_new_tensor(ctx, type, n_dims, ne);
    ggml_set_name(tensor, name.c_str());
    // tensor->data = NULL
    // tensor->buffer = NULL
    // tensor->op = GGML_OP_NONE

    // 4. 记录：这个张量的数据在文件中的哪个位置
    //    offs = GGUF 数据区起始偏移 + 张量在数据区内的偏移
    weights_map[name] = {
        .idx    = file_index,               // 第几个分片文件
        .offs   = data_offset + tensor_offset,  // 文件中的绝对偏移
        .tensor = tensor
    };

    return tensor;
    // 注意：此时 tensor->data 仍然是 NULL！
    //       实际的数据加载在后面的 load_all_data() 中完成
}
```

**通俗理解**：`create_tensor` 就像在餐厅点菜——你告诉服务员你要什么菜（张量名称），服务员记下菜单（weights_map），但菜还没上桌（data 还是 NULL）。

### 11.4 阶段 B：init_mappings — 建立 mmap 映射

```cpp
void llama_model_loader::init_mappings(bool prefetch, llama_mlocks * mlock_mmaps) {
    if (use_mmap) {
        for (const auto & file : files) {
            // 为每个分片文件创建 mmap 映射
            auto mapping = std::make_unique<llama_mmap>(
                file.get(),
                prefetch ? -1 : 0,   // -1 表示预取全部，0 表示按需
                is_numa
            );
            mappings.emplace_back(std::move(mapping));
        }
    }
}
```

`llama_mmap` 内部的平台实现：

```
Linux:
  fd = open("model.gguf", O_RDONLY);
  addr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
  // addr 现在指向整个文件内容的虚拟内存地址
  // 操作系统负责：当你访问 addr[offset] 时，自动从磁盘读入对应的页

Windows:
  hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
  addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
  // 原理相同：addr 指向文件的虚拟内存映射
```

**mmap 之后的内存状态**：

```
虚拟地址空间：
  addr ──→ [GGUF文件头|KV元数据|张量描述表|token_embd数据|blk.0.attn_q数据|...]
           ↑                                                   ↑
           offset=0                              offset=0x1A3F00

此时还没有任何物理内存被占用（除非 prefetch=true）
操作系统会在你真正访问某个地址时，才从磁盘读入对应的数据页（缺页中断）
```

### 11.5 阶段 C：load_all_data — 把数据关联到张量

这是最关键的函数，遍历所有张量，根据路径执行不同的加载逻辑。

```cpp
void llama_model_loader::load_all_data(...) {
    // 遍历 ggml_context 中的所有张量
    for (ggml_tensor * cur = ggml_get_first_tensor(ctx);
         cur != NULL;
         cur = ggml_get_next_tensor(ctx, cur)) {

        // 1. 查找这个张量在 weights_map 中的记录
        const auto * weight = get_weight(ggml_get_name(cur));
        if (weight == nullptr) continue;  // 非权重张量跳过

        size_t n_size = ggml_nbytes(cur);  // 这个张量占多少字节

        if (use_mmap) {
            // ═══════════════════════════════════════
            // 路径 A：mmap 加载（CPU 上的权重）
            // ═══════════════════════════════════════

            // 2a. 计算 mmap 中这个张量的数据地址
            const auto & mapping = mappings.at(weight->idx);
            uint8_t * data = (uint8_t *)mapping->addr() + weight->offs;
            //             ↑ mmap基地址            ↑ 张量在文件中的偏移

            // data 现在指向 mmap 区域中这个张量的数据
            // 但 cur->data 还是 NULL，需要分配一个"包装"buffer

            // 3a. 分配 buffer 并设置 data 指针
            ggml_backend_tensor_alloc(buf_mmap, cur, data);
            // 内部做两件事：
            //   cur->buffer = buf_mmap;    ← 标记数据在 CPU mmap buffer 中
            //   cur->data   = data;        ← 指向 mmap 中的实际数据

        } else {
            // ═══════════════════════════════════════
            // 路径 B/C：read 加载（需要显式读文件）
            // ═══════════════════════════════════════

            const auto & file = files.at(weight->idx);

            if (ggml_backend_buffer_is_host(cur->buffer)) {
                // ─── 路径 B：CPU 缓冲区 ───

                // 直接读到 tensor->data（之前已经分配好了 CPU 缓冲区）
                file->seek(weight->offs, SEEK_SET);   // 定位到文件中张量数据的位置
                file->read_raw(cur->data, n_size);     // 读 n_size 字节到内存
                // 现在 cur->data 包含了权重数据

            } else {
                // ─── 路径 C：GPU 显存 ───

                // 不能直接读文件到 GPU，需要两步：
                // 步骤 1：读文件到 CPU 临时缓冲区（staging buffer）
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(staging_buffer, n_size);

                // 步骤 2：从 CPU 缓冲区异步拷贝到 GPU
                ggml_backend_tensor_set_async(backend_gpu, cur, staging_buffer, 0, n_size);
                // 内部调用 cudaMemcpyAsync 或类似 API
                // 将数据从 CPU 内存拷贝到 GPU 显存
                // cur->data 现在指向 GPU 显存中的地址
            }
        }
    }
}
```

### 11.6 用具体例子走一遍

假设加载 LLaMA-7B 模型，处理 `blk.0.attn_q.weight` 这个张量：

```
张量信息（来自 GGUF 张量描述表）：
  名称 = "blk.0.attn_q.weight"
  类型 = Q4_0（4-bit 量化）
  形状 = [4096, 4096]（4096 x 4096 矩阵）
  文件偏移 = 0x1A3F00（相对于文件开头）
  数据大小 = 4096 * 4096 * 0.5 bytes ≈ 8MB（Q4_0 约每权重 0.5 字节）
```

#### 路径 A：mmap（默认，最快）

```
步骤 1：create_tensor 阶段
  weights_map["blk.0.attn_q.weight"] = {
    idx = 0,                    // 在第 0 个文件中
    offs = data_start + 0x1A3F00,  // 绝对偏移
    tensor = <刚创建的 ggml_tensor>
  }
  tensor->data = NULL  ← 还没有数据

步骤 2：init_mappings 阶段
  mappings[0] = mmap(model.gguf)
  // 整个文件被映射到虚拟地址空间
  // 假设映射基地址 = 0x7F0000000000

步骤 3：load_all_data 阶段
  // 计算 mmap 中这个张量的地址
  data = mappings[0].addr() + weight->offs
       = 0x7F0000000000 + 0x1A3F00
       = 0x7F00001A3F00

  // 分配 buffer 并设置指针
  ggml_backend_tensor_alloc(buf_mmap, tensor, data)
    → tensor->buffer = buf_mmap
    → tensor->data = 0x7F00001A3F00

结果：
  tensor->data 直接指向 mmap 区域中的权重数据
  没有任何数据拷贝！
  操作系统在你真正访问这些数据时才从磁盘读入（缺页中断）

内存布局：
  mmap 区域：[...其他数据...| blk.0.attn_q.weight 的 8MB 数据 | ...其他数据...]
                            ↑
                     tensor->data 指向这里
```

#### 路径 B：read 到 CPU（use_mmap=false，CPU 模式）

```
步骤 1-2：同上（但跳过 mmap 创建）

步骤 3：load_all_data 阶段
  // 先分配 CPU 缓冲区（在之前的 alloc 阶段已完成）
  tensor->buffer = cpu_buffer;
  tensor->data = malloc(8MB);  // 实际是 buffer 内的偏移

  // 从文件读取数据到缓冲区
  file->seek(0x1A3F00, SEEK_SET);
  file->read_raw(tensor->data, 8MB);
  // 8MB 数据从磁盘 → CPU 内存

结果：
  tensor->data 指向 CPU 缓冲区中的 8MB 权重数据
  数据已经完全在物理内存中
```

#### 路径 C：read + 拷贝到 GPU

```
步骤 1-2：同上

步骤 3：load_all_data 阶段
  // 先分配 GPU 缓冲区
  tensor->buffer = cuda_buffer;
  tensor->data = cudaMalloc(8MB);  // GPU 显存中的地址

  // 从文件读到 CPU 临时缓冲区
  file->seek(0x1A3F00, SEEK_SET);
  file->read_raw(staging_buffer, 8MB);  // 磁盘 → CPU 内存

  // 从 CPU 异步拷贝到 GPU
  cudaMemcpyAsync(tensor->data, staging_buffer, 8MB, ...);
  // CPU 内存 → GPU 显存

结果：
  tensor->data 指向 GPU 显存中的权重数据
  数据流：磁盘 → CPU 内存 → GPU 显存（两次拷贝）
```

### 11.7 加载后的张量状态对比

```
                    mmap 路径           read(CPU) 路径       read(GPU) 路径
                    ─────────           ────────────         ──────────────
tensor->buffer     CPU mmap buffer      CPU buffer           CUDA buffer
tensor->data       指向 mmap 地址       指向 malloc 地址     指向 GPU 显存地址
数据拷贝次数        0 次                1 次（文件→内存）     2 次（文件→CPU→GPU）
加载速度            最快（按需加载）     中等                  最慢
内存占用            虚拟内存（按需）     物理内存（立即）      物理内存 + 显存
```

### 11.8 完整时间线

```
时间 ─────────────────────────────────────────────────────────────────────►

gguf_init_from_file()
  │ 读文件头、KV、张量描述表（几 MB，瞬间完成）
  │
  ▼
load_arch() / load_hparams() / load_vocab()
  │ 从 KV 元数据中读取（纯内存操作）
  │
  ▼
load_tensors() 开始
  │
  ├─ create_tensor() × N 次
  │    │ 为每个权重创建 ggml_tensor（data=NULL）
  │    │ 在 weights_map 中记录文件偏移
  │    │ ← 此时还没有任何磁盘 I/O
  │
  ├─ 分配缓冲区（ggml_backend_alloc_ctx_tensors）
  │    │ 为每个张量分配 buffer（CPU/GPU）
  │    │ tensor->buffer 被设置
  │    │ tensor->data 被设置为 buffer 内的地址（但还没有实际数据）
  │
  ├─ init_mappings()（如果 use_mmap=true）
  │    │ mmap 整个文件（瞬间完成，不读数据）
  │    │ mappings[i] = mmap(files[i])
  │
  └─ load_all_data()
       │ 遍历所有张量，执行实际的数据加载
       │
       │ mmap 路径：tensor->data = mmap_addr + offset（瞬间）
       │ read 路径：file->read(tensor->data, size)（读磁盘）
       │ GPU  路径：read + cudaMemcpy（读磁盘 + 传到 GPU）
       │
       ▼
  所有张量就绪
  tensor->data 指向可用的权重数据
  tensor->buffer 标识数据所在的后端
  → 模型加载完成，可以开始推理
```

---

## 12. 推理全流程：从用户输入到模型输出

### 12.1 一句话总结

```
用户文本 → 分词（tokenize）→ 查嵌入表 → Transformer 层计算 → 输出 logits → 采样选词 → 反分词（detokenize）→ 输出文本
```

### 12.2 用具体例子走一遍

假设用户输入 "你好"，模型回复 "世界"。

#### 第 1 步：分词（tokenize）

```c
// API: llama_tokenize(vocab, text, ..., tokens, ..., true, true)
// 输入："你好"
// 输出：token_ids = [1024, 3847]

// 背后发生的事：
//   1. 预处理：按预分词规则拆分文本
//   2. 查词表：用 BPE/SPM 算法将子串映射为 token ID
//   3. 添加特殊 token：如 BOS（句子开头）→ [1, 1024, 3847]
```

#### 第 2 步：构建 batch 并提交 decode

```c
// 将 token ID 包装成 batch
llama_batch batch = llama_batch_get_one(tokens, n_tokens);
// batch.token  = [1, 1024, 3847]  ← 3 个 token ID
// batch.n_tokens = 3
// batch.pos    = NULL（自动管理位置）
// batch.logits = NULL（只有最后一个 token 需要输出 logits）

// 调用 decode 执行推理
llama_decode(ctx, batch);
```

#### 第 3 步：decode 内部流程（核心）

```cpp
// src/llama-context.cpp: llama_context::decode()

int llama_context::decode(const llama_batch & batch_inp) {

    // 3a. 初始化批次分配器
    balloc->init(batch_inp, vocab, memory, n_embd, n_seq_max, output_all);

    // 3b. 初始化内存上下文（管理 KV 缓存）
    mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);

    // 3c. 将批次拆分为微批次（ubatch），逐个处理
    do {
        const auto & ubatch = mctx->get_ubatch();

        // 处理一个微批次
        auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);

        // 提取 logits
        auto * t_logits = res->get_logits();
        // 从后端缓冲区拷贝 logits 到 CPU
        if (logits.data && t_logits && n_outputs > 0) {
            ggml_backend_tensor_get(t_logits, logits.data, ...);
        }
    } while (...);
}
```

#### 第 4 步：process_ubatch 内部（图构建与执行）

```cpp
// src/llama-context.cpp: llama_context::process_ubatch()

llm_graph_result * process_ubatch(ubatch, gtype, mctx, ret) {

    // 4a. 构建或复用计算图
    if (can_reuse(gparams)) {
        n_reused++;  // 复用上次的图
    } else {
        gf = model.build_graph(gparams);           // 构建新图
        ggml_backend_sched_alloc_graph(sched, gf);  // 分配后端缓冲区
    }

    // 4b. 填入输入数据
    res->set_inputs(&ubatch);
    // 内部做的事：
    //   将 ubatch.token (token IDs) 写入 inp_tokens 张量
    //   将 ubatch.pos   (位置信息)  写入 inp_pos 张量
    //   构建/更新注意力掩码写入 inp_attn 张量

    // 4c. 执行计算图
    graph_compute(gf, ubatch.n_tokens > 1);
    // 内部：ggml_backend_sched_graph_compute_async(sched, gf)
    // 按拓扑序执行所有计算节点，结果写入各节点的 data

    return res;
}
```

#### 第 5 步：set_inputs 做了什么（token ID → 嵌入向量）

```cpp
// src/llama-graph.cpp: llm_graph_input_embd::set_input()

void set_input(const llama_ubatch * ubatch) {
    // 将 token IDs 拷贝到图的输入张量
    ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens * sizeof(int32_t));
    // tokens 张量现在包含 [1, 1024, 3847]
}

// 计算图中，inpL 张量由 token embedding 操作生成：
// inpL = tok_embd[tokens]  ← 查嵌入表
// 即：对于每个 token ID，从 tok_embd 矩阵中取出对应行
// tok_embd 形状 [n_embd, n_vocab] = [4096, 32000]
// inpL = tok_embd[:, tokens] → 形状 [4096, 3]
```

#### 第 6 步：Transformer 层计算（回顾第 4 节）

```
inpL [4096 x 3]
  │
  │  以下对 inpL 的每一列（每个 token）执行相同的操作
  │
  ├── Layer 0: RMS Norm → Q/K/V 投影 → RoPE → 注意力 → 残差 → FFN → 残差
  ├── Layer 1: 同上
  ├── ...
  └── Layer 31: 同上
       │
       ▼
  output RMS Norm → output 投影
       │
       ▼
  logits [32000 x 3]  ← 3 个位置上，每个位置 32000 个词的得分
       │
       └── 只取最后一个位置的 logits（因为只有最后一个 token 需要预测下一个词）
           logits_last = logits[:, 2]  形状 [32000]
```

#### 第 7 步：采样（从 logits 选出下一个 token）

```c
// API: llama_sampler_sample(smpl, ctx, -1)
// ctx 中的 logits.data 现在包含 32000 个分数

llama_token llama_sampler_sample(smpl, ctx, idx) {
    // 获取 logits 指针
    float * logits = llama_get_logits_ith(ctx, idx);
    // logits = [0.12, -0.34, 2.15, 0.87, ...]  32000 个数

    // 采样链依次执行：
    // 1. Temperature: logits /= temperature
    //    temperature=0.8 → 让分布更尖锐（更确定）
    //
    // 2. Top-K: 只保留概率最高的 K 个
    //    top_k=40 → 从 32000 个中只留前 40 个
    //
    // 3. Top-P: 从高到低累加概率，截断到 P
    //    top_p=0.95 → 累积概率超过 95% 后的 token 被丢弃
    //
    // 4. 重复惩罚：降低已出现 token 的概率
    //
    // 5. 从剩余 token 中随机采样
    //    假设选中了 token_id = 5023

    return token_id;  // 5023
}
```

#### 第 8 步：反分词（token → 文本）

```c
// API: llama_token_to_piece(vocab, token_id, buf, ...)

// token_id = 5023 → 查词表 → "世界"
char buf[32];
llama_token_to_piece(vocab, 5023, buf, sizeof(buf), 0, true);
// buf = "世界"

// 输出给用户："世界"
```

### 12.3 自回归生成循环

上面的步骤只生成一个 token。实际生成多个 token 时，是一个**自回归循环**：

```
用户输入："你好"
                │
                ▼
  ┌── tokenize ──→ [1, 1024, 3847]
  │
  │   ┌── decode(batch=[1, 1024, 3847]) ──→ logits[3] ──→ sample ──→ token 5023 ("世") ──┐
  │   │                                                                                    │
  │   │   ┌── decode(batch=[5023]) ──→ logits[1] ──→ sample ──→ token 6102 ("界") ──┐     │
  │   │   │                                                                          │     │
  │   │   │   ┌── decode(batch=[6102]) ──→ logits[1] ──→ sample ──→ token 2 (EOS) ─┐│     │
  │   │   │   │                                                                    ││     │
  │   │   │   │   EOS token → 停止生成                                              ││     │
  │   │   │   └────────────────────────────────────────────────────────────────────┘│     │
  │   │   └─────────────────────────────────────────────────────────────────────────┘     │
  │   └──────────────────────────────────────────────────────────────────────────────────┘
  │
  │   输出：token 5023 → "世"
  │         token 6102 → "界"
  │         token 2    → <eos>（停止，不输出）
  │
  ▼
  最终输出："世界"
```

**关键细节**：
- 第一次 decode 处理所有输入 token（"提示词阶段"），可以批量处理
- 后续每次 decode 只处理 1 个新生成的 token（"生成阶段"），因为要等上一步的采样结果
- 每次 decode 的 KV 缓存会累加，之前 token 的 K/V 不需要重新计算

### 12.4 完整推理时间线

```
时间 ──────────────────────────────────────────────────────────────────────────────►

"你好"
  │
  ▼
llama_tokenize("你好")
  │ 输出：token_ids = [1, 1024, 3847]
  │
  ▼
llama_decode(batch = {token: [1, 1024, 3847], n_tokens: 3})
  │
  ├─ set_inputs()
  │    ├─ token IDs → inp_tokens 张量
  │    └─ 位置 [0,1,2] → inp_pos 张量
  │
  ├─ graph_compute()
  │    │
  │    ├─ inpL = tok_embd[:, [1, 1024, 3847]]      ← 查嵌入表
  │    │
  │    ├─ Layer 0:  norm → QKV → rope → attn → add → ffn → add
  │    ├─ Layer 1:  ...
  │    ├─ ...
  │    └─ Layer 31: ...
  │         │
  │         └─ logits = output^T × last_layer_out   ← [32000 x 3]
  │
  ├─ 提取最后一个位置的 logits
  │    └─ logits_last = logits[:, 2]                 ← [32000]
  │
  └─ 返回
  │
  ▼
llama_sampler_sample(smpl, ctx, -1)
  │ Temperature → Top-K → Top-P → 采样
  │ 输出：token_id = 5023
  │
  ▼
llama_token_to_piece(vocab, 5023)
  │ 输出："世"
  │
  ▼
llama_decode(batch = {token: [5023], n_tokens: 1})
  │ ← 注意：只送 1 个新 token，之前的 KV 缓存保留
  │ ← 新 token 的位置 = 3（接在 [0,1,2] 之后）
  │
  ├─ graph_compute()
  │    └─ ... 得到 logits[32000 x 1]
  │
  ▼
llama_sampler_sample(smpl, ctx, -1)
  │ 输出：token_id = 6102
  │
  ▼
llama_token_to_piece(vocab, 6102)
  │ 输出："界"
  │
  ▼
llama_decode(batch = {token: [6102], n_tokens: 1})
  │
  ├─ graph_compute()
  │    └─ ... 得到 logits
  │
  ▼
llama_sampler_sample(smpl, ctx, -1)
  │ 输出：token_id = 2 (EOS)
  │
  ▼
检测到 EOS → 停止生成

最终输出："世界"
```

### 12.5 数据流转总结

```
用户文本
  │
  │ [分词器：BPE/SPM]
  ▼
token IDs（整数数组）
  │
  │ [set_inputs：拷贝到 inp_tokens 张量]
  ▼
inp_tokens（ggml_tensor, I32）
  │
  │ [计算图内部：tok_embd[:, tokens] 查表]
  ▼
inpL 嵌入向量（ggml_tensor, F32）      ← [n_embd x n_tokens]
  │
  │ [32 层 Transformer：norm → attn → ffn → ...]
  ▼
hidden_states（ggml_tensor, F32）      ← [n_embd x n_tokens]
  │
  │ [output 投影：output^T × hidden_states]
  ▼
logits（ggml_tensor, F32）             ← [n_vocab x n_tokens]
  │
  │ [提取最后一个位置的 logits]
  │ [采样器：temperature → top-k → top-p → 采样]
  ▼
next_token_id（单个整数）
  │
  │ [反分词器：查词表]
  ▼
输出文本
```

每一步中 ggml_tensor 的角色：
- `inp_tokens`：INPUT 标志的输入张量，由 `set_inputs()` 填入 token IDs
- `tok_embd`：叶子节点（权重），查表时作为 src[0] 被引用
- `inpL`：计算节点（GET_ROWS 操作），src[0]=tok_embd, src[1]=inp_tokens
- 中间层各节点：计算节点，src[] 链式连接
- `logits`：OUTPUT 标志的输出张量，执行后由 `ggml_backend_tensor_get()` 拷贝到 CPU
