# ggml_tensor 结构体详解

`ggml_tensor` 是 ggml 张量库的核心结构体，它同时承担两个角色：
1. **数据容器**：存储张量的形状、数据类型和实际数据指针
2. **计算图节点**：通过 `src[]` 指针表达算子依赖，构成有向无环图（DAG）

本文档逐字段分析 `ggml_tensor` 的设计，并配合实际源码说明各字段在算子中的使用方式。

---

## 完整定义

```c
// ggml/include/ggml.h:660-692
struct ggml_tensor {
    enum ggml_type type;                          // [字段1] 数据类型

    struct ggml_backend_buffer * buffer;          // [字段2] 后端内存缓冲区

    int64_t ne[GGML_MAX_DIMS];                    // [字段3] 各维度元素数
    size_t  nb[GGML_MAX_DIMS];                    // [字段4] 各维度字节步长

    enum ggml_op op;                              // [字段5] 算子类型

    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]; // [字段6] 算子参数

    int32_t flags;                                // [字段7] 标志位

    struct ggml_tensor * src[GGML_MAX_SRC];       // [字段8] 输入张量（图的边）

    struct ggml_tensor * view_src;                // [字段9] 视图源张量
    size_t               view_offs;               // [字段10] 视图偏移

    void * data;                                  // [字段11] 数据指针

    char name[GGML_MAX_NAME];                     // [字段12] 名称（调试用）

    void * extra;                                 // [字段13] 扩展数据

    char padding[8];                              // [字段14] 对齐填充
};
```

相关常量：

```c
#define GGML_MAX_DIMS           4     // 最多 4 维
#define GGML_MAX_SRC            10    // 每个节点最多 10 个输入
#define GGML_MAX_OP_PARAMS      64    // 算子参数最多 64 字节
#define GGML_MAX_NAME           64    // 名称最多 64 字符
```

---

## 逐字段详解

### 字段 1：`type` — 数据类型

```c
enum ggml_type type;
```

指定张量中每个元素的数据类型，决定了存储大小和计算精度。

**完整类型列表**（`ggml.h:389-433`）：

| 类型 | 说明 | 每元素字节 |
|------|------|-----------|
| `GGML_TYPE_F32` | 32 位浮点 | 4 |
| `GGML_TYPE_F16` | 16 位浮点（半精度） | 2 |
| `GGML_TYPE_BF16` | BFloat16 | 2 |
| `GGML_TYPE_F64` | 64 位浮点 | 8 |
| `GGML_TYPE_I8` / `I16` / `I32` / `I64` | 整数类型 | 1/2/4/8 |
| `GGML_TYPE_Q4_0` | 4-bit 量化（类型 0） | ~0.5 |
| `GGML_TYPE_Q4_1` | 4-bit 量化（类型 1） | ~0.56 |
| `GGML_TYPE_Q5_0` / `Q5_1` | 5-bit 量化 | ~0.56 |
| `GGML_TYPE_Q8_0` | 8-bit 量化 | 1 |
| `GGML_TYPE_Q2_K` ~ `Q8_K` | K-量化（分块量化） | 0.25~1 |
| `GGML_TYPE_IQ2_XXS` ~ `IQ4_XS` | 超低比特率量化 | ~0.25~0.5 |
| `GGML_TYPE_TQ1_0` / `TQ2_0` | T-量化（1-bit/2-bit） | ~0.125/0.25 |
| `GGML_TYPE_MXFP4` / `NVFP4` | MX/NV FP4 格式 | ~0.5 |

**实际影响**：
- `type` 决定了 `nb[0]` 的值（第一维的字节步长 = `ggml_type_size(type) / ggml_blck_size(type)`）
- 后端根据 `type` 选择不同的计算核函数（如 CUDA 有针对 Q4_0 和 F16 的不同矩阵乘法核）
- 量化类型在计算前会被反量化（dequantize）为 F32 或 F16

**源码中的使用**：

```c
// ggml_mul_mat 中：结果张量固定为 F32 类型
struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
```

```c
// Flash Attention 中：mask 必须是 F16 类型
GGML_ASSERT(mask->type == GGML_TYPE_F16);
```

---

### 字段 2：`buffer` — 后端内存缓冲区

```c
struct ggml_backend_buffer * buffer;
```

指向张量数据所在的**后端内存缓冲区**。`buffer` 标识数据存储在哪个硬件设备上。

**取值含义**：
- `NULL`：张量数据尚未分配到任何后端缓冲区（数据指针 `data` 可能为 NULL）
- 非 NULL：指向一个 `ggml_backend_buffer`，表示数据已分配到特定后端（CPU 内存、GPU 显存等）

**生命周期**：

```
1. 张量创建时：buffer = NULL, data = NULL（或指向 ggml_context 的临时内存）
2. 图调度阶段：ggml_backend_sched_alloc_graph() 为张量分配后端缓冲区
3. 分配后：buffer 指向具体缓冲区，data 指向缓冲区内的实际地址
4. 执行时：后端通过 buffer 知道数据在哪个设备上，决定用哪个核函数
```

**与 `data` 的关系**：`buffer` 回答"数据在哪个设备"，`data` 回答"数据的具体地址"。`data` 通常是 `buffer` 内部某处的偏移。

---

### 字段 3：`ne[4]` — 各维度元素数（Number of Elements）

```c
int64_t ne[GGML_MAX_DIMS];  // GGML_MAX_DIMS = 4
```

描述张量在每个维度上的元素个数。ggml 张量最多 4 维。

**维度含义约定**：

```
ne[0] — 最内层维度（如嵌入维度、特征维度）
ne[1] — 第二维度（如序列长度、token 数）
ne[2] — 第三维度（如注意力头数）
ne[3] — 最外层维度（如批次数）
```

**具体例子**：

```
一个形状为 [hidden_dim=4096, n_tokens=128] 的嵌入矩阵：
  ne[0] = 4096    ← 每个token的嵌入维度
  ne[1] = 128     ← token 数量
  ne[2] = 1
  ne[3] = 1

一个形状为 [head_dim=128, n_heads=32, n_tokens=128] 的 Q 张量：
  ne[0] = 128     ← 每个头的维度
  ne[1] = 32      ← 注意力头数
  ne[2] = 128     ← token 数量
  ne[3] = 1
```

**源码中的使用**：

```c
// ggml_mul_mat 中：通过输入张量的 ne 推导输出张量的形状
// 若 a 的形状为 [ne0_a, ne1_a, ne2_a, ne3_a]
//    b 的形状为 [ne0_b, ne1_b, ne2_b, ne3_b]
// 则结果的形状为：
const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
// 即 [ne1_a, ne1_b, ne2_b, ne3_b]
struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
```

---

### 字段 4：`nb[4]` — 各维度字节步长（Stride in Bytes）

```c
size_t nb[GGML_MAX_DIMS];
```

描述沿每个维度前进一个位置需要跳过的字节数。这是张量的**内存布局**描述。

**计算规则**（定义在 `ggml.h:666-669`）：

```
nb[0] = ggml_type_size(type) / ggml_blck_size(type)  (+ 可能的 padding)
nb[1] = nb[0] * ne[0]
nb[2] = nb[1] * ne[1]
nb[3] = nb[2] * ne[2]
```

**具体例子**（F32 类型，`nb[0] = 4` 字节）：

```
形状 [3, 2, 2] 的 F32 张量：

ne[0]=3, ne[1]=2, ne[2]=2

nb[0] = 4          ← 前进一列：跳过 1 个 float = 4 字节
nb[1] = 4 * 3 = 12 ← 前进一行：跳过 3 个 float = 12 字节
nb[2] = 12 * 2 = 24 ← 前进一个矩阵：跳过 2 行 = 24 字节

内存布局（连续）：
  [a00 a01 a02 | a10 a11 a12 | b00 b01 b02 | b10 b11 b12]
   ^---第0行---^  ^---第1行---^  ^---第0行---^  ^---第1行---^
   |_____ 矩阵A _____|         |_____ 矩阵B _____|
```

**为什么需要 `nb[]` 而不是只用 `ne[]`**：

1. **非连续张量（视图/转置）**：`nb[]` 允许表达非连续内存布局。例如转置操作只需交换 `nb[1]` 和 `nb[2]`，无需复制数据
2. **量化类型**：`nb[0]` 的值不等于 `sizeof(element)`，因为量化类型以块（block）为单位存储

**非连续布局示例**：转置一个 `[3, 2]` 的矩阵

```
原始张量 A：
  nb[0] = 4, nb[1] = 12
  内存：[a00 a01 a02 | a10 a11 a12]

转置张量 A^T（通过 ggml_transpose 创建）：
  ne 变为 [2, 3]
  nb[0] = 12（= 原来的 nb[1]）  ← 前进一列 = 跳过原来的一行
  nb[1] = 4 （= 原来的 nb[0]）  ← 前进一行 = 跳过原来的一列
  内存：仍然指向同一块数据，只是解读方式变了
```

**源码中的使用**：

```c
// ggml_view_tensor 中：视图继承原始张量的 nb
for (int i = 0; i < GGML_MAX_DIMS; i++) {
    result->nb[i] = src->nb[i];
}
```

---

### 字段 5：`op` — 算子类型

```c
enum ggml_op op;
```

标识该张量是通过什么**算子**计算得到的。取值定义在 `ggml.h:473` 的 `ggml_op` 枚举中（共 80+ 种）。

**关键取值及含义**：

| 算子 | 含义 | src 使用 |
|------|------|----------|
| `GGML_OP_NONE` | 无算子（叶子节点：常量/输入/权重） | 不使用 src |
| `GGML_OP_ADD` | 逐元素加法 | src[0]=a, src[1]=b |
| `GGML_OP_MUL` | 逐元素乘法 | src[0]=a, src[1]=b |
| `GGML_OP_MUL_MAT` | 矩阵乘法 | src[0]=W, src[1]=x |
| `GGML_OP_RMS_NORM` | RMS 归一化 | src[0]=a |
| `GGML_OP_SILU` | SiLU 激活函数 | src[0]=a |
| `GGML_OP_SOFT_MAX` | Softmax | src[0]=a, src[1]=mask(可选) |
| `GGML_OP_ROPE` | 旋转位置编码 | src[0]=a, src[1]=pos, src[2]=freq(可选) |
| `GGML_OP_FLASH_ATTN_EXT` | 融合注意力 | src[0]=Q, src[1]=K, src[2]=V, src[3]=mask |
| `GGML_OP_RESHAPE` | 改变形状 | src[0]=a |
| `GGML_OP_PERMUTE` | 维度重排 | src[0]=a |
| `GGML_OP_VIEW` | 视图（共享内存） | src[0]=a |
| `GGML_OP_CPY` | 数据拷贝 | src[0]=src, src[1]=dst |

**`op` 决定了计算图如何遍历和执行**：

- `op == GGML_OP_NONE` → 叶子节点，不需要计算，加入图的 `leafs[]`
- `op != GGML_OP_NONE` → 计算节点，需要执行，加入图的 `nodes[]`

---

### 字段 6：`op_params[]` — 算子参数

```c
int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];  // 16 个 int32
```

存储算子执行所需的**额外参数**。不同的算子使用不同的参数。以 `int32_t` 数组分配以保持对齐。

**各算子的 `op_params` 使用**：

#### RMS Norm

```c
// ggml_rms_norm_impl 中：
float eps = 1e-5f;
ggml_set_op_params(result, &eps, sizeof(eps));
// op_params[0] = *(int32_t*)&eps  ← 将 float 以 int32 形式存储
```

#### RoPE（旋转位置编码）

```c
// ggml_rope_impl 中：使用 15 个 int32 参数
int32_t params[15] = {
    /*[0]  n_past  */  0,
    /*[1]  n_dims  */  n_dims,
    /*[2]  mode    */  mode,
    /*[3]  n_ctx   */  0,
    /*[4]  n_ctx_orig */ n_ctx_orig,
    /*[5]  freq_base   */ *(int32_t*)&freq_base,
    /*[6]  freq_scale  */ *(int32_t*)&freq_scale,
    /*[7]  ext_factor  */ *(int32_t*)&ext_factor,
    /*[8]  attn_factor */ *(int32_t*)&attn_factor,
    /*[9]  beta_fast   */ *(int32_t*)&beta_fast,
    /*[10] beta_slow   */ *(int32_t*)&beta_slow,
    /*[11-14] mrope_sections */
};
ggml_set_op_params(result, params, sizeof(params));
```

#### Flash Attention

```c
// ggml_flash_attn_ext 中：使用 3 个 float 参数
float params[] = { scale, max_bias, logit_softcap };
ggml_set_op_params(result, params, sizeof(params));
// op_params 存储：scale, max_bias, logit_softcap
```

#### 矩阵乘法精度

```c
// ggml_mul_mat_set_prec 中：
void ggml_mul_mat_set_prec(struct ggml_tensor * a, enum ggml_prec prec) {
    ggml_set_op_params_i32(a, 0, (int32_t)prec);
}
```

**设计理由**：将算子参数内嵌在张量结构体中，避免额外的内存分配。执行时直接从 `op_params` 读取参数即可。

---

### 字段 7：`flags` — 标志位

```c
int32_t flags;
```

用位掩码标记张量的特殊属性。

**定义的标志**（`ggml.h:638-642`）：

```c
GGML_TENSOR_FLAG_INPUT   = 1   // 张量是计算图的输入
GGML_TENSOR_FLAG_OUTPUT  = 2   // 张量是计算图的输出
GGML_TENSOR_FLAG_PARAM   = 4   // 张量包含可训练参数
GGML_TENSOR_FLAG_LOSS    = 8   // 张量定义损失（用于数值优化）
GGML_TENSOR_FLAG_COMPUTE = 16  // 张量必须被计算
```

**使用场景**：

- `GGML_TENSOR_FLAG_INPUT`：标记输入张量（如 token embedding），图调度时需要为它分配输入缓冲区
- `GGML_TENSOR_FLAG_OUTPUT`：标记输出张量（如 logits），执行后需要将结果拷贝回主机
- `GGML_TENSOR_FLAG_PARAM`：标记可训练参数，图构建时即使 `op == GGML_OP_NONE` 也被视为计算节点（而非叶子）
- `GGML_TENSOR_FLAG_COMPUTE`：在 `ggml_visit_parents_graph` 中动态设置，标记该节点需要实际执行计算

```c
// 图构建时动态设置 COMPUTE 标志
if (node->op != GGML_OP_NONE && compute) {
    node->flags |= GGML_TENSOR_FLAG_COMPUTE;
}
```

---

### 字段 9-10：`view_src` + `view_offs` — 视图机制

```c
struct ggml_tensor * view_src;   // 视图源张量（NULL 表示不是视图）
size_t               view_offs;  // 在源张量数据中的字节偏移
```

视图（View）是一种**零拷贝共享内存**机制。一个视图张量与源张量共享同一块内存，只是从不同的偏移位置或以不同的形状来解读数据。

**视图链**：视图可以嵌套，`ggml_new_tensor_impl` 会自动追踪到最底层的源张量：

```c
// ggml_new_tensor_impl 中：
if (view_src != NULL && view_src->view_src != NULL) {
    view_offs += view_src->view_offs;  // 累加偏移
    view_src   = view_src->view_src;   // 追溯到最底层
}
```

**常见视图操作**：

```c
// ggml_reshape：改变形状，共享内存
// 内部调用 ggml_view_tensor → view_src 指向原始张量

// ggml_transpose：交换维度步长，共享内存
// 修改 nb[] 的顺序，但 view_src 指向原始张量

// ggml_view_2d 等：创建子区域视图
// view_offs 指定子区域的起始偏移
```

**与 `buffer` 的关系**：视图张量的 `buffer` 最终与最底层源张量的 `buffer` 相同，因为它们共享同一块设备内存。

**与 `data` 的关系**：视图张量的 `data` = `view_src->data + view_offs`。

---

### 字段 11：`data` — 数据指针

```c
void * data;
```

指向张量实际数据的内存地址。

**生命周期变化**：

```
1. 创建时：data = NULL（或指向 ggml_context 的 arena 内存）
2. 模型加载后：data 指向 mmap 的模型文件区域（权重数据）
3. 图调度分配后：data 指向后端缓冲区内的具体地址
4. 视图张量：data = view_src->data + view_offs
```

**关键约束**：
- 对于非视图张量，`data` 由后端缓冲区管理，不应手动 `free`
- 对于视图张量，`data` 指向源张量的内存区域
- 执行算子时，后端将计算结果写入输出张量的 `data` 指向的位置

---

### 字段 12：`name` — 名称

```c
char name[GGML_MAX_NAME];  // 最多 64 字符
```

张量的名称，主要用于**调试和日志**。不影响计算逻辑。

**自动命名**：如果创建时未指定名称，图构建阶段会自动生成：

```c
// ggml_visit_parents_graph 中：
if (strlen(node->name) == 0) {
    ggml_format_name(node, "node_%d", cgraph->n_nodes);  // 计算节点
    // 或
    ggml_format_name(node, "leaf_%d", cgraph->n_leafs);  // 叶子节点
}
```

**模型权重的命名**：加载 GGUF 模型时，权重张量会被赋予 GGUF 中定义的名称，如：
- `token_embd.weight` — 嵌入层权重
- `blk.0.attn_q.weight` — 第 0 层 Q 投影权重
- `output_norm.weight` — 输出归一化权重

---

### 字段 13：`extra` — 扩展数据

```c
void * extra;
```

供后端实现存储**额外的元数据**。不同后端可以在此存储各自的信息。

**典型用途**：
- CUDA 后端：可能存储 CUDA tensor descriptor、cuBLAS handle 等
- Vulkan 后端：可能存储 Vulkan buffer 引用

这个字段是后端私有的，核心 ggml 代码不使用它。

---

### 字段 14：`padding` — 对齐填充

```c
char padding[8];
```

8 字节的填充，确保 `sizeof(ggml_tensor)` 满足对齐要求。

---

## `src[]` 的详细说明

### 核心概念

`src[]` 是 `ggml_tensor` 最关键的设计——它让每个张量"记住自己是由谁算出来的"，从而在散落的张量节点之间形成计算图。

```c
struct ggml_tensor * src[GGML_MAX_SRC];  // 最多 10 个输入
```

### 不同算子的 `src[]` 使用方式

```
                    ┌──────────────────────────────────────────┐
                    │              src[] 使用方式               │
                    ├──────────────┬───────────────────────────┤
                    │   算子       │  src[0]    src[1]   src[2] │
                    ├──────────────┼───────────────────────────┤
 GGML_OP_ADD       │ 加法         │ 左操作数   右操作数   -     │
 GGML_OP_MUL       │ 乘法         │ 左操作数   右操作数   -     │
 GGML_OP_MUL_MAT   │ 矩阵乘法     │ 权重矩阵   输入向量   -     │
 GGML_OP_RMS_NORM  │ RMS归一化    │ 输入       -          -     │
 GGML_OP_SILU      │ SiLU激活     │ 输入       -          -     │
 GGML_OP_RESHAPE   │ 改变形状     │ 输入       -          -     │
 GGML_OP_PERMUTE   │ 维度重排     │ 输入       -          -     │
 GGML_OP_ROPE      │ 旋转位置编码 │ 输入       位置向量   频率   │
 GGML_OP_CPY       │ 数据拷贝     │ 源张量     目标张量   -     │
 GGML_OP_FLASH_ATTN│ 融合注意力   │ Q          K          V      │  ← src[3]=mask
                    └──────────────┴───────────────────────────┘
```

### 实际源码中的赋值

```c
// 加法：ggml_add_impl（ggml.c:2014）
result->op     = GGML_OP_ADD;
result->src[0] = a;       // 左操作数
result->src[1] = b;       // 右操作数
// src[2..9] 保持为 NULL

// 矩阵乘法：ggml_mul_mat（ggml.c:3235）
result->op     = GGML_OP_MUL_MAT;
result->src[0] = a;       // 权重矩阵 W
result->src[1] = b;       // 输入向量 x
// 含义：result = W^T × x

// RMS归一化：ggml_rms_norm_impl（ggml.c:3112）
result->op     = GGML_OP_RMS_NORM;
result->src[0] = a;       // 输入张量
// 仅 1 个输入

// RoPE：ggml_rope_impl（ggml.c:4115）
result->op     = GGML_OP_ROPE;
result->src[0] = a;       // Q 或 K 张量
result->src[1] = b;       // 位置 ID 向量（I32 类型）
result->src[2] = c;       // 频率因子（F32，可选）

// Flash Attention：ggml_flash_attn_ext（ggml.c:5314）
result->op     = GGML_OP_FLASH_ATTN_EXT;
result->src[0] = q;       // Query
result->src[1] = k;       // Key
result->src[2] = v;       // Value
result->src[3] = mask;    // 注意力掩码（可为 NULL）
```

### 如何构成计算图

通过 `src[]` 的指针链接，从任意输出张量出发，可以递归追踪到所有参与计算的张量：

```
假设计算 result = softmax(W_out × (residual + attention(W_v × softmax(Q*K^T) × ...)))

result (SOFT_MAX)
  │
  └── src[0] ─→ mm_out (MUL_MAT)          W_out × hidden
                  ├── src[0] ─→ W_out      (权重，叶子)
                  └── src[1] ─→ add_out (ADD)    residual + attn_out
                                  ├── src[0] ─→ residual (叶子)
                                  └── src[1] ─→ attn_out
                                                  └── ... 递归展开
```

`ggml_build_forward_expand(graph, result)` 就是沿着这条链递归遍历，自动构建出完整的拓扑有序计算图。

### "指向父节点"的含义

在计算图的术语中：
- **父节点（parent/upstream）**= 数据来源方向的上游张量
- `result->src[0] = W` 意味着：W 是 result 的上游，数据从 W 流向 result

这不是调用关系上的"父"，而是**数据依赖**方向上的"上游"。

---

## 张量创建的完整流程

以 `ggml_mul_mat(ctx, W, x)` 为例，完整展示张量的创建过程：

```c
struct ggml_tensor * ggml_mul_mat(ctx, W, x) {
    // 1. 根据输入形状推导输出形状
    const int64_t ne[4] = { W->ne[1], x->ne[1], x->ne[2], x->ne[3] };

    // 2. 分配新的 ggml_tensor 结构体（从 ggml_context 的内存池中）
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    // 此时 result 的状态：
    //   type   = GGML_TYPE_F32
    //   buffer = NULL（未分配后端缓冲区）
    //   ne     = { W->ne[1], x->ne[1], x->ne[2], x->ne[3] }
    //   nb     = 自动计算（基于 ne 和 type）
    //   op     = GGML_OP_NONE（还未设置）
    //   src[]  = 全部 NULL
    //   data   = NULL 或指向 arena 内存

    // 3. 设置算子类型
    result->op = GGML_OP_MUL_MAT;

    // 4. 连接输入张量（构成图的边）
    result->src[0] = W;   // 权重矩阵
    result->src[1] = x;   // 输入向量

    return result;
    // 此时 result 是一个"悬浮"的计算节点：
    //   - 有明确的算子和输入
    //   - 但尚未加入任何计算图
    //   - 尚未被分配到任何后端
}
```

之后的流程：

```
ggml_mul_mat() 返回 result
  ↓
... 更多算子链接（add, norm, 等）...
  ↓
ggml_build_forward_expand(graph, final_output)
  → 从 final_output 递归追踪 src[]，将所有节点加入 graph
  ↓
ggml_backend_sched_alloc_graph(sched, graph)
  → 为每个节点分配后端（CPU/CUDA），分配缓冲区，设置 data 指针
  ↓
ggml_backend_sched_graph_compute_async(sched, graph)
  → 按拓扑序执行每个节点的算子，结果写入各张量的 data
```

---

> ggml_tensor 与模型的对应关系分析已独立保存至 [llama-learn-ggml-model-tensor.md](llama-learn-ggml-model-tensor.md)。
