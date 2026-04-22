# ggml 张量库计算图分析

本文档深入分析 ggml 计算图的核心数据结构、构建流程、执行机制，以及在 llama.cpp 模型引擎中的实际使用方式。

---

## 1. 核心数据结构

### 1.1 张量（ggml_tensor）

张量是计算图的基本单元，既承载数据，也通过 `src[]` 指针表达算子依赖关系，构成有向无环图（DAG）的边。

```c
// ggml/include/ggml.h
struct ggml_tensor {
    enum ggml_type type;                     // 数据类型：F32, F16, Q4_0, Q8_0 等
    struct ggml_backend_buffer * buffer;      // 所属的后端内存缓冲区

    int64_t ne[GGML_MAX_DIMS];               // 各维度元素数（最多 4 维）
    size_t  nb[GGML_MAX_DIMS];               // 各维度的字节步长（stride）

    enum ggml_op op;                         // 关联的算子类型（如 GGML_OP_MUL_MAT）
    int32_t op_params[GGML_MAX_OP_PARAMS];   // 算子参数
    int32_t flags;                           // 标志位（如 GGML_TENSOR_FLAG_COMPUTE）

    struct ggml_tensor * src[GGML_MAX_SRC];  // 输入张量 → 图的边，指向父节点

    struct ggml_tensor * view_src;           // 视图操作的源张量
    size_t view_offs;                        // 视图偏移

    void * data;                             // 数据指针
    char name[GGML_MAX_NAME];               // 张量名称（调试用）
};
```

**关键设计**：`src[]` 数组构成了图的边。每个张量通过 `src[0]`, `src[1]` 等指向其输入张量。一个 `MUL_MAT` 节点的 `src[0]` 是权重矩阵，`src[1]` 是输入向量。

**图的两种节点**：
- **叶子节点（leaf）**：`op == GGML_OP_NONE` 且无参数标志 → 常量/输入数据（如模型权重）
- **计算节点（node）**：有明确算子 → 需要执行计算

### 1.2 计算图（ggml_cgraph）

```c
// ggml/src/ggml-impl.h
struct ggml_cgraph {
    int size;                                // 最大节点/叶子容量
    int n_nodes;                             // 当前计算节点数
    int n_leafs;                             // 当前叶子节点数

    struct ggml_tensor ** nodes;             // 计算节点数组（拓扑序）
    struct ggml_tensor ** leafs;             // 叶子节点数组
    struct ggml_tensor ** grads;             // 梯度张量（用于训练）
    struct ggml_tensor ** grad_accs;         // 梯度累加器
    int32_t * use_counts;                    // 每个张量的引用计数（用于内存优化）

    struct ggml_hash_set visited_hash_set;   // 已访问节点的哈希集合
    enum ggml_cgraph_eval_order order;       // 求值顺序（左到右/右到左）
    uint64_t uid;                            // 图的唯一标识
};
```

**拓扑排序保证**：`nodes[]` 数组中的节点按依赖顺序排列——父节点一定排在子节点之前。这保证了执行时从前往后遍历即可。

**引用计数**：`use_counts[]` 记录每个张量被多少个节点引用。当一个张量的引用计数归零时，其内存可以被复用。

---

## 2. 计算图生命周期

计算图的完整生命周期分为三个阶段：**构建 → 调度 → 执行**。

```
┌─────────────────────────────────────────────────────────────────┐
│                        构建阶段                                  │
│                                                                 │
│  ggml_new_graph()  →  创建空图                                  │
│       ↓                                                         │
│  ggml_mul_mat(ctx, W, x)  →  创建张量节点，设置 src[0]=W, src[1]=x  │
│       ↓                                                         │
│  ggml_add(ctx, result, bias)  →  链式构建                       │
│       ↓                                                         │
│  ggml_build_forward_expand(graph, output)  →  递归回溯，拓扑排序  │
│       ↓                                                         │
│  得到 nodes[] 和 leafs[] 的完整计算图                            │
├─────────────────────────────────────────────────────────────────┤
│                        调度阶段                                  │
│                                                                 │
│  ggml_backend_sched_alloc_graph()                               │
│       ↓                                                         │
│  为每个节点分配后端（CPU/CUDA/Vulkan...）                         │
│       ↓                                                         │
│  按后端边界切分子图（split）                                     │
│       ↓                                                         │
│  分配内存缓冲区                                                  │
├─────────────────────────────────────────────────────────────────┤
│                        执行阶段                                  │
│                                                                 │
│  ggml_graph_compute() / ggml_backend_sched_graph_compute()      │
│       ↓                                                         │
│  按拓扑序遍历 nodes[]                                           │
│       ↓                                                         │
│  各后端并行执行算子                                              │
│       ↓                                                         │
│  结果写入输出张量的 data 指针                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 构建阶段

#### 2.1.1 创建计算图

```c
// ggml.c
struct ggml_cgraph * ggml_new_graph_custom(struct ggml_context * ctx, size_t size, bool grads) {
    // 从 ggml_context 中分配图对象
    struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, ...);
    struct ggml_cgraph * cgraph = (struct ggml_cgraph *)((char *)ctx->mem_buffer + obj->offs);

    // 分配节点、叶子、引用计数等数组
    cgraph->nodes     = /* 分配 size 个指针 */;
    cgraph->leafs     = /* 分配 size 个指针 */;
    cgraph->use_counts = /* 分配哈希表大小的 int32 */;
    cgraph->visited_hash_set = ggml_hash_set_new(hash_size);
    cgraph->n_nodes   = 0;
    cgraph->n_leafs   = 0;

    return cgraph;
}

// 简化接口
struct ggml_cgraph * ggml_new_graph(struct ggml_context * ctx) {
    return ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}
```

#### 2.1.2 创建算子节点

调用 `ggml_mul_mat()`、`ggml_add()` 等函数时，会创建一个新的 `ggml_tensor`，设置其 `op` 和 `src[]`，但**不会立即加入计算图**：

```c
// 创建矩阵乘法节点
struct ggml_tensor * ggml_mul_mat(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,     // src[0]：权重矩阵
    struct ggml_tensor  * b      // src[1]：输入向量
) {
    struct ggml_tensor * result = ggml_new_tensor(ctx, ...);
    result->op = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}
```

此时图中只有散落的张量节点，通过 `src[]` 链接但尚未被组织成有序图。

#### 2.1.3 递归构建计算图（核心）

```c
void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

static void ggml_build_forward_impl(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor,
                                     bool expand, bool compute) {
    ggml_visit_parents_graph(cgraph, tensor, compute);
}
```

核心递归函数 `ggml_visit_parents_graph` 从输出节点出发，沿着 `src[]` 反向递归访问所有祖先节点：

```c
static size_t ggml_visit_parents_graph(struct ggml_cgraph * cgraph,
                                        struct ggml_tensor * node, bool compute) {
    // 1. 标记节点需要计算
    if (node->op != GGML_OP_NONE && compute) {
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    // 2. 哈希表去重：已访问过则直接返回
    if (ggml_bitset_get(visited_hash_set.used, node_hash_pos)) {
        return node_hash_pos;
    }

    // 3. 标记为已访问
    cgraph->visited_hash_set.keys[node_hash_pos] = node;
    ggml_bitset_set(visited_hash_set.used, node_hash_pos);

    // 4. 递归访问所有 src 父节点（先处理依赖）
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        struct ggml_tensor * src = node->src[i];
        if (src) {
            ggml_visit_parents_graph(cgraph, src, compute);
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    // 5. 所有父节点处理完毕后，将当前节点加入图
    if (node->op == GGML_OP_NONE) {
        cgraph->leafs[cgraph->n_leafs++] = node;     // 叶子节点
    } else {
        cgraph->nodes[cgraph->n_nodes++] = node;     // 计算节点
    }
}
```

**关键特性**：
- **递归后序遍历**：先处理所有 `src`（父节点），再把自己加入图 → 天然产生拓扑排序
- **哈希去重**：通过 `visited_hash_set` 防止重复访问，支持 DAG（有分支汇合的图）
- **引用计数**：每访问一次某个张量，其 `use_count` 加一 → 后续可据此优化内存复用

### 2.2 调度阶段

#### 2.2.1 后端调度器

```c
// ggml-backend.cpp
struct ggml_backend_sched {
    int n_backends;
    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];           // 可用后端列表
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];  // 各后端的缓冲区类型
    ggml_gallocr_t galloc;                                       // 内存分配器

    int * node_backend_ids;    // 每个计算节点分配的后端 ID
    int * leaf_backend_ids;    // 每个叶子节点分配的后端 ID

    struct ggml_backend_sched_split * splits;  // 切分后的子图
    int n_splits;
};
```

#### 2.2.2 调度流程

```
原始计算图
  │
  ├─ 1. 为每个节点分配后端 ID
  │     └─ 根据张量所在设备和 supports_op() 决定
  │
  ├─ 2. 检测后端边界（相邻节点属于不同后端）
  │     └─ 例：nodes[0..5] 在 CUDA，nodes[6..10] 在 CPU
  │
  ├─ 3. 按后端边界切分为子图（splits）
  │     └─ 每个子图只包含同一后端的节点
  │
  ├─ 4. 在子图边界插入数据拷贝
  │     └─ 将需要跨后端的张量拷贝到目标设备
  │
  └─ 5. 分配各子图的内存缓冲区
        └─ 调用 ggml_gallocr_alloc_graph()
```

**分配策略**：
- 模型权重张量通常驻留在 GPU → 对应节点分配到 CUDA 后端
- 小型操作（如 norm、add）可能在 CPU 上更高效
- 调度器自动根据 `supports_op()` 和数据位置做决策

### 2.3 执行阶段

#### 2.3.1 入口函数

```c
// ggml-cpu.c — CPU 后端的执行入口
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    // 1. 创建或复用线程池
    if (threadpool == NULL) {
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    }

    // 2. 启动工作线程
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // 3. 主线程也参与计算
    ggml_graph_compute_thread(&threadpool->workers[0]);

    return threadpool->ec;  // 返回执行状态
}
```

#### 2.3.2 线程执行逻辑

```c
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    const struct ggml_cgraph * cgraph = tp->cgraph;

    // 按拓扑序遍历所有计算节点
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        // 跳过空操作
        if (ggml_op_is_empty(node->op)) continue;

        // 查找该算子的计算函数
        const ggml_compute_fn_t * compute_fn = ggml_get_compute_fn(node->op, node->type);

        // 执行计算
        compute_fn(node->data, node->nb, node->op_params, node->src, &params);
    }
}
```

**多线程协作**：对于大型算子（如矩阵乘法），工作被切分为多个 chunk，不同线程处理不同 chunk，通过原子计数器同步。

#### 2.3.3 多后端联合执行

```c
// ggml-backend.cpp — 跨后端执行入口
enum ggml_status ggml_backend_sched_graph_compute_async(
    ggml_backend_sched_t sched,
    struct ggml_cgraph * cgraph) {

    // 按子图顺序执行
    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        ggml_backend_t backend = sched->backends[split->backend_id];

        // 在对应后端上执行子图
        ggml_backend_graph_compute(backend, split->graph);
    }
}
```

---

## 3. 在 llama.cpp 中的实际使用

### 3.1 图构建的触发时机

在 `llama-context.cpp` 的推理流程中：

```cpp
// 简化后的 process_ubatch() 逻辑
if (graph_reuse_disable || !res->can_reuse(gparams)) {
    // 构建新的计算图
    gf = model.build_graph(gparams);
    ggml_backend_sched_alloc_graph(sched.get(), gf);
} else {
    // 复用已有图（只需更新输入张量数据）
    n_reused++;
}

// 设置输入张量
res->set_inputs(&ubatch);

// 执行计算图
ggml_backend_sched_graph_compute_async(sched.get(), gf);
```

**图复用机制**：相同的批次配置下，计算图结构不变，只需更新输入数据即可复用。这避免了每步推理都重建图的开销。

### 3.2 模型层级的图构建

`llm_graph_context` 类提供了构建各层的通用方法：

```
llm_graph_context
  ├─ build_norm()        → 归一化层（RMS Norm / Layer Norm / Group Norm）
  ├─ build_lora_mm()     → 矩阵乘法（支持 LoRA 适配器注入）
  ├─ build_qkv()         → Q/K/V 投影
  ├─ build_attn_mha()    → 多头注意力
  ├─ build_ffn()         → 前馈网络
  ├─ build_cvec()        → 控制向量注入
  └─ build_rope()        → 旋转位置编码
```

### 3.3 具体示例：LLaMA 模型的图构建

以下展示 LLaMA 模型单层 Transformer 的完整图构建过程：

```cpp
// src/models/llama.cpp（简化）
for (int il = 0; il < n_layer; ++il) {
    ggml_tensor * inpSA = inpL;  // 保存当前层输入，用于残差连接

    // ── 注意力前归一化 ──
    cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

    // ── Q/K/V 投影 ──
    auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, ...);

    // ── RoPE 旋转位置编码 ──
    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, ...);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, ...);

    // ── 存入 KV 缓存 ──
    ggml_tensor * K_pe = ...; // 从 KV 缓存读取历史 K
    ggml_tensor * V_pe = ...; // 从 KV 缓存读取历史 V

    // ── 多头注意力 ──
    cur = build_attn(..., Qcur, K_pe, V_pe, ...);
    // 内部：Q*K^T → softmax → *V

    // ── 残差连接 ──
    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);

    // ── FFN 前归一化 ──
    cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);

    // ── 前馈网络（SiLU 门控）──
    cur = build_ffn(cur,
        model.layers[il].ffn_up,    // 上投影权重
        model.layers[il].ffn_gate,  // 门控权重
        model.layers[il].ffn_down,  // 下投影权重
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    // 内部：gate(x) * SiLU(up(x)) → down(...)

    // ── 残差连接 ──
    cur = ggml_add(ctx0, cur, ffn_inp);

    inpL = cur;  // 传递到下一层
}

// 最终层归一化 + 输出投影
cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
cur = build_lora_mm(model.output, cur);
```

对应的计算图结构：

```
                    inpL (输入 token 嵌入)
                      │
            ┌─────────┤──────────┐
            │    RMS Norm         │
            │         │          │
            │    ┌────┼────┐     │
            │    Q投影 K投影 V投影 │
            │    │     │     │   │
            │   RoPE RoPE   │   │
            │    │     │     │   │
            │    └──┬──┘     │   │
            │   Q*K^T→softmax→*V │  ← 注意力核心
            │       │            │
            │    输出投影(wo)     │
            │       │            │
            └───────┼────────────┘
                    │
                 add(cur, inpSA)  ← 残差连接
                    │
            ┌───────┤
            │  RMS Norm
            │       │
            │  gate投影  up投影    ← FFN
            │    │        │
            │  SiLU(gate)*up
            │       │
            │   down 投影
            │       │
            └───────┤
                    │
              add(cur, ffn_inp)  ← 残差连接
                    │
                 inpL (传入下一层)
```

### 3.4 关键构建模式的图结构

#### 3.4.1 矩阵乘法 + LoRA

```cpp
ggml_tensor * build_lora_mm(ggml_tensor * w, ggml_tensor * cur, ...) {
    // 主路径：W × cur
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    // LoRA 旁路（如果存在适配器）
    for (const auto & lora : *loras) {
        ggml_tensor * ab_cur = ggml_mul_mat(ctx0, lora.b,
                             ggml_mul_mat(ctx0, lora.a, cur));
        res = ggml_add(ctx0, res, ggml_scale(ctx0, ab_cur, scale));
    }

    return res;
}
```

图结构：
```
     cur                w (权重)
      │                  │
      ├── mul_mat ───────┤        主路径：W × cur
      │                  │
      ├── mul_mat ──┐    │        LoRA A × cur
      │             │    │
      │        mul_mat ──┤        LoRA B × (A×cur)
      │             │    │
      │        scale     │
      │             │    │
      └──── add ────┘           主路径 + LoRA 旁路
                │
              result
```

#### 3.4.2 多头注意力

```cpp
ggml_tensor * build_attn_mha(q, k, v, mask, ...) {
    // 维度重排
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head, n_batch]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    if (use_flash_attn) {
        // 融合注意力算子（单步完成 QK^T + softmax + *V）
        return ggml_flash_attn_ext(ctx0, q, k, v, mask, scale, ...);
    } else {
        // 标准分解式注意力
        ggml_tensor * kq   = ggml_mul_mat(ctx0, k, q);              // Q × K^T
        ggml_tensor * kq_s = ggml_soft_max_ext(ctx0, kq, mask, scale, ...);  // softmax
        ggml_tensor * kqv  = ggml_mul_mat(ctx0, v, kq_s);          // attn × V
        return kqv;
    }
}
```

标准注意力图结构：
```
   Q       K       V       mask
   │       │       │         │
 permute permute permute     │
   │       │       │         │
   └── mul_mat ──┘           │   ← Q × K^T
          │                  │
      soft_max ──────────────┘   ← softmax(带 mask)
          │
          └── mul_mat ─────────── V   ← attn_weights × V
                  │
                kqv
```

Flash Attention 将三步合并为单个算子，减少内存访问和中间结果存储。

#### 3.4.3 归一化层

```cpp
ggml_tensor * build_norm(cur, mw, mb, type, il) {
    switch (type) {
        case LLM_NORM_RMS: cur = ggml_rms_norm(ctx0, cur, eps); break;
        case LLM_NORM:     cur = ggml_norm(ctx0, cur, eps);     break;
        case LLM_NORM_GROUP: cur = ggml_group_norm(ctx0, cur, ...); break;
    }
    if (mw) cur = ggml_mul(ctx0, cur, mw);   // 乘以权重
    if (mb) cur = ggml_add(ctx0, cur, mb);   // 加偏置
    return cur;
}
```

#### 3.4.4 前馈网络（FFN）

```cpp
ggml_tensor * build_ffn(cur, up, gate, down, ...) {
    // 门控 FFN（如 LLaMA 的 SwiGLU）：
    // output = down(SiLU(gate(x)) * up(x))

    ggml_tensor * up_proj  = build_lora_mm(up, cur);    // up 投影
    ggml_tensor * gate_proj = build_lora_mm(gate, cur);  // gate 投影

    // SiLU 门控：gate_output = SiLU(gate_proj) * up_proj
    cur = ggml_swiglu_split(ctx0, gate_proj, up_proj);

    // down 投影
    cur = build_lora_mm(down, cur);
    return cur;
}
```

图结构：
```
         cur (输入)
        ┌──┴──┐
        │     │
   up 投影  gate 投影
        │     │
        │   SiLU
        │     │
        └── mul ┘     ← 逐元素相乘
            │
       down 投影
            │
         output
```

---

## 4. 完整调用链路

以一次 `llama_decode()` 调用为例，从入口到硬件执行：

```
llama_decode(ctx, batch)
  │
  ├─ llama_context::decode(batch)
  │     │
  │     ├─ process_ubatch()
  │     │     │
  │     │     ├─ model.build_graph(gparams)         ── 构建计算图
  │     │     │     │
  │     │     │     ├─ ggml_new_graph(ctx0)          创建空图
  │     │     │     ├─ for each layer:               逐层构建
  │     │     │     │   ├─ build_norm()
  │     │     │     │   ├─ build_qkv()
  │     │     │     │   ├─ build_attn_mha()
  │     │     │     │   ├─ build_ffn()
  │     │     │     │   └─ ...
  │     │     │     ├─ build_lora_mm(output, cur)    输出投影
  │     │     │     └─ ggml_build_forward_expand(gf, output)  递归建图
  │     │     │
  │     │     ├─ ggml_backend_sched_alloc_graph()    ── 调度与内存分配
  │     │     │     ├─ 分配后端 ID
  │     │     │     ├─ 切分子图
  │     │     │     └─ 分配缓冲区
  │     │     │
  │     │     ├─ set_inputs(&ubatch)                  ── 填入输入数据
  │     │     │
  │     │     └─ graph_compute(gf)                    ── 执行计算
  │     │           │
  │     │           └─ ggml_backend_sched_graph_compute_async()
  │     │                 │
  │     │                 ├─ for each split:           逐子图执行
  │     │                 │   └─ ggml_backend_graph_compute(backend, split)
  │     │                 │       │
  │     │                 │       └─ for each node:    按拓扑序执行
  │     │                 │           └─ compute_fn(op, type)
  │     │                 │               ├─ CUDA kernel    (GPU 节点)
  │     │                 │               └─ CPU SIMD ops   (CPU 节点)
  │     │                 │
  │     │                 └─ 返回结果到 logits 张量
  │     │
  │     └─ 返回 logits
  │
  └─ 采样器选择下一个 token
```

---

## 5. 关键设计总结

| 设计点 | 实现方式 | 优势 |
|--------|----------|------|
| **图的表示** | 张量通过 `src[]` 指针连接 | 零额外开销，自然构成 DAG |
| **拓扑排序** | 递归后序遍历自动产生 | 保证执行顺序正确 |
| **去重** | 哈希集合 `visited_hash_set` | 支持 DAG（如残差连接的汇合点） |
| **内存优化** | `use_counts[]` 引用计数 | 可判断何时释放中间结果 |
| **后端抽象** | 每个后端实现 `supports_op()` | 自动选择最优硬件 |
| **多后端协同** | 图切分 + 边界数据拷贝 | 可同时利用 CPU 和 GPU |
| **图复用** | 相同批次配置下不重建图 | 避免重复构建开销 |
| **算子融合** | Flash Attention 等融合算子 | 减少中间结果和内存访问 |
| **LoRA 注入** | 在矩阵乘法旁路添加分支 | 不修改原始权重即可适配 |
| **多线程** | 大算子切 chunk 并行执行 | 充分利用多核 CPU |
