# llama.cpp 项目架构分析

本文档从整体到细节，分析 llama.cpp 的项目结构、各模块作用及其整合方式。

---

## 1. 项目全景

llama.cpp 是一个用 C/C++ 实现的大语言模型推理框架，核心目标是：在各类硬件上高效运行 LLM 推理。项目采用分层架构，自底向上分为四层：

```
┌─────────────────────────────────────────────────┐
│                  工具层 (tools/)                  │  ← HTTP 服务器、量化工具、基准测试等
├─────────────────────────────────────────────────┤
│                公共库层 (common/)                 │  ← 参数解析、采样、聊天模板、PEG 解析器
├─────────────────────────────────────────────────┤
│               模型引擎层 (src/)                   │  ← 模型加载、推理、词表、采样器
├─────────────────────────────────────────────────┤
│              张量计算库层 (ggml/)                  │  ← 张量定义、后端抽象、硬件加速
└─────────────────────────────────────────────────┘
```

**关键数据流**：用户请求 → 工具层（如 server）→ 公共库（参数/模板处理）→ 模型引擎（加载权重/构建计算图）→ ggml（调度到具体硬件后端执行）

---

## 2. 目录结构与职责

```
llama.cpp/
├── ggml/                  # 核心张量计算库（独立子模块）
│   ├── include/           #   公共 API 头文件 (ggml.h, ggml-alloc.h, ggml-backend.h 等)
│   └── src/               #   各后端实现
│       ├── ggml-cpu/      #     CPU 后端（x86/ARM/RISC-V 等 SIMD 优化）
│       ├── ggml-cuda/     #     NVIDIA CUDA 后端（61 个 .cu 核函数文件）
│       ├── ggml-vulkan/   #     Vulkan 计算后端（SPIR-V 着色器）
│       ├── ggml-metal/    #     Apple Metal 后端
│       ├── ggml-sycl/     #     Intel SYCL 后端
│       ├── ggml-hip/      #     AMD HIP 后端
│       ├── ggml-cann/     #     华为昇腾 CANN 后端
│       ├── ggml-opencl/   #     OpenCL 后端
│       ├── ggml-webgpu/   #     WebGPU 后端
│       ├── ggml-blas/     #     BLAS 加速后端
│       └── ggml-rpc/      #     RPC 远程后端
│
├── include/               # llama 公共 API 头文件
│   ├── llama.h            #   主 API（约 81KB，定义了全部公开接口）
│   └── llama-cpp.h        #   C++ 封装
│
├── src/                   # 模型引擎层（核心推理逻辑）
│   ├── llama.cpp          #   主入口：模型创建、上下文管理
│   ├── llama-model.cpp    #   模型加载与架构实现（554KB，最大文件）
│   ├── llama-context.cpp  #   推理上下文与计算图执行
│   ├── llama-graph.cpp    #   计算图构建
│   ├── llama-sampler.cpp  #   采样策略实现
│   ├── llama-vocab.cpp    #   词表与分词器
│   ├── llama-chat.cpp     #   聊天模板处理
│   ├── llama-quant.cpp    #   量化实现
│   ├── llama-arch.cpp     #   架构特定代码
│   └── models/            #   各模型架构的具体构建器
│
├── common/                # 公共工具库（被 tools/ 和 examples/ 共享）
│   ├── arg.cpp/h          #   统一命令行参数解析
│   ├── common.cpp/h       #   通用工具函数
│   ├── chat.cpp/h         #   聊天消息处理、工具调用解析
│   ├── sampling.cpp/h     #   高级采样封装
│   ├── peg-parser.cpp/h   #   PEG 解析器（替代正则表达式）
│   ├── console.cpp/h      #   控制台 I/O
│   ├── speculative.cpp/h  #   推测解码支持
│   ├── ngram-*.cpp/h      #   N-gram 缓存
│   ├── json-partial.cpp/h #   JSON 增量解析
│   ├── regex-partial.cpp/h#   正则增量解析
│   └── jinja/             #   Jinja2 模板引擎（用于聊天模板渲染）
│
├── tools/                 # 命令行工具集
│   ├── server/            #   HTTP 推理服务器（OpenAI/Anthropic API 兼容）
│   ├── quantize/          #   模型量化工具
│   ├── llama-bench/       #   性能基准测试
│   ├── tokenize/          #   分词测试工具
│   ├── perplexity/        #   困惑度评估
│   ├── imatrix/           #   重要性矩阵计算（指导量化）
│   ├── gguf-split/        #   GGUF 模型分片/合并
│   ├── export-lora/       #   LoRA 权重导出
│   ├── rpc/               #   RPC 服务器（分布式推理）
│   ├── mtmd/              #   多模态处理守护进程
│   ├── tts/               #   文本转语音工具
│   ├── completion/        #   简单补全测试
│   ├── parser/            #   模板解析分析工具
│   ├── cvector-generator/ #   代码向量生成器
│   └── fit-params/        #   参数拟合工具
│
├── tests/                 # 测试套件
│   ├── test-backend-ops.cpp    # 后端算子测试（386KB）
│   ├── test-chat.cpp           # 聊天功能测试
│   ├── test-gguf.cpp           # GGUF 格式测试
│   ├── test-quantize-*.cpp     # 量化测试
│   ├── test-grammar-*.cpp      # 语法解析测试
│   ├── test-sampling.cpp       # 采样测试
│   ├── test-tokenizer-*.cpp    # 分词器测试
│   ├── test-opt.cpp            # 优化器测试
│   └── peg-parser/             # PEG 解析器专项测试
│
├── examples/              # 示例程序
│   ├── simple/            #   基本文本生成
│   ├── embedding/         #   嵌入提取
│   ├── batched/           #   批处理推理
│   ├── parallel/          #   并行处理
│   ├── speculative/       #   推测解码示例
│   ├── retrieval/         #   检索增强生成
│   ├── lookup/            #   KV 缓存查找优化
│   ├── diffusion/         #   扩散模型示例
│   └── ...
│
├── docs/                  # 项目文档
├── cmake/                 # CMake 构建配置
├── scripts/               # 构建和工具脚本
│
├── CMakeLists.txt         # 主构建文件
├── convert_hf_to_gguf.py  # HuggingFace 模型 → GGUF 格式转换
└── README.md
```

---

## 3. 核心模块详解

### 3.1 ggml — 张量计算库

ggml 是整个项目的计算底座，提供张量定义、算子系统和硬件后端抽象。

#### 3.1.1 张量结构

```c
struct ggml_tensor {
    enum ggml_type type;              // 数据类型：F32, F16, Q4_0, Q8_0 等量化类型
    struct ggml_backend_buffer * buffer; // 所属的后端内存缓冲区
    int64_t ne[GGML_MAX_DIMS];        // 各维度元素数（最多 4 维）
    size_t nb[GGML_MAX_DIMS];         // 各维度的字节步长
    enum ggml_op op;                  // 关联的算子类型
    int32_t op_params[GGML_MAX_OP_PARAMS]; // 算子参数
    struct ggml_tensor * src[GGML_MAX_SRC]; // 输入张量（构建计算图）
    void * data;                      // 数据指针
    char name[GGML_MAX_NAME];         // 张量名称
};
```

#### 3.1.2 算子系统

ggml 定义了 80+ 种算子（`ggml_op` 枚举），涵盖：

| 类别 | 典型算子 |
|------|----------|
| 基础运算 | ADD, SUB, MUL, DIV |
| 逐元素 | SQR, SQRT, LOG, SIN, COS, ABS |
| 规约 | SUM, MEAN, ARGMAX, ARGMIN |
| 矩阵运算 | MUL_MAT（矩阵乘法）, OUT_PROD（外积） |
| 归一化 | NORM, RMS_NORM, GROUP_NORM |
| 激活函数 | RELU, GELU, SILU, TANH |
| 卷积 | CONV_1D, CONV_2D, CONV_3D |
| 注意力 | SOFT_MAX, FLASH_ATTN_EXT |
| 形状操作 | RESHAPE, VIEW, PERMUTE, TRANSPOSE, CONCAT |
| 量化 | QUANTIZE, DEQUANTIZE |

#### 3.1.3 后端抽象

ggml 通过统一的后端接口屏蔽硬件差异：

```
ggml_backend_t           → 后端句柄（CPU / CUDA / Vulkan / Metal 等）
ggml_backend_buffer_t    → 后端上的内存缓冲区
ggml_backend_buffer_type_t → 缓冲区类型（分配参数）
```

**后端注册机制**：各后端通过 `GGML_BACKEND_REGISTER()` 在运行时动态注册，主程序无需硬编码依赖。

**算子调度**：每个后端实现 `supports_op()` 来声明支持的算子。计算图执行时，调度器根据算子类型和数据类型自动选择最合适的后端。

#### 3.1.4 计算图

ggml 使用有向无环图（DAG）组织计算：

```
struct ggml_cgraph {
    struct ggml_tensor ** nodes;     // 需要计算的节点
    struct ggml_tensor ** leafs;     // 常量/输入张量（叶子节点）
    struct ggml_tensor ** grads;     // 梯度节点（用于训练）
    int32_t * use_counts;            // 引用计数（优化用）
};
```

构建方式：调用 `ggml_build_forward_expand()` 将算子逐个加入图。执行时通过 `ggml_graph_compute()` 驱动各后端完成计算。

#### 3.1.5 硬件加速

**CPU SIMD 优化**：
- x86: AVX, AVX2, AVX-512, AMX
- ARM: NEON, SVE
- RISC-V: RVV
- 其他: PowerPC, S390, LoongArch

**GPU 后端**：CUDA（NVIDIA）、Vulkan（跨平台）、Metal（Apple Silicon）、SYCL（Intel）、HIP（AMD）、CANN（华为昇腾）

### 3.2 src/ — 模型引擎层

模型引擎层位于 ggml 之上，负责模型加载、推理执行和结果采样。

#### 3.2.1 核心数据结构

```
llama_model      → 已加载的模型（权重、超参数、词表、层定义）
llama_context     → 推理上下文（KV 缓存、计算调度器、输出缓冲区）
llama_vocab       → 词表与分词器
llama_sampler     → 采样策略链
```

#### 3.2.2 模型加载流程

```
GGUF 文件
  │
  ├─ 1. 读取元数据（架构类型、超参数、词表）
  ├─ 2. 架构检测（从 140+ 种架构中识别）
  ├─ 3. 创建张量结构（根据架构规格）
  ├─ 4. 加载量化权重（支持 mmap、直接 I/O）
  └─ 5. 设备分配（将张量分布到可用设备）
       │
       └─ llama_model 就绪
```

**支持的模型架构**（140+ 种）：
- 通用：LLaMA, LLaMA4, Mistral, Gemma, Phi
- 中文：QWEN2, QWEN3, ChatGLM, GLM4
- 代码：StarCoder, Refact, CodeShell
- MoE：QWEN2MoE, GLM4_MoE, OpenAI_MoE
- 视觉：QWEN2VL, CogVLM
- 特殊：RWKV, Mamba, T5, BERT, BitNet

#### 3.2.3 推理执行

```
输入 Token 序列
  │
  ├─ 1. llama_decode() 入口
  ├─ 2. 构建计算图（llama-graph.cpp）
  │     └─ 根据模型架构构建前向传播图
  ├─ 3. 后端调度器分配算子到各硬件后端
  ├─ 4. 执行计算（ggml_graph_compute）
  ├─ 5. 获取 logits 输出
  └─ 6. 采样器选择下一个 token
       │
       └─ 输出 Token
```

#### 3.2.4 内存管理

KV 缓存是推理的核心内存结构，支持多种类型：
- **标准 KV 缓存**：用于 Transformer 架构
- **SWA（滑动窗口注意力）**：限制缓存长度以节省内存
- **循环缓存**：用于 RWKV/Mamba 等循环架构
- **混合缓存**：结合多种策略

#### 3.2.5 采样系统

采样采用链式架构，多个采样器串联工作：

```
logits → Temperature → Top-K → Top-P → Min-P → Typical → Grammar → Penalty → 输出 Token
```

| 采样器 | 作用 |
|--------|------|
| Greedy | 贪心选择概率最高的 token |
| Temperature | 调节概率分布的锐度 |
| Top-K | 只从概率最高的 K 个 token 中采样 |
| Top-P (Nucleus) | 从累积概率达到 P 的最小 token 集中采样 |
| Min-P | 过滤掉概率低于阈值的 token |
| Typical | 局部典型采样 |
| Mirostat | 自适应困惑度控制 |
| Grammar | 基于语法约束的生成（如 JSON Schema） |
| Penalty | 重复频率惩罚、存在惩罚 |

#### 3.2.6 分词器

支持多种分词方案：

| 类型 | 说明 | 典型模型 |
|------|------|----------|
| SPM | SentencePiece BPE | LLaMA 系列 |
| BPE | GPT-2 字节级 BPE | GPT 系列 |
| WPM | WordPiece | BERT |
| UGM | Unigram | T5 |
| RWKV | 贪心分词 | RWKV |
| PLAMO2 | Aho-Corasick + DP | PLaMo2 |

共有 62 种预分词模式，适配不同模型家族。

### 3.3 common/ — 公共工具库

common 层是连接引擎层与应用层的桥梁，提供所有工具共享的基础设施。

#### 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 参数解析 | arg.cpp/h | 统一的命令行参数处理，所有工具共用 |
| 聊天处理 | chat.cpp/h | 消息格式化、工具调用解析、增量 diff 追踪 |
| 采样封装 | sampling.cpp/h | 对 llama_sampler 的高级封装 |
| PEG 解析器 | peg-parser.cpp/h | 解析模型输出的 PEG 解析器（替代正则） |
| 模板引擎 | jinja/ | Jinja2 模板渲染（用于聊天模板） |
| 控制台 | console.cpp/h | 终端 I/O 处理 |
| 推测解码 | speculative.cpp/h | 推测解码支持逻辑 |
| JSON 解析 | json-partial.cpp/h | 增量 JSON 解析 |

### 3.4 tools/server/ — HTTP 推理服务器

服务器是项目中最大的工具，提供与 OpenAI/Anthropic 兼容的 API。

#### 架构模式

```
HTTP 请求 → httplib 解析 → server_queue（任务队列）→ server_slot（推理槽）→ llama_context → 响应
```

**多槽并行**：服务器维护多个 `server_slot`，每个槽绑定一个推理上下文，实现并发请求处理。

**核心组件**：

| 文件 | 职责 |
|------|------|
| server.cpp | 主入口，启动与路由 |
| server-context.h/cpp | llama 上下文生命周期管理 |
| server-http.h/cpp | HTTP 服务器实现 |
| server-task.h/cpp | 任务类型定义（补全、嵌入、重排等） |
| server-queue.h/cpp | 线程安全任务队列 |
| server-slot.h/cpp | 推理槽管理 |

**API 端点**：

| 端点 | 兼容性 | 用途 |
|------|--------|------|
| `/v1/chat/completions` | OpenAI | 聊天补全 |
| `/v1/completions` | OpenAI | 文本补全 |
| `/v1/embeddings` | OpenAI | 文本嵌入 |
| `/v1/messages` | Anthropic | 消息 API |
| `/completion` | 原生 | 补全 |
| `/embedding` | 原生 | 嵌入 |

**两种运行模式**：
- **单模型模式**：加载一个模型直接服务
- **路由模式**：编排多个模型实例跨进程工作

---

## 4. 模块整合方式

### 4.1 调用链路

以一次聊天补全请求为例，完整调用链路：

```
用户请求 (HTTP POST /v1/chat/completions)
  │
  ├─ [tools/server] HTTP 解析与路由
  │
  ├─ [common/chat] 消息格式化、模板渲染 (Jinja2)
  │
  ├─ [src/llama-vocab] 将文本分词为 token 序列
  │
  ├─ [src/llama-context] 构建推理请求
  │   ├─ [src/llama-graph] 构建计算图
  │   │
  │   ├─ [ggml] 调度器将算子分配到后端
  │   │   ├─ [ggml-cuda] GPU 执行矩阵乘法、注意力
  │   │   └─ [ggml-cpu]  CPU 执行其他算子
  │   │
  │   └─ 获取 logits
  │
  ├─ [src/llama-sampler] 采样下一个 token
  │   └─ [common/sampling] 采样参数封装
  │
  ├─ [src/llama-vocab] 将 token 反向解码为文本
  │
  └─ [tools/server] HTTP 流式/非流式响应
```

### 4.2 构建系统集成

```
CMakeLists.txt（根）
  ├─ add_subdirectory(ggml)       → 构建各后端库 (ggml, ggml-base, ggml-cpu, ggml-cuda 等)
  ├─ add_subdirectory(src)        → 构建 llama 库 (依赖 ggml)
  ├─ add_subdirectory(common)     → 构建 common 库 (依赖 llama)
  ├─ add_subdirectory(tools)      → 构建各工具可执行文件 (依赖 common + llama)
  ├─ add_subdirectory(examples)   → 构建示例程序
  ├─ add_subdirectory(tests)      → 构建测试
  └─ cmake/                       → 平台/编译器配置
```

依赖关系：
```
ggml (无外部依赖)
  ↑
llama (依赖 ggml)
  ↑
common (依赖 llama)
  ↑
tools / examples (依赖 common)
```

### 4.3 跨语言支持

- **C API**：`include/llama.h` 定义完整的 C 接口
- **C++ 封装**：`include/llama-cpp.h`
- **Python 绑定**：通过 `convert_hf_to_gguf.py` 等脚本桥接
- **其他语言**：通过 C ABI 暴露接口，各语言可绑定

---

## 5. GGUF 模型格式

GGUF 是 llama.cpp 的模型文件格式，设计用于高效存储和加载：

- **元数据**：模型架构类型、超参数、词表定义
- **张量数据**：支持多种量化格式存储的权重
- **分片支持**：大模型可拆分为多个文件
- **版本**：GGUF v1、v2、v3

转换工具：`convert_hf_to_gguf.py` 将 HuggingFace 格式模型转换为 GGUF。

---

## 6. 量化体系

llama.cpp 支持丰富的量化类型，在模型体积与推理精度间灵活权衡：

| 量化类型 | 每权重比特数 | 特点 |
|----------|-------------|------|
| Q4_0 / Q4_1 | 4 bit | 基础 4-bit 量化 |
| Q5_0 / Q5_1 | 5 bit | 5-bit 量化 |
| Q8_0 | 8 bit | 高精度 8-bit |
| Q2_K / Q3_K / Q4_K / Q5_K / Q6_K | 2-6 bit | K-量化（分块量化） |
| IQ2_XXS / IQ3_XS / IQ4_NL | 2-4 bit | 超低比特率量化 |
| F16 | 16 bit | 半精度浮点（无量化） |
| F32 | 32 bit | 全精度浮点 |

量化工具：`tools/quantize/`

重要性矩阵（imatrix）：通过 `tools/imatrix/` 计算，可指导量化时保留关键权重精度。

---

## 7. 关键设计特点

1. **后端无关性**：核心推理逻辑不依赖具体硬件，通过 ggml 后端抽象自动适配
2. **图计算模式**：基于计算图的前向传播，支持算子融合和跨后端调度
3. **零拷贝优化**：尽可能使用原地操作减少内存拷贝
4. **量化友好**：从底层张量开始就为量化设计，支持运行时反量化/重量化
5. **多设备支持**：智能张量并行和模型分片，跨 GPU 分布推理
6. **流式处理**：从服务器到分词器全链路支持流式输出
7. **多模态扩展**：通过 mtmd 模块支持图像、音频等多模态输入
