# embedding-server

基于 ONNX Runtime 的 Qwen3-Embedding-0.6B 推理服务，提供 OpenAI 兼容的 `POST /v1/embeddings` 接口。

- **输出维度 1024**（支持 `dimensions` 按 Matryoshka 截断到任意维度）
- 模型：`onnx-community/Qwen3-Embedding-0.6B-ONNX` 的 `model_int8.onnx`（586 MB）
- 推理：ONNX Runtime Java **1.19.2**（CPU）
- 分词：纯 Java 移植的 Qwen2 Byte-Level BPE，只读 `tokenizer.json`，不依赖 DJL 或其他原生库
- 技术栈：Java 21 + Spring Boot 3.5.3

## 模型下载

模型文件不随仓库提交，克隆后先下载到 `models/` 目录。

| 来源 | 地址 |
| --- | --- |
| ModelScope（推荐） | https://www.modelscope.cn/models/onnx-community/Qwen3-Embedding-0.6B-ONNX |
| 模型文件页 | https://www.modelscope.cn/models/onnx-community/Qwen3-Embedding-0.6B-ONNX/tree/master/onnx |
| HuggingFace | https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX |

运行所需的最小文件是 **`model_int8.onnx` + `tokenizer.json`**：模型权重与全部分词信息（词表、合并表、特殊 token、正则、后处理模板）都在 `tokenizer.json` 里，其余 JSON 本服务不读取。

一键下载（Linux / macOS / Windows Git Bash）：

```bash
mkdir -p models
BASE="https://www.modelscope.cn/models/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/master"
curl -L -o models/model_int8.onnx "$BASE/onnx/model_int8.onnx"
curl -L -o models/tokenizer.json "$BASE/tokenizer.json"
# 可选：模型结构信息，排错或换模型时参考
curl -L -o models/config.json "$BASE/config.json"
```

Windows PowerShell 可调用 `curl.exe`：

```powershell
New-Item -ItemType Directory -Force models | Out-Null
$base = "https://www.modelscope.cn/models/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/master"
curl.exe -L -o models/model_int8.onnx "$base/onnx/model_int8.onnx"
curl.exe -L -o models/tokenizer.json "$base/tokenizer.json"
# 可选
curl.exe -L -o models/config.json "$base/config.json"
```

下载后校验：

| 文件 | 大小 | SHA256 |
| --- | --- | --- |
| `models/model_int8.onnx` | 613,527,539 字节 | `6d0ea863f78b4a84afa3c7fcba1ec341572b5e28121aef77b7092b1dfdf679c7` |
| `models/tokenizer.json` | 约 11 MB | `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a` |

```bash
sha256sum models/model_int8.onnx
```

## 快速开始

要求 JDK 21、Maven 3.9+。

```bash
# 开发运行（在项目根目录，默认读取 models/ 下的文件）
mvn spring-boot:run

# 打包运行
mvn -DskipTests package
java -Xmx2g -jar target/embedding-server-1.0.0.jar
```

默认端口 `8000`，启动时加载模型约 2 秒并自动预热。

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"Qwen3-Embedding-0.6B","dimension":1024}
```

模型路径是相对路径，按启动工作目录解析；从其他目录启动时用参数覆盖：

```bash
java -Xmx2g -jar target/embedding-server-1.0.0.jar \
  --embedding.model-path=/path/to/models/model_int8.onnx \
  --embedding.tokenizer-path=/path/to/models/tokenizer.json
```

## 可选：GPU 推理（默认关闭）

`pom.xml` 默认只引用 CPU 版 `onnxruntime`，GPU 依赖以注释形式保留。需要 GPU 时：

1. 注释掉 `pom.xml` 中的 `onnxruntime` 依赖；
2. 取消注释 `onnxruntime_gpu` 依赖；
3. 重新执行 `mvn -DskipTests package`。

运行前提：

- NVIDIA GPU 与驱动
- CUDA 12.x runtime
- **cuDNN 8.x**：ONNX Runtime 1.19.2 的 CUDA EP 依赖 cuDNN 8，**不能用 cuDNN 9**
- cuDNN 8 的 `bin` 目录加入 `PATH`（或在启动脚本中前置）

```bash
# Linux / Git Bash
export PATH=/path/to/cudnn8/bin:$PATH
java -Xmx2g -jar target/embedding-server-1.0.0.jar --embedding.gpu-id=0
```

```powershell
# PowerShell
$env:PATH = "D:\path\to\cudnn8\bin;" + $env:PATH
java -Xmx2g -jar target/embedding-server-1.0.0.jar --embedding.gpu-id=0
```

默认模型仍是 `models/model_int8.onnx`。`embedding.gpu-id` 默认 `-1`（CPU），`0` 表示第一块 GPU；也可用环境变量 `EMBEDDING_GPU_ID=0`。

### 实测提示

测试机：GTX 1650 4 GB / CUDA 12.6 / cuDNN 8.9.7 / `-Xmx2g`：

| 组合 | 8 token | 1024 token |
| --- | --- | --- |
| **CPU + int8（默认）** | 0.23 s | 8.0 s |
| GPU + int8 | 0.57 s | 8.7 s |
| GPU + fp16（参考） | 0.16 s | 2.7 s |

int8 在 GPU 上反而更慢，不是显卡问题：ONNX Runtime 的 CUDA EP 不支持动态量化算子（`MatMulInteger` 等），大量节点会回退 CPU，日志中可见数百个 Memcpy 节点，数据在显存与内存之间来回搬运。int8 的速度优势来自 CPU 的 QGEMM 内核或 TensorRT INT8 引擎，不是 ORT CUDA EP 的路径。

因此默认继续使用 **CPU + int8**。如果确实需要 GPU 加速，建议换 fp16 模型（同一仓库的 `model_fp16.onnx` + `model_fp16.onnx_data`），fp16 的全算子都在 CUDA EP 上，实测长文本可快约 3 倍。

## 向量维度

模型隐藏层维度为 **1024**，默认返回已 L2 归一化的 1024 维向量。

传 `dimensions`（1~1024）会按 Matryoshka 截断并重新归一化，例如 `dimensions: 256` 返回 256 维、模长为 1 的向量。截断结果与全量向量的前 256 维归一化完全一致（已验证 `cos=1.000000000`）。

## API

### 请求

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Embedding-0.6B","input":["今天天气不错","明天可能下雨"]}'
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `input` | 是 | 字符串或字符串数组，数组上限由 `embedding.max-inputs` 控制（默认 64） |
| `model` | 否 | 回显用；不传则返回 `embedding.model-name` |
| `encoding_format` | 否 | `float`（默认）或 `base64`（little-endian float32） |
| `dimensions` | 否 | 1~1024，截断后重新归一化 |
| `instruction` | 否 | 扩展字段；传入时按官方格式拼成 `Instruct: {instruction}\nQuery:{text}` |

未知字段（例如 OpenAI SDK 的 `user`）会被忽略。暂不支持 token id 数组作为 `input`。

### 响应

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, "..."] }
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": { "prompt_tokens": 12, "total_tokens": 12 }
}
```

`usage` 统计实际参与推理的 token 数（含句末模板 token，超长文本按 `max-tokens` 截断后统计）。

### 错误码

| 状态码 | 场景 |
| --- | --- |
| `400` | 参数错误、非法 JSON、空数组、超限、非法 `dimensions`、非法 `encoding_format` |
| `401` | 已启用 API Key 时缺少或错误的 `Authorization` |
| `405` | 非 POST 方法 |
| `413` | 请求体超过 `max-request-bytes` |
| `415` | `Content-Type` 不是 `application/json` |
| `500` | 内部推理异常（响应不暴露内部细节，详情见服务日志） |
| `503` | 并发已满且排队超过 `queue-timeout-seconds` |

错误体：

```json
{ "error": { "message": "input 不能为空", "type": "invalid_request_error", "code": null } }
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")
resp = client.embeddings.create(
    model="Qwen3-Embedding-0.6B",
    input=["今天天气不错", "明天可能下雨"],
    dimensions=256,          # 可选
)
print(len(resp.data[0].embedding))  # 256
```

已用 `openai 2.48.0` 实测：单字符串、数组、`dimensions`、SDK 默认的 base64 编码、鉴权错误均正常。

## 配置

`src/main/resources/application.yml` 中 `embedding.*`：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `model-name` | `Qwen3-Embedding-0.6B` | 响应中回显的模型名 |
| `model-path` | `models/model_int8.onnx` | ONNX 模型路径 |
| `tokenizer-path` | `models/tokenizer.json` | 分词器文件 |
| `max-tokens` | `1024` | 单条文本上限（含句末模板 token），超出截断 |
| `max-inputs` | `64` | 单次请求文本条数上限 |
| `max-concurrency` | `4` | 同时执行的推理数，建议按逻辑核数的 1/3~1/2 设置 |
| `intra-op-threads` | `0` | ONNX 算子内线程数，0 表示自动 |
| `dimension` | `1024` | 模型输出维度识别失败时的兜底值 |
| `warmup` | `true` | 启动后预热一次 |
| `max-request-bytes` | `4194304` | 请求体大小上限，超出返回 413 |
| `queue-timeout-seconds` | `30` | 并发已满时的排队等待秒数，超时返回 503 |
| `api-key` | 空 | 非空时校验 `Authorization: Bearer <key>`；默认关闭 |
| `gpu-id` | `-1` | GPU 设备号，`-1` 为 CPU；启用 GPU 需取消注释 GPU 依赖并安装 CUDA 12 + cuDNN 8 |

开启 API Key：

```bash
java -Xmx2g -jar target/embedding-server-1.0.0.jar --embedding.api-key=your-secret
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret" \
  -d '{"input":"hello"}'
```

## 生产部署建议

- **JVM**：`-Xmx2g` 足够（实测常驻 610~635 MB，含 ORT native 内存）；长文本高并发可给 4g。
- **并发**：`max-concurrency` 默认 4，适合短文本；1024 token 长文本并发几乎无吞吐收益，建议降到 1~2，避免单条延迟被放大。
- **反向代理**：建议 Nginx 前置，限制 body 大小与超时，并做 TLS/鉴权；应用层的 `max-request-bytes`、`api-key` 是第二道保护。
- **进程管理**：systemd / 容器托管，存活探针用 `GET /health`；服务已开启 `server.shutdown=graceful`，收到 SIGTERM 后最多等 30 s 处理完在途请求。
- **日志**：访问日志输出到 `logs/access*.log`（`%h %t "%r" %s %b %D`，`%D` 为微秒耗时），应用日志走标准输出；`logs/` 已加入 `.gitignore`。
- **资源**：单实例内存约 0.7 GB、模型加载约 2 s，适合内网离线部署。

### 压测参考

测试机：Intel i7-9750H（6 核 12 线程）、`-Xmx4g`、20 并发、短文本（约 10 token）：

| `max-concurrency` | 吞吐 | P50 | P95 |
| --- | --- | --- | --- |
| 2 | 8.8 req/s | 2.22 s | 3.40 s |
| 4（默认） | 13.0 req/s | 1.51 s | 1.77 s |
| 6 | 13.4 req/s | 1.43 s | 1.99 s |

1024 token 长文本：单条约 6~8 s；3 个并发时每条约 20 s（CPU 已饱和，只增加延迟）。500 次请求压测零错误，内存稳定。

## 实现说明

1. **分词器**：严格对齐 HuggingFace tokenizers 的处理顺序——NFC 规范化 → added_tokens 最长匹配 → Split 正则预切分 → ByteLevel 字节映射 → BPE 合并 → 追加 post_processor 模板 token。词表、合并表、正则全部从 `tokenizer.json` 读取。
2. **Unicode 语义**：Split 正则以 `Pattern.UNICODE_CHARACTER_CLASS` 编译，使 `\s` 等与 Rust 分词器的 Unicode 语义一致（否则 NBSP、全角空格等会产生不同分词结果）。
3. **KV cache**：该导出图除 `input_ids` / `attention_mask` / `position_ids` 外，还有 28 层 `past_key_values.*.key/value`。每次推理传入形状 `[1, 8, 0, 128]` 的空缓存走无缓存前向，输出中的 `present.*` 丢弃。
4. **池化**：取最后一个有效 token（句末是模板追加的 `<|endoftext|>`）的隐藏状态，再 L2 归一化；`dimensions` 截断在归一化之前。
5. **逐条推理**：该图在 batch>1 时存在跨样本注意力串扰——两条无关文本放进同一次 `session.run`，即使等长无 padding，各自向量也会偏离单条结果（余弦约 0.90~0.92）；仅修改另一行的 pad token id 也会影响目标行。因此固定逐条推理，吞吐靠 `max-concurrency` 提升。**不要为了吞吐改回批量推理。**

## 验证结果

| 项目 | 结果 |
| --- | --- |
| 分词对拍 | 500 条随机文本（含多语言、emoji、组合字符、控制字符、特殊 token、长文本）+ 30 条定向样本，与 HuggingFace `tokenizers 0.22.2` 逐 id 一致 |
| 推理对拍 | 与 Python `onnxruntime 1.19.2` 同模型输出余弦 `1.000000` |
| 批量一致性 | 批量请求与逐条请求结果完全一致（`1.0`），顺序正确 |
| 归一化 | 默认与 `dimensions` 截断后模长均为 `1.0` |
| 截断语义 | `dimensions=256` 与全量前 256 维归一化 `cos=1.000000000` |
| 生产检查 | 38 项断言通过：鉴权、健康检查、多语言、NFC、空串、控制字符、非法 JSON、深嵌套、5 MB 请求、非法维度、错误方法/类型、长文本截断、并发一致性等 |
| 稳定性 | 500 请求 / 20 并发：零错误，P95 3.4 s，RSS 610~635 MB，无泄漏 |
| 限流 | 并发满载且排队超时返回 `503 / rate_limit_error` |
| 故障启动 | 模型文件缺失、端口占用均快速失败并输出明确原因 |
| 兼容性 | OpenAI Python SDK `2.48.0` 全通过 |
| 语义 | int8：`What is the capital of China?` / `The capital of China is Beijing.` 相似度 `0.7739`；fp16 `0.8014` |
| GPU（可选） | int8 在 GPU 上无加速（量化节点回退 CPU，8.7 s vs CPU 8.0 s）；GPU 加速需改用 fp16 |

## 已知限制

- **instruction 字段**：0.6B int8 模型对官方指令格式不敏感，实测 query→doc 平均相似度与不加指令基本持平（0.674 vs 0.677）。字段保留用于兼容和后续替换更大模型，检索场景可不传。
- **逐条推理**：单请求内文本顺序推理；64 条短文本约 6~7 s。提高 `max-concurrency` 可提升并发吞吐。
- **长文本并发**：1024 token 推理 CPU 密集，多并发只增加延迟，不增加吞吐；长文本服务建议 `max-concurrency=1~2`。
- **int8 精度**：排序与语义正常，相似度绝对值略低于 fp32；换 `model.onnx`（约 2.4 GB，需与 `model.onnx_data` 同目录）可提高精度，改 `embedding.model-path` 即可，无需改代码。
- **GPU + int8**：int8 模型在 ORT CUDA EP 上没有收益（量化算子回退 CPU），默认组合是 CPU + int8；需要 GPU 加速请换 fp16 模型。
- **cuDNN 版本**：ORT 1.19.2 要求 cuDNN 8；cuDNN 9 会导致 CUDA EP 初始化失败。
- **请求体限制**：`max-request-bytes` 依赖 `Content-Length`；chunked 传输的大请求建议在 Nginx 层限制。
- ONNX Runtime 固定 1.19.2：本机 Windows/JDK 已验证可加载；1.21+ 可能因 MSVC 运行库版本出现 `onnxruntime.dll` 初始化失败。

## 目录结构

```
embedding-server/
├── models/                               # 不提交到仓库，按上文下载
│   ├── model_int8.onnx                   # 必需
│   ├── tokenizer.json                    # 必需
│   └── config.json                       # 可选，排错用
├── src/main/java/com/police/embedding/
│   ├── EmbeddingServerApplication.java
│   ├── config/EmbeddingProperties.java   # 配置项
│   ├── core/QwenTokenizer.java           # 纯 Java BPE 分词器
│   ├── core/EmbeddingEngine.java         # ONNX 会话、推理、池化与归一化
│   └── web/
│       ├── EmbeddingController.java      # /v1/embeddings、/health、DTO
│       └── AccessGuardFilter.java        # API Key 与请求体大小保护
├── src/main/resources/application.yml
├── .gitignore
├── pom.xml
└── README.md
```

## 提交到 GitHub

`.gitignore` 已排除 `target/`、`logs/` 与 `models/`，仓库体积保持在 MB 以内。克隆后按「模型下载」一节拉取模型即可运行。

## 许可证

未指定许可证，如需开源请按项目要求补充 `LICENSE`。
