# HarnessEngineeringTest

围绕 **Qwen3-8B (Instruct, 非思考模式)** 构建的单文件 LLM Harness：在每次 `call_llm` 的 prompt 不超过 **2048 token** 的硬约束下，完成 exact-match 文本分类。

模型权重保持冻结，所有"学习"沉淀在 Harness 的外部状态中（增量倒排索引 + 字符 n-gram + prompt 装配模板 + 输出兜底）。

---

## 项目结构

```
.
├── src/                     # Python 源码
│   ├── solution.py          # MyHarness 业务实现（核心，单文件交付）
│   ├── harness_base.py      # Harness 基类（外部接口）
│   ├── llm_client.py        # OpenAI Compatible 客户端 + Qwen tokenizer 计数
│   └── run.py               # 本地评测脚本（多线程 ThreadPoolExecutor）
├── data/                    # DEV 集
│   ├── train_dev.jsonl      # 训练流（231 条，77 类银行客服意图）
│   └── test_dev.jsonl       # 验证集（539 条）
├── tokenizer/               # 本地 Qwen tokenizer（精确 token 计数）
├── tests/                   # MyHarness 单元测试（pytest）
├── report/                  # 探索报告（report.tex / report.pdf / figures）
├── requirements.txt         # 运行依赖：openai / transformers / numpy
├── requirements-dev.txt     # 开发依赖：pytest
├── pyproject.toml           # 项目元数据 + pytest 配置
├── .env.example             # LLM 接入环境变量模板
├── LICENSE                  # MIT
└── README.md
```

## 设计要点

- **单文件交付**：业务全部位于 `solution.py` 的 `MyHarness` 类，imports 限制为 Python 标准库 + NumPy + 给定的 `harness_base`。
- **检索 + LLM 决策**：`update` 阶段维护 word 与 character n-gram 两套增量倒排索引；`predict` 阶段用 min-max 归一化的 BM25 + 字符 n-gram 余弦混合检索拿到候选样本。
- **四节式 prompt**：`[ALLOWED LABELS] / [EXAMPLES] / [INPUT] / label:` 由显式 sentinel 隔开，`label:` 充当输出锚点。
- **抗注入**：把入参 `text` 当作不可信数据，用 `<example>` / `<text>` 标签隔离；system prompt 明确声明忽略文本内的角色篡改。
- **输出恢复**：清洗 → 精确匹配 → 子串匹配 → Jaccard 近似四层兜底，从模型输出中恢复合法标签。
- **自适应 token 预算**：根据 query 长度动态分配 labels / examples / input 的 token 预算，并在长 query 上做防御性裁剪，避免评测系统从尾部截断 `label:` 锚点。

详细设计、消融实验与错例分析见 [`report/report.pdf`](report/report.pdf)。

## 快速上手

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 LLM endpoint（推荐用环境变量；可参考 .env.example）
export LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
export LLM_API_KEY="your_api_key"
export LLM_MODEL="qwen3-8b"

# 3. 连通性测试
python src/llm_client.py

# 4. 完整本地评测（4 轮平均，从项目根目录运行）
python src/run.py

# 5. 快速迭代
python src/run.py --runs 1
python src/run.py --workers 50      # 提升并发，注意上游 API 速率限制
```

## 单元测试

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## 接口约定

`MyHarness` 继承 `harness_base.Harness`，由运行时注入：

| 属性 | 用途 |
|---|---|
| `self.call_llm(messages)` | OpenAI 风格 messages → 回复文本。prompt 超过 `max_prompt_tokens` 会从尾部截断。 |
| `self.count_tokens(text)` | 单条文本的 token 数。 |
| `self.count_messages_tokens(messages)` | messages 列表中所有 `content` 拼接后的总 token 数。 |
| `self.max_prompt_tokens` | prompt token 预算（默认 2048）。 |
| `self.memory` | `list[tuple[str, str]]`，由 `update` 累积。 |

## 约束

- 仅依赖 Python 标准库、NumPy、`harness_base`，不引入额外第三方库。
- 不进行任何磁盘 I/O；运行时状态全部维护在 `self.memory` 与实例属性中。
- LLM 调用 prompt token 数严格控制在 `max_prompt_tokens` 以内（实测 616 条评测样本无一条溢出 2048）。
