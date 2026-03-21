# 基于图检索增强生成的跨学科知识发现研究

中文版说明。当前内容与 `README.md` 保持同一口径，聚焦你的论文实现版本，而不是上游通用 GraphRAG 框架介绍。

## 项目简介
本项目是一个面向跨学科知识发现的 GraphRAG 研究原型系统，围绕论文《基于图检索增强生成的跨学科知识发现研究》展开实现。系统以科学文献为对象，通过“大语言模型知识抽取 + 知识图谱构建 + 图检索增强问答 + 评测分析”的完整链路，支持从非结构化学术文本中发现潜在的跨学科关联。

当前仓库的实现重点，已经从通用 GraphRAG 框架进一步收敛到以下研究目标：

- 面向跨学科科学文献构建结构化知识图谱
- 支持基于图谱的复杂问题检索、分解与推理
- 以 AIGC/大语言模型教育应用文献为主要实验样本
- 提供知识图谱构建评测与问答评测两套实验工具链
- 提供 Web 原型界面，方便上传数据、构图、问答与可视化分析

## 研究定位
结合论文当前内容，本项目聚焦的是“跨学科科学知识发现”而不是通用聊天问答。核心问题包括：

1. 如何基于文献元数据自动构建跨学科科学知识图谱。
2. 如何利用 GraphRAG 提升跨学科知识发现的效率、覆盖率与可解释性。
3. 如何围绕知识图谱构建质量和问答质量设计评估体系，并支持不同模型之间的对比实验。

当前研究样本以“大语言模型在教育领域中的应用研究”为主，因此你会在仓库中看到 `AIGC-EDU`、`AIGC-EDU-test` 等数据集，以及围绕它们组织的构图与评测结果。

## 当前系统能力

### 1. 文献上传与数据集管理
- 支持通过 Web 界面上传 `.txt`、`.md`、`.json`、`.pdf`、`.docx`、`.doc`
- 自动生成 `data/uploaded/<dataset_name>/corpus.json`
- 支持数据集列表、删除、重建图谱、上传自定义 schema
- 内置 `demo` 数据集，便于快速验证流程

### 2. 知识图谱构建
- 主构图入口：`main.py` 与 `backend.py`
- 核心构图模块：`models/constructor/kt_gen.py`
- 支持基于 schema 的实体、关系、属性抽取
- 支持跨文档连接、社区检测、分块审计与图谱输出
- 输出落盘到 `output/graphs/`、`output/chunks/`、`output/logs/`

### 3. 图检索增强问答
- 支持 `agent` 与 `noagent` 两种模式
- `agent` 模式包含问题分解、子问题处理、迭代检索与推理
- 检索模块基于图谱、FAISS 与 chunk 证据共同工作
- 支持图谱可视化、问答过程展示与检索结果回显

### 4. 评测与实验
- `eval/kg_eval/`：知识图谱构建质量评测
- `eval/rag_eval/`：问答质量评测
- `eval/utils/sample_kg_eval_stratified.py`：分层随机抽样
- 支持 gold 生成、候选模型对比、跨文档审阅模板导出、Markdown 报告生成

## 技术路线与实现映射
论文中的三条主线，在代码中的对应关系如下：

| 研究主线 | 对应实现 |
| --- | --- |
| 科学文献知识抽取与语义融合 | `models/constructor/kt_gen.py`、`utils/document_parser.py`、`schemas/` |
| 图检索增强生成与跨学科问答 | `models/retriever/agentic_decomposer.py`、`models/retriever/enhanced_kt_retriever.py`、`backend.py` |
| 评估体系构建与模型比较 | `eval/kg_eval/`、`eval/rag_eval/`、`test_kg_eval.py` |

## 项目结构
下面是当前仓库中最重要的目录与文件：

```text
youtu-graphrag/
├─ backend.py                  # FastAPI 后端与 Web 接口
├─ main.py                     # 命令行主入口：构图 / 检索 / 评测前运行
├─ config/
│  ├─ base_config.yaml         # 主配置文件
│  └─ config_loader.py         # 配置加载与路径标准化
├─ frontend/
│  ├─ index_new.html           # 前端页面
│  ├─ script.js                # 前端交互逻辑
│  └─ style.css                # 页面样式
├─ models/
│  ├─ constructor/
│  │  └─ kt_gen.py             # 知识图谱构建核心
│  └─ retriever/
│     ├─ agentic_decomposer.py # 问题分解
│     ├─ enhanced_kt_retriever.py
│     └─ faiss_filter.py       # FAISS 检索
├─ utils/
│  ├─ document_parser.py       # 多格式文档解析
│  ├─ call_llm_api.py          # LLM 调用封装
│  ├─ dataset_audit.py         # 数据集一致性审计
│  ├─ tree_comm.py             # 社区检测相关工具
│  └─ paths.py                 # 仓库根路径解析
├─ data/
│  ├─ demo/                    # demo 数据集
│  └─ uploaded/                # Web 上传后的数据集
├─ schemas/                    # 数据集 schema
├─ output/
│  ├─ graphs/                  # 图谱输出
│  ├─ chunks/                  # chunk 输出
│  └─ logs/                    # 运行日志
├─ eval/
│  ├─ kg_eval/                 # 构图评测
│  ├─ rag_eval/                # 问答评测
│  ├─ utils/                   # 评测辅助脚本
│  └─ results/                 # 评测结果
├─ test_kg_eval.py
└─ test_sample_kg_eval_stratified.py
```

## 环境要求
- Python 3.10+
- 推荐使用虚拟环境
- CPU 可运行；如需更快嵌入或构图，可自行扩展 GPU 环境
- 若需要更好的文档解析兼容性，建议准备 Java 运行时

主要依赖见：

- `requirements.txt`
- `requirements-server.txt`
- `requirements-optional.txt`

## LLM 环境变量
项目支持按任务拆分模型配置。

### 通用默认配置
```env
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_key
```

### 可选：构图与问答分开配置
```env
KG_LLM_MODEL=qwen3-max
KG_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
KG_LLM_API_KEY=your_key

RAG_LLM_MODEL=deepseek-chat
RAG_LLM_BASE_URL=https://api.deepseek.com
RAG_LLM_API_KEY=your_key
```

如使用 Azure OpenAI，可补充：

```env
OPENAI_PROVIDER=azure
API_VERSION=2025-01-01-preview
```

评测模块使用单独的环境文件：

- `eval/.env`
- `eval/rag_eval/.env`

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

如需处理中文文本，建议安装中文 spaCy 模型：

```bash
python -m spacy download zh_core_web_lg
```

### 2. 配置环境变量
可直接参考根目录的 `.env.example`。

### 3. 启动 Web 原型
```bash
python backend.py
```

启动后访问：

```text
http://localhost:8000
```

### 4. 使用命令行构图/检索
```bash
python main.py --config config/base_config.yaml --datasets demo
```

如只想运行特定流程，可结合 `triggers` 使用 `--override`：

```bash
python main.py --datasets demo --override "{\"triggers\": {\"constructor_trigger\": true, \"retrieve_trigger\": false}}"
```

## Web 使用流程
当前前端界面主要支持以下流程：

1. 上传文档并生成数据集
2. 为数据集上传自定义 schema
3. 构建知识图谱
4. 查看图谱可视化
5. 选择数据集进行研究问答
6. 对已有数据集执行重建或删除

后端核心接口位于 `backend.py`，包括：

- `GET /api/datasets`
- `POST /api/upload`
- `POST /api/construct-graph`
- `POST /api/ask-question`
- `GET /api/graph/{dataset_name}`
- `GET /api/dataset-audit/{dataset_name}`

## 配置说明
主配置文件是 `config/base_config.yaml`，当前默认设置体现了你的研究场景：

- `active_dataset: demo`
- `construction.mode: agent`
- `nlp.spacy_model: zh_core_web_lg`
- `datasets.demo` 指向 `data/demo/`
- 输出目录统一落到 `output/`

可优先关注以下几组参数：

- `construction.*`：构图、分块、跨文档连接、并发控制
- `retrieval.*`：检索参数、召回路径、缓存目录
- `triggers.mode`：`agent` / `noagent`
- `datasets.*`：语料、问答集、schema、图谱输出位置

## 评测流程

### 1. 知识图谱构建评测
配置文件：

- `eval/kg_eval/config.yaml`

常用命令：

```bash
python -m eval.kg_eval.run generate_gold
python -m eval.kg_eval.run run
python -m eval.kg_eval.run cross_doc_review
```

该模块用于：

- gold 标注草稿生成
- 候选抽取结果与 gold 对比
- 图结构与跨文档连接质量分析
- 自动生成评测报告

### 2. 问答评测
配置文件：

- `eval/rag_eval/config.yaml`

常用命令：

```bash
python -m eval.rag_eval.run
python -m eval.rag_eval.run --dataset AIGC-EDU-test --qa-mode agent
```

该模块用于：

- 读取问题集
- 调用当前 GraphRAG 流程生成答案
- 使用评审模型从准确性、完整性、逻辑性、可解释性、跨学科性等维度打分
- 生成结构化结果与汇总报告

## 测试
当前仓库中已经存在的基础测试包括：

```bash
python test_kg_eval.py
python test_sample_kg_eval_stratified.py
```

如果只想验证后端能否正常提供数据集和页面，也可以直接运行：

```bash
python backend.py
```

## 当前更贴近论文的理解方式
如果要用一句话概括当前项目，可以这样理解：

> 这是一个面向跨学科科学文献知识发现的 GraphRAG 实验平台，重点解决“如何把学术文本转成图、如何基于图做问答、如何对构图和问答效果进行评测”这三个问题。

相比原始通用 GraphRAG 介绍，当前仓库更强调：

- 学术文献与跨学科知识发现
- 教育场景下的大语言模型相关研究样本
- 评测可复现性
- Web 原型与实验工具的结合

## 相关文档
- `README.md`：当前主 README
- `README-JA.md`：日文版说明
- `FULLGUIDE-CN.md`：较完整的中文使用说明
- `FULLGUIDE.md`：英文完整说明
- `AGENTS.md`：面向开发代理的仓库协作说明

## 说明
- 当前 README-CN 以“你的研究实现版本”为中心编写，不再沿用上游通用项目表述。
- 如果后续你准备将仓库公开发表，建议补充：
  - 研究数据来源说明
  - 模型选择依据
  - 复现实验表格
  - 典型问答案例
  - 论文引用与版本对应关系
