# 🚀 Academic GraphRAG Full Guide

<div align="center">
  <img src="assets/logo.png" alt="Logo" width="100">
  
  **Complete Guide from Installation to Usage: A GraphRAG Research Prototype for Interdisciplinary Knowledge Discovery**
  
  [⬅️ Back to Home](README.md) | [🌐 中文版指南](FULLGUIDE.md)
</div>

---

## 📋 Table of Contents
- <a href="#environment-preparation">⚙️ Environment Preparation</a>
- <a href="#web-interface-experience">💻 Web Interface Usage Flow</a>
- <a href="#command-line-usage">🛠️ Command Line Usage (CLI)</a>
- <a href="#evaluation-workflow">⚖️ Evaluation Workflow</a>
- <a href="#configuration-details">⚙️ Advanced Configuration Analysis</a>
- <a href="#troubleshooting">🔧 Troubleshooting and Optimization</a>

---

<a id="environment-preparation"></a>
## ⚙️ Environment Preparation

This project supports Docker deployment and local Python environment installation. We recommend using a **virtual environment** to avoid dependency conflicts.

### 1. Basic Installation (Local)
```bash
# Clone the project
git clone https://github.com/Morris1029/Academic_graphRAG.git
cd academic-graphrag

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-server.txt

# Download Chinese NLP model (Core for academic document processing)
python -m spacy download zh_core_web_lg
```

### 2. Core Environment Variables Configuration (`.env`)
Copy `.env.example` and rename it to `.env`. The project supports assigning different LLM models by task:

```env
# General Default Configuration
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_key

# [ADVANCED] Configure building and questioning separately (Recommended)
KG_LLM_MODEL=qwen-max
KG_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
KG_LLM_API_KEY=sk-xxxx

RAG_LLM_MODEL=deepseek-reasoner
RAG_LLM_BASE_URL=https://api.deepseek.com
RAG_LLM_API_KEY=sk-xxxx
```

### 3. Multi-format Document Parsing Support (Optional)
If you need to process complex formats like `.pdf`, `.docx`, `.doc`, etc., we recommend running the environment initialization script:
```bash
chmod +x setup_env.sh
./setup_env.sh
```
*This script will automatically detect and attempt to install Java (for Tika support) and Antiword (for .doc support).*

---

<a id="web-interface-experience"></a>
## 💻 Web Interface Usage Flow

The Web interface provides an intuitive way to manage datasets and visualize the graph construction process.

<div align="center">
  <img src="assets/homepage.png" alt="Homepage Overview">
</div>

### Start Service
```bash
# Using startup script
chmod +x start.sh
./start.sh
# Or run backend directly
python backend.py
```
Access URL: `http://localhost:8000`

### Core Operation Steps
1.  **Data Upload**: In the "Upload Documents" tab, you can drag and drop documents like `.pdf`, `.txt`, `.json`, etc. The system will automatically identify the encoding and format it.
    <div align="center"><img src="assets/dataupload.png" alt="Data Upload Panel"></div>
2.  **Schema Definition**: A general Academic Schema is associated with each dataset by default. If you have special needs, you can upload a custom Schema separately after uploading the dataset.
3.  **Graph Construction**: Click "Construct Graph" in the "Knowledge Tree Visualization" tab. Since building graphs involves many LLM calls, it is recommended to adjust `max_concurrent_llm_requests` in `base_config.yaml` based on your API rate limits.
    <div align="center"><img src="assets/Kgvisual.png" alt="Visual Construction"></div>
4.  **Research Q&A**: Enter the "Research Q&A" interface and select a constructed dataset. In `agent` mode, the system will automatically decompose questions and perform multi-round iterative retrieval.
    <div align="center"><img src="assets/reseachQA.png" alt="Q&A Panel"></div>

---

<a id="command-line-usage"></a>
## 🛠️ Command Line Usage (CLI)

The CLI is suitable for executing batch tasks or running in a server environment.

### 1. Full Process Start
```bash
# Perform Construction + Retrieval for the demo dataset
python main.py --datasets demo
```

### 2. Behavior Customization (Override)
You can modify any setting in `base_config.yaml` in real-time via the `--override` parameter:
```bash
# Build graph only, no retrieval (disable retrieval trigger)
python main.py --datasets demo --override "{\"triggers\": {\"constructor_trigger\": true, \"retrieve_trigger\": false}}"

# Use basic mode (noagent) to improve response speed
python main.py --datasets demo --override "{\"triggers\": {\"mode\": \"noagent\"}}"
```

---

<a id="evaluation-workflow"></a>
## ⚖️ Evaluation Workflow

As a research prototype, this project provides independent evaluation frameworks located in the `eval/` directory.

### 1. Knowledge Graph Construction Evaluation (`kg_eval`)
Measures the consistency of entities/triples extracted by different LLMs with Gold data.
```bash
# 1. Generate Gold data draft
python -m eval.kg_eval.run generate_gold
# 2. Start batch extraction and comparative evaluation
python -m eval.kg_eval.run run
```

### 2. QA Quality Evaluation (`rag_eval`)
Based on LLM-as-a-Judge, scores generated answers multi-dimensionally.
```bash
# Execute fully automated answering and scoring based on the AIGC-EDU-test dataset
python -m eval.rag_eval.run --dataset AIGC-EDU-test --qa-mode agent
```
*Evaluation results will be saved in the `eval/results/` directory.*

---

<a id="configuration-details"></a>
## ⚙️ Advanced Configuration Analysis (`base_config.yaml`)

| Parameter Group | Core Parameter | Description |
| :--- | :--- | :--- |
| **construction** | `chunk_size` | Size of document segments. 1000 is recommended for academic literature to maintain context coherence. |
| | `mode: agent` | `agent` mode uses CoT for finer extraction, slower but higher quality. |
| | `max_workers` | Local CPU concurrency (controls parsing/IO). |
| **retrieval** | `top_k: 20` | Initial number of Chunks and triples recalled. |
| | `similarity_threshold` | Similarity threshold for FAISS search. |
| **embeddings** | `device: cpu` | If video memory is sufficient, can be set to `cuda` to accelerate vectorization. |

---

<a id="troubleshooting"></a>
## 🔧 Troubleshooting and Optimization

### 1. FAISS Error `Segmentation fault: 11`
When processing large-scale nodes (e.g., 5000+), an OpenMP concurrency conflict may cause a crash.
**Solution**: Execute `export OMP_NUM_THREADS=1` before running any script.

### 2. Construction Failure or Incomplete LLM Response
- Check `llm_timeout_seconds` in `config/base_config.yaml`. Recommend 90s or more for complex extraction tasks.
- If the network is unstable, adjust `retry_attempts` to control automatic retries.

### 3. Data Consistency Check
If you suspect issues during graph construction, run:
```bash
# Check Source/Chunk/Graph consistency for a specific dataset
python backend.py --dataset-audit <name>
```

---

<div align="center">
  
  **🌟 Sincerely welcome contributions or feedback 🌟**
  
  [⬅️ Back to Home](README.md)
  
</div>