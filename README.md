# Local RAG & Declarative LLM Workflows (`LLM-RAG-Implementation`)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![DSPy](https://img.shields.io/badge/DSPy-Framework-blueviolet?style=for-the-badge)](https://github.com/stanfordnlp/dspy)

A complete end-to-end framework for building, evaluating, and deploying **Local Retrieval-Augmented Generation (RAG)** systems and **Declarative LLM Workflows**. 

This repository covers the full RAG lifecycle: from raw PDF parsing, text chunking, and embedding generation using Hugging Face models (`all-mpnet-base-v2`), to GPU-accelerated similarity search, dynamic prompt formatting, local/cloud LLM inference (`Phi-3.5`, `Gemma`, `GPT-4o-mini`), interactive Streamlit web UI deployment, and advanced agentic evaluation using DSPy.

---

## 💡 Overview & Key Architecture

Traditional Large Language Models (LLMs) often suffer from knowledge cutoffs and hallucinations when queried on domain-specific documents. This project implements a privacy-focused, zero-cost, local-first RAG pipeline that operates directly on standard consumer GPU hardware.

```
                    +------------------------------------+
                    |  Document Ingestion & Chunking     |
                    | (PyMuPDF + spaCy Sentence Split)   |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |   Vector Embedding Generation      |
                    |   (all-mpnet-base-v2, 768-dim)     |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    | PyTorch Tensor Vector Store (GPU)  |
                    +-----------------+------------------+
                                      |
  +------------------+                v                +-------------------+
  | User Query (UI/  | ----> Top-K Similarity Search   | Context-Augmented |
  | Streamlit App)   |       (Dot Product / Top-K) --> | Prompt Generation |
  +------------------+                                 +---------+---------+
                                                                 |
                                                                 v
                                                       +-------------------+
                                                       | LLM Generation    |
                                                       | (Phi-3.5 / Gemma) |
                                                       +-------------------+
```

---

## ✨ Features

- 📑 **PDF Ingestion & Smart Text Chunking**: Extracts text from large PDFs (e.g., 1,200+ page textbooks) using PyMuPDF (`fitz`) and splits them into semantically coherent sentence groups via spaCy.
- ⚡ **GPU Vector Indexing**: Encodes document passages into 768-dimensional dense vectors using `sentence-transformers/all-mpnet-base-v2` and performs sub-millisecond similarity scoring using PyTorch tensor operations.
- 🎯 **Context-Augmented Few-Shot Generation**: Augments input queries with retrieved top-$k$ context chunks and formatted few-shot prompt templates to guarantee factual accuracy and eliminate hallucinations.
- 💻 **Interactive Streamlit Web Interface**: Complete web application (`streamlit/main.py`) featuring response streaming, session state management, and real-time query retrieval.
- 🛠️ **Declarative RAG & ReAct Agents (DSPy)**: Incorporates Stanford's DSPy framework for programmatic prompt optimization, multi-hop reasoning, and tool-augmented ReAct agents (`PythonInterpreter`, `ColBERTv2`).
- 🔍 **Embedding Knowledge Base & Intent Classifier**: Intent recognition engine (`debugger embeddings/`) mapping incoming user prompts to structured knowledge base tags.

---

## 📁 Repository Structure

```
.
├── 00-simple-local-rag.ipynb      # Main step-by-step notebook: RAG from scratch
├── Learn RAG.ipynb                 # Interactive learning notebook & pipeline walkthrough
├── requirements.txt               # Project dependencies
├── human-nutrition-text.pdf       # Sample 1,200-page dataset (Human Nutrition textbook)
├── text_chunks_and_embeddings.csv  # Pre-computed document embeddings CSV
├── streamlit/                     # Full-stack Web Application
│   ├── main.py                    # Streamlit frontend & chat interface
│   ├── rag.py                     # Retrieval and HF Inference client integration
│   └── read_embeddings.py         # Fast CSV-to-GPU tensor vector loader
├── docker-dataset/                # Advanced DSPy & Benchmark Workflows
│   ├── RAG DSPy.ipynb             # Declarative RAG module with DSPy & FAISS
│   ├── SOP Evaluation.ipynb       # ReAct Agents & SOP evaluation pipelines
│   └── Dockerfiles-dataset.ipynb  # Tech corpus QA benchmark datasets
├── debugger embeddings/           # Semantic Intent Recognition Module
│   └── Embedding Knowledge Base.ipynb # Intent classification via vector distance
└── images/                        # Architecture diagrams & visual assets
```

---

## 🛠️ Tech Stack & Dependencies

- **Core Language**: Python 3.10+
- **Deep Learning Framework**: PyTorch (CUDA supported)
- **Embedding & NLP Models**: `sentence-transformers` (`all-mpnet-base-v2`), `spaCy`, `PyMuPDF`
- **LLMs Supported**: Microsoft `Phi-3.5-mini-instruct`, Google `Gemma-2B/7B`, OpenAI `GPT-4o-mini`
- **Agent & Prompt Optimization**: DSPy, FAISS, ColBERTv2
- **Web UI & Streaming**: Streamlit
- **Data Engineering**: Pandas, NumPy, Tqdm

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.10 or higher installed. An NVIDIA GPU with CUDA support is recommended for local acceleration.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/KareemEl-Giushy/LLM-RAG-Implementation.git
cd LLM-RAG-Implementation

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for text chunking
python -m spacy download en_core_web_sm
```

---

## 🏃 Running the Project

### Option A: Interactive Notebooks (End-to-End Walkthrough)

Launch Jupyter Notebook to explore the RAG implementation step-by-step:

```bash
jupyter notebook 00-simple-local-rag.ipynb
```

Key stages inside the notebook:
1. PDF Text Extraction & Cleaning
2. Sentence Segmentation & Overlapping Chunk Creation
3. Vector Embedding Generation & GPU Indexing
4. Top-K Vector Search (`torch.topk` & Dot Product Similarity)
5. Hugging Face / PyTorch Local LLM Generation

### Option B: Streamlit Web Application

Run the interactive chat web UI:

```bash
cd streamlit
streamlit run main.py
```

Open your browser at `http://localhost:8501` to start asking questions against the indexed document knowledge base in real-time.

---

## 📊 Evaluation & DSPy Workflows

Inside `docker-dataset/`, DSPy modules are evaluated for automated prompt compilation and agent reasoning:

- **DSPy RAG Chain**: Replaces manual prompt construction with declarative module signatures (`context, question -> response`).
- **ReAct Agent**: Integrates dynamic tools like `PythonInterpreter` and `ColBERTv2` vector search to solve complex multi-step technical queries.

---

## 🤝 Contributing

Contributions, feedback, and issue reports are welcome! Feel free to open a PR or submit an issue to improve pipeline performance, add vector database adapters (Chroma/Qdrant), or extend evaluation benchmarks.

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).