# 🤖 AI Engineer RAG Chatbot

A **production-ready RAG chatbot** for AI/ML topics and mock interviews, built from scratch with:

- 🔍 **RAG Pipeline** — FAISS vector store + OpenAI embeddings
- 🧠 **OpenAI GPT-4o** — for generating answers
- 📊 **RAGAS Evaluation** — automated quality metrics
- 🌐 **FastAPI backend** + **Streamlit frontend**
- 🐳 **Docker** + ☁️ **AWS deployment** guide
- 📚 **20 curated AI/ML papers** auto-downloaded from ArXiv

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│              (Q&A mode + Mock Interview mode)            │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP / SSE
┌─────────────────────▼───────────────────────────────────┐
│                    FastAPI Backend                       │
│  POST /chat  │  POST /chat/stream  │  POST /interview   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   RAG Pipeline                          │
│                                                         │
│  Query → Embed → FAISS Search → Build Context → LLM    │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
┌──────────▼──────────┐    ┌──────────▼──────────────────┐
│    FAISS Index      │    │   OpenAI GPT-4o              │
│  (text-embedding-   │    │   (gpt-4o / gpt-4-turbo)    │
│    3-small)         │    │                             │
└──────────┬──────────┘    └─────────────────────────────┘
           │
┌──────────▼──────────┐
│   PDF Documents     │
│  (20 ArXiv papers)  │
│  RAG, Agents, LLMs  │
│  Transformers, LoRA │
└─────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

```bash
python >= 3.10
pip
An OpenAI API key (https://platform.openai.com/api-keys)
```

### 2. Install dependencies

```bash
git clone https://github.com/your-repo/rag-production-chatbot.git
cd rag-production-chatbot

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 4. Download AI research papers (20 curated papers)

```bash
python scripts/download_pdfs.py
```

Papers include: RAG survey, Attention is All You Need, GPT-3/4, Llama 2, ReAct,
Toolformer, LoRA, QLoRA, Chain-of-Thought, RAGAS, and more.

### 5. Build the vector store

```bash
python scripts/ingest_data.py
```

This embeds all PDFs using OpenAI `text-embedding-3-small` and builds a FAISS index.

### 6. Start the API

```bash
python app/main.py
# API available at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### 7. Start the Streamlit UI (new terminal)

```bash
streamlit run frontend/streamlit_app.py
# Open http://localhost:8501
```

---

## Features

### 💬 Q&A Mode

Ask any AI/ML engineering question. The chatbot:
1. Embeds your query with OpenAI
2. Retrieves the top-5 most relevant chunks from the vector store
3. Passes retrieved context + chat history to GPT-4o
4. Returns a grounded, cited answer

**Example questions:**
- "Explain the difference between LoRA and full fine-tuning"
- "How does FAISS perform approximate nearest neighbour search?"
- "What are the key RAGAS metrics and what do they measure?"
- "Walk me through the ReAct agent framework"

### 🎤 Mock Interview Mode

Simulates a real AI Engineer technical interview:
- Choose your level: Junior / Mid / Senior
- Choose a topic: RAG, LLMs, Agents, MLOps, etc.
- Get one question at a time with a 1–10 score
- Detailed feedback + follow-up questions
- Full assessment after 5 questions

### 📊 RAGAS Evaluation

```bash
python scripts/evaluate_rag.py
```

Evaluates your RAG system on 8 curated AI/ML test cases:

| Metric              | Description                                    |
|---------------------|------------------------------------------------|
| Faithfulness        | Answer grounded in retrieved context (0–1)    |
| Answer Relevancy    | Answer addresses the question (0–1)           |
| Context Precision   | Retrieved chunks are relevant (0–1)           |
| Context Recall      | Context covers the necessary info (0–1)       |
| Answer Correctness  | Factual accuracy vs ground truth (0–1)        |

---

## API Reference

### `POST /chat`
```json
{
  "question": "What is RAG?",
  "chat_history": [
    {"role": "user", "content": "Tell me about transformers"},
    {"role": "assistant", "content": "..."}
  ],
  "top_k": 5
}
```

Response:
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) combines...",
  "sources": ["rag_knowledge_intensive_nlp.pdf"],
  "retrieval_scores": [0.92, 0.88, 0.85, 0.81, 0.79],
  "latency_ms": 1243.5
}
```

### `POST /chat/stream`
Same request as `/chat`, returns Server-Sent Events for streaming responses.

### `POST /interview`
Same request as `/chat`, uses the mock interview system prompt.

### `GET /health`
```json
{
  "status": "healthy",
  "pipeline_ready": true,
  "doc_count": 1842,
  "model": "gpt-4o"
}
```

---

## Configuration

All settings are in `.env` (see `.env.example`):

| Variable              | Default                  | Description                         |
|-----------------------|--------------------------|-------------------------------------|
| `OPENAI_API_KEY`      | **required**             | Your OpenAI API key                 |
| `LLM_MODEL`           | `gpt-4o`                 | OpenAI chat model                   |
| `EMBEDDING_MODEL`     | `text-embedding-3-small` | OpenAI embedding model              |
| `CHUNK_SIZE`          | `1000`                   | Characters per document chunk       |
| `CHUNK_OVERLAP`       | `200`                    | Overlap between chunks              |
| `TOP_K_RETRIEVAL`     | `5`                      | Number of chunks to retrieve        |
| `LLM_TEMPERATURE`     | `0.0`                    | LLM randomness (0 = deterministic)  |

---

## Project Structure

```
rag_production_chatbot/
├── app/
│   ├── main.py              # FastAPI backend (entry point)
│   ├── rag_pipeline.py      # Core RAG orchestration
│   ├── embeddings.py        # OpenAI embeddings + caching
│   ├── vector_store.py      # FAISS vector store
│   ├── document_processor.py # PDF loading & chunking
│   ├── llm.py               # OpenAI LLM client + prompts
│   ├── evaluation.py        # RAGAS evaluation module
│   └── logger.py            # Loguru logging setup
│
├── frontend/
│   └── streamlit_app.py     # Streamlit chat UI
│
├── scripts/
│   ├── download_pdfs.py     # Download 20 ArXiv papers
│   ├── ingest_data.py       # Build FAISS vector store
│   └── evaluate_rag.py      # Run RAGAS evaluation
│
├── tests/
│   └── test_rag.py          # Unit & integration tests
│
├── docker/
│   ├── Dockerfile           # API container
│   ├── Dockerfile.frontend  # Frontend container
│   └── docker-compose.yml   # Full stack compose
│
├── aws/
│   └── deployment_guide.md  # EC2, ECS, Lambda guides
│
├── data/
│   ├── pdfs/                # Downloaded PDFs (gitignored)
│   └── vectorstore/         # FAISS index (gitignored)
│
├── config.py                # Centralised settings
├── requirements.txt
├── .env.example
└── README.md
```

---

## Docker

```bash
# Build and start all services
cp .env.example .env  # fill in OPENAI_API_KEY

# Run ingestion first
docker compose -f docker/docker-compose.yml run --rm api \
  python scripts/download_pdfs.py

docker compose -f docker/docker-compose.yml run --rm api \
  python scripts/ingest_data.py

# Start services
docker compose -f docker/docker-compose.yml up -d

# View logs
docker compose -f docker/docker-compose.yml logs -f
```

---

## AWS Deployment

See [`aws/deployment_guide.md`](aws/deployment_guide.md) for:
- **Option 1**: EC2 (quickest, dev/staging)
- **Option 2**: ECS Fargate (production, recommended)
- **Option 3**: Lambda + API Gateway (serverless)
- Auto-scaling, CloudWatch monitoring, CI/CD with GitHub Actions

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=app --cov-report=html
```

---

## Adding More Documents

```bash
# Place any PDF in data/pdfs/
cp your_paper.pdf data/pdfs/

# Rebuild the vector store
python scripts/ingest_data.py --rebuild
```

---

## Knowledge Base Papers

The chatbot is pre-loaded with these papers:

| Topic                    | Paper                                               |
|--------------------------|-----------------------------------------------------|
| Transformers             | Attention Is All You Need                           |
| Transformers             | BERT                                               |
| LLMs                     | GPT-3 (Language Models are Few-Shot Learners)      |
| LLMs                     | Llama 2                                            |
| LLMs                     | GPT-4 Technical Report                             |
| RAG                      | RAG for Knowledge-Intensive NLP Tasks              |
| RAG                      | Self-RAG                                           |
| RAG                      | Corrective RAG (CRAG)                              |
| RAG                      | Advanced RAG Survey                                |
| Agents                   | ReAct: Synergizing Reasoning and Acting            |
| Agents                   | Toolformer                                         |
| Agents                   | Generative Agents                                  |
| Agents                   | LLM-based Autonomous Agents Survey                 |
| Prompt Engineering       | Chain-of-Thought Prompting                         |
| Prompt Engineering       | Tree of Thoughts                                   |
| Fine-tuning              | LoRA                                               |
| Fine-tuning              | QLoRA                                              |
| Evaluation               | RAGAS                                              |
| Instruction Tuning       | FLAN (Finetuned Language Models)                   |

---

## Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| LLM              | OpenAI GPT-4o                       |
| Embeddings       | OpenAI text-embedding-3-small       |
| Vector Store     | FAISS (CPU)                         |
| Document Loading | pypdf + RecursiveTextSplitter       |
| Evaluation       | RAGAS                               |
| API Backend      | FastAPI + Uvicorn                   |
| Frontend         | Streamlit                           |
| Containerisation | Docker + Docker Compose             |
| Cloud            | AWS (EC2 / ECS Fargate / Lambda)    |
| Logging          | Loguru                              |
| Testing          | pytest                              |
