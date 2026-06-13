# 🐍 Python Q&A Assistant

An AI-powered question-answering system for Python programming queries, grounded in Stack Overflow data using a production-grade **RAG (Retrieval-Augmented Generation)** pipeline.

---

## 🏗️ Architecture

```
User Question
     ↓
FastAPI  POST /ask
     ↓
LRU Cache check  ──── HIT ────────────────────────→ Cached Response
     ↓ MISS
Query Rewriting  (Groq)
     ↓
Multi-Query Generation  (Groq → 3 variants)       
     ↓
Async Parallel Retrieval  (per variant)           
  ├── FAISS semantic search  (HyDE embedding)
  └── BM25 keyword search
     ↓
RRF Fusion  (all variants merged)
     ↓
Cohere Rerank  → top-5
     ↓
Confidence Check  (fallback if score < threshold)
     ↓
Context Compression  (Groq strips noise)           
     ↓
Groq llama-3.3-70b  → Grounded Answer
     ↓
LRU Cache store  →  Response
```

---

## 🔧 Full Stack

| Component | Tool | Purpose |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | Semantic chunk + query embedding |
| Vector Store | FAISS (IndexFlatIP) | Fast approximate nearest-neighbour search |
| Keyword Search | BM25Okapi (rank-bm25) | Exact keyword / error message matching |
| Reranker | Cohere Rerank v3 | Cross-encoder precision filter |
| LLM | Groq llama-3.3-70b | Query rewrite, HyDE, compression, generation |
| API | FastAPI | REST endpoints |
| Cache | In-memory LRU (512 slots) | Instant repeat-query responses |

---

## 📁 Project Structure

```
python-qa-assistant/
├── app/
│   ├── main.py          # FastAPI app — /ask, /health, /cache/stats
│   ├── rag.py           # Full RAG pipeline
│   ├── ingest.py        # One-time ingestion — builds FAISS + BM25
│   └── models.py        # Pydantic request/response schemas
├── data/                # Place Kaggle CSVs here
├── faiss_index/         # Auto-generated after running ingest.py
│   ├── index.faiss
│   ├── index.pkl
│   └── bm25.pkl
├── tests/
│   └── test_api.py      # 40+ pytest tests
├── notebooks/
│   └── test_results.ipynb
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/Tetrax-427/python-qa-assistant.git
cd python-qa-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment variables

```bash
cp .env.example .env
# Edit .env and fill in:
# GROQ_API_KEY    → https://console.groq.com       
# COHERE_API_KEY  → https://dashboard.cohere.com   
```

### 4. Download dataset

[Stack Overflow Python Questions — Kaggle](https://www.kaggle.com/datasets/stackoverflow/pythonquestions)

Place inside `data/`:
```
data/
├── Questions.csv
└── Answers.csv
```

### 5. Run ingestion (one-time)

```bash
python -m app.ingest
```

Builds `faiss_index/index.faiss`, `index.pkl`, `bm25.pkl`.
Expected time: 20–40 min for full dataset.

### 6. Start API

```bash
uvicorn app.main:app --reload
```

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

---

## 🔌 API Reference

### `GET /health`
```json
{
  "status": "healthy",
  "faiss_index_loaded": true,
  "bm25_index_loaded": true,
  "message": "All systems ready."
}
```

### `POST /ask`

**Request:**
```json
{
  "question": "How do I reverse a list in Python?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "How do I reverse a list in Python?",
  "rewritten_query": "Python list reversal methods",
  "query_variants": [
    "Python list reversal methods",
    "how to reverse a list in Python",
    "list reverse in-place vs copy Python"
  ],
  "answer": "You can reverse a list using `list[::-1]` (returns a new list) or `list.reverse()` (in-place)...",
  "sources": [
    {
      "title": "Reverse a list in Python",
      "score": 0.9423,
      "snippet": "Use list.reverse() or slicing...",
      "quality_band": "highly-voted",
      "topics": "list, sort",
      "answer_score": 312,
      "question_score": 198
    }
  ],
  "model": "llama-3.3-70b-versatile",
  "cache_hit": false,
  "low_confidence": false
}
```

### `GET /cache/stats`
```json
{
  "size": 12,
  "hits": 34,
  "misses": 21,
  "hit_rate": 0.618
}
```

---

## 🧪 Running Tests

```bash
pytest tests/test_api.py -v
```

---

## 🚀 RAG Improvements (v3 — all cumulative)

| # | Improvement | Where | Benefit |
|---|---|---|---|
| 1 | Query Rewriting | `rag.py` | Cleaner, retrieval-optimised queries |
| 2 | HyDE | `rag.py` | Doc-to-doc embedding matching |
| 3 | Hybrid Search (FAISS + BM25) | `rag.py` + `ingest.py` | Catches both semantic + exact keyword matches |
| 4 | RRF Fusion | `rag.py` | Merges both ranked lists optimally |
| 5 | Cohere Rerank | `rag.py` | Cross-encoder precision on top-20 candidates |
| 6 | Confidence Check | `rag.py` | No hallucination on low-quality retrieval |
| 7 | LRU Cache | `rag.py` | Instant repeat responses, lower API costs |
| 8 | Multi-Query Generation | `rag.py` | 3 variants → broader recall (+14% retrieval accuracy) |
| 9 | Async Parallel Retrieval | `rag.py` | All variants retrieved concurrently, lower latency |
| 10 | Context Compression | `rag.py` | Noise-free docs → better LLM focus |
| 11 | Contextual Chunk Headers | `ingest.py` | Quality + topic context baked into embeddings |
| 12 | Rich Metadata | `ingest.py` | Quality band + vote scores exposed in response |

---

## 📈 Scaling to 100+ Concurrent Users

| Concern | Solution |
|---|---|
| Async endpoints | FastAPI async + Groq fast inference |
| FAISS bottleneck | Replace with Pinecone / Qdrant for distributed search |
| Repeat queries | LRU cache (in-memory) → Redis for distributed cache |
| LLM cost | Cache + batch similar queries |
| Rate limits | Request queuing via Celery + Redis |
| Horizontal scale | Docker + load balancer (Nginx) |
| Observability | Add structured logging + Prometheus metrics |
