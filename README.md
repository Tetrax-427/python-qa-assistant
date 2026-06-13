# 🐍 Python Q&A Assistant

An AI-powered question-answering system for Python programming queries, grounded in Stack Overflow data using a production-grade RAG (Retrieval-Augmented Generation) pipeline.

---

## 🏗️ Architecture

```
User Question
     ↓
FastAPI  POST /ask
     ↓
LRU Cache Check ──── HIT ──────────────────────→ Cached Response
     ↓ MISS
Query Rewriting  (Groq)
     ↓
Multi-Query Generation  (Groq → 3 variants)
     ↓
Async Parallel Retrieval  (per variant, concurrent)
  ├── FAISS  semantic search  (HyDE embedding)
  └── BM25   keyword search
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
LRU Cache Store  →  Response
```

---

## 🔧 Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (IndexFlatIP, persisted to disk) |
| Keyword Search | BM25Okapi (rank-bm25) |
| Reranker | Cohere Rerank v3 |
| LLM | Groq `llama-3.3-70b-versatile` |
| API Framework | FastAPI |
| Cache | In-memory LRU (512 slots) |

---

## 📁 Project Structure

```
python-qa-assistant/
├── app/
│   ├── main.py          # FastAPI app — /ask, /health, /cache/stats
│   ├── rag.py           # Core RAG pipeline
│   ├── ingest.py        # One-time ingestion script
│   └── models.py        # Pydantic schemas
├── faiss_index/         # Auto-generated after running ingest.py
│   ├── index.faiss
│   ├── index.pkl
│   └── bm25.pkl
├── data/                # Kaggle CSVs 
├── tests/
│   └── test_api.py      # 38 pytest tests
├── notebooks/
│   └── test_results.ipynb
├── conftest.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tetrax-427/python-qa-assistant.git
cd python-qa-assistant
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Fill in your API keys in `.env`:

```env
GROQ_API_KEY=your_groq_api_key        # https://console.groq.com (free)
COHERE_API_KEY=your_cohere_api_key    # https://dashboard.cohere.com (free)
```

### 5. Download the dataset

Download from Kaggle: [Stack Overflow Python Questions](https://www.kaggle.com/datasets/stackoverflow/pythonquestions)

Place inside `data/`:
```
data/
├── Questions.csv
└── Answers.csv
```

### 6. Run ingestion (one-time)

```bash
python -m app.ingest
```

Builds `faiss_index/index.faiss`, `index.pkl`, and `bm25.pkl`.
Uses top 50k highest-voted questions for optimal quality.

### 7. Start the API

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
  "answer": "You can reverse a list using `list[::-1]` (new list) or `.reverse()` (in-place)...",
  "sources": [
    {
      "title": "Best way to create a reversed list in Python?",
      "score": 0.9995,
      "snippet": "newlist = oldlist[::-1]...",
      "quality_band": "highly-voted",
      "topics": "list, sort",
      "answer_score": 120,
      "question_score": 43
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

38 tests covering health, schema validation, diverse queries, cache behaviour, rich metadata, and edge cases.

---

## 🚀 RAG Pipeline — Design Decisions

| Technique | Why |
|---|---|
| **Query Rewriting** | Raw user queries are often vague — rewriting improves retrieval precision |
| **HyDE** | Embedding a hypothetical answer instead of the query gives doc-to-doc matching, improving recall by up to 42% |
| **Hybrid Search (FAISS + BM25)** | FAISS catches semantic similarity; BM25 catches exact keywords, error messages, function names |
| **Multi-Query Generation** | 3 query variants cover different angles of the same question — broader recall |
| **Async Parallel Retrieval** | All variants retrieved concurrently via ThreadPoolExecutor — lower latency |
| **RRF Fusion** | Reciprocal Rank Fusion optimally merges multiple ranked lists |
| **Cohere Rerank** | Cross-encoder re-scores top-20 candidates for precision — outperforms bi-encoder alone |
| **Confidence Check** | Prevents hallucination when retrieval quality is poor |
| **Context Compression** | Strips noise from retrieved docs before LLM sees them — better focus, less hallucination |
| **Contextual Chunk Headers** | Quality band + topic context baked into embeddings at ingest time — zero runtime cost |
| **LRU Cache** | Repeated queries served instantly — reduces latency and API costs |

---

## 📈 Scaling to 100+ Concurrent Users

| Concern | Solution |
|---|---|
| Async endpoints | FastAPI async + Groq fast inference |
| FAISS at scale | Replace with Pinecone / Qdrant for distributed vector search |
| Repeated queries | Upgrade LRU cache to Redis for distributed caching |
| LLM cost | Cache + batch similar queries |
| Rate limits | Request queuing via Celery + Redis |
| Horizontal scaling | Docker + Nginx load balancer |
| Observability | Structured logging + Prometheus metrics |