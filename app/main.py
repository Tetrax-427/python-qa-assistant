import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    QuestionRequest, AnswerResponse, HealthResponse,
    SourceDocument, CacheStatsResponse,
)
from app.rag import rag_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up — loading RAG pipeline …")
    try:
        rag_pipeline.load()
        logger.info("✅ RAG pipeline loaded")
    except RuntimeError as e:
        logger.warning(f"⚠️  Could not load RAG pipeline at startup: {e}")
    yield
    logger.info("🛑 Shutting down")


app = FastAPI(
    title="Python Q&A Assistant",
    description="""
AI-powered Q&A system for Python programming, grounded in Stack Overflow data.

**RAG Pipeline:**
1. Query Rewriting → cleaner retrieval query
2. Multi-Query Generation → 3 diverse variants for broader recall  
3. Async Parallel Retrieval → FAISS + BM25 per variant, all concurrent  
4. RRF Fusion → merges all ranked lists  
5. Cohere Rerank → cross-encoder precision filter  
6. Confidence Check → graceful fallback if score below threshold  
7. Context Compression → strips noise before LLM sees it  
8. Groq llama-3.3-70b → grounded answer generation  
+ LRU Response Cache → repeated queries served instantly
    """,
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check — reports FAISS and BM25 index load status."""
    loaded = rag_pipeline.is_loaded
    return HealthResponse(
        status             = "healthy" if loaded else "degraded",
        faiss_index_loaded = loaded,
        bm25_index_loaded  = loaded,
        message            = "All systems ready." if loaded else "Run: python -m app.ingest",
    )


@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
def ask_question(request: QuestionRequest):
    """
    Answer a Python question using the full v3 RAG pipeline.

    Returns the answer, source documents with rich metadata,
    the rewritten query, all query variants used, cache status,
    and a low_confidence flag if retrieval quality was poor.
    """
    if not rag_pipeline.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not ready. Run: python -m app.ingest",
        )

    try:
        logger.info(f"Received: {request.question[:80]}")
        result = rag_pipeline.ask(question=request.question, top_k=request.top_k)

        sources = [
            SourceDocument(
                title          = s["title"],
                score          = s["score"],
                snippet        = s["snippet"],
                quality_band   = s.get("quality_band",   "unknown"),
                topics         = s.get("topics",         ""),
                answer_score   = s.get("answer_score",   0),
                question_score = s.get("question_score", 0),
            )
            for s in result["sources"]
        ]

        return AnswerResponse(
            question        = request.question,
            rewritten_query = result["rewritten_query"],
            query_variants  = result.get("query_variants", []),
            answer          = result["answer"],
            sources         = sources,
            cache_hit       = result["cache_hit"],
            low_confidence  = result["low_confidence"],
        )

    except Exception as e:
        logger.exception("Error in /ask")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats", response_model=CacheStatsResponse, tags=["Cache"])
def cache_stats():
    """LRU cache statistics — size, hits, misses, hit rate."""
    return CacheStatsResponse(**rag_pipeline.cache_stats())
