from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="A Python-related question to be answered",
        examples=["How do I reverse a list in Python?"]
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of top documents to retrieve after reranking"
    )


class SourceDocument(BaseModel):
    title:          str   = Field(description="Title of the source Stack Overflow question")
    score:          float = Field(description="Relevance score from Cohere reranker")
    snippet:        str   = Field(description="Relevant excerpt from the answer")
    quality_band:   str   = Field(description="Answer quality band: highly-voted / well-voted / positively-voted / low-voted")
    topics:         str   = Field(description="Detected Python topics in this document")
    answer_score:   int   = Field(description="Raw Stack Overflow answer vote score")
    question_score: int   = Field(description="Raw Stack Overflow question vote score")


class AnswerResponse(BaseModel):
    question:         str                  = Field(description="Original question")
    rewritten_query:  str                  = Field(description="Query after rewriting optimisation")
    query_variants:   List[str]            = Field(description="All query variants used for multi-query retrieval")
    answer:           str                  = Field(description="Grounded answer from the LLM")
    sources:          List[SourceDocument] = Field(description="Top source documents used (with rich metadata)")
    model:            str                  = Field(default="llama-3.3-70b-versatile", description="LLM used")
    cache_hit:        bool                 = Field(default=False, description="True if served from LRU cache")
    low_confidence:   bool                 = Field(default=False, description="True if retrieval confidence was below threshold")


class HealthResponse(BaseModel):
    status:             str  = Field(description="Service health status: healthy / degraded")
    faiss_index_loaded: bool = Field(description="FAISS index loaded")
    bm25_index_loaded:  bool = Field(description="BM25 index loaded")
    message:            str  = Field(description="Human-readable status message")


class CacheStatsResponse(BaseModel):
    size:     int   = Field(description="Current number of cached entries")
    hits:     int   = Field(description="Total cache hits since startup")
    misses:   int   = Field(description="Total cache misses since startup")
    hit_rate: float = Field(description="Cache hit rate between 0.0 and 1.0")
