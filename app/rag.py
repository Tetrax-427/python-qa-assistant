"""
rag.py — Advanced RAG Pipeline v3

Full pipeline:
  query
    → cache check
    → query rewrite (Groq)                         
    → multi-query generation (Groq, 3 variants)    
    → async parallel retrieval per query variant   
    → RRF fusion across all query results
    → Cohere rerank → top-5
    → confidence check
    → context compression (Groq)                  
    → Groq answer generation
    → cache store
    → response
"""

import os
import pickle
import hashlib
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional
from collections import OrderedDict

import faiss
import numpy as np
import cohere
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Paths ───
BASE_DIR         = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = BASE_DIR / "faiss_index" / "index.faiss"
METADATA_PATH    = BASE_DIR / "faiss_index" / "index.pkl"
BM25_INDEX_PATH  = BASE_DIR / "faiss_index" / "bm25.pkl"

# ── Config ───
EMBEDDING_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL            = "llama-3.3-70b-versatile"
FAISS_TOP_K           = 20
BM25_TOP_K            = 20
RERANK_TOP_K          = 5
CONFIDENCE_THRESHOLD  = 0.20
CACHE_MAX_SIZE        = 512
NUM_QUERY_VARIANTS    = 3       # multi-query: number of variants to generate
MAX_WORKERS           = 4       # thread pool for async parallel retrieval

# ── Prompts ───
REWRITE_PROMPT = """You are a search query optimizer for a Python Q&A system.
Rewrite the user's question into a clean, precise, retrieval-optimized query.
- 1-2 sentences max
- Preserve exact technical terms, function names, error messages
- Remove filler words and conversational tone
- Output the rewritten query ONLY, no explanation

User question: {question}
Rewritten query:"""

MULTI_QUERY_PROMPT = """You are a search query expert for a Python Stack Overflow Q&A system.
Given a user question, generate {n} diverse search query variations that approach 
the same topic from different angles. This improves retrieval recall.

Rules:
- Each query on its own line, no numbering or bullets
- Vary phrasing, level of specificity, and perspective
- Include both conceptual and code-focused variants
- Keep each query concise and retrieval-friendly

Question: {question}
Generate {n} query variations:"""

HYDE_PROMPT = """You are a Python expert. Write a concise, direct answer (3-5 sentences max)
to the following question. Include a brief code example if relevant.
This is used for document retrieval only — not shown to users.

Question: {question}
Answer:"""

COMPRESS_PROMPT = """You are a context extractor for a Python Q&A system.
Given a question and a retrieved Stack Overflow document, extract ONLY the 
sentences and code snippets that are directly relevant to answering the question.
Remove irrelevant tangents, unrelated examples, and noise.
Keep all code blocks that are relevant. Output the compressed context only.

Question: {question}

Document:
{document}

Relevant context:"""

SYSTEM_PROMPT = """You are a helpful Python programming assistant grounded in Stack Overflow knowledge.

Rules:
- Answer ONLY based on the provided context documents.
- If the context doesn't fully address the question, say so honestly.
- Be concise and accurate. Include working code examples in markdown code blocks.
- Do not hallucinate libraries, functions, or APIs that don't exist.
- Prefer answers backed by highly-voted Stack Overflow content."""

FALLBACK_ANSWER = (
    "I wasn't able to find sufficiently relevant information in the Stack Overflow "
    "dataset to answer this confidently. Please try rephrasing, or check the official "
    "Python documentation at https://docs.python.org"
)


# ── LRU Cache ───
class LRUCache:
    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self._cache: OrderedDict = OrderedDict()
        self._max  = max_size
        self.hits  = 0
        self.misses = 0

    def _key(self, question: str, top_k: int) -> str:
        return hashlib.md5(f"{question.strip().lower()}|{top_k}".encode()).hexdigest()

    def get(self, question: str, top_k: int) -> Optional[dict]:
        k = self._key(question, top_k)
        if k in self._cache:
            self._cache.move_to_end(k)
            self.hits += 1
            return self._cache[k]
        self.misses += 1
        return None

    def set(self, question: str, top_k: int, value: dict) -> None:
        k = self._key(question, top_k)
        self._cache[k] = value
        self._cache.move_to_end(k)
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size":     self.size,
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


# ── RAG Pipeline ────
class RAGPipeline:

    def __init__(self):
        self._faiss_index = None
        self._bm25_index  = None
        self._chunks      = None
        self._metadata    = None
        self._embedder    = None
        self._cohere      = None
        self._groq        = None
        self._executor    = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._loaded      = False
        self._cache       = LRUCache()

    def load(self) -> None:
        if self._loaded:
            return
        for p in [FAISS_INDEX_PATH, METADATA_PATH, BM25_INDEX_PATH]:
            if not p.exists():
                raise RuntimeError(
                    f"Missing index: {p}\n"
                    "Run ingestion first:  python -m app.ingest"
                )

        logger.info("Loading FAISS index …")
        self._faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

        logger.info("Loading metadata + chunks …")
        with open(METADATA_PATH, "rb") as f:
            data = pickle.load(f)
        self._chunks   = data["chunks"]
        self._metadata = data["metadata"]

        logger.info("Loading BM25 index …")
        with open(BM25_INDEX_PATH, "rb") as f:
            self._bm25_index = pickle.load(f)

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL} …")
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Initialising Cohere client …")
        self._cohere = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

        logger.info("Initialising Groq client …")
        self._groq = Groq(api_key=os.environ["GROQ_API_KEY"])

        self._loaded = True
        logger.info(f"✅ RAG pipeline ready — {self._faiss_index.ntotal:,} vectors indexed")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def cache_stats(self) -> dict:
        return self._cache.stats()

    # Query preparation
    def _rewrite_query(self, question: str) -> str:
        """Rewrite raw user question into a retrieval-optimised query."""
        try:
            resp = self._groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
                temperature=0.0,
                max_tokens=128,
            )
            rewritten = resp.choices[0].message.content.strip()
            logger.info(f"Rewritten: '{question[:50]}' → '{rewritten[:50]}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original: {e}")
            return question

    # Multi-Query Generation
    def _generate_multi_queries(self, rewritten: str) -> List[str]:
        """
        Generate N diverse query variants from the rewritten query.
        All variants are retrieved in parallel — broader recall, same latency.
        """
        try:
            resp = self._groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{
                    "role": "user",
                    "content": MULTI_QUERY_PROMPT.format(
                        question=rewritten,
                        n=NUM_QUERY_VARIANTS,
                    ),
                }],
                temperature=0.5,   # some diversity
                max_tokens=256,
            )
            raw     = resp.choices[0].message.content.strip()
            queries = [q.strip() for q in raw.splitlines() if q.strip()][:NUM_QUERY_VARIANTS]
            # Always include the rewritten query itself
            all_queries = list(dict.fromkeys([rewritten] + queries))
            logger.info(f"Multi-query variants ({len(all_queries)}): {all_queries}")
            return all_queries
        except Exception as e:
            logger.warning(f"Multi-query generation failed, using single query: {e}")
            return [rewritten]

    def _generate_hyde_doc(self, question: str) -> str:
        """Generate a hypothetical answer for embedding-based retrieval."""
        try:
            resp = self._groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
                temperature=0.3,
                max_tokens=256,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"HyDE failed, using query: {e}")
            return question

    # Retrieval
    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.encode([text], normalize_embeddings=True).astype("float32")

    def _faiss_search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        scores, indices = self._faiss_index.search(query_vec, top_k)
        return [(int(idx), float(s)) for s, idx in zip(scores[0], indices[0]) if idx != -1]

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        tokens = query.lower().split()
        scores = self._bm25_index.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

    # Async Parallel Retrieval 
    def _retrieve_for_query(self, query: str) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Run FAISS + BM25 for a single query.
        Used as the unit of work in async parallel retrieval.
        Note: HyDE embedding used for FAISS; raw query used for BM25.
        """
        hyde_doc  = self._generate_hyde_doc(query)
        hyde_vec  = self._embed(hyde_doc)
        faiss_res = self._faiss_search(hyde_vec, FAISS_TOP_K)
        bm25_res  = self._bm25_search(query, BM25_TOP_K)
        return faiss_res, bm25_res

    async def _async_retrieve_all(self, queries: List[str]) -> List[Tuple[List, List]]:
        """
        Dispatch retrieval for ALL query variants concurrently using a thread pool.
        FAISS and sentence-transformer are CPU-bound — ThreadPoolExecutor keeps
        the event loop unblocked while they run in parallel.
        """
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self._executor, self._retrieve_for_query, q)
            for q in queries
        ]
        results = await asyncio.gather(*tasks)
        logger.info(f"Async retrieval complete for {len(queries)} query variants")
        return list(results)

    def _rrf_fusion(
        self,
        all_faiss: List[List[Tuple[int, float]]],
        all_bm25:  List[List[Tuple[int, float]]],
        k:         int = 60,
        top_n:     int = 25,
    ) -> List[int]:
        """
        RRF across ALL query variants' FAISS + BM25 results.
        More query variants = broader candidate pool = better final recall.
        """
        rrf: dict[int, float] = {}

        for result_list in all_faiss + all_bm25:
            for rank, (doc_idx, _) in enumerate(result_list):
                rrf[doc_idx] = rrf.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_docs[:top_n]]

    def _rerank(self, query: str, doc_indices: List[int], top_k: int) -> List[Tuple[int, float]]:
        docs = [self._chunks[i] for i in doc_indices]
        resp = self._cohere.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=top_k,
        )
        return [(doc_indices[hit.index], hit.relevance_score) for hit in resp.results]

    def _is_confident(self, reranked: List[Tuple[int, float]]) -> bool:
        if not reranked:
            return False
        top_score = reranked[0][1]
        if top_score < CONFIDENCE_THRESHOLD:
            logger.warning(f"Low confidence: top score {top_score:.4f} < {CONFIDENCE_THRESHOLD}")
            return False
        return True

    # Context Compression 
    def _compress_context(self, question: str, reranked: List[Tuple[int, float]]) -> List[Tuple[int, float, str]]:
        """
        For each top-ranked doc, call Groq to extract only the sentences/code
        directly relevant to the question. Returns (idx, score, compressed_text).

        Benefits:
        - Reduces noise in the prompt → LLM focuses on what matters
        - Fits more relevant docs into the context window
        - Reduces hallucination from irrelevant tangents
        """
        compressed = []
        for idx, score in reranked:
            raw_doc = self._chunks[idx]
            try:
                resp = self._groq.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{
                        "role": "user",
                        "content": COMPRESS_PROMPT.format(
                            question=question,
                            document=raw_doc[:2000],  # cap input length
                        ),
                    }],
                    temperature=0.0,
                    max_tokens=400,
                )
                compressed_text = resp.choices[0].message.content.strip()
                # Fallback: if compression returns too little, use original
                if len(compressed_text) < 50:
                    compressed_text = raw_doc
            except Exception as e:
                logger.warning(f"Compression failed for doc {idx}: {e}")
                compressed_text = raw_doc

            compressed.append((idx, score, compressed_text))

        logger.info(f"Context compression done for {len(compressed)} docs")
        return compressed

    # Generation
    def _generate(self, question: str, compressed_docs: List[Tuple[int, float, str]]) -> str:
        context = "\n\n---\n\n".join(
            f"[Doc {i+1} | relevance: {score:.3f} | {self._metadata[idx]['quality_band']}]\n{text}"
            for i, (idx, score, text) in enumerate(compressed_docs)
        )
        user_msg = (
            f"Context from Stack Overflow:\n\n{context}\n\n---\n\n"
            f"Question: {question}\n\n"
            f"Answer based on the context above:"
        )
        resp = self._groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()

    # Public API
    def ask(self, question: str, top_k: int = RERANK_TOP_K) -> dict:
        """
        Full v3 RAG pipeline (synchronous entry point).
        Internally runs async parallel retrieval via asyncio.
        """
        self.load()

        # ── Cache check ──
        cached = self._cache.get(question, top_k)
        if cached:
            logger.info("Cache hit")
            return {**cached, "cache_hit": True}

        # ── Step 1: Query rewrite ──
        rewritten = self._rewrite_query(question)

        # ── Step 2: Multi-query generation ──
        query_variants = self._generate_multi_queries(rewritten)

        # ── Step 3: Async parallel retrieval per variant ───
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        retrieval_results = loop.run_until_complete(
            self._async_retrieve_all(query_variants)
        )

        all_faiss = [r[0] for r in retrieval_results]
        all_bm25  = [r[1] for r in retrieval_results]

        # ── Step 4: RRF across all variants ───
        fused_indices = self._rrf_fusion(all_faiss, all_bm25)

        if not fused_indices:
            return self._build_fallback(rewritten, query_variants)

        # ── Step 5: Cohere rerank ────
        reranked = self._rerank(rewritten, fused_indices, top_k)

        # ── Step 6: Confidence check ───
        if not self._is_confident(reranked):
            result = self._build_fallback(rewritten, query_variants)
            self._cache.set(question, top_k, result)
            return result

        # ── Step 7: Context compression ───
        compressed_docs = self._compress_context(question, reranked)

        # ── Step 8: Generate answer ───
        answer = self._generate(question, compressed_docs)

        sources = [
            {
                "title":          self._metadata[idx]["title"],
                "score":          round(score, 4),
                "snippet":        self._metadata[idx]["snippet"],
                "quality_band":   self._metadata[idx]["quality_band"],
                "topics":         self._metadata[idx]["topics"],
                "answer_score":   self._metadata[idx]["answer_score"],
                "question_score": self._metadata[idx]["question_score"],
            }
            for idx, score, _ in compressed_docs
        ]

        result = {
            "answer":          answer,
            "sources":         sources,
            "cache_hit":       False,
            "rewritten_query": rewritten,
            "query_variants":  query_variants,
            "low_confidence":  False,
        }

        self._cache.set(question, top_k, result)
        return result

    def _build_fallback(self, rewritten: str, query_variants: List[str]) -> dict:
        return {
            "answer":          FALLBACK_ANSWER,
            "sources":         [],
            "cache_hit":       False,
            "rewritten_query": rewritten,
            "query_variants":  query_variants,
            "low_confidence":  True,
        }


# Singleton shared across FastAPI requests
rag_pipeline = RAGPipeline()
