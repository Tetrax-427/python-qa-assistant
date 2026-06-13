"""
tests/test_api.py — Full test suite v3

Covers:
  - Health endpoint (FAISS + BM25)
  - /ask response schema (all v3 fields)
  - 10 diverse Python queries
  - Multi-query variants in response
  - Rich source metadata (quality_band, topics, scores)
  - Context compression reflected in sources
  - Cache hit/miss behaviour
  - Cache stats endpoint
  - Confidence fallback
  - Edge cases (validation, 503, 422)

Run:
    pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)


# ── Mock helpers ───
def make_mock_source():
    return {
        "title":          "How to reverse a list in Python",
        "score":          0.95,
        "snippet":        "Use list[::-1] or list.reverse() to reverse a list in place.",
        "quality_band":   "highly-voted",
        "topics":         "list, sort",
        "answer_score":   312,
        "question_score": 198,
    }


def make_mock_result(cache_hit=False, low_confidence=False):
    return {
        "answer":          "You can reverse a list using `[::-1]` or `.reverse()`.",
        "sources":         [make_mock_source()],
        "cache_hit":       cache_hit,
        "rewritten_query": "Python reverse list methods",
        "query_variants":  [
            "Python reverse list methods",
            "how to reverse a list in Python",
            "list reversal Python examples",
        ],
        "low_confidence":  low_confidence,
    }


def get_mock_pipeline(cache_hit=False, low_confidence=False):
    mock = MagicMock()
    mock.is_loaded = True
    mock.ask.return_value = make_mock_result(cache_hit=cache_hit, low_confidence=low_confidence)
    mock.cache_stats.return_value = {
        "size": 3, "hits": 5, "misses": 10, "hit_rate": 0.333
    }
    return mock


# ── Health endpoint ───
class TestHealthEndpoint:
    def test_returns_200(self):
        with patch("app.main.rag_pipeline", get_mock_pipeline()):
            assert client.get("/health").status_code == 200

    def test_schema(self):
        with patch("app.main.rag_pipeline", get_mock_pipeline()):
            data = client.get("/health").json()
        for field in ["status", "faiss_index_loaded", "bm25_index_loaded", "message"]:
            assert field in data

    def test_healthy_when_loaded(self):
        with patch("app.main.rag_pipeline", get_mock_pipeline()):
            data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["faiss_index_loaded"] is True
        assert data["bm25_index_loaded"] is True

    def test_degraded_when_not_loaded(self):
        mock = MagicMock()
        mock.is_loaded = False
        with patch("app.main.rag_pipeline", mock):
            data = client.get("/health").json()
        assert data["status"] == "degraded"


# ── /ask response schema ────
class TestAskSchema:
    def _ask(self, question="How do I reverse a list in Python?", top_k=5, **kw):
        with patch("app.main.rag_pipeline", get_mock_pipeline(**kw)):
            return client.post("/ask", json={"question": question, "top_k": top_k})

    def test_returns_200(self):
        assert self._ask().status_code == 200

    def test_has_question(self):
        q = "How do I reverse a list in Python?"
        assert self._ask(q).json()["question"] == q

    def test_has_rewritten_query(self):
        data = self._ask().json()
        assert "rewritten_query" in data
        assert len(data["rewritten_query"]) > 0

    def test_has_query_variants(self):
        data = self._ask().json()
        assert "query_variants" in data
        assert isinstance(data["query_variants"], list)
        assert len(data["query_variants"]) >= 1

    def test_has_answer(self):
        assert len(self._ask().json()["answer"]) > 10

    def test_has_sources_list(self):
        assert isinstance(self._ask().json()["sources"], list)

    def test_has_model_field(self):
        assert "model" in self._ask().json()

    def test_has_cache_hit(self):
        assert "cache_hit" in self._ask().json()

    def test_has_low_confidence(self):
        assert "low_confidence" in self._ask().json()

    def test_cache_hit_false_first_call(self):
        assert self._ask(cache_hit=False).json()["cache_hit"] is False

    def test_cache_hit_true_on_cached(self):
        assert self._ask(cache_hit=True).json()["cache_hit"] is True

    def test_low_confidence_false_normally(self):
        assert self._ask(low_confidence=False).json()["low_confidence"] is False

    def test_low_confidence_true_when_flagged(self):
        assert self._ask(low_confidence=True).json()["low_confidence"] is True


# ── Source document rich metadata ───
class TestSourceMetadata:
    def _sources(self):
        with patch("app.main.rag_pipeline", get_mock_pipeline()):
            return client.post("/ask", json={"question": "How do I reverse a list in Python?"}).json()["sources"]

    def test_sources_not_empty(self):
        assert len(self._sources()) > 0

    def test_source_has_title(self):
        assert "title" in self._sources()[0]

    def test_source_has_score(self):
        assert "score" in self._sources()[0]

    def test_source_has_snippet(self):
        assert "snippet" in self._sources()[0]

    def test_source_has_quality_band(self):
        assert "quality_band" in self._sources()[0]

    def test_source_has_topics(self):
        assert "topics" in self._sources()[0]

    def test_source_has_answer_score(self):
        assert "answer_score" in self._sources()[0]

    def test_source_has_question_score(self):
        assert "question_score" in self._sources()[0]

    def test_quality_band_is_valid(self):
        valid_bands = {"highly-voted", "well-voted", "positively-voted", "low-voted", "unknown"}
        assert self._sources()[0]["quality_band"] in valid_bands

    def test_scores_are_numeric(self):
        src = self._sources()[0]
        assert isinstance(src["answer_score"],   int)
        assert isinstance(src["question_score"], int)


# ── Diverse query coverage ───
class TestDiverseQueries:
    QUERIES = [
        "What is a list comprehension in Python and how do I use it?",
        "How do I handle exceptions in Python using try except?",
        "How do I read and write files in Python?",
        "How do I merge two dictionaries in Python?",
        "How do I filter rows in a pandas DataFrame by column value?",
        "How do Python decorators work and how do I create one?",
        "What is the difference between a generator and an iterator in Python?",
        "How do I use async and await in Python?",
        "How do I sort a dictionary by value in Python?",
        "How do lambda functions work in Python?",
    ]

    def test_all_queries_return_200(self):
        for q in self.QUERIES:
            with patch("app.main.rag_pipeline", get_mock_pipeline()):
                r = client.post("/ask", json={"question": q})
            assert r.status_code == 200, f"Failed for: {q}"


# ── Cache stats endpoint ───
class TestCacheStats:
    def test_returns_200(self):
        with patch("app.main.rag_pipeline", get_mock_pipeline()):
            assert client.get("/cache/stats").status_code == 200

    def test_schema(self):
        with patch("app.main.rag_pipeline", get_mock_pipeline()):
            data = client.get("/cache/stats").json()
        for field in ["size", "hits", "misses", "hit_rate"]:
            assert field in data

    def test_hit_rate_in_range(self):
        with patch("app.main.rag_pipeline", get_mock_pipeline()):
            data = client.get("/cache/stats").json()
        assert 0.0 <= data["hit_rate"] <= 1.0

    def test_counts_are_non_negative(self):
        with patch("app.main.rag_pipeline", get_mock_pipeline()):
            data = client.get("/cache/stats").json()
        assert data["hits"]   >= 0
        assert data["misses"] >= 0
        assert data["size"]   >= 0


# ── Edge cases ────
class TestEdgeCases:
    def test_too_short_question_rejected(self):
        assert client.post("/ask", json={"question": "Hi"}).status_code == 422

    def test_missing_question_rejected(self):
        assert client.post("/ask", json={}).status_code == 422

    def test_top_k_too_large_rejected(self):
        assert client.post("/ask", json={"question": "How do I reverse a list?", "top_k": 99}).status_code == 422

    def test_top_k_zero_rejected(self):
        assert client.post("/ask", json={"question": "How do I reverse a list?", "top_k": 0}).status_code == 422

    def test_503_when_pipeline_not_loaded(self):
        mock = MagicMock()
        mock.is_loaded = False
        with patch("app.main.rag_pipeline", mock):
            r = client.post("/ask", json={"question": "How do I reverse a list in Python?"})
        assert r.status_code == 503

    def test_fallback_response_structure(self):
        with patch("app.main.rag_pipeline", get_mock_pipeline(low_confidence=True)):
            data = client.post("/ask", json={"question": "How do I reverse a list in Python?"}).json()
        assert data["low_confidence"] is True
        assert "answer" in data
