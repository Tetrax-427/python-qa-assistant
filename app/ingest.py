import re
import pickle
import logging
from pathlib import Path

import pandas as pd
import faiss
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──
BASE_DIR         = Path(__file__).resolve().parent.parent
DATA_DIR         = BASE_DIR / "data"

INDEX_DIR        = BASE_DIR / "faiss_index"
INDEX_DIR.mkdir(exist_ok=True)

QUESTIONS_CSV    = DATA_DIR / "Questions.csv"
ANSWERS_CSV      = DATA_DIR / "Answers.csv"
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH    = INDEX_DIR / "index.pkl"
BM25_INDEX_PATH  = INDEX_DIR / "bm25.pkl"

# ── Config ──
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE      = 512
MAX_CHUNK_CHARS = 1000


# ── Helpers ──
def clean_html(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def score_band(score: int) -> str:
    """Convert numeric vote score into a human-readable quality band."""
    if score >= 100:
        return "highly-voted"
    elif score >= 20:
        return "well-voted"
    elif score >= 5:
        return "positively-voted"
    else:
        return "low-voted"


def extract_keywords(title: str) -> str:
    """
    Extract likely Python keywords/topics from the question title.
    Used for metadata and contextual headers.
    """
    PYTHON_TOPICS = [
        "list", "dict", "tuple", "set", "string", "str", "int", "float",
        "class", "function", "lambda", "generator", "iterator", "decorator",
        "async", "await", "thread", "multiprocessing", "regex", "re",
        "file", "io", "json", "csv", "pandas", "numpy", "matplotlib",
        "django", "flask", "fastapi", "sqlalchemy", "requests", "urllib",
        "exception", "error", "import", "module", "package", "pip",
        "loop", "for", "while", "comprehension", "map", "filter", "sort",
        "index", "slice", "format", "print", "type", "isinstance",
        "inheritance", "polymorphism", "dataclass", "enum", "abc",
        "pathlib", "os", "sys", "argparse", "logging", "unittest", "pytest",
    ]
    title_lower = title.lower()
    found = [kw for kw in PYTHON_TOPICS if kw in title_lower]
    return ", ".join(found[:5]) if found else "python"


def build_chunks(df: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    """
    Build:
      - raw_chunks   : original text (used for BM25 + display)
      - embed_chunks : text WITH contextual header prepended (used for FAISS embedding)
      - metadata     : rich metadata per chunk

    Contextual Chunk Header format (prepended before embedding):
      [Stack Overflow | Python | {topic_keywords} | Q-score: {band} | A-score: {band}]

    This gives the embedding model document-level context, dramatically
    improving retrieval precision without any runtime cost.
    """
    raw_chunks   = []
    embed_chunks = []
    metadata     = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building chunks"):
        title   = clean_html(str(row["Title"]))
        q_body  = clean_html(str(row["QuestionBody"])[:400])
        a_body  = clean_html(str(row["AnswerBody"])[:MAX_CHUNK_CHARS])

        q_score = int(row.get("QuestionScore", 0) or 0)
        a_score = int(row.get("AnswerScore",   0) or 0)
        topics  = extract_keywords(title)

        header = (
            f"[Stack Overflow | Python | Topics: {topics} | "
            f"Question quality: {score_band(q_score)} | "
            f"Answer quality: {score_band(a_score)}]"
        )

        raw_chunk   = f"Question: {title}\n{q_body}\n\nAnswer: {a_body}"
        embed_chunk = f"{header}\n\n{raw_chunk}"   # header only prepended for embedding

        raw_chunks.append(raw_chunk)
        embed_chunks.append(embed_chunk)

        metadata.append({
            "question_id":    int(row["QuestionId"]),
            "title":          title,
            "snippet":        a_body[:300],
            "question_score": q_score,
            "answer_score":   a_score,
            "quality_band":   score_band(a_score),
            "topics":         topics,
        })

    logger.info(f"Built {len(raw_chunks):,} chunks with contextual headers")
    return raw_chunks, embed_chunks, metadata


def load_and_merge() -> pd.DataFrame:
    """Load Questions + Answers, merge top answer per question."""
    logger.info("Loading Questions.csv …")
    questions = pd.read_csv(
        QUESTIONS_CSV,
        usecols=["Id", "Title", "Body", "Score"],
        encoding="latin-1",
        low_memory=False,
    )
    questions.rename(columns={
        "Id":    "QuestionId",
        "Body":  "QuestionBody",
        "Score": "QuestionScore",
    }, inplace=True)

    logger.info("Loading Answers.csv …")
    answers = pd.read_csv(
        ANSWERS_CSV,
        usecols=["Id", "ParentId", "Body", "Score"],
        encoding="latin-1",
        low_memory=False,
    )
    answers.rename(columns={
        "Id":       "AnswerId",
        "ParentId": "QuestionId",
        "Body":     "AnswerBody",
        "Score":    "AnswerScore",
    }, inplace=True)

    logger.info("Selecting top answer per question …")
    top_answers = (
        answers.sort_values("AnswerScore", ascending=False)
        .groupby("QuestionId", as_index=False)
        .first()
    )

    logger.info("Merging …")
    merged = questions.merge(
        top_answers[["QuestionId", "AnswerBody", "AnswerScore"]],
        on="QuestionId",
        how="inner",
    )
    merged.dropna(subset=["Title", "AnswerBody"], inplace=True)
    logger.info(f"Merged dataset: {len(merged):,} rows")
    return merged


def build_bm25_index(raw_chunks: list[str]) -> BM25Okapi:
    """Build BM25 index from raw chunks (no header — keyword matching)."""
    logger.info("Tokenising for BM25 …")
    tokenized = [c.lower().split() for c in tqdm(raw_chunks, desc="BM25 tokenise")]
    bm25 = BM25Okapi(tokenized)
    logger.info(f"BM25 index built: {len(tokenized):,} docs")
    return bm25


def embed_and_index(
    raw_chunks:   list[str],
    embed_chunks: list[str],
    metadata:     list[dict],
) -> None:
    """
    Embed header-enriched chunks → FAISS.
    Index raw chunks → BM25.
    Save everything to disk.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info(f"Embedding {len(embed_chunks):,} header-enriched chunks …")
    all_embeddings = []
    for i in tqdm(range(0, len(embed_chunks), BATCH_SIZE), desc="Embedding"):
        batch = embed_chunks[i : i + BATCH_SIZE]
        vecs  = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(vecs)

    all_embeddings = np.vstack(all_embeddings).astype("float32")
    dim = all_embeddings.shape[1]
    logger.info(f"Embeddings shape: {all_embeddings.shape}")

    logger.info("Building FAISS IndexFlatIP …")
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)
    logger.info(f"FAISS: {index.ntotal:,} vectors")

    logger.info(f"Saving FAISS → {FAISS_INDEX_PATH}")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    logger.info(f"Saving metadata → {METADATA_PATH}")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump({"chunks": raw_chunks, "metadata": metadata}, f)

    bm25 = build_bm25_index(raw_chunks)
    logger.info(f"Saving BM25 → {BM25_INDEX_PATH}")
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    logger.info("✅ Ingestion complete — FAISS (header-enriched) + BM25 (raw) saved.")


def main():
    for path in [QUESTIONS_CSV, ANSWERS_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Download from https://www.kaggle.com/datasets/stackoverflow/pythonquestions\n"
                "and place Questions.csv + Answers.csv in the data/ folder."
            )

    df = load_and_merge()
    raw_chunks, embed_chunks, meta = build_chunks(df)
    embed_and_index(raw_chunks, embed_chunks, meta)


if __name__ == "__main__":
    main()
