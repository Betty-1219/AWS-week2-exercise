"""
test_pipeline.py
================
Unit tests for the Mini Semantic Search pipeline.

Run with:
    pytest tests/ -v

Most tests are fully offline (no model download required).
The model-dependent tests are marked with @pytest.mark.slow and skipped
by default.  To run them:
    pytest tests/ -v -m slow
"""

from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers — create temporary files
# ---------------------------------------------------------------------------

SAMPLE_CSV_CONTENT = textwrap.dedent("""\
    id,question,answer
    1,What is Python?,Python is a high-level programming language.
    2,What is Git?,Git is a distributed version control system.
    3,What is an API?,An API is an interface that allows programs to communicate.
""")

SAMPLE_JSON_CONTENT = json.dumps(
    [
        {
            "id": "1",
            "question": "What is Python?",
            "answer": "Python is a high-level programming language.",
        },
        {
            "id": "2",
            "question": "What is Git?",
            "answer": "Git is a distributed version control system.",
        },
    ]
)


# ===========================================================================
# 1. Data Loader tests
# ===========================================================================

class TestLoadCSV:
    """Tests for src.data_loader.load_csv."""

    def test_load_valid_csv(self, tmp_path: Path) -> None:
        """A well-formed CSV is loaded into a DataFrame."""
        from src.data_loader import load_csv

        csv_file = tmp_path / "faqs.csv"
        csv_file.write_text(SAMPLE_CSV_CONTENT, encoding="utf-8")
        df = load_csv(csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "question" in df.columns
        assert "answer" in df.columns

    def test_load_csv_file_not_found(self) -> None:
        """Loading a nonexistent file raises FileNotFoundError."""
        from src.data_loader import load_csv

        with pytest.raises(FileNotFoundError, match="not found"):
            load_csv("nonexistent_file.csv")

    def test_load_csv_missing_columns(self, tmp_path: Path) -> None:
        """CSV without required columns raises ValueError."""
        from src.data_loader import load_csv

        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col_a,col_b\nfoo,bar\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required column"):
            load_csv(bad_csv)

    def test_load_csv_empty_file(self, tmp_path: Path) -> None:
        """Empty CSV (header only) raises ValueError."""
        from src.data_loader import load_csv

        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("question,answer\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_csv(empty_csv)

    def test_load_csv_strips_column_names(self, tmp_path: Path) -> None:
        """Column names with whitespace are normalised."""
        from src.data_loader import load_csv

        csv_file = tmp_path / "spaces.csv"
        csv_file.write_text(
            " Question , Answer \nWhat is Git?,Git is VCS.\n",
            encoding="utf-8",
        )
        df = load_csv(csv_file)
        assert "question" in df.columns
        assert "answer" in df.columns

    def test_load_csv_removes_exact_duplicates(self, tmp_path: Path) -> None:
        """Exact duplicate rows are removed."""
        from src.data_loader import load_csv

        content = (
            "question,answer\n"
            "What is Python?,Python is great.\n"
            "What is Python?,Python is great.\n"  # duplicate
            "What is Git?,Git is VCS.\n"
        )
        csv_file = tmp_path / "dups.csv"
        csv_file.write_text(content, encoding="utf-8")
        df = load_csv(csv_file)
        assert len(df) == 2

    def test_load_csv_handles_missing_values(self, tmp_path: Path) -> None:
        """Rows with missing values in one column are kept; both-missing rows dropped."""
        from src.data_loader import load_csv

        content = (
            "question,answer\n"
            "What is Python?,Python is great.\n"
            "What is Git?,\n"          # empty answer — kept
            ",Standalone answer.\n"    # empty question — kept
            ",\n"                       # both empty — dropped
        )
        csv_file = tmp_path / "missing.csv"
        csv_file.write_text(content, encoding="utf-8")
        df = load_csv(csv_file)
        # Row with both missing is dropped; two rows with one missing are kept
        assert len(df) == 3


class TestLoadJSON:
    """Tests for src.data_loader.load_json."""

    def test_load_valid_json_array(self, tmp_path: Path) -> None:
        """A JSON array of FAQ objects is loaded correctly."""
        from src.data_loader import load_json

        json_file = tmp_path / "faqs.json"
        json_file.write_text(SAMPLE_JSON_CONTENT, encoding="utf-8")
        df = load_json(json_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "question" in df.columns
        assert "answer" in df.columns

    def test_load_valid_json_with_faqs_key(self, tmp_path: Path) -> None:
        """JSON with a top-level 'faqs' key is loaded correctly."""
        from src.data_loader import load_json

        data = {
            "faqs": [
                {"question": "Q1?", "answer": "A1."},
                {"question": "Q2?", "answer": "A2."},
            ]
        }
        json_file = tmp_path / "faqs_keyed.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")
        df = load_json(json_file)
        assert len(df) == 2

    def test_load_json_file_not_found(self) -> None:
        """Loading a nonexistent JSON file raises FileNotFoundError."""
        from src.data_loader import load_json

        with pytest.raises(FileNotFoundError):
            load_json("no_file.json")

    def test_load_json_invalid_structure(self, tmp_path: Path) -> None:
        """JSON with unsupported structure raises ValueError."""
        from src.data_loader import load_json

        json_file = tmp_path / "bad.json"
        json_file.write_text('{"not_faqs": 123}', encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported JSON structure"):
            load_json(json_file)

    def test_load_json_empty_array(self, tmp_path: Path) -> None:
        """JSON with an empty array raises ValueError."""
        from src.data_loader import load_json

        json_file = tmp_path / "empty.json"
        json_file.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="no records"):
            load_json(json_file)

    def test_load_json_missing_columns(self, tmp_path: Path) -> None:
        """JSON without required fields raises ValueError."""
        from src.data_loader import load_json

        data = [{"title": "Q?", "body": "A."}]
        json_file = tmp_path / "wrong_fields.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required column"):
            load_json(json_file)


class TestLoadFAQs:
    """Tests for src.data_loader.load_faqs (auto-detect by extension)."""

    def test_detects_csv(self, tmp_path: Path) -> None:
        from src.data_loader import load_faqs

        csv_file = tmp_path / "faqs.csv"
        csv_file.write_text(SAMPLE_CSV_CONTENT, encoding="utf-8")
        df = load_faqs(csv_file)
        assert len(df) == 3

    def test_detects_json(self, tmp_path: Path) -> None:
        from src.data_loader import load_faqs

        json_file = tmp_path / "faqs.json"
        json_file.write_text(SAMPLE_JSON_CONTENT, encoding="utf-8")
        df = load_faqs(json_file)
        assert len(df) == 2

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        from src.data_loader import load_faqs

        txt_file = tmp_path / "faqs.txt"
        txt_file.write_text("some text", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_faqs(txt_file)


# ===========================================================================
# 2. Text Cleaner tests
# ===========================================================================

class TestCleanText:
    """Tests for src.text_cleaner.clean_text."""

    def test_strips_leading_trailing_whitespace(self) -> None:
        from src.text_cleaner import clean_text

        assert clean_text("  hello  ") == "hello"

    def test_normalises_windows_line_endings(self) -> None:
        from src.text_cleaner import clean_text

        result = clean_text("line1\r\nline2")
        assert "\r" not in result
        assert "line1" in result
        assert "line2" in result

    def test_collapses_multiple_blank_lines(self) -> None:
        from src.text_cleaner import clean_text

        result = clean_text("para1\n\n\n\n\npara2")
        # Should have at most one blank line between paragraphs
        assert "\n\n\n" not in result
        assert "para1" in result
        assert "para2" in result

    def test_preserves_technical_terms(self) -> None:
        from src.text_cleaner import clean_text

        text = "Use pip install requests==2.31.0 and check os.path.join()."
        result = clean_text(text)
        # None of these should be altered
        assert "pip" in result
        assert "requests==2.31.0" in result
        assert "os.path.join()" in result

    def test_preserves_url(self) -> None:
        from src.text_cleaner import clean_text

        text = "Visit https://docs.python.org/3/ for docs."
        result = clean_text(text)
        assert "https://docs.python.org/3/" in result

    def test_handles_none_input(self) -> None:
        from src.text_cleaner import clean_text

        assert clean_text(None) == ""  # type: ignore[arg-type]

    def test_handles_empty_string(self) -> None:
        from src.text_cleaner import clean_text

        assert clean_text("") == ""

    def test_collapses_extra_spaces(self) -> None:
        from src.text_cleaner import clean_text

        result = clean_text("hello   world")
        assert result == "hello world"


class TestCleanDataFrame:
    """Tests for src.text_cleaner.clean_dataframe."""

    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_cleans_question_and_answer(self) -> None:
        from src.text_cleaner import clean_dataframe

        df = self._make_df([
            {"id": 1, "question": "  What is Python? ", "answer": "  A language.  "},
        ])
        result = clean_dataframe(df)
        assert result.loc[0, "question"] == "What is Python?"
        assert result.loc[0, "answer"] == "A language."

    def test_drops_rows_with_empty_answer_after_cleaning(self) -> None:
        from src.text_cleaner import clean_dataframe

        df = self._make_df([
            {"id": 1, "question": "What is Python?", "answer": "  A language.  "},
            {"id": 2, "question": "Empty answer?", "answer": "   "},
        ])
        result = clean_dataframe(df)
        assert len(result) == 1
        assert result.loc[0, "question"] == "What is Python?"

    def test_raises_on_missing_columns(self) -> None:
        from src.text_cleaner import clean_dataframe

        df = pd.DataFrame({"col_a": ["x"], "col_b": ["y"]})
        with pytest.raises(ValueError, match="missing"):
            clean_dataframe(df)

    def test_does_not_mutate_original(self) -> None:
        from src.text_cleaner import clean_dataframe

        df = self._make_df([
            {"id": 1, "question": "  Q?  ", "answer": "  A.  "},
        ])
        original_q = df.loc[0, "question"]
        clean_dataframe(df)
        assert df.loc[0, "question"] == original_q  # unchanged


# ===========================================================================
# 3. Chunker tests
# ===========================================================================

class TestChunkText:
    """Tests for src.chunker.chunk_text."""

    def test_short_text_returns_single_chunk(self) -> None:
        from src.chunker import chunk_text

        text = "Short answer."
        result = chunk_text(text, chunk_size=500, overlap=50)
        assert result == [text]

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        from src.chunker import chunk_text

        text = "A" * 1100  # definitely longer than default 500
        result = chunk_text(text, chunk_size=500, overlap=50)
        assert len(result) > 1

    def test_overlap_is_respected(self) -> None:
        from src.chunker import chunk_text

        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26 chars
        # chunk_size=10, overlap=5 → step=5
        # Chunk 0: chars 0-9  → "ABCDEFGHIJ"
        # Chunk 1: chars 5-14 → "FGHIJKLMNO"
        result = chunk_text(text, chunk_size=10, overlap=5)
        assert len(result) >= 2
        # The start of chunk 1 should share chars with the end of chunk 0
        assert result[1][0] == result[0][5]

    def test_empty_text_returns_empty_list(self) -> None:
        from src.chunker import chunk_text

        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_none_returns_empty_list(self) -> None:
        from src.chunker import chunk_text

        assert chunk_text(None) == []  # type: ignore[arg-type]

    def test_invalid_chunk_size_raises(self) -> None:
        from src.chunker import chunk_text

        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("hello", chunk_size=0)

    def test_overlap_gte_chunk_size_raises(self) -> None:
        from src.chunker import chunk_text

        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello", chunk_size=10, overlap=10)

    def test_no_empty_chunks(self) -> None:
        from src.chunker import chunk_text

        text = "x" * 200
        result = chunk_text(text, chunk_size=50, overlap=10)
        assert all(len(c) > 0 for c in result)


class TestChunkFAQRow:
    """Tests for src.chunker.chunk_faq_row."""

    def test_returns_chunk_records(self) -> None:
        from src.chunker import chunk_faq_row

        records = chunk_faq_row(
            faq_id=1,
            question="What is Python?",
            answer="Python is a language. " * 30,
            chunk_size=100,
            overlap=20,
        )
        assert len(records) > 0
        for r in records:
            assert "chunk_id" in r
            assert "faq_id" in r
            assert "question" in r
            assert "chunk_index" in r
            assert "text" in r

    def test_chunk_ids_are_unique(self) -> None:
        from src.chunker import chunk_faq_row

        records = chunk_faq_row(
            faq_id=5,
            question="Q?",
            answer="Answer. " * 50,
            chunk_size=100,
            overlap=20,
        )
        ids = [r["chunk_id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_chunk_id_format(self) -> None:
        from src.chunker import chunk_faq_row

        records = chunk_faq_row(
            faq_id=7,
            question="Q?",
            answer="Short answer.",
        )
        assert records[0]["chunk_id"] == "faq_7_chunk_0"

    def test_empty_answer_returns_empty_list(self) -> None:
        from src.chunker import chunk_faq_row

        assert chunk_faq_row(faq_id=1, question="Q?", answer="") == []
        assert chunk_faq_row(faq_id=1, question="Q?", answer="   ") == []

    def test_faq_id_stored_correctly(self) -> None:
        from src.chunker import chunk_faq_row

        records = chunk_faq_row(faq_id=42, question="Q?", answer="A.")
        assert records[0]["faq_id"] == 42

    def test_question_stored_correctly(self) -> None:
        from src.chunker import chunk_faq_row

        records = chunk_faq_row(faq_id=1, question="My question?", answer="A.")
        assert records[0]["question"] == "My question?"

    def test_chunk_index_is_sequential(self) -> None:
        from src.chunker import chunk_faq_row

        records = chunk_faq_row(
            faq_id=1,
            question="Q?",
            answer="A" * 600,
            chunk_size=200,
            overlap=20,
        )
        indices = [r["chunk_index"] for r in records]
        assert indices == list(range(len(records)))


class TestChunkDataframe:
    """Tests for src.chunker.chunk_dataframe."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": [1, 2],
                "question": ["What is Python?", "What is Git?"],
                "answer": [
                    "Python is a programming language.",
                    "Git is a version control system.",
                ],
            }
        )

    def test_returns_list_of_dicts(self) -> None:
        from src.chunker import chunk_dataframe

        df = self._make_df()
        result = chunk_dataframe(df)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_correct_number_of_chunks_for_short_answers(self) -> None:
        from src.chunker import chunk_dataframe

        df = self._make_df()
        # Both answers are short — each produces exactly one chunk
        result = chunk_dataframe(df, chunk_size=500, overlap=50)
        assert len(result) == 2

    def test_raises_on_missing_columns(self) -> None:
        from src.chunker import chunk_dataframe

        df = pd.DataFrame({"question": ["Q?"], "answer": ["A."]})
        # Missing 'id' column
        with pytest.raises(ValueError, match="missing"):
            chunk_dataframe(df)


# ===========================================================================
# 4. Semantic Search unit tests (offline — no model download)
# ===========================================================================

# Skip these tests if chromadb is not installed yet.
# They do not need the embedding model, but they do need chromadb.
chromadb_available = pytest.importorskip  # used below per-test


def _chromadb_installed() -> bool:
    try:
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False


requires_chromadb = pytest.mark.skipif(
    not _chromadb_installed(),
    reason="chromadb not installed — run: pip install chromadb",
)


@requires_chromadb
class TestSemanticSearchValidation:
    """Validate query-checking logic without a live model or database.

    Requires chromadb to be installed; skipped automatically otherwise.
    """

    def test_empty_query_raises_value_error(self, tmp_path: Path) -> None:
        """search() with an empty string raises ValueError."""
        from src.semantic_search import SemanticSearch

        ss = SemanticSearch(db_dir=tmp_path / "db")
        with pytest.raises(ValueError, match="non-empty"):
            ss.search("")

    def test_whitespace_only_query_raises_value_error(self, tmp_path: Path) -> None:
        from src.semantic_search import SemanticSearch

        ss = SemanticSearch(db_dir=tmp_path / "db")
        with pytest.raises(ValueError, match="non-empty"):
            ss.search("   ")

    def test_collection_size_on_empty_db(self, tmp_path: Path) -> None:
        """collection_size() returns 0 on a fresh (empty) database."""
        from src.semantic_search import SemanticSearch

        ss = SemanticSearch(db_dir=tmp_path / "db")
        assert ss.collection_size() == 0

    def test_search_on_empty_collection_raises_runtime_error(
        self, tmp_path: Path
    ) -> None:
        """Searching an empty collection raises RuntimeError."""
        from src.semantic_search import SemanticSearch

        ss = SemanticSearch(db_dir=tmp_path / "db")
        with pytest.raises(RuntimeError, match="empty"):
            ss.search("test query")


# ===========================================================================
# 5. Integration-style tests (marked slow — require model download)
# ===========================================================================

@pytest.mark.slow
@requires_chromadb
class TestSemanticSearchIntegration:
    """End-to-end tests that download the embedding model.

    Run with:  pytest -m slow
    Skip with: pytest -m "not slow"   (default)
    """

    def test_build_and_search(self, tmp_path: Path) -> None:
        """Build an index from 3 chunks, search, and verify top result."""
        from src.chunker import ChunkRecord
        from src.semantic_search import SemanticSearch

        chunks: list[ChunkRecord] = [
            ChunkRecord(
                chunk_id="faq_1_chunk_0",
                faq_id=1,
                question="What is Python?",
                chunk_index=0,
                text="Python is a high-level programming language known for its readability.",
            ),
            ChunkRecord(
                chunk_id="faq_2_chunk_0",
                faq_id=2,
                question="What is Git?",
                chunk_index=0,
                text="Git is a distributed version control system for tracking changes in code.",
            ),
            ChunkRecord(
                chunk_id="faq_3_chunk_0",
                faq_id=3,
                question="What is machine learning?",
                chunk_index=0,
                text="Machine learning is a subfield of AI where models learn from data.",
            ),
        ]

        ss = SemanticSearch(
            collection_name="test_collection",
            db_dir=tmp_path / "db",
        )
        ss.build_index(chunks, reset=True)

        results = ss.search("version control for code", top_k=1, print_results=False)
        assert len(results) == 1
        assert results[0]["faq_id"] == 2  # Git answer is the most relevant

    def test_top_k_respects_limit(self, tmp_path: Path) -> None:
        from src.chunker import ChunkRecord
        from src.semantic_search import SemanticSearch

        chunks = [
            ChunkRecord(
                chunk_id=f"faq_{i}_chunk_0",
                faq_id=i,
                question=f"Question {i}?",
                chunk_index=0,
                text=f"Answer number {i} with some content.",
            )
            for i in range(1, 6)  # 5 chunks
        ]

        ss = SemanticSearch(
            collection_name="test_topk",
            db_dir=tmp_path / "db2",
        )
        ss.build_index(chunks, reset=True)

        results = ss.search("some content", top_k=3, print_results=False)
        assert len(results) == 3

    def test_results_have_expected_keys(self, tmp_path: Path) -> None:
        from src.chunker import ChunkRecord
        from src.semantic_search import SemanticSearch

        chunks = [
            ChunkRecord(
                chunk_id="faq_1_chunk_0",
                faq_id=1,
                question="What is an API?",
                chunk_index=0,
                text="An API is a set of rules that allows programs to communicate.",
            )
        ]

        ss = SemanticSearch(
            collection_name="test_keys",
            db_dir=tmp_path / "db3",
        )
        ss.build_index(chunks, reset=True)

        results = ss.search("how to communicate between programs", top_k=1, print_results=False)
        assert len(results) == 1
        result = results[0]
        for key in ("rank", "question", "chunk_text", "faq_id", "chunk_index", "distance"):
            assert key in result, f"Missing key: {key}"

    def test_distance_is_float(self, tmp_path: Path) -> None:
        from src.chunker import ChunkRecord
        from src.semantic_search import SemanticSearch

        chunks = [
            ChunkRecord(
                chunk_id="faq_1_chunk_0",
                faq_id=1,
                question="What is Python?",
                chunk_index=0,
                text="Python is a programming language.",
            )
        ]

        ss = SemanticSearch(
            collection_name="test_distance",
            db_dir=tmp_path / "db4",
        )
        ss.build_index(chunks, reset=True)

        results = ss.search("programming language", top_k=1, print_results=False)
        assert isinstance(results[0]["distance"], float)
