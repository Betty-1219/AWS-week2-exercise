"""
chunker.py
==========
Splits FAQ answer text into overlapping character-based chunks.

Why chunk?
----------
Language models and vector databases have practical limits on input length.
Chunking also helps precision: if a user searches for something that only
appears in one paragraph of a long answer, retrieving that specific chunk
gives a more focused result than retrieving the entire answer.

Chunking strategy
-----------------
This implementation uses **character-based sliding windows** with overlap:

  chunk_size  – maximum number of characters per chunk (default: 500)
  overlap     – number of characters from the end of one chunk that are
                repeated at the start of the next chunk (default: 50)

Example with chunk_size=20, overlap=5 on text "ABCDEFGHIJKLMNOPQRSTU":
  Chunk 0: "ABCDEFGHIJKLMNOPQRST"  (chars 0–19)
  Chunk 1: "PQRSTUVWXYZ..."        (chars 15–34)  ← 5-char overlap

Short answers that fit within one chunk are returned as a single chunk.
Empty text produces no chunks.

Each chunk is returned as a dictionary with the following keys:
  - chunk_id:    A unique string ID, e.g. "faq_3_chunk_0"
  - faq_id:      The integer id of the source FAQ row
  - question:    The FAQ question (for display and metadata)
  - chunk_index: Position of the chunk within this FAQ's answer (0-based)
  - text:        The chunk text

Usage example
-------------
>>> from src.chunker import chunk_faq_row, chunk_dataframe
>>> chunks = chunk_faq_row(faq_id=1, question="What is Python?",
...                        answer="Python is a language...", chunk_size=100)
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd


# ---------------------------------------------------------------------------
# Type alias for a single chunk record
# ---------------------------------------------------------------------------

class ChunkRecord(TypedDict):
    chunk_id: str
    faq_id: int
    question: str
    chunk_index: int
    text: str


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split a string into overlapping character-level chunks.

    Parameters
    ----------
    text:       The text to split.
    chunk_size: Maximum characters per chunk.  Must be > 0.
    overlap:    How many characters from the end of the previous chunk to
                repeat at the start of the next one.  Must be >= 0 and
                < chunk_size.

    Returns
    -------
    A list of non-empty string chunks.  Returns [] for empty/None input.

    Notes
    -----
    The chunks are character-based, not word-based.  A word may be split
    across a boundary in rare cases.  For this FAQ dataset (plain prose)
    this is acceptable and keeps the implementation simple and predictable.
    Word-based splitting would require a tokeniser and adds complexity.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}.")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})."
        )

    text = text.strip()

    # If the entire text fits in one chunk, return it as-is (no splitting)
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - overlap  # how far to advance the window each time
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # never add an empty chunk
            chunks.append(chunk)
        start += step

    return chunks


def chunk_faq_row(
    faq_id: int,
    question: str,
    answer: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[ChunkRecord]:
    """Chunk a single FAQ answer and attach metadata.

    Parameters
    ----------
    faq_id:     Integer identifier of the source FAQ row.
    question:   The FAQ question text (stored as metadata).
    answer:     The FAQ answer text to chunk.
    chunk_size: Maximum characters per chunk.
    overlap:    Character overlap between consecutive chunks.

    Returns
    -------
    A list of ChunkRecord dictionaries.  Returns [] if the answer is empty.
    """
    raw_chunks = chunk_text(answer, chunk_size=chunk_size, overlap=overlap)

    records: list[ChunkRecord] = []
    for idx, chunk in enumerate(raw_chunks):
        records.append(
            ChunkRecord(
                chunk_id=f"faq_{faq_id}_chunk_{idx}",
                faq_id=int(faq_id),
                question=str(question),
                chunk_index=idx,
                text=chunk,
            )
        )
    return records


def chunk_dataframe(
    df: pd.DataFrame,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[ChunkRecord]:
    """Chunk every FAQ row in a DataFrame.

    Iterates over each row and calls chunk_faq_row.  Progress is logged.

    Parameters
    ----------
    df:         A DataFrame with at least 'id', 'question', and 'answer'
                columns (as produced by data_loader + text_cleaner).
    chunk_size: Maximum characters per chunk.
    overlap:    Character overlap between consecutive chunks.

    Returns
    -------
    A flat list of ChunkRecord dictionaries ready for embedding.

    Raises
    ------
    ValueError: If required columns are missing.
    """
    required = {"id", "question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"chunk_dataframe requires columns {required}; missing: {missing}"
        )

    all_chunks: list[ChunkRecord] = []
    for _, row in df.iterrows():
        chunks = chunk_faq_row(
            faq_id=row["id"],
            question=row["question"],
            answer=row["answer"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)

    total_faqs = len(df)
    total_chunks = len(all_chunks)
    avg = total_chunks / total_faqs if total_faqs else 0
    print(
        f"[chunker] Created {total_chunks} chunk(s) from {total_faqs} FAQ(s) "
        f"(avg {avg:.1f} chunks/FAQ, chunk_size={chunk_size}, overlap={overlap})."
    )
    return all_chunks
