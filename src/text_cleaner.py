"""
text_cleaner.py
===============
Functions for cleaning raw FAQ text before chunking and embedding.

Design goals
------------
- Remove noise (extra whitespace, inconsistent line endings) without
  destroying meaningful content.
- Preserve technical terms, code identifiers, URLs, punctuation, and
  numbers — these carry meaning in a programming FAQ dataset.
- Be transparent about what is changed: the functions are small, readable,
  and easy to reason about.

Usage example
-------------
>>> from src.text_cleaner import clean_text, clean_dataframe
>>> clean_text("  Hello\\n\\n  World!  ")
'Hello  World!'
>>> import pandas as pd
>>> df = pd.DataFrame({"id":[1],"question":["  What is Python? "],"answer":["  A language. "]})
>>> clean_dataframe(df)
"""

from __future__ import annotations

import re

import pandas as pd


# ---------------------------------------------------------------------------
# Single-string cleaner
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lightly clean a single string of text.

    Steps performed
    ~~~~~~~~~~~~~~~
    1. Return an empty string immediately for None or non-string inputs.
    2. Strip leading and trailing whitespace.
    3. Normalise Windows (\\r\\n) and old Mac (\\r) line endings to \\n.
    4. Collapse sequences of 3 or more blank lines to a single blank line
       (preserves intentional paragraph breaks).
    5. Collapse runs of horizontal whitespace (spaces/tabs) on each line to
       a single space — this removes accidental double-spaces without
       destroying sentence structure or technical identifiers.

    What is intentionally NOT changed
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - Punctuation (parentheses, colons, commas, hyphens, underscores, etc.)
    - URLs and file paths
    - Code-like identifiers (snake_case, CamelCase, ALL_CAPS)
    - Numbers and version strings (e.g. Python 3.13, requests==2.31.0)
    - Single or double blank lines that separate paragraphs

    Parameters
    ----------
    text: The raw text to clean.

    Returns
    -------
    Cleaned text as a string.  Returns '' for empty/None input.
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. Strip outer whitespace
    text = text.strip()

    # 2. Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Collapse runs of 3+ consecutive blank lines to 2 (one blank line)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Collapse horizontal whitespace runs to a single space per line
    #    We process line-by-line to avoid collapsing intentional indentation
    #    in code blocks.  For this FAQ dataset that is not critical, but it
    #    is good practice.
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in lines]
    text = "\n".join(lines)

    # 5. Final strip
    return text.strip()


# ---------------------------------------------------------------------------
# DataFrame-level cleaner
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to the 'question' and 'answer' columns of a DataFrame.

    Rows where the answer becomes empty after cleaning are dropped with a
    warning (an empty answer cannot produce useful search results).

    Rows where the question becomes empty are warned about but kept — the
    answer text alone can still be indexed.

    Parameters
    ----------
    df: A DataFrame produced by data_loader.load_csv / load_json.
        Must contain 'question' and 'answer' columns.

    Returns
    -------
    A new DataFrame with cleaned text columns.  The original DataFrame is
    not modified (a copy is returned).

    Raises
    ------
    ValueError: If required columns are missing.
    """
    required = {"question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"clean_dataframe requires columns {required}; "
            f"missing: {missing}"
        )

    df = df.copy()
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)

    # Drop rows whose answer is now empty (they cannot produce useful chunks)
    before = len(df)
    df = df[df["answer"] != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(
            f"[text_cleaner] Dropped {dropped} row(s) whose answer was "
            "empty after cleaning."
        )

    # Warn about empty questions (but keep the rows)
    empty_q = (df["question"] == "").sum()
    if empty_q:
        print(
            f"[text_cleaner] Warning: {empty_q} row(s) have an empty "
            "question after cleaning; they will still be indexed."
        )

    print(f"[text_cleaner] Cleaned {len(df)} FAQ rows.")
    return df
