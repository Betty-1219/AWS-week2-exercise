"""
data_loader.py
==============
Functions for loading and validating FAQ datasets from CSV and JSON files.

The dataset must contain at minimum two columns (or JSON fields):
  - ``question``: the FAQ question text
  - ``answer``:   the FAQ answer text

Both CSV and JSON loaders return a cleaned pandas DataFrame with those two
columns guaranteed plus an ``id`` column.  The id column is an integer index
used to track which FAQ each chunk came from.

Usage example
-------------
>>> from src.data_loader import load_csv, load_json
>>> df = load_csv("data/faqs.csv")
>>> df.head(3)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"question", "answer"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate that a DataFrame has the required columns and non-empty data.

    Parameters
    ----------
    df:     The DataFrame to validate.
    source: A descriptive label used in error messages (e.g. the file path).

    Returns
    -------
    A cleaned DataFrame with an 'id' column added (1-based integer).

    Raises
    ------
    ValueError: If required columns are missing, or the file is empty.
    """
    if df.empty:
        raise ValueError(
            f"The dataset loaded from '{source}' is empty. "
            "Please provide a file with at least one FAQ row."
        )

    # Normalise column names: strip whitespace, lower-case
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset '{source}' is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}. "
            "Rename your columns to 'question' and 'answer'."
        )

    # Drop rows where BOTH question AND answer are completely absent
    before = len(df)
    df = df.dropna(subset=["question", "answer"], how="all")
    dropped = before - len(df)
    if dropped:
        print(f"[data_loader] Dropped {dropped} row(s) where both question and answer were missing.")

    # Convert to string, filling remaining NaN with empty string temporarily
    # Use .loc to avoid SettingWithCopyWarning on DataFrame slices
    df = df.copy()
    df.loc[:, "question"] = df["question"].fillna("").astype(str).str.strip()
    df.loc[:, "answer"] = df["answer"].fillna("").astype(str).str.strip()

    # Warn about rows with empty questions or answers (but keep them for now;
    # later processing will skip empty chunks)
    empty_q = (df["question"] == "").sum()
    empty_a = (df["answer"] == "").sum()
    if empty_q:
        print(f"[data_loader] Warning: {empty_q} row(s) have an empty question.")
    if empty_a:
        print(f"[data_loader] Warning: {empty_a} row(s) have an empty answer.")

    # Remove exact duplicate rows (same question AND answer)
    before = len(df)
    df = df.drop_duplicates(subset=["question", "answer"])
    dups = before - len(df)
    if dups:
        print(f"[data_loader] Removed {dups} exact duplicate FAQ row(s).")

    # Reset index and assign a clean integer 'id' column (1-based)
    df = df.reset_index(drop=True)
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    else:
        # If an id column already exists keep it but ensure it is an int
        df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(filepath: str | Path) -> pd.DataFrame:
    """Load FAQ data from a CSV file.

    The CSV must have at minimum a 'question' column and an 'answer' column.
    Column names are case-insensitive and leading/trailing whitespace is
    stripped.  An optional 'id' column is preserved if present; otherwise a
    sequential integer id is added automatically.

    Parameters
    ----------
    filepath: Path to the CSV file (str or pathlib.Path).

    Returns
    -------
    A validated pandas DataFrame with columns: id, question, answer
    (plus any extra columns the file may contain).

    Raises
    ------
    FileNotFoundError: If the file does not exist.
    ValueError:        If required columns are missing or the file is empty.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"CSV file not found: '{filepath}'. "
            "Check the path and make sure the file exists."
        )

    try:
        df = pd.read_csv(filepath, encoding="utf-8", dtype=str)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file '{filepath}': {exc}") from exc

    print(f"[data_loader] Loaded {len(df)} rows from '{filepath}'.")
    return _validate_dataframe(df, str(filepath))


def load_json(filepath: str | Path) -> pd.DataFrame:
    """Load FAQ data from a JSON file.

    Supported JSON shapes:
      1. A JSON array of objects: [{"question": "...", "answer": "..."}, ...]
      2. A JSON object with a top-level 'faqs' key:
         {"faqs": [{"question": "...", "answer": "..."}, ...]}

    Parameters
    ----------
    filepath: Path to the JSON file (str or pathlib.Path).

    Returns
    -------
    A validated pandas DataFrame with columns: id, question, answer.

    Raises
    ------
    FileNotFoundError: If the file does not exist.
    ValueError:        If the JSON structure is unrecognised or required
                       fields are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"JSON file not found: '{filepath}'. "
            "Check the path and make sure the file exists."
        )

    try:
        with filepath.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse JSON file '{filepath}': {exc}"
        ) from exc

    # Normalise to a list of dicts
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and "faqs" in raw:
        records = raw["faqs"]
    else:
        raise ValueError(
            f"Unsupported JSON structure in '{filepath}'. "
            "Expected a JSON array or an object with a 'faqs' key."
        )

    if not records:
        raise ValueError(f"JSON file '{filepath}' contains no records.")

    df = pd.DataFrame(records).astype(str)
    print(f"[data_loader] Loaded {len(df)} rows from '{filepath}'.")
    return _validate_dataframe(df, str(filepath))


def load_faqs(filepath: str | Path) -> pd.DataFrame:
    """Convenience loader that detects CSV or JSON by file extension.

    Parameters
    ----------
    filepath: Path to a .csv or .json file.

    Returns
    -------
    A validated pandas DataFrame.

    Raises
    ------
    ValueError: If the file extension is not .csv or .json.
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    if ext == ".csv":
        return load_csv(filepath)
    elif ext == ".json":
        return load_json(filepath)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}' for '{filepath}'. "
            "Use a .csv or .json file."
        )
