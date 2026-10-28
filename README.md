# Mini Semantic Search

A beginner-friendly semantic search system for programming FAQs, built with
Python, Sentence Transformers, ChromaDB, and Pandas.

---

## Overview

This project demonstrates how to build a **semantic search engine** from
scratch. Unlike keyword search (which matches exact words), semantic search
understands the *meaning* of a query. Asking _"How do I save code online?"_
will correctly match _"Upload your project to GitHub"_ — even though no
words overlap.

The system loads a dataset of programming FAQs, converts each answer into
a numerical embedding using a pretrained AI model, stores the embeddings in
a local vector database, and retrieves the most relevant results for any
natural-language question.

---

## Problem statement

Students and developers often search documentation or FAQs using exact
keywords and miss relevant answers because the wording is different.
This project shows how embedding-based retrieval solves that problem
for a small, practical FAQ dataset.

---

## Features

- Load FAQs from CSV or JSON using Pandas
- Validate data automatically (missing columns, empty rows, duplicates)
- Clean text while preserving code, URLs, and technical identifiers
- Split long answers into overlapping character chunks
- Embed each chunk using `all-MiniLM-L6-v2` (~90 MB, 384 dimensions)
- Store and query embeddings with persistent local ChromaDB
- Return the top 3 most relevant results with distance scores
- Full Jupyter Notebook walkthrough
- pytest test suite (offline-friendly)

---

## Architecture and pipeline

```
faqs.csv
   │
   ▼
data_loader.py      ← load_csv() / load_json() / load_faqs()
   │   Validates columns, handles missing values, removes duplicates
   ▼
text_cleaner.py     ← clean_dataframe()
   │   Strips whitespace, normalises line endings
   ▼
chunker.py          ← chunk_dataframe()
   │   Splits answers into overlapping 500-char chunks
   ▼
semantic_search.py  ← SemanticSearch.build_index()
   │   Embeds chunks with all-MiniLM-L6-v2
   │   Stores vectors + metadata in ChromaDB (cosine space)
   ▼
SemanticSearch.search("your question")
   │   Embeds query → nearest-neighbour search → top-3 results
   ▼
Ranked results with question, chunk text, FAQ id, cosine distance
```

---

## Project structure

```
mini-semantic-search/
├── data/
│   └── faqs.csv               ← 24-row FAQ dataset
├── notebooks/
│   └── mini_semantic_search.ipynb  ← full walkthrough
├── src/
│   ├── __init__.py
│   ├── data_loader.py         ← CSV / JSON loading and validation
│   ├── text_cleaner.py        ← whitespace normalisation
│   ├── chunker.py             ← character-based chunking
│   └── semantic_search.py     ← embedding, indexing, querying
├── tests/
│   └── test_pipeline.py       ← pytest test suite
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

> The `chroma_db/` directory is created automatically when you run the
> notebook or index a dataset.  It is excluded from Git.

---

## Prerequisites

- Python 3.10 or later (developed on 3.13)
- Git
- An internet connection for the first model download (~90 MB)

---

## Installation (Windows PowerShell)

Open PowerShell in the project folder and run these commands one at a time.

### Step 1 — Create a virtual environment

```powershell
python -m venv .venv
```

### Step 2 — Activate the virtual environment

```powershell
.venv\Scripts\activate
```

Your prompt changes to `(.venv) PS C:\...` to confirm it is active.

### Step 3 — Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 4 — Install all dependencies

```powershell
pip install -r requirements.txt
```

This installs Pandas, Sentence Transformers (with PyTorch), ChromaDB,
Jupyter, and pytest.  The first install takes a few minutes.

### Step 5 — Register the Jupyter kernel (so the notebook sees your packages)

```powershell
python -m ipykernel install --user --name mini-search --display-name "Mini Semantic Search"
```

### Step 6 — Start Jupyter Notebook

```powershell
jupyter notebook
```

Your browser opens at `http://localhost:8888`.

### Step 7 — Open and run the notebook

Navigate to `notebooks/mini_semantic_search.ipynb`.
Select the **Mini Semantic Search** kernel.
Click **Kernel → Restart & Run All** to run every cell from top to bottom.

---

## Running the tests

```powershell
# Activate the virtual environment first (.venv\Scripts\activate)

# Run all offline tests (fast — no model download needed)
pytest tests/ -v

# Run only offline tests explicitly
pytest tests/ -v -m "not slow"

# Run the slow integration tests (downloads the model)
pytest tests/ -v -m slow
```

Expected output (offline tests only):

```
tests/test_pipeline.py::TestLoadCSV::test_load_valid_csv           PASSED
tests/test_pipeline.py::TestLoadCSV::test_load_csv_file_not_found  PASSED
tests/test_pipeline.py::TestLoadCSV::test_load_csv_missing_columns PASSED
... (all offline tests PASSED)
```

---

## Sample input data

The file `data/faqs.csv` contains 24 FAQ rows covering:

| id | question | answer (excerpt) |
|----|----------|-----------------|
| 1  | What is Python? | Python is a high-level, interpreted... |
| 3  | How do I use Git to track changes? | Git is a distributed version control... |
| 5  | What is an API and how do I use one? | An API (Application Programming Interface)... |
| 9  | What is semantic search? | Keyword search matches exact words... |
| 23 | What is Pandas and what is it used for? | Pandas is a powerful open-source Python library... |

The CSV has three columns: `id`, `question`, `answer`.

---

## Example queries and representative output

The output below is representative of what the system produces. Exact
distances will vary slightly depending on the model version and hardware.

---

**Query:** `"How do I upload my code to GitHub?"`

```
──────────────────────────────────────────────────────────────────────
Query: "How do I upload my code to GitHub?"
──────────────────────────────────────────────────────────────────────

[Rank 1]  FAQ #4 (chunk 0)  |  Distance: 0.0821  (lower = more similar)
Question : How do I upload my code to GitHub?
Answer   : To push your code to GitHub: (1) Create a new repository on
           github.com ... git push -u origin main.

[Rank 2]  FAQ #3 (chunk 0)  |  Distance: 0.3142  (lower = more similar)
Question : How do I use Git to track changes in my project?
Answer   : Git is a distributed version control system...

[Rank 3]  FAQ #24 (chunk 0)  |  Distance: 0.4017  (lower = more similar)
Question : What is version control and why is it important?
Answer   : Version control is the practice of tracking and managing changes...
```

---

**Query:** `"What is the difference between a list and a tuple in Python?"`

```
[Rank 1]  FAQ #2 (chunk 0)  |  Distance: 0.0312  (lower = more similar)
Question : What is the difference between a list and a tuple in Python?
Answer   : In Python, both lists and tuples are ordered sequences...
```

---

**Query:** `"How can my application communicate with an API?"`

```
[Rank 1]  FAQ #6 (chunk 0)  |  Distance: 0.0517  (lower = more similar)
Question : How can my application communicate with an external REST API?
Answer   : Your application can communicate with a REST API using HTTP requests...
```

---

## Concepts explained

### Embeddings

An embedding is a list of numbers (a vector) that represents the meaning
of a piece of text.  Two sentences with similar meanings produce vectors
that are close together in high-dimensional space.

### Sentence Transformers

`sentence-transformers/all-MiniLM-L6-v2` is a pretrained model that
converts any English text into a 384-dimensional embedding.  It was
trained on millions of text pairs to produce embeddings where semantically
similar sentences are geometrically close.

### Vector databases

ChromaDB stores vectors alongside metadata and finds the nearest vectors
to a query vector efficiently using approximate nearest-neighbour (ANN)
algorithms.  Unlike a regular database that searches by value, ChromaDB
searches by geometric proximity.

### Chunking

Long documents are split into smaller overlapping pieces.  This lets the
search system retrieve a precise, focused passage rather than an entire
article.  The overlap (50 characters by default) ensures that sentences
near a boundary appear in at least one complete chunk.

### Cosine distance

ChromaDB measures similarity using cosine distance: `1 − cosine_similarity`.

- Distance **0.0** = vectors are identical (perfect match).
- Distance **< 0.3** = very similar (strong match).
- Distance **> 0.7** = not very similar.

A distance is **not** a percentage.  The score is labelled as
"distance (lower = more similar)" in all output to avoid misinterpretation.

---

## First-run model download

The first time you run the notebook (or any code that calls
`SemanticSearch`), the `all-MiniLM-L6-v2` model (~90 MB) is downloaded
from HuggingFace and cached locally.  An internet connection is required
for this one-time download.  Subsequent runs use the local cache.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'src'` | Make sure your virtual environment is active and you started Jupyter from the project root. |
| `FileNotFoundError: ... faqs.csv` | The notebook must be able to find `data/faqs.csv`. Check that `project_root` in the first cell resolves correctly. |
| `RuntimeError: collection is empty` | Run the **Build the ChromaDB index** cell before running a search. |
| Slow first run | The model is downloading (~90 MB). Check your internet connection. |
| `pip install` fails on Windows | Try running PowerShell as Administrator, or check that Visual C++ Build Tools are installed for packages that compile native extensions. |
| ChromaDB import error | Upgrade pip: `python -m pip install --upgrade pip`, then re-run `pip install -r requirements.txt`. |

---

## Limitations

- The dataset has 24 FAQs — a real system needs many more for broad coverage.
- Character-based chunking may split words at boundaries.
- Results are ordered by embedding distance only — no re-ranking.
- The model is English-only.
- No web interface is provided (notebook only).

---

## How to demonstrate the project to your instructor

1. Open PowerShell in the project folder.
2. Activate the virtual environment: `.venv\Scripts\activate`
3. Run the tests: `pytest tests/ -v -m "not slow"`
4. Start Jupyter: `jupyter notebook`
5. Open `notebooks/mini_semantic_search.ipynb`.
6. Select **Kernel → Restart & Run All**.
7. Show the query cells (section 8) and explain that the system finds
   relevant answers even when the exact words don't match.
8. Change one of the query strings to your own question and re-run that
   cell to show a live demonstration.

---

## License

MIT — see [LICENSE](LICENSE).
