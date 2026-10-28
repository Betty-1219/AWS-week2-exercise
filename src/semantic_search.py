"""
semantic_search.py
==================
End-to-end semantic search pipeline using Sentence Transformers and ChromaDB.

Pipeline overview
-----------------
1.  Load a pretrained sentence-transformer model
    (default: 'sentence-transformers/all-MiniLM-L6-v2', 384 dimensions).
2.  Accept a list of ChunkRecord dictionaries produced by chunker.py.
3.  Generate embeddings for each chunk's text using the model.
4.  Store chunks + embeddings in a persistent local ChromaDB collection.
5.  Accept a natural-language query string.
6.  Embed the query with the same model.
7.  Find the top-k nearest neighbours in the collection (cosine distance).
8.  Return and display ranked results.

About scores
------------
ChromaDB returns a *distance* value (lower = more similar) when using the
default cosine space.  Specifically, ChromaDB uses:

    distance = 1 - cosine_similarity

So a distance of 0.0 means identical vectors and a distance of 2.0 means
perfectly opposite vectors.  A typical "good" match is below 0.5.

This module labels scores as "distance (lower is more similar)" to avoid
the common mistake of calling a cosine distance a "similarity percentage".

Usage example
-------------
>>> from src.semantic_search import SemanticSearch
>>> ss = SemanticSearch()
>>> ss.build_index(chunks)          # chunks from chunker.chunk_dataframe()
>>> results = ss.search("How do I push code to GitHub?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.chunker import ChunkRecord


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "faq_chunks"
DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
DEFAULT_TOP_K = 3


# ---------------------------------------------------------------------------
# SemanticSearch class
# ---------------------------------------------------------------------------

class SemanticSearch:
    """Manages embedding, indexing, and querying of FAQ chunks.

    Parameters
    ----------
    model_name:       HuggingFace model id for sentence-transformers.
    collection_name:  Name of the ChromaDB collection.
    db_dir:           Directory where ChromaDB persists its data.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        db_dir: str | Path = DEFAULT_DB_DIR,
    ) -> None:
        self.model_name = model_name
        self.collection_name = collection_name
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self._model: SentenceTransformer | None = None
        self._client: chromadb.ClientAPI | None = None
        self._collection: Any | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            print(f"[semantic_search] Loading model '{self.model_name}' …")
            print(
                "  (First run downloads the model from HuggingFace; "
                "subsequent runs use the local cache.)"
            )
            self._model = SentenceTransformer(self.model_name)
            print("[semantic_search] Model loaded.")
        return self._model

    def _get_client(self) -> chromadb.ClientAPI:
        """Create or reuse a persistent ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.db_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self, create: bool = False) -> Any:
        """Get the ChromaDB collection, optionally creating it."""
        client = self._get_client()
        if create or self._collection is None:
            # get_or_create returns an existing collection or makes a new one.
            # We use cosine space: distance = 1 - cosine_similarity
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_collection(self) -> None:
        """Delete and recreate the ChromaDB collection.

        Call this when you want to re-index a changed dataset from scratch.
        All existing embeddings are deleted.
        """
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            print(
                f"[semantic_search] Deleted existing collection "
                f"'{self.collection_name}'."
            )
        except Exception:
            pass  # Collection did not exist — that is fine
        self._collection = None
        self._get_collection(create=True)
        print(
            f"[semantic_search] Created fresh collection "
            f"'{self.collection_name}'."
        )

    def build_index(
        self,
        chunks: list[ChunkRecord],
        reset: bool = False,
        batch_size: int = 64,
    ) -> None:
        """Embed all chunks and store them in ChromaDB.

        Parameters
        ----------
        chunks:     List of ChunkRecord dicts from chunker.chunk_dataframe().
        reset:      If True, wipe the collection first (full re-index).
                    If False, only add chunks whose IDs are not already
                    present — safe to call multiple times on the same data.
        batch_size: Number of chunks to embed and upsert at once.  Reduce
                    this on low-memory machines.

        Notes
        -----
        ChromaDB upsert() inserts new documents and updates existing ones
        with the same ID, so re-running on the same data is idempotent.
        """
        if not chunks:
            print("[semantic_search] No chunks provided; index not built.")
            return

        if reset:
            self.reset_collection()

        collection = self._get_collection()
        model = self._get_model()

        print(
            f"[semantic_search] Indexing {len(chunks)} chunk(s) "
            f"in batches of {batch_size} …"
        )

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]

            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [
                {
                    "faq_id": str(c["faq_id"]),
                    "question": c["question"],
                    "chunk_index": str(c["chunk_index"]),
                }
                for c in batch
            ]

            # Generate embeddings (returns a numpy array, shape [N, dim])
            embeddings = model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()  # ChromaDB expects plain Python lists

            # upsert = insert or update — safe to call multiple times
            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            end = min(batch_start + batch_size, len(chunks))
            print(
                f"  Upserted chunks {batch_start + 1}–{end} / {len(chunks)}"
            )

        total = collection.count()
        print(
            f"[semantic_search] Index ready.  "
            f"Collection '{self.collection_name}' contains {total} chunk(s)."
        )

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        print_results: bool = True,
    ) -> list[dict[str, Any]]:
        """Search the index for chunks most relevant to the query.

        Parameters
        ----------
        query:         Natural-language question or search string.
        top_k:         Number of top results to return.
        print_results: If True, print a formatted summary to stdout.

        Returns
        -------
        A list of result dictionaries (length <= top_k), each with:
          - rank:       int, 1-based rank
          - question:   str, the source FAQ question
          - chunk_text: str, the matching answer chunk
          - faq_id:     int, source FAQ id
          - chunk_index:int, chunk position within this FAQ
          - distance:   float, cosine distance (0 = identical, 2 = opposite)
                        LOWER distance means MORE similar.

        Raises
        ------
        ValueError: If the query is empty.
        RuntimeError: If the collection has not been built yet.
        """
        query = query.strip() if isinstance(query, str) else ""
        if not query:
            raise ValueError(
                "Query must be a non-empty string. "
                "Please provide a question or search phrase."
            )

        collection = self._get_collection()
        count = collection.count()
        if count == 0:
            raise RuntimeError(
                "The ChromaDB collection is empty.  "
                "Call build_index() before searching."
            )

        # Clamp top_k to the number of available documents
        effective_k = min(top_k, count)

        # Embed the query with the same model used for documents
        model = self._get_model()
        query_embedding = model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        # Query ChromaDB — returns distances in the same cosine space
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        # Unpack ChromaDB response structure:
        # results['documents'][0] is a list of document strings
        # results['metadatas'][0] is a list of metadata dicts
        # results['distances'][0] is a list of float distances
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        formatted: list[dict[str, Any]] = []
        for rank, (doc, meta, dist) in enumerate(
            zip(documents, metadatas, distances), start=1
        ):
            formatted.append(
                {
                    "rank": rank,
                    "question": meta.get("question", "N/A"),
                    "chunk_text": doc,
                    "faq_id": int(meta.get("faq_id", 0)),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "distance": round(float(dist), 4),
                }
            )

        if print_results:
            self._print_results(query, formatted)

        return formatted

    # ------------------------------------------------------------------
    # Display helper
    # ------------------------------------------------------------------

    @staticmethod
    def _print_results(
        query: str, results: list[dict[str, Any]]
    ) -> None:
        """Pretty-print search results to stdout."""
        sep = "─" * 70
        print(f"\n{sep}")
        print(f"Query: \"{query}\"")
        print(sep)
        if not results:
            print("No results found.")
            return
        for r in results:
            print(
                f"\n[Rank {r['rank']}]  FAQ #{r['faq_id']} "
                f"(chunk {r['chunk_index']})"
                f"  |  Distance: {r['distance']:.4f}  "
                "(lower = more similar)"
            )
            print(f"Question : {r['question']}")
            print(f"Answer   : {r['chunk_text'][:300]}"
                  + ("…" if len(r['chunk_text']) > 300 else ""))
        print(f"\n{sep}\n")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def collection_size(self) -> int:
        """Return the number of chunks currently stored in the collection."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0
