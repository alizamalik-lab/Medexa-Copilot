from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.progress import progress_bar

# Chroma rejects upserts larger than this limit.
CHROMA_MAX_BATCH = 5000


class VectorStore:
    """ChromaDB wrapper with persistence and existence checks."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_function: Embeddings,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self._store: Chroma | None = None

    def exists(self) -> bool:
        """Return True if the collection already has indexed documents."""
        if not self.persist_dir.exists():
            return False
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        try:
            collection = client.get_collection(self.collection_name)
            return collection.count() > 0
        except Exception:
            return False

    def get_store(self) -> Chroma:
        if self._store is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=str(self.persist_dir),
            )
        return self._store

    def add_documents(
        self,
        documents: list[Document],
        *,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> None:
        store = self.get_store()
        if not documents:
            return

        batch_size = max(1, min(batch_size, CHROMA_MAX_BATCH))
        total = len(documents)
        num_batches = (total + batch_size - 1) // batch_size
        progress = progress_bar(
            total=total,
            desc="Embedding & storing",
            unit="chunk",
            disable=not show_progress,
        )

        for batch_num, start in enumerate(range(0, total, batch_size), start=1):
            batch = documents[start : start + batch_size]
            store.add_documents(batch)
            progress.update(len(batch))
            progress.set_postfix(batch=f"{batch_num}/{num_batches}", refresh=False)

        progress.close()

    def as_retriever(self, top_k: int = 5, score_threshold: float | None = None):
        search_type = "similarity"
        search_kwargs: dict[str, float | int] = {"k": top_k}
        if score_threshold is not None:
            search_type = "similarity_score_threshold"
            search_kwargs["score_threshold"] = score_threshold

        return self.get_store().as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def get_by_metadata(
        self, field: str, value: str, limit: int = 50
    ) -> list[Document]:
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection = client.get_collection(self.collection_name)
        results = collection.get(
            where={field: value},
            limit=limit,
            include=["documents", "metadatas"],
        )
        return self._to_documents(results)

    def get_by_document_contains(
        self, text: str, limit: int = 50
    ) -> list[Document]:
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection = client.get_collection(self.collection_name)
        results = collection.get(
            where_document={"$contains": text},
            limit=limit,
            include=["documents", "metadatas"],
        )
        return self._to_documents(results)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        store = self.get_store()
        results = store.similarity_search_with_score(query, k=k)
        if score_threshold is not None:
            results = [(doc, score) for doc, score in results if score <= score_threshold]
        return results

    def _to_documents(self, results: dict) -> list[Document]:
        documents: list[Document] = []
        for content, metadata in zip(
            results.get("documents", []) or [],
            results.get("metadatas", []) or [],
        ):
            if content:
                documents.append(
                    Document(page_content=content, metadata=metadata or {})
                )
        return documents