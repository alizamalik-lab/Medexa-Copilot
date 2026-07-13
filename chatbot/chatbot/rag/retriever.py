from langchain_core.documents import Document

from rag.hybrid_retriever import HybridRetriever
from rag.vectordb import VectorStore


class ContextRetriever:
    """Hybrid retrieval facade used by the chatbot."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model_name: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ):
        self.top_k = top_k
        self._hybrid = HybridRetriever(
            vector_store=vector_store,
            embedding_model_name=embedding_model_name,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    def retrieve(self, question: str) -> list[Document]:
        return self._hybrid.retrieve(question)

    def format_context(self, documents: list[Document]) -> str:
        if not documents:
            return ""

        blocks = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "unknown")
            doc_type = doc.metadata.get("doc_type", "")
            cpt_code = doc.metadata.get("cpt_code", "")
            method = doc.metadata.get("retrieval_method", "unknown")
            blocks.append(
                f"--- Context {i} (source: {source}, type: {doc_type}, "
                f"cpt_code: {cpt_code}, retrieval: {method}) ---\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(blocks)

    def extract_sources(self, documents: list[Document]) -> list[str]:
        seen = set()
        sources = []
        for doc in documents:
            src = doc.metadata.get("source")
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
        return sources
