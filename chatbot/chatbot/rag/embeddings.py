from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingModel:
    """Wraps sentence-transformers via LangChain HuggingFaceEmbeddings."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        # BGE models work better with this prefix for queries (handled in retriever)
        self.model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        return self._embeddings