from app.config import settings
from rag.loader import DocumentLoader
from rag.chunker import DocumentChunker
from rag.embeddings import EmbeddingModel
from rag.progress import log_progress, progress_bar
from rag.vectordb import VectorStore


class DocumentIndexer:
    """Indexes documents into ChromaDB. Skips if DB already populated."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embedding_model = EmbeddingModel(settings.embedding_model)
        self.vector_store = VectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.collection_name,
            embedding_function=self.embedding_model.embeddings,
        )

    def index(self, force: bool = False) -> None:
        if self.vector_store.exists() and not force:
            print("Vector DB already exists. Skipping indexing.")
            return

        log_progress("Loading documents...")
        documents = self.loader.load_all(
            settings.json_dir, settings.pdf_dir, show_progress=True
        )
        if not documents:
            raise FileNotFoundError(
                f"No documents found in {settings.json_dir} or {settings.pdf_dir}"
            )

        log_progress(f"Loaded {len(documents)} documents.")
        chunks = self.chunker.chunk(documents, show_progress=True)
        log_progress(f"Created {len(chunks)} chunks.")

        self.vector_store.add_documents(chunks, show_progress=True)
        log_progress("Indexing complete.")