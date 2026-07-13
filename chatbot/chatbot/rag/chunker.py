from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.progress import progress_bar


class DocumentChunker:
    """Splits documents into overlapping chunks while preserving metadata."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(
        self, documents: list[Document], show_progress: bool = False
    ) -> list[Document]:
        chunks: list[Document] = []
        doc_iter = (
            progress_bar(documents, desc="Chunking documents", unit="doc")
            if show_progress
            else documents
        )

        for doc in doc_iter:
            chunks.extend(self.splitter.split_documents([doc]))

        for chunk in chunks:
            chunk.metadata.setdefault("source", "unknown")
            chunk.metadata.setdefault("doc_type", "unknown")
        return chunks