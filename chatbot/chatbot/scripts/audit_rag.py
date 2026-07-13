"""RAG pipeline audit script."""
from collections import Counter
from pathlib import Path

import chromadb

from app.config import settings
from rag.chunker import DocumentChunker
from rag.embeddings import EmbeddingModel
from rag.loader import DocumentLoader
from rag.retriever import ContextRetriever
from rag.vectordb import VectorStore

QUERY = "What is CPT 97110?"


def audit_loading() -> list:
    loader = DocumentLoader()
    print("=" * 60)
    print("1. FILE LOADING AUDIT")
    print("=" * 60)

    json_paths = sorted(settings.json_dir.glob("*.json"))
    pdf_paths = sorted(settings.pdf_dir.glob("*.pdf"))
    print(f"JSON files on disk: {len(json_paths)}")
    for p in json_paths:
        print(f"  - {p.name}")
    print(f"PDF files on disk: {len(pdf_paths)}")
    for p in pdf_paths:
        print(f"  - {p.name}")

    json_docs = loader.load_json_files(settings.json_dir)
    pdf_docs = loader.load_pdfs(settings.pdf_dir)
    print(f"\nPDF documents created:")
    for p in pdf_paths:
        count = sum(1 for d in pdf_docs if d.metadata.get("source") == p.name)
        print(f"  Loaded {p.name}: {count} document(s)")

    all_docs = json_docs + pdf_docs
    print(f"\nTotal LangChain documents before chunking: {len(all_docs)}")

    # Per-file CPT 97110 in loaded docs
    print("\nCPT 97110 in loaded documents (by metadata cpt_code):")
    for source in sorted({d.metadata.get("source") for d in json_docs}):
        matches = [
            d for d in json_docs
            if d.metadata.get("source") == source and d.metadata.get("cpt_code") == "97110"
        ]
        print(f"  {source}: {len(matches)} record(s) with cpt_code=97110")
        if matches:
            print(f"    preview: {matches[0].page_content[:200].replace(chr(10), ' ')}")

    return all_docs


def audit_chunking(documents: list) -> list:
    print("\n" + "=" * 60)
    print("2. CHUNKING AUDIT")
    print("=" * 60)
    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.chunk(documents)
    print(f"chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap}")
    print(f"Total chunks: {len(chunks)}")

    by_source = Counter(c.metadata.get("source", "unknown") for c in chunks)
    print("Chunks per source file:")
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count}")

    cpt97110_chunks = [
        c for c in chunks
        if c.metadata.get("cpt_code") == "97110" or "cpt_code: 97110" in c.page_content
    ]
    print(f"\nChunks containing CPT 97110: {len(cpt97110_chunks)}")
    for i, c in enumerate(cpt97110_chunks[:5], 1):
        print(f"  chunk {i}: source={c.metadata.get('source')} cpt_code={c.metadata.get('cpt_code')!r}")
    return chunks


def audit_chroma():
    print("\n" + "=" * 60)
    print("3. CHROMADB AUDIT")
    print("=" * 60)
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    try:
        col = client.get_collection(settings.collection_name)
    except Exception as e:
        print(f"Collection missing or error: {e}")
        return

    total = col.count()
    print(f"Collection: {settings.collection_name}")
    print(f"Total indexed chunks: {total}")

    # Sample metadata keys
    sample = col.get(limit=3, include=["metadatas", "documents"])
    if sample["metadatas"]:
        print(f"Sample metadata keys: {list(sample['metadatas'][0].keys())}")

    # Search for cpt_code metadata
    try:
        by_meta = col.get(where={"cpt_code": "97110"}, include=["metadatas", "documents"])
        print(f"\nChunks with metadata cpt_code='97110': {len(by_meta['ids'])}")
        for i, (meta, doc) in enumerate(zip(by_meta["metadatas"], by_meta["documents"]), 1):
            print(f"  [{i}] source={meta.get('source')} cpt_code={meta.get('cpt_code')}")
            print(f"      preview: {doc[:150].replace(chr(10), ' ')}")
    except Exception as e:
        print(f"Metadata filter cpt_code=97110 failed: {e}")

    # Full scan for 97110 in document text (if collection is manageable)
    if total <= 10000:
        all_data = col.get(include=["metadatas", "documents"])
        text_matches = [
            (m, d) for m, d in zip(all_data["metadatas"], all_data["documents"])
            if "97110" in d
        ]
        print(f"\nChunks containing '97110' in text (full scan): {len(text_matches)}")
        by_src = Counter(m.get("source") for m, _ in text_matches)
        for src, cnt in sorted(by_src.items()):
            print(f"  {src}: {cnt}")
    else:
        print(f"\nSkipping full text scan ({total} chunks too large)")


def audit_retrieval():
    print("\n" + "=" * 60)
    print("4. RETRIEVAL AUDIT")
    print("=" * 60)
    print(f"top_k={settings.top_k}")
    print(f"similarity_threshold={settings.similarity_threshold}")
    print(f"embedding_model={settings.embedding_model}")

    emb = EmbeddingModel(settings.embedding_model)
    vs = VectorStore(
        settings.chroma_persist_dir,
        settings.collection_name,
        emb.embeddings,
    )
    store = vs.get_store()
    retriever = ContextRetriever(
        vs.as_retriever(
            top_k=settings.top_k,
            score_threshold=settings.similarity_threshold,
        ),
        top_k=settings.top_k,
    )

    # Check BGE prefix logic
    vectorstore_str = str(getattr(retriever.retriever, "vectorstore", "")).lower()
    bge_prefix_active = "bge" in vectorstore_str
    print(f"BGE query prefix active: {bge_prefix_active}")
    print(f"  vectorstore repr contains 'bge': {bge_prefix_active}")

    # Similarity search with scores
    print(f"\nQuery: {QUERY!r}")
    results = store.similarity_search_with_score(QUERY, k=settings.top_k)
    print(f"\nTop {settings.top_k} results (similarity_search_with_score):")
    for i, (doc, score) in enumerate(results, 1):
        cpt = doc.metadata.get("cpt_code", "")
        has_97110 = "97110" in doc.page_content
        print(f"  [{i}] score={score:.4f} source={doc.metadata.get('source')} "
              f"cpt_code={cpt!r} has_97110={has_97110}")
        print(f"      preview: {doc.page_content[:180].replace(chr(10), ' ')}")

    docs = retriever.retrieve(QUERY)
    print(f"\nContextRetriever.retrieve() returned {len(docs)} chunks:")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] source={doc.metadata.get('source')} "
              f"cpt_code={doc.metadata.get('cpt_code')!r} "
              f"has_97110={'97110' in doc.page_content}")
        print(f"      {doc.page_content[:200].replace(chr(10), ' ')}")

    # Metadata filter test - should NOT be applied in retriever
    print("\nMetadata filter in retriever: NONE (no where clause in as_retriever)")


if __name__ == "__main__":
    docs = audit_loading()
    audit_chunking(docs)
    audit_chroma()
    audit_retrieval()
