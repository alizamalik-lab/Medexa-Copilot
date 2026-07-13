from app.config import settings
from rag.embeddings import EmbeddingModel
from rag.vectordb import VectorStore

QUERY = "What is CPT 97110?"
emb = EmbeddingModel(settings.embedding_model)
vs = VectorStore(settings.chroma_persist_dir, settings.collection_name, emb.embeddings)
store = vs.get_store()

all_results = store.similarity_search_with_score(QUERY, k=50)
print("Rank of cpt_general_info.json / 97110 in top 50:")
for i, (doc, score) in enumerate(all_results, 1):
    if (
        doc.metadata.get("cpt_code") == "97110"
        and doc.metadata.get("source") == "cpt_general_info.json"
    ):
        print(f"  Rank {i}, score={score:.4f}")
        print(f"  content: {doc.page_content[:250]}")

print("\nAll 97110 chunks in top 50:")
for i, (doc, score) in enumerate(all_results, 1):
    if doc.metadata.get("cpt_code") == "97110":
        has_header = "cpt_code: 97110" in doc.page_content
        print(
            f"  Rank {i}, score={score:.4f}, source={doc.metadata.get('source')}, "
            f"has_cpt_header={has_header}"
        )

prefixed = f"Represent this sentence for searching relevant passages: {QUERY}"
prefixed_results = store.similarity_search_with_score(prefixed, k=10)
print("\nWith BGE query prefix, top 10:")
for i, (doc, score) in enumerate(prefixed_results, 1):
    print(
        f"  [{i}] score={score:.4f} cpt={doc.metadata.get('cpt_code')} "
        f"source={doc.metadata.get('source')}"
    )
