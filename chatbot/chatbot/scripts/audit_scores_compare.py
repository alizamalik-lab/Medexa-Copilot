from app.config import settings
from rag.embeddings import EmbeddingModel
from rag.vectordb import VectorStore

QUERY = "What is CPT 97110?"
emb = EmbeddingModel(settings.embedding_model)
store = VectorStore(
    settings.chroma_persist_dir, settings.collection_name, emb.embeddings
).get_store()

for source, cpt in [
    ("cpt_general_info.json", "97110"),
    ("cpt_icd10_info.json", "97610"),
    ("cpt_icd10_info.json", "97139"),
    ("cpt_icd10_info.json", "97110"),
]:
    results = store.similarity_search_with_score(
        QUERY, k=1, filter={"source": source, "cpt_code": cpt}
    )
    if results:
        doc, score = results[0]
        print(f"{source} / {cpt}: score={score:.4f}")
        print(f"  {doc.page_content[:180]}\n")
    else:
        print(f"{source} / {cpt}: NOT FOUND\n")
