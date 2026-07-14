from abc import ABC, abstractmethod
from typing import Protocol

from langchain_core.documents import Document

from rag.query_router import CodeType, DetectedCode, QueryRoute
from rag.vectordb import VectorStore

SOURCE_PRIORITY: dict[str, int] = {
    "pt_ot_slp_billing_categories.json": 0,
    "medexa_cpt_lookup.json": 1,
    "cpt_general_info.json": 2,
    "cpt_aoc_info.json": 3,
    "cpt_mue_info.json": 4,
    "cpt_ptp_info.json": 5,
    "cpt_icd10_info.json": 6,
    "healthcare.pdf": 6,
    "medexa.pdf": 7,
    "Medexa_Physical_Therapy_Knowledge_Base.docx": 8,
    "medexa_ot_knowledge_base.docx": 9,
    "medexa_slp_knowledge_base.docx": 10,
    "medexa_therapy_billing_rules_knowledge_base.docx": 11,
    "NCCI_Knowledge_Base_Medexa.docx": 12,
    "medexa_Claim_Denials_knowledge_base.docx": 13,
    "medexa_emr_ehr_knowledge_base.docx": 14,
    "medexa_hipaa_knowledge_base.docx": 15,
}


class RetrievalTool(ABC):
    """Base class for pluggable retrieval strategies (NCCI, MUE, etc.)."""

    name: str = "base"

    @abstractmethod
    def retrieve(
        self,
        route: QueryRoute,
        top_k: int,
    ) -> list[Document]:
        raise NotImplementedError


class MetadataCodeRetrievalTool(RetrievalTool):
    """Exact-match retrieval via Chroma metadata or document content."""

    name = "metadata_code"

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, route: QueryRoute, top_k: int) -> list[Document]:
        if not route.codes:
            return []

        docs: list[Document] = []
        for detected in route.codes:
            docs.extend(self._retrieve_code(detected, top_k))

        return self._tag_results(self._collapse_by_source(docs), retrieval_method="metadata")

    def _collapse_by_source(
        self, docs: list[Document], per_source: int = 2
    ) -> list[Document]:
        buckets: dict[str, list[Document]] = {}
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            buckets.setdefault(source, []).append(doc)

        collapsed: list[Document] = []
        for source in sorted(buckets, key=lambda s: SOURCE_PRIORITY.get(s, 99)):
            bucket = buckets[source]
            bucket.sort(
                key=lambda d: (
                    0 if "cpt_code:" in d.page_content[:120] else 1,
                    len(d.page_content),
                )
            )
            collapsed.extend(bucket[:per_source])
        return collapsed

    def _retrieve_code(self, detected: DetectedCode, top_k: int) -> list[Document]:
        if detected.metadata_field:
            return self.vector_store.get_by_metadata(
                detected.metadata_field,
                detected.code,
                limit=top_k,
            )

        if detected.code_type == CodeType.ICD10:
            return self.vector_store.get_by_document_contains(
                detected.code,
                limit=top_k,
            )

        if detected.code_type == CodeType.MODIFIER:
            return self.vector_store.get_by_document_contains(
                detected.code,
                limit=top_k,
            )

        return []

    def _tag_results(
        self, documents: list[Document], retrieval_method: str
    ) -> list[Document]:
        tagged: list[Document] = []
        for doc in documents:
            metadata = dict(doc.metadata)
            metadata["retrieval_method"] = retrieval_method
            tagged.append(Document(page_content=doc.page_content, metadata=metadata))
        return tagged


STRUCTURED_JSON_SOURCES = frozenset(
    {
        "medexa_cpt_lookup.json",
        "cpt_general_info.json",
        "cpt_aoc_info.json",
        "cpt_mue_info.json",
        "cpt_ptp_info.json",
        "cpt_icd10_info.json",
    }
)


class SemanticRetrievalTool(RetrievalTool):
    """Dense vector similarity search."""

    name = "semantic"

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model_name: str,
        score_threshold: float | None = None,
    ):
        self.vector_store = vector_store
        self.embedding_model_name = embedding_model_name
        self.score_threshold = score_threshold

    def retrieve(self, route: QueryRoute, top_k: int) -> list[Document]:
        query = route.question
        if "bge" in self.embedding_model_name.lower():
            query = (
                "Represent this sentence for searching relevant passages: "
                f"{route.question}"
            )

        results = self.vector_store.similarity_search_with_score(
            query,
            k=top_k,
            score_threshold=self.score_threshold,
            filter={"source": {"$nin": list(STRUCTURED_JSON_SOURCES)}},
        )

        docs: list[Document] = []
        for doc, score in results:
            metadata = dict(doc.metadata)
            metadata["retrieval_method"] = "semantic"
            metadata["similarity_score"] = score
            docs.append(Document(page_content=doc.page_content, metadata=metadata))
        return docs


class RuleEngineTool(Protocol):
    """Hook for future rule engines: NCCI, MUE, Modifier 59, 8-minute rule."""

    def applies_to(self, route: QueryRoute) -> bool: ...

    def retrieve(self, route: QueryRoute, top_k: int) -> list[Document]: ...
