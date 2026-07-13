from langchain_core.documents import Document

from rag.query_router import CodeType, QueryRoute, QueryRouter
from rag.retrieval_tools import MetadataCodeRetrievalTool, SemanticRetrievalTool
from rag.vectordb import VectorStore

# Structured JSON tables are served by deterministic billing tools, not embeddings.
STRUCTURED_JSON_SOURCES = frozenset(
    {
        "cpt_knowledge.json",
        "cpt_general_info.json",
        "cpt_aoc_info.json",
        "cpt_mue_info.json",
        "cpt_ptp_info.json",
        "cpt_icd10_info.json",
    }
)

# Prefer descriptive knowledge-base sources before large reference tables.
SOURCE_PRIORITY: dict[str, int] = {
    "cpt_knowledge.json": 0,
    "cpt_general_info.json": 1,
    "cpt_aoc_info.json": 2,
    "cpt_mue_info.json": 3,
    "cpt_ptp_info.json": 4,
    "cpt_icd10_info.json": 5,
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


class HybridRetriever:
    """Query-routed hybrid retrieval: metadata first, semantic fallback/combine."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model_name: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.router = QueryRouter()
        self.metadata_tool = MetadataCodeRetrievalTool(vector_store)
        self.semantic_tool = SemanticRetrievalTool(
            vector_store,
            embedding_model_name=embedding_model_name,
            score_threshold=score_threshold,
        )
        self.rule_tools: list = []  # Future: NCCI, MUE, Modifier 59, 8-minute rule

    def retrieve(self, question: str) -> list[Document]:
        route = self.router.route(question)
        print(
            f"[router] mode={route.mode} codes="
            f"{[(c.code_type.value, c.code) for c in route.codes]}"
        )

        if route.mode == "semantic":
            docs = self.semantic_tool.retrieve(route, self.top_k)
        elif route.mode == "metadata":
            docs = self.metadata_tool.retrieve(route, self.top_k)
            if not docs:
                docs = self.semantic_tool.retrieve(route, self.top_k)
        else:
            metadata_docs = self.metadata_tool.retrieve(route, self.top_k * 3)
            semantic_docs = self.semantic_tool.retrieve(route, self.top_k)
            docs = self._merge_and_rank(metadata_docs, semantic_docs)

        rule_docs = self._run_rule_tools(route)
        if rule_docs:
            docs = self._merge_and_rank(rule_docs, docs)

        docs = self._exclude_structured_json(docs)
        return docs[: self.top_k]

    def _exclude_structured_json(self, docs: list[Document]) -> list[Document]:
        return [
            doc
            for doc in docs
            if doc.metadata.get("source") not in STRUCTURED_JSON_SOURCES
        ]

    def _run_rule_tools(self, route: QueryRoute) -> list[Document]:
        docs: list[Document] = []
        for tool in self.rule_tools:
            if tool.applies_to(route):
                docs.extend(tool.retrieve(route, self.top_k))
        return docs

    def _merge_and_rank(
        self,
        metadata_docs: list[Document],
        semantic_docs: list[Document],
    ) -> list[Document]:
        merged: dict[str, Document] = {}

        for doc in metadata_docs:
            key = self._doc_key(doc)
            merged[key] = doc

        for doc in semantic_docs:
            key = self._doc_key(doc)
            if key not in merged:
                merged[key] = doc

        ranked = sorted(
            merged.values(),
            key=self._rank_key,
        )
        return ranked

    def _doc_key(self, doc: Document) -> str:
        source = doc.metadata.get("source", "")
        cpt_code = doc.metadata.get("cpt_code", "")
        record_index = doc.metadata.get("record_index", "")
        preview = doc.page_content[:120]
        return f"{source}|{cpt_code}|{record_index}|{hash(preview)}"

    def _rank_key(self, doc: Document) -> tuple:
        method = doc.metadata.get("retrieval_method", "semantic")
        method_rank = 0 if method == "metadata" else 1
        source = doc.metadata.get("source", "unknown")
        source_rank = SOURCE_PRIORITY.get(source, 99)
        score = doc.metadata.get("similarity_score", 1.0)
        has_header = 1 if "cpt_code:" in doc.page_content[:80] else 0
        return (method_rank, source_rank, -has_header, score)
