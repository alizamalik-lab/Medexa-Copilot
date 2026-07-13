from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from rag.billing_tools import BillingTools
from rag.chatbot import RAGChatbot, create_llm
from rag.embeddings import EmbeddingModel
from rag.indexer import DocumentIndexer
from rag.memory import ConversationMemory
from rag.retriever import ContextRetriever
from rag.vectordb import VectorStore


chatbot: RAGChatbot | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot

    # Index on startup only if chroma_db is empty
    indexer = DocumentIndexer()
    indexer.index()

    embedding_model = EmbeddingModel(settings.embedding_model)
    vector_store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.collection_name,
        embedding_function=embedding_model.embeddings,
    )

    retriever = ContextRetriever(
        vector_store=vector_store,
        embedding_model_name=settings.embedding_model,
        top_k=settings.top_k,
        score_threshold=settings.similarity_threshold,
    )

    llm = create_llm(
        provider=settings.llm_provider,
        openai_key=settings.openai_api_key,
        anthropic_key=settings.anthropic_api_key,
        google_api_key=settings.google_api_key,
        groq_api_key=settings.groq_api_key,
        openai_model=settings.openai_model,
        anthropic_model=settings.anthropic_model,
        gemini_model=settings.gemini_model,
        groq_model=settings.groq_model,
    )

    memory = ConversationMemory(max_exchanges=settings.max_history_exchanges)
    billing_tools = BillingTools(json_dir=settings.json_dir)
    chatbot = RAGChatbot(
        llm=llm,
        retriever=retriever,
        memory=memory,
        billing_tools=billing_tools,
    )
    yield


app = FastAPI(title="Medical Billing RAG Copilot", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = Field(
        default=None,
        description="Conversation session id. Omit to start a new session.",
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    session_id = request.session_id or str(uuid4())
    result = chatbot.ask(request.question, session_id=session_id)
    return ChatResponse(**result)


@app.delete("/chat/{session_id}")
async def clear_chat_session(session_id: str):
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    chatbot.memory.clear_session(session_id)
    return {"status": "ok", "session_id": session_id}


@app.get("/health")
async def health():
    return {"status": "ok"}