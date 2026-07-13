# Medexa RAG service

This folder is the home for the **RAG backend** that Medexa Express calls via `RAG_URL`.

On Hugging Face, the Docker image expects the RAG at **flat** `chatbot/` (not `chatbot/chatbot/`).

## Local stand-in (still nested in this repo)

Until you flatten locally, the current app may still live at `chatbot/chatbot/`:

```bash
cd chatbot/chatbot
# create/activate venv, pip install -r requirements.txt, set .env with GROQ_API_KEY
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

For HF / Docker, upload contents so the Space has:

```text
chatbot/app
chatbot/rag
chatbot/data
chatbot/requirements.txt
chatbot/chroma_db   # optional
```

## Required HTTP contract

- `POST /chat` body `{ "question": "...", "session_id": "..." | null }`
  → `{ "answer": "...", "sources": [...], "session_id": "..." }`
- `DELETE /chat/{session_id}` — clear memory
- `GET /health` → `{ "status": "ok" }`

Express only talks to this service; Knowledgebase JSON injection has been removed from Node.
