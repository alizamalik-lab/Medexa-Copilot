---
title: Medexa Chatbot
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Medexa Chatbot

Medexa is a PT/OT/SLP-focused US clinical coding assistant built with:

- Vite frontend
- Express backend (UI + `/api/chat` proxy)
- Separate Python **RAG** service (retrieve → prompt → LLM)
- CPT / ICD-10 / MUE / PTP / AOC data owned by the RAG under `chatbot/`

## Architecture

```
Browser → Express (:3001 / :7860) → RAG_URL (:8000) → Chroma + LLM
```

Express no longer injects `Knowledgebase/` into prompts. All answering happens in the RAG process.

## Local development (two processes)

### 1. RAG service

Stand-in (current nested app):

```bash
cd chatbot/chatbot
python -m venv .venv
# activate venv, then:
pip install -r requirements.txt
# set GROQ_API_KEY in chatbot/chatbot/.env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

When the new RAG is ready, place it flat under `chatbot/` and run uvicorn from there instead. Keep the same HTTP contract (`POST /chat`, `DELETE /chat/{id}`, `GET /health`). See [chatbot/README.md](chatbot/README.md).

### 2. Medexa UI + proxy

```bash
# repo root
cp .env.example .env
# ensure RAG_URL=http://localhost:8000
npm install
npm run dev
```

Or production-style:

```bash
npm run build
npm start
```

Open `http://localhost:5173` (dev) or `http://localhost:3001` (built).

### Env (Express)

| Variable | Purpose |
|----------|---------|
| `PORT` | Express port (default `3001`) |
| `RAG_URL` | Base URL of the RAG service (required for chat) |
| `RAG_CHAT_PATH` | Chat path (default `/chat`) |
| `RAG_TIMEOUT_MS` | Request timeout (default `60000`) |

LLM API keys belong on the **RAG** service, not Express.

## Hugging Face Space Setup

This repo is configured for a **Docker Space**. The Dockerfile runs **both** processes in one container:

- Express UI + `/api/chat` proxy on port **7860**
- Python RAG (uvicorn) on **127.0.0.1:8000**
- `RAG_URL=http://127.0.0.1:8000` is set inside the image (do not leave this unset)

### Required Space secret

In the Space **Settings → Secrets**:

- `GROQ_API_KEY` — required for chat answers

Optional Space variables:

- `GROQ_MODEL` (e.g. `llama-3.3-70b-versatile`)
- `LLM_PROVIDER` (default `groq`)

### What must be in the Space repo

Push the full repo (or connect GitHub). The Dockerfile expects:

- `server/`, `src/`, `index.html`, `package.json`, `Dockerfile`, `scripts/start-hf.sh`
- `chatbot/chatbot/app/`, `chatbot/chatbot/rag/`, `chatbot/chatbot/data/`, `chatbot/chatbot/requirements.txt`

The vector index is built **during the Docker image build** (no need to commit `chroma_db/`).

Do **not** upload `.venv/`, `.env`, or `node_modules/`.

### Create the Space (one-time)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. **Space name:** e.g. `medexa-copilot`
3. **SDK:** Docker
4. **Hardware:** CPU basic (free) or **CPU upgrade** if builds fail on memory
5. Connect this GitHub repo, or push with:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/medexa-copilot
   git push space main
   ```
6. **Settings → Repository secrets:** add `GROQ_API_KEY`
7. Wait for the Docker build (first build can take 15–30+ minutes)

If you only upload the Node app without the RAG folders / new Dockerfile, Express will show:  
`RAG_URL is not configured` (or RAG will be unreachable).

## Folder swap (new RAG)

1. Stop the old RAG.
2. Replace `chatbot/chatbot/` with the new RAG at `chatbot/`.
3. Keep `RAG_URL` the same.
4. Restart uvicorn from `chatbot/`.
5. No Express rewrite required if the HTTP contract matches.

## Notes

- `/api/health` reports whether `RAG_URL` is set and whether `/health` on the RAG is reachable.
- Clear chat clears the UI and best-effort `DELETE`s the RAG session.
