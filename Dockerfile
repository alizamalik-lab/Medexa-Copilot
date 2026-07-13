# Medexa: Express (UI + proxy) + Python RAG in one container.
# Browser → $PORT (Express) → http://127.0.0.1:8000 (RAG)
#
# Works on Render, Hugging Face Spaces, and similar Docker hosts.
# Required secret: GROQ_API_KEY

FROM node:20-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Node app ---
COPY package*.json ./
RUN npm ci
COPY index.html vite.config.js ./
COPY src ./src
COPY server ./server
RUN npm run build && npm prune --omit=dev

# --- RAG app + data ---
COPY chatbot/chatbot/app ./chatbot/chatbot/app
COPY chatbot/chatbot/rag ./chatbot/chatbot/rag
COPY chatbot/chatbot/data ./chatbot/chatbot/data
COPY chatbot/chatbot/requirements.txt ./chatbot/chatbot/requirements.txt

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r chatbot/chatbot/requirements.txt

# Pre-build the vector index so cold starts are fast (no indexing at runtime).
ENV DATA_DIR=./data
ENV CHROMA_PERSIST_DIR=./chroma_db
RUN cd /app/chatbot/chatbot \
    && /opt/venv/bin/python -c "from rag.indexer import DocumentIndexer; DocumentIndexer().index()"

COPY scripts/start.sh /app/scripts/start.sh
RUN chmod +x /app/scripts/start.sh \
    && sed -i 's/\r$//' /app/scripts/start.sh

ENV PATH="/opt/venv/bin:$PATH"
ENV NODE_ENV=production
ENV RAG_URL=http://127.0.0.1:8000
ENV RAG_CHAT_PATH=/chat
ENV RAG_DIR=/app/chatbot/chatbot
ENV LLM_PROVIDER=groq

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=5 \
  CMD sh -c 'curl -fsS "http://127.0.0.1:${PORT:-10000}/api/health" || exit 1'

CMD ["/app/scripts/start.sh"]
