# Hugging Face Docker Space: Express (UI + proxy) + Python RAG in one container.
# Browser → :7860 (Express) → http://127.0.0.1:8000 (RAG)
#
# RAG layout expected at: chatbot/app, chatbot/rag, chatbot/data, ...
# Required Space secret: GROQ_API_KEY

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

# --- RAG app + data (flat chatbot/) ---
COPY chatbot/app ./chatbot/app
COPY chatbot/rag ./chatbot/rag
COPY chatbot/data ./chatbot/data
COPY chatbot/requirements.txt ./chatbot/requirements.txt
COPY chatbot/chroma_db ./chatbot/chroma_db

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r chatbot/requirements.txt

COPY scripts/start-hf.sh /app/scripts/start-hf.sh
RUN chmod +x /app/scripts/start-hf.sh \
    && sed -i 's/\r$//' /app/scripts/start-hf.sh

ENV PATH="/opt/venv/bin:$PATH"
ENV NODE_ENV=production
ENV PORT=7860
ENV RAG_URL=http://127.0.0.1:8000
ENV RAG_CHAT_PATH=/chat
ENV RAG_DIR=/app/chatbot
ENV LLM_PROVIDER=groq
ENV DATA_DIR=./data
ENV CHROMA_PERSIST_DIR=./chroma_db

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=10 \
  CMD curl -fsS http://127.0.0.1:7860/api/health || exit 1

CMD ["/app/scripts/start-hf.sh"]
