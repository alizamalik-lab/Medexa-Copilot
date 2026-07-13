# Medical Billing RAG Copilot (Milestone 1)

Local RAG chatbot over CPT JSON files and healthcare PDFs.

## Setup

1. Python 3.11+
2. `python -m venv .venv`
3. Activate venv
4. `pip install -r requirements.txt`
5. Copy `.env.example` → `.env` and set API keys
6. Place files:
   - `data/json/*.json` (5 CPT files)
   - `data/pdf/*.pdf` (Healthcare + Medexa docs)

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000