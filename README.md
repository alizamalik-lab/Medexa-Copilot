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
- Express backend
- Groq's OpenAI-compatible chat completions API
- Local CPT / ICD-10 / MUE / PTP / AOC knowledge files

## Hugging Face Space Setup

This repo is configured for a **Docker Space**.

### Required secret

Add this Space secret in Hugging Face:

- `GROQ_API_KEY`

Optional variable:

- `GROQ_MODEL` (defaults to `llama-3.1-8b-instant`)

### Required local knowledge files

This app expects these files under `Knowledgebase/`:

- `cpt_general_info.json`
- `cpt_aoc_info.json`
- `cpt_mue_info.json`
- `cpt_icd10_info.json`
- `cpt_ptp_info.json`
- optional text-searchable `.pdf` files

Important: in this local repo, `Knowledgebase/*.json` and `Knowledgebase/*.pdf` are gitignored. For the Hugging Face Space to work, you must **upload those files into the Space repository** or otherwise make them available inside the container at build/runtime.

### How it runs on Hugging Face

- Docker builds the app
- `npm run build` generates the frontend `dist/`
- Express serves the built frontend and `/api/chat`
- Hugging Face exposes the app on port `7860`

## Local production-style run

```bash
npm install
npm run build
npm start
```

Then open:

- `http://localhost:3001` locally

## Notes

- The frontend and backend are bundled into one container for easier free hosting.
- The large ICD-10 and PTP JSON files are indexed lazily on first relevant request.
- The app will start without a PDF, but it will not answer chat requests without `GROQ_API_KEY`.
