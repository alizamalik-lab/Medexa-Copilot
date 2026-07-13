/**
 * Medexa ↔ RAG HTTP contract (matches chatbot FastAPI):
 *
 * POST {RAG_URL}{RAG_CHAT_PATH}  default /chat
 *   body: { question: string, session_id?: string | null }
 *   200:  { answer: string, sources?: string[], session_id?: string }
 *
 * DELETE {RAG_URL}/chat/{session_id}
 *   clears conversation memory on the RAG side (no-op if unsupported)
 *
 * GET {RAG_URL}/health
 *   used by Express /api/health for optional readiness
 *
 * When swapping to a new RAG under chatbot/, keep this contract (or remap only here).
 */

import '../loadEnv.js';

function getRagBaseUrl() {
  return (process.env.RAG_URL || '').replace(/\/$/, '');
}

function getChatPath() {
  return process.env.RAG_CHAT_PATH || '/chat';
}

function getTimeoutMs() {
  return Number(process.env.RAG_TIMEOUT_MS) || 60_000;
}

export function isRagConfigured() {
  return Boolean(getRagBaseUrl());
}

export function getRagUrl() {
  return getRagBaseUrl() || null;
}

function chatEndpoint() {
  const base = getRagBaseUrl();
  const path = getChatPath().startsWith('/') ? getChatPath() : `/${getChatPath()}`;
  return `${base}${path}`;
}

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timeoutMs = getTimeoutMs();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function normalizeAnswer(payload) {
  if (!payload || typeof payload !== 'object') {
    throw new Error('RAG returned an empty or invalid payload.');
  }

  const answer =
    payload.answer ??
    payload.response ??
    payload.text ??
    payload.message;

  if (typeof answer !== 'string' || !answer.trim()) {
    throw new Error('RAG response did not include an answer string.');
  }

  const sources = Array.isArray(payload.sources)
    ? payload.sources.filter((s) => typeof s === 'string')
    : [];

  const sessionId =
    payload.session_id ??
    payload.sessionId ??
    null;

  return {
    answer: answer.trim(),
    sources,
    sessionId: typeof sessionId === 'string' && sessionId ? sessionId : null
  };
}

/**
 * @param {{ question: string, sessionId?: string | null }} args
 */
export async function askRag({ question, sessionId = null }) {
  if (!getRagBaseUrl()) {
    throw new Error('RAG_URL is not configured. Set RAG_URL in the environment.');
  }

  let response;
  try {
    response = await fetchWithTimeout(chatEndpoint(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        question,
        session_id: sessionId || null
      })
    });
  } catch (err) {
    if (err?.name === 'AbortError') {
      throw new Error(`RAG request timed out after ${getTimeoutMs()}ms.`);
    }
    throw new Error(`RAG service unavailable: ${err.message}`);
  }

  let payload = null;
  const text = await response.text();
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text };
    }
  }

  if (!response.ok) {
    const detail =
      payload?.detail ||
      payload?.error ||
      payload?.message ||
      `HTTP ${response.status}`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }

  return normalizeAnswer(payload);
}

/**
 * Best-effort session clear. Ignores 404 / network failures so UI clear always works.
 */
export async function clearRagSession(sessionId) {
  const base = getRagBaseUrl();
  if (!base || !sessionId) return { ok: false, skipped: true };

  try {
    const response = await fetchWithTimeout(
      `${base}/chat/${encodeURIComponent(sessionId)}`,
      { method: 'DELETE', headers: { Accept: 'application/json' } }
    );
    return { ok: response.ok, status: response.status };
  } catch {
    return { ok: false, skipped: true };
  }
}

export async function pingRagHealth() {
  const base = getRagBaseUrl();
  if (!base) {
    return { configured: false, reachable: false };
  }

  try {
    const response = await fetchWithTimeout(`${base}/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' }
    });
    return { configured: true, reachable: response.ok, status: response.status };
  } catch {
    return { configured: true, reachable: false };
  }
}
