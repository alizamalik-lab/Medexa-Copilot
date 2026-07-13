import { Router } from 'express';
import { askRag, clearRagSession, isRagConfigured } from '../services/ragClient.js';

const router = Router();

const sendEvent = (res, event, data) => {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
};

router.post('/', async (req, res) => {
  const { message, sessionId = null } = req.body ?? {};

  if (!message || typeof message !== 'string' || !message.trim()) {
    return res.status(400).json({ error: 'A non-empty "message" string is required.' });
  }

  if (!isRagConfigured()) {
    return res.status(503).json({
      error: 'RAG_URL is not configured. Start the RAG service and set RAG_URL.'
    });
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  try {
    const result = await askRag({
      question: message.trim(),
      sessionId: typeof sessionId === 'string' && sessionId ? sessionId : null
    });

    // RAG is non-streaming; emit one delta so the existing UI SSE client still works.
    sendEvent(res, 'delta', { text: result.answer });
    sendEvent(res, 'done', {
      sessionId: result.sessionId,
      sources: result.sources
    });
    res.end();
  } catch (err) {
    console.error('[chat route] RAG error:', err.message);
    sendEvent(res, 'error', {
      message: err.message || 'RAG service unavailable.'
    });
    res.end();
  }
});

router.delete('/session/:sessionId', async (req, res) => {
  const { sessionId } = req.params;
  if (!sessionId) {
    return res.status(400).json({ error: 'sessionId is required.' });
  }

  const result = await clearRagSession(sessionId);
  res.json({ status: 'ok', ...result });
});

export default router;
