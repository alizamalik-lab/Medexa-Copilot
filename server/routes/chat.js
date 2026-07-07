import { Router } from 'express';
import OpenAI from 'openai';
import { extractCptCodes, extractIcd10Codes, searchLargeFiles } from '../services/searchService.js';
import { buildSystemPrompt } from '../services/promptBuilder.js';

const router = Router();
const DEFAULT_MODEL = process.env.GROQ_MODEL || process.env.OPENAI_MODEL || 'llama-3.1-8b-instant';
const API_KEY = process.env.GROQ_API_KEY || process.env.OPENAI_API_KEY;

let openai = null;
function getOpenAIClient() {
  if (!openai) {
    openai = new OpenAI({
      apiKey: API_KEY,
      baseURL: 'https://api.groq.com/openai/v1'
    });
  }
  return openai;
}

const MAX_HISTORY_MESSAGES = 10;

router.post('/', async (req, res) => {
  const { message, history = [] } = req.body ?? {};

  if (!message || typeof message !== 'string' || !message.trim()) {
    return res.status(400).json({ error: 'A non-empty "message" string is required.' });
  }

  if (!API_KEY) {
    return res.status(500).json({ error: 'Server is missing GROQ_API_KEY.' });
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  const sendEvent = (event, data) => {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  let stream;

  try {
    const trimmedHistory = history.slice(-MAX_HISTORY_MESSAGES);
    const allTexts = [
      message,
      ...trimmedHistory.map((m) => m.content ?? '')
    ];

    const cptCodes = extractCptCodes(allTexts);
    const icd10Codes = extractIcd10Codes(allTexts);
    const dynamicContext = await searchLargeFiles(cptCodes, icd10Codes);
    const systemPrompt = buildSystemPrompt({ cptCodes, dynamicContext });

    const messages = trimmedHistory.map((m) => ({
      role: m.role === 'user' ? 'user' : 'assistant',
      content: m.content
    }));

    stream = await getOpenAIClient().chat.completions.create({
      model: DEFAULT_MODEL,
      stream: true,
      temperature: 0.3,
      messages: [
        { role: 'system', content: systemPrompt },
        ...messages,
        { role: 'user', content: message }
      ]
    });

    req.on('close', () => {
      stream?.controller?.abort?.();
    });

    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) sendEvent('delta', { text: delta });
    }

    sendEvent('done', {});
    res.end();
  } catch (err) {
    console.error('[chat route] Error:', err);
    sendEvent('error', { message: 'Something went wrong generating a response.' });
    res.end();
  }
});

export default router;
