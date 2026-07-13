import './loadEnv.js';
import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'node:url';
import chatRouter from './routes/chat.js';
import { isRagConfigured, pingRagHealth, getRagUrl } from './services/ragClient.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const DIST_DIR = path.join(ROOT_DIR, 'dist');
const INDEX_HTML = path.join(DIST_DIR, 'index.html');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json({ limit: '2mb' }));

app.get('/api/health', async (req, res) => {
  const ragConfigured = isRagConfigured();
  const ragPing = await pingRagHealth();

  res.json({
    status: 'ok',
    rag: {
      configured: ragConfigured,
      url: getRagUrl(),
      reachable: ragPing.reachable
    }
  });
});

app.use('/api/chat', chatRouter);

if (fs.existsSync(DIST_DIR)) {
  app.use(express.static(DIST_DIR));

  app.use((req, res, next) => {
    if (req.path.startsWith('/api/')) {
      return next();
    }
    return res.sendFile(INDEX_HTML);
  });
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`[Medexa] Server listening on http://0.0.0.0:${PORT}`);
  if (isRagConfigured()) {
    console.log(`[Medexa] RAG_URL=${getRagUrl()}`);
  } else {
    console.warn('[Medexa] RAG_URL is not set — /api/chat will return 503 until configured.');
  }
});
