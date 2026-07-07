import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'node:url';
import chatRouter from './routes/chat.js';
import { loadStaticKnowledge } from './services/dataLoader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST_DIR = path.resolve(__dirname, '../dist');
const INDEX_HTML = path.join(DIST_DIR, 'index.html');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json({ limit: '2mb' }));

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
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

async function start() {
  console.log('[Medexa] Loading static knowledge base...');
  await loadStaticKnowledge();
  console.log('[Medexa] Static knowledge loaded.');

  app.listen(PORT, () => {
    console.log(`[Medexa] Server listening on http://localhost:${PORT}`);
  });
}

start().catch((err) => {
  console.error('[Medexa] Fatal startup error:', err);
  process.exit(1);
});
