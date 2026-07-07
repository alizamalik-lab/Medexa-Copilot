import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import pdfParse from 'pdf-parse/lib/pdf-parse.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const KB_DIR = path.resolve(__dirname, '../../Knowledgebase');

const MAX_PDF_CHARS = 20000;

const state = {
  generalInfo: null,
  aocInfo: null,
  mueInfo: null,
  pdfText: '',
  loaded: false
};

async function readJsonSafe(filename) {
  const filePath = path.join(KB_DIR, filename);
  try {
    const raw = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(raw);
  } catch (err) {
    console.warn(`[dataLoader] Could not load ${filename}: ${err.message}`);
    return null;
  }
}

async function loadPdfs() {
  let combinedText = '';
  let entries;

  try {
    entries = await fs.readdir(KB_DIR);
  } catch (err) {
    console.warn(`[dataLoader] Knowledgebase folder not found: ${err.message}`);
    return combinedText;
  }

  const pdfFiles = entries.filter((f) => f.toLowerCase().endsWith('.pdf'));

  for (const file of pdfFiles) {
    try {
      const buffer = await fs.readFile(path.join(KB_DIR, file));
      const parsed = await pdfParse(buffer);
      combinedText += `\n\n--- Source: ${file} ---\n${parsed.text}\n`;
      console.log(`[dataLoader] Parsed PDF: ${file} (${parsed.text.length} chars)`);
    } catch (err) {
      console.warn(`[dataLoader] Failed to parse ${file}: ${err.message}`);
    }
  }

  if (combinedText.length > MAX_PDF_CHARS) {
    console.warn(`[dataLoader] PDF text truncated from ${combinedText.length} to ${MAX_PDF_CHARS} chars.`);
    combinedText = `${combinedText.slice(0, MAX_PDF_CHARS)}\n[...truncated...]`;
  }

  return combinedText;
}

export async function loadStaticKnowledge() {
  const [generalInfo, aocInfo, mueInfo, pdfText] = await Promise.all([
    readJsonSafe('cpt_general_info.json'),
    readJsonSafe('cpt_aoc_info.json'),
    readJsonSafe('cpt_mue_info.json'),
    loadPdfs()
  ]);

  state.generalInfo = generalInfo;
  state.aocInfo = aocInfo;
  state.mueInfo = mueInfo;
  state.pdfText = pdfText;
  state.loaded = true;

  console.log(
    '[dataLoader] Loaded:',
    `general=${generalInfo ? 'ok' : 'missing'}`,
    `aoc=${aocInfo ? 'ok' : 'missing'}`,
    `mue=${mueInfo ? 'ok' : 'missing'}`,
    `pdfChars=${pdfText.length}`
  );
}

export function getStaticKnowledge() {
  return state;
}
