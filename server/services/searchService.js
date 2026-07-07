import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const KB_DIR = path.resolve(__dirname, '../../Knowledgebase');

const ICD10_CODE_FIELD = 'cpt_code';
const PTP_CODE_FIELD = 'cpt_code';

const CODE_REGEX = /\b(?:[0-9]{5}|G[0-9]{4})\b/g;
const ICD10_REGEX = /\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?\b/gi;

const LARGE_ICD_THRESHOLD = 500;
const ICD_SAMPLE_SIZE = 50;

let icd10Index = null;
let ptpIndex = null;
let indexBuildPromise = null;

function buildIndex(data, codeField) {
  const index = new Map();
  const records = Array.isArray(data) ? data : Object.values(data ?? {});

  for (const record of records) {
    const code = record?.[codeField];
    if (!code) continue;
    if (!index.has(code)) index.set(code, []);
    index.get(code).push(record);
  }

  return index;
}

async function buildIndexesOnce() {
  if (indexBuildPromise) return indexBuildPromise;

  indexBuildPromise = (async () => {
    console.log('[searchService] Indexing large knowledge files (one-time cost)...');
    const start = Date.now();

    const [icd10Raw, ptpRaw] = await Promise.all([
      fs.readFile(path.join(KB_DIR, 'cpt_icd10_info.json'), 'utf-8').catch(() => null),
      fs.readFile(path.join(KB_DIR, 'cpt_ptp_info.json'), 'utf-8').catch(() => null)
    ]);

    if (icd10Raw) icd10Index = buildIndex(JSON.parse(icd10Raw), ICD10_CODE_FIELD);
    if (ptpRaw) ptpIndex = buildIndex(JSON.parse(ptpRaw), PTP_CODE_FIELD);

    console.log(
      `[searchService] Indexed ICD10=${icd10Index?.size ?? 0} codes,`,
      `PTP=${ptpIndex?.size ?? 0} codes in ${Date.now() - start}ms`
    );
  })();

  return indexBuildPromise;
}

export function extractCptCodes(texts) {
  const textArray = Array.isArray(texts) ? texts : [texts];
  const matches = new Set();

  for (const text of textArray) {
    if (!text || typeof text !== 'string') continue;
    const found = text.match(CODE_REGEX) || [];
    for (const code of found) matches.add(code);
  }

  return [...matches];
}

export function extractIcd10Codes(texts) {
  const textArray = Array.isArray(texts) ? texts : [texts];
  const matches = new Set();

  for (const text of textArray) {
    if (!text || typeof text !== 'string') continue;
    const found = text.match(ICD10_REGEX) || [];
    for (const code of found) matches.add(code.toUpperCase());
  }

  return [...matches];
}

function shapeIcd10Record(record, mentionedIcd10s) {
  const codes = record.valid_icd10_codes ?? [];
  const total = codes.length;

  if (mentionedIcd10s.length > 0) {
    const results = mentionedIcd10s.map((icd) => {
      const isValid = codes.some((c) => c.code?.toUpperCase() === icd.toUpperCase());
      return {
        cpt_code: record.cpt_code,
        icd10: icd,
        is_valid: isValid,
        total_valid_count: total
      };
    });
    return results.length === 1 ? results[0] : results;
  }

  if (total < LARGE_ICD_THRESHOLD) return record;

  return {
    cpt_code: record.cpt_code,
    total_count: total,
    sample_codes: codes.slice(0, ICD_SAMPLE_SIZE).map((c) => c.code),
    note: 'Full list too large for context. Ask about a specific ICD-10 code to validate.'
  };
}

export async function searchLargeFiles(cptCodes, icd10Codes = []) {
  if (!cptCodes || cptCodes.length === 0) {
    return { icd10: [], ptp: [] };
  }

  await buildIndexesOnce();

  const icd10Results = [];
  const ptpResults = [];

  for (const code of cptCodes) {
    if (icd10Index?.has(code)) {
      for (const record of icd10Index.get(code)) {
        icd10Results.push(shapeIcd10Record(record, icd10Codes));
      }
    }
    if (ptpIndex?.has(code)) {
      ptpResults.push(...ptpIndex.get(code));
    }
  }

  return { icd10: icd10Results, ptp: ptpResults };
}
