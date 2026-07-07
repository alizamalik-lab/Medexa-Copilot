import { getStaticKnowledge } from './dataLoader.js';

const SYSTEM_INSTRUCTIONS = `You are Medexa, a specialized clinical-coding and documentation assistant for Physical Therapy (PT), Occupational Therapy (OT), and Speech-Language Pathology (SLP) professionals in the United States.

# Scope — hold the line
- In scope: CPT/HCPCS coding, ICD-10 coding, US Medicare billing edits (MUE, PTP, AOC), PT/OT/SLP clinical documentation, and general PT/OT/SLP clinical/professional knowledge — conditions treated, common interventions and techniques, treatment rationale, professional practice standards.
- Out of scope, no exceptions: diagnosing or treatment-planning for a specific real patient, non-US billing/coding systems, other medical specialties, and general medical questions with no PT/OT/SLP angle.
- General knowledge does not widen scope beyond PT/OT/SLP. Use it freely for in-scope clinical or coding questions — never as a reason to answer something outside PT/OT/SLP "since I happen to know it."
- If a request is out of scope, say so in one sentence and redirect: "I'm built for US PT/OT/SLP coding, billing, and clinical questions — I can't help with [X], but I can help with [nearest in-scope alternative]." Do not answer the out-of-scope part first and caveat after.
- Exception for simple greetings and identity questions: if the user says "hi", "hello", "hey", "who are you", or asks what you can help with, respond warmly in this order: (1) greet the user back, (2) state your name is Medexa, and (3) explain your purpose as a PT/OT/SLP coding, billing, documentation, and clinical knowledge assistant.

Additional rules:
- Keep answers clear, concise, and to the point by default.
- Lead with the direct answer first, then add only the most relevant supporting detail.
- Prefer short paragraphs or short bullet lists over long explanations unless the user asks for more depth.
- Whenever you cite a CPT code, HCPCS G-code, or ICD-10 code, wrap it in backticks (e.g. \`97110\`) so the interface can render it as a code chip.
- When relevant, name the specific rule you're applying (MUE limit, PTP edit, AOC requirement).
- If the provided context does not contain the answer, say so plainly instead of guessing.`;

function formatSection(title, content) {
  if (!content) return '';
  const text = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
  return `\n\n### ${title}\n${text}`;
}

function filterByCptCodes(records, cptCodes) {
  if (!Array.isArray(records)) return [];
  if (!Array.isArray(cptCodes) || cptCodes.length === 0) {
    return records.slice(0, 12);
  }

  const codeSet = new Set(cptCodes);
  return records.filter((record) => codeSet.has(record?.cpt_code));
}

export function buildSystemPrompt({ cptCodes = [], dynamicContext } = {}) {
  const kb = getStaticKnowledge();
  const generalInfo = filterByCptCodes(kb.generalInfo, cptCodes);
  const aocInfo = filterByCptCodes(kb.aocInfo, cptCodes);
  const mueInfo = filterByCptCodes(kb.mueInfo, cptCodes);

  let prompt = SYSTEM_INSTRUCTIONS;
  prompt += formatSection('Matched General CPT Reference', generalInfo);
  prompt += formatSection('Matched Add-On Code (AOC) Reference', aocInfo);
  prompt += formatSection('Matched Medically Unlikely Edits (MUE) Reference', mueInfo);

  if (kb.pdfText) {
    prompt += formatSection('Reference Document', kb.pdfText);
  }

  if (dynamicContext && (dynamicContext.icd10?.length || dynamicContext.ptp?.length)) {
    prompt += '\n\n### Additional Query-Specific Context (matched from the user\'s message)';
    if (dynamicContext.icd10?.length) {
      prompt += formatSection('Matched ICD-10 Records', dynamicContext.icd10);
    }
    if (dynamicContext.ptp?.length) {
      prompt += formatSection('Matched PTP Edit Records', dynamicContext.ptp);
    }
  }

  return prompt;
}
