# Medexa PT/OT/SLP Chatbot — Detailed Implementation Plan

## Goal Description
Build a local-first medical chatbot specializing in PT/OT/SLP US healthcare topics. The UI will be warm, modern, and Gemini-like (Vanilla JS + CSS). The backend (Express.js) will securely handle OpenAI GPT-4o-mini API calls and inject knowledge from local JSON and PDF files. 

Crucially, because two JSON files (`cpt_icd10_info.json` at 40MB and `cpt_ptp_info.json` at 3.5MB) are too large for any LLM context window, the backend will implement a **Hybrid Knowledge Injection Strategy**: loading 3 small JSONs into memory completely, and using Regex-based "search-then-inject" for the massive ICD-10 and PTP files on a per-query basis.

## User Review Required
> [!IMPORTANT]
> **API Key Setup**: Before execution begins, an OpenAI API key must be placed in a `.env` file in the root directory.
> **PDF Knowledge**: You mentioned using a PDF for the 6th file. Ensure it is text-searchable (not a scanned image) and placed in `Medexa-Chatbot/Knowledgebase/`.

## Proposed Changes

---

### Phase 1: Project Scaffolding & Dependencies

The project will be built in `c:\Medexa-Copilot\Medexa-Chatbot\`.

#### [NEW] [package.json](file:///c:/Medexa-Copilot/Medexa-Chatbot/package.json)
Initialize a Node.js project.
**Dependencies**: `express` (server), `cors` (cross-origin), `dotenv` (env vars), `openai` (LLM), `pdf-parse` (PDF extraction), `marked` (markdown rendering in frontend).
**Dev Dependencies**: `vite` (frontend tooling), `concurrently` (running backend and frontend together).
**Scripts**:
- `"start:backend": "node server/index.js"`
- `"start:frontend": "vite"`
- `"dev": "concurrently \"npm run start:backend\" \"npm run start:frontend\""`

#### [NEW] [.env](file:///c:/Medexa-Copilot/Medexa-Chatbot/.env)
Store the OpenAI API key (`OPENAI_API_KEY=your_key_here`) and port (`PORT=3001`).

#### [NEW] [vite.config.js](file:///c:/Medexa-Copilot/Medexa-Chatbot/vite.config.js)
Configure Vite to proxy `/api` requests to `http://localhost:3001`.

---

### Phase 2: Backend Implementation (Express + OpenAI)

#### [NEW] [server/services/dataLoader.js](file:///c:/Medexa-Copilot/Medexa-Chatbot/server/services/dataLoader.js)
**Purpose**: Load the small JSON files and parse the PDF at server startup.
**Implementation Steps**:
1. Use `fs.readFileSync` to parse `Knowledgebase/cpt_general_info.json`, `cpt_aoc_info.json`, and `cpt_mue_info.json`.
2. Use `pdf-parse` to extract text from any `.pdf` files in the `Knowledgebase/` folder.
3. Export an object containing this fully loaded data so the prompt builder can easily append it.

#### [NEW] [server/services/searchService.js](file:///c:/Medexa-Copilot/Medexa-Chatbot/server/services/searchService.js)
**Purpose**: Handle the massive 40MB and 3.5MB JSON files on-demand.
**Implementation Steps**:
1. Implement `extractCptCodes(text)` using a Regex like `/\b(?:9[0-9]{4}|G[0-9]{4})\b/g` (to find 5-digit CPT codes and G-codes).
2. Implement `searchLargeFiles(cptCodes)`: Use the `fs` module (or JSONStream if memory is an issue, but Node can hold 40MB in memory via `require`) to find and return the specific ICD-10 and PTP records for *only* the matched CPT codes.

#### [NEW] [server/services/promptBuilder.js](file:///c:/Medexa-Copilot/Medexa-Chatbot/server/services/promptBuilder.js)
**Purpose**: Construct the system prompt.
**Implementation Steps**:
1. Start with the core instructions: "You are Medexa, a US medical chatbot for PT/OT/SLP. Only answer medical questions. Use the following context..."
2. Append the small JSON data (General, AOC, MUE).
3. Append the PDF text.
4. Accept dynamically retrieved ICD-10/PTP data (from `searchService`) and append it as "Additional Query-Specific Context".

#### [NEW] [server/routes/chat.js](file:///c:/Medexa-Copilot/Medexa-Chatbot/server/routes/chat.js)
**Purpose**: The API endpoint handling frontend requests.
**Implementation Steps**:
1. Expose `POST /api/chat`.
2. Accept `{ message, history }` from the client.
3. Pass `message` to `searchService` to get dynamic ICD-10/PTP context.
4. Pass everything to `promptBuilder` to get the final System Prompt.
5. Initialize the `openai` client.
6. Call `openai.chat.completions.create` with `model: "gpt-4o-mini"`, `stream: true`, and the messages array (System Prompt + History + User Message).
7. Stream the SSE (Server-Sent Events) chunks back to the Express response using `res.write()`.

#### [NEW] [server/index.js](file:///c:/Medexa-Copilot/Medexa-Chatbot/server/index.js)
**Purpose**: Server entry point.
**Implementation Steps**:
1. Init Express app. Use `express.json()` and `cors()`.
2. Call `dataLoader.init()` to preload the small files.
3. Register the `/api/chat` router.
4. `app.listen(3001)`.

---

### Phase 3: Frontend Implementation (Vite + Vanilla JS)

#### [NEW] [index.html](file:///c:/Medexa-Copilot/Medexa-Chatbot/index.html)
**Purpose**: Semantic UI structure.
**Implementation Steps**:
1. Link to `style.css` and `main.js` (module type).
2. Create a layout:
   - Header with Logo and "Clear Chat" button.
   - Main scrollable `#chat-container`.
   - Initial `#welcome-screen` with a greeting and 3 clickable starter chips (e.g., "What is CPT 97110?").
   - Bottom `#input-container` with a textarea and a Send button.

#### [NEW] [src/style.css](file:///c:/Medexa-Copilot/Medexa-Chatbot/src/style.css)
**Purpose**: Warm, modern, premium Gemini-like aesthetics.
**Implementation Steps**:
1. **Variables**: Set warm tones. `--bg: #FAFAF8`, `--primary: #2D7D6F` (teal), `--accent: #E8835A` (coral).
2. **Typography**: Use `'Inter'` for body, `'Outfit'` for headings (import from Google Fonts).
3. **Animations**: Add `@keyframes slideUp` for new messages, and a bouncing dot animation for the typing indicator.
4. **Message Bubbles**: 
   - User: Right-aligned, colored background (coral), white text.
   - AI: Left-aligned, light grey background, dark text. Markdown lists/tables styled neatly.
5. **Scrollbar**: Hide default scrollbar or style it minimally.

#### [NEW] [src/ui.js](file:///c:/Medexa-Copilot/Medexa-Chatbot/src/ui.js)
**Purpose**: DOM manipulation.
**Implementation Steps**:
1. Export functions: `appendUserMessage(text)`, `appendAIMessage()`.
2. `appendAIMessage()` should return a DOM element reference so the streaming text can be injected into it in real-time.
3. Implement `showTypingIndicator()` and `hideTypingIndicator()`.
4. Implement `scrollToBottom()`.
5. Use the `marked` library to render markdown inside the AI message bubble whenever new chunks arrive.

#### [NEW] [src/api.js](file:///c:/Medexa-Copilot/Medexa-Chatbot/src/api.js)
**Purpose**: Handle the fetch and SSE streaming.
**Implementation Steps**:
1. Track the last 10 messages in an array to send as `history`.
2. Make a `fetch('/api/chat', { method: 'POST' })`.
3. Read the stream using the Fetch API `response.body.getReader()`.
4. Parse the SSE chunks, append them to the AI message DOM element, and parse via `marked`.

#### [NEW] [src/main.js](file:///c:/Medexa-Copilot/Medexa-Chatbot/src/main.js)
**Purpose**: Wire it all together.
**Implementation Steps**:
1. Attach event listeners to the Send button, Enter key (Shift+Enter for newline), and Starter Chips.
2. Disable the input while streaming is happening.

---

## Verification Plan

### Automated Tests
- Run `npm run dev`. Both Vite and Express should start without errors.
- Confirm `dataLoader` prints success messages to the terminal indicating JSONs and PDFs were parsed.

### Manual Verification
1. Open `http://localhost:5173`.
2. Click a starter chip: "What is CPT 97110?"
3. Verify the chat bubble appears instantly, typing indicator shows, and stream begins.
4. Check that the UI looks premium (fonts load, colors match the warm palette, animations are smooth).
5. Verify context retention by asking a follow-up: "Are there any MUE limits for that code?"
6. Verify the Hybrid Search by asking: "What are the valid ICD-10 codes for 90901?" and ensuring it pulls from the 40MB file.
