import { marked } from 'marked';
import './codeChips.js';

const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome-screen');
const chatContainer = document.getElementById('chat-container');

export function hideWelcomeScreen() {
  welcomeEl?.classList.add('is-hidden');
}

export function appendUserMessage(text) {
  hideWelcomeScreen();
  const bubble = document.createElement('div');
  bubble.className = 'message message-user';
  bubble.textContent = text;
  messagesEl.appendChild(bubble);
  scrollToBottom();
}

export function appendAIMessage() {
  hideWelcomeScreen();

  const wrapper = document.createElement('div');
  wrapper.className = 'message message-ai';

  const avatar = document.createElement('span');
  avatar.className = 'ai-avatar';
  avatar.innerHTML =
    '<svg viewBox="0 0 32 32" width="18" height="18"><path d="M4 24 A14 14 0 0 1 28 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/><circle cx="16" cy="24" r="2" fill="currentColor"/></svg>';

  const content = document.createElement('div');
  content.className = 'message-content';

  wrapper.appendChild(avatar);
  wrapper.appendChild(content);
  messagesEl.appendChild(wrapper);
  scrollToBottom();

  return content;
}

export function renderMarkdownInto(el, rawText) {
  el.innerHTML = marked.parse(rawText);
}

let typingEl = null;

export function showTypingIndicator() {
  if (typingEl) return;

  typingEl = document.createElement('div');
  typingEl.className = 'typing-indicator';
  typingEl.innerHTML = `
    <span class="ai-avatar" aria-hidden="true">
      <svg viewBox="0 0 32 32" width="18" height="18"><path d="M4 24 A14 14 0 0 1 28 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/><circle cx="16" cy="24" r="2" fill="currentColor"/></svg>
    </span>
    <div class="typing-bubble">
      <svg viewBox="0 0 60 24" width="52" height="20" class="rom-sweep" aria-hidden="true">
        <path class="arc arc-1" d="M4 20 A10 10 0 0 1 20 12" fill="none" stroke-width="2.5" stroke-linecap="round"/>
        <path class="arc arc-2" d="M22 20 A10 10 0 0 1 38 12" fill="none" stroke-width="2.5" stroke-linecap="round"/>
        <path class="arc arc-3" d="M40 20 A10 10 0 0 1 56 12" fill="none" stroke-width="2.5" stroke-linecap="round"/>
      </svg>
    </div>`;
  messagesEl.appendChild(typingEl);
  scrollToBottom();
}

export function hideTypingIndicator() {
  typingEl?.remove();
  typingEl = null;
}

export function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

messagesEl?.addEventListener('click', (e) => {
  const chip = e.target.closest('.code-chip');
  if (!chip) return;

  const code = chip.dataset.code;
  navigator.clipboard?.writeText(code).then(() => {
    chip.classList.add('is-copied');
    setTimeout(() => chip.classList.remove('is-copied'), 1200);
  });
});
