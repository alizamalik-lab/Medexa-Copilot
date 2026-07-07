import {
  appendUserMessage,
  appendAIMessage,
  renderMarkdownInto,
  showTypingIndicator,
  hideTypingIndicator,
  scrollToBottom
} from './ui.js';
import { streamChat, resetHistory } from './api.js';

const form = document.getElementById('input-container');
const input = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const clearBtn = document.getElementById('clear-chat-btn');
const chips = document.querySelectorAll('.chip');
const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome-screen');

let isStreaming = false;

function setStreamingState(streaming) {
  isStreaming = streaming;
  input.disabled = streaming;
  sendBtn.disabled = streaming;
}

function autoGrow() {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

async function handleSend(text) {
  const trimmed = (text ?? '').trim();
  if (!trimmed || isStreaming) return;

  appendUserMessage(trimmed);
  input.value = '';
  autoGrow();

  setStreamingState(true);
  showTypingIndicator();

  const aiContentEl = appendAIMessage();
  let started = false;

  await streamChat(trimmed, {
    onDelta: (fullText) => {
      if (!started) {
        hideTypingIndicator();
        started = true;
      }
      renderMarkdownInto(aiContentEl, fullText);
      scrollToBottom();
    },
    onDone: () => {
      hideTypingIndicator();
      setStreamingState(false);
      input.focus();
    },
    onError: (err) => {
      hideTypingIndicator();
      renderMarkdownInto(aiContentEl, `_Something went wrong: ${err.message}_`);
      setStreamingState(false);
    }
  });
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  handleSend(input.value);
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend(input.value);
  }
});

input.addEventListener('input', autoGrow);

chips.forEach((chip) => {
  chip.addEventListener('click', () => handleSend(chip.dataset.prompt));
});

clearBtn.addEventListener('click', () => {
  messagesEl.innerHTML = '';
  resetHistory();
  welcomeEl?.classList.remove('is-hidden');
});
