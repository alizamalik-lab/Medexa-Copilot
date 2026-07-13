const HISTORY_LIMIT = 2;
let history = [];
let sessionId = null;

function newSessionId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `sess_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

function ensureSessionId() {
  if (!sessionId) sessionId = newSessionId();
  return sessionId;
}

export function resetHistory() {
  history = [];
  const previous = sessionId;
  sessionId = null;

  if (previous) {
    fetch(`/api/chat/session/${encodeURIComponent(previous)}`, {
      method: 'DELETE'
    }).catch(() => {});
  }
}

export async function streamChat(message, { onDelta, onDone, onError } = {}) {
  const historyBeforeThisMessage = history.slice(-HISTORY_LIMIT);
  history.push({ role: 'user', content: message });
  history = history.slice(-HISTORY_LIMIT);

  const currentSessionId = ensureSessionId();

  let response;
  try {
    response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        sessionId: currentSessionId,
        history: historyBeforeThisMessage
      })
    });
  } catch (err) {
    onError?.(err);
    return;
  }

  if (!response.ok || !response.body) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const errBody = await response.json();
      if (errBody?.error) detail = errBody.error;
    } catch {
      // ignore
    }
    onError?.(new Error(detail));
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullText = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split('\n\n');
    buffer = events.pop() ?? '';

    for (const rawEvent of events) {
      const lines = rawEvent.split('\n');
      const eventLine = lines.find((l) => l.startsWith('event:'));
      const dataLine = lines.find((l) => l.startsWith('data:'));
      if (!eventLine || !dataLine) continue;

      const eventType = eventLine.replace('event:', '').trim();
      const data = JSON.parse(dataLine.replace('data:', '').trim());

      if (eventType === 'delta') {
        fullText += data.text;
        onDelta?.(fullText, data.text);
      } else if (eventType === 'done') {
        if (data.sessionId) sessionId = data.sessionId;
        history.push({ role: 'assistant', content: fullText });
        history = history.slice(-HISTORY_LIMIT);
        onDone?.(fullText);
      } else if (eventType === 'error') {
        onError?.(new Error(data.message));
      }
    }
  }
}
