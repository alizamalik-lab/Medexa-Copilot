const HISTORY_LIMIT = 2;
let history = [];

export function resetHistory() {
  history = [];
}

export async function streamChat(message, { onDelta, onDone, onError } = {}) {
  const historyBeforeThisMessage = history.slice(-HISTORY_LIMIT);
  history.push({ role: 'user', content: message });
  history = history.slice(-HISTORY_LIMIT);

  let response;
  try {
    response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history: historyBeforeThisMessage })
    });
  } catch (err) {
    onError?.(err);
    return;
  }

  if (!response.ok || !response.body) {
    onError?.(new Error(`Request failed with status ${response.status}`));
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
        history.push({ role: 'assistant', content: fullText });
        history = history.slice(-HISTORY_LIMIT);
        onDone?.(fullText);
      } else if (eventType === 'error') {
        onError?.(new Error(data.message));
      }
    }
  }
}
