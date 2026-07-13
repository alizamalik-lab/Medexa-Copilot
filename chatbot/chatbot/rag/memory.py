from dataclasses import dataclass
from threading import Lock
from typing import Literal


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class PendingClarification:
    original_question: str
    intent_name: str


class ConversationMemory:
    """In-memory per-session conversation history and session state."""

    def __init__(self, max_exchanges: int = 15):
        self.max_exchanges = max(1, max_exchanges)
        self._sessions: dict[str, list[ChatMessage]] = {}
        self._focus_codes: dict[str, str] = {}
        self._pending: dict[str, PendingClarification] = {}
        self._lock = Lock()

    def get_messages(self, session_id: str) -> list[ChatMessage]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def get_focus_code(self, session_id: str) -> str | None:
        with self._lock:
            return self._focus_codes.get(session_id)

    def set_focus_code(self, session_id: str, cpt_code: str | None) -> None:
        with self._lock:
            if cpt_code:
                self._focus_codes[session_id] = cpt_code
            else:
                self._focus_codes.pop(session_id, None)

    def get_pending_clarification(self, session_id: str) -> PendingClarification | None:
        with self._lock:
            return self._pending.get(session_id)

    def set_pending_clarification(
        self, session_id: str, original_question: str, intent_name: str
    ) -> None:
        with self._lock:
            self._pending[session_id] = PendingClarification(
                original_question=original_question,
                intent_name=intent_name,
            )

    def clear_pending_clarification(self, session_id: str) -> None:
        with self._lock:
            self._pending.pop(session_id, None)

    def add_exchange(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            messages = self._sessions.setdefault(session_id, [])
            messages.append(ChatMessage(role="user", content=question))
            messages.append(ChatMessage(role="assistant", content=answer))
            max_messages = self.max_exchanges * 2
            if len(messages) > max_messages:
                self._sessions[session_id] = messages[-max_messages:]

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._focus_codes.pop(session_id, None)
            self._pending.pop(session_id, None)

    def format_history(self, session_id: str) -> str:
        messages = self.get_messages(session_id)
        if not messages:
            return "(no prior messages)"

        lines: list[str] = []
        for msg in messages:
            label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{label}: {msg.content}")
        return "\n".join(lines)
