"""In-memory session store.

A plain ``dict[str, Session]`` indexed by ``session_id``. The class exists
mostly to give the codebase a single seam to swap later (Redis, Postgres, …)
without churning the routers and the service.

Volatility (state lost on process restart) is intentional for this phase —
persistence is module-3 work. A ``threading.Lock`` protects mutations because
FastAPI runs sync endpoints in a threadpool; a single worker is still the
supported deployment (``uvicorn --workers=1``). Multiple workers would each
hold their own copy of the store and break the conversational guarantee.
"""

from __future__ import annotations

from threading import Lock

from app.sessions.models import ConversationHistory, Session


class SessionNotFoundError(KeyError):
    """Raised by ``SessionStore.get_or_404`` when the id is unknown."""


class SessionStore:
    def __init__(self, *, max_turns: int = 6) -> None:
        self._sessions: dict[str, Session] = {}
        self._max_turns = max_turns
        self._lock = Lock()

    def create(self) -> Session:
        session = Session(history=ConversationHistory(max_turns=self._max_turns))
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_404(self, session_id: str) -> Session:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise SessionNotFoundError(session_id) from exc

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
