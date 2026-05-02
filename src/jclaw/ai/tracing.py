from __future__ import annotations

from datetime import datetime


MAX_RENDERED_TRACE_EVENTS = 18


class AgentTracingMixin:
    def _start_execution_trace(self, chat_id: str, text: str) -> str:
        if self.db.get_trace_mode(chat_id) == "off":
            return ""
        session = self.db.create_execution_trace_session(chat_id, text)
        self._active_trace_ids[chat_id] = session.trace_id
        self._active_trace_statuses[chat_id] = "running"
        self.db.append_execution_trace_event(
            session.trace_id,
            event_type="turn_started",
            summary="Received a new user turn.",
            payload={"request": text},
        )
        return session.trace_id

    def _append_execution_trace_event(
        self,
        chat_id: str,
        event_type: str,
        summary: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        trace_id = self._active_trace_ids.get(chat_id, "")
        if not trace_id:
            return
        self.db.append_execution_trace_event(
            trace_id,
            event_type=event_type,
            summary=summary,
            payload=payload,
        )

    def _set_execution_trace_status(self, chat_id: str, status: str) -> None:
        trace_id = self._active_trace_ids.get(chat_id, "")
        if not trace_id:
            return
        self._active_trace_statuses[chat_id] = status

    def _finish_execution_trace(self, chat_id: str, *, final_reply: str) -> None:
        trace_id = self._active_trace_ids.pop(chat_id, "")
        status = self._active_trace_statuses.pop(chat_id, "running")
        if not trace_id:
            return
        self.db.finish_execution_trace_session(trace_id, status=status, final_reply=final_reply)

    def render_running_trace(self, chat_id: str) -> str:
        session = self.db.get_latest_execution_trace_session(chat_id, status="running")
        if session is None:
            return ""
        return self._render_trace_session(session.trace_id)

    def render_latest_trace(self, chat_id: str) -> str:
        session = self.db.get_latest_execution_trace_session(chat_id)
        if session is None:
            return ""
        return self._render_trace_session(session.trace_id)

    def _render_trace_session(self, trace_id: str) -> str:
        session = self.db.get_execution_trace_session(trace_id)
        if session is None:
            return ""
        events = self.db.list_execution_trace_events(trace_id)
        if not events:
            return "```text\nTrace\n(no events yet)\n```"
        lines = [f"Trace [{session.status}]"]
        request = session.user_text.strip().replace("\n", " ")
        if request:
            lines.append(f"Request: {request[:120]}{'...' if len(request) > 120 else ''}")
        if len(events) > MAX_RENDERED_TRACE_EVENTS:
            skipped = len(events) - MAX_RENDERED_TRACE_EVENTS
            lines.append(f"... {skipped} earlier event(s) omitted ...")
            render_events = events[-MAX_RENDERED_TRACE_EVENTS:]
        else:
            render_events = events
        for event in render_events:
            lines.append(f"{event.event_index}. {event.summary}")
        return "```text\n" + "\n".join(lines) + "\n```"

    def _controller_now(self) -> datetime:
        return datetime.now().astimezone()
