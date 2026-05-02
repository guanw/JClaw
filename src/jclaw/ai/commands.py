from __future__ import annotations

from typing import Any

from jclaw.tools.base import ToolContext


class AgentCommandsMixin:
    def _handle_command(self, chat_id: str, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return "Send a message or use /help for command syntax."

        command, _, remainder = stripped.partition(" ")
        if command.startswith("/"):
            command = command.split("@", 1)[0]
        handler = self._command_handlers().get(command)
        if handler is None:
            return None
        return handler(chat_id, remainder)

    def _command_handlers(self) -> dict[str, Any]:
        return {
            "/help": lambda chat_id, remainder: self._help_text(),
            "help": lambda chat_id, remainder: self._help_text(),
            "/remember": lambda chat_id, remainder: self._remember(chat_id, remainder),
            "/forget": lambda chat_id, remainder: self._forget(chat_id, remainder),
            "/memory": lambda chat_id, remainder: self._memory(chat_id),
            "/debug": lambda chat_id, remainder: self._debug_trace(chat_id, remainder),
            "/trace": lambda chat_id, remainder: self._trace_last(chat_id, remainder),
            "/approve": lambda chat_id, remainder: self._approve(chat_id, remainder),
            "/deny": lambda chat_id, remainder: self._deny(chat_id, remainder),
            "/grants": lambda chat_id, remainder: self._grants(),
            "/revoke": lambda chat_id, remainder: self._revoke(remainder),
            "/abort": lambda chat_id, remainder: self._abort(chat_id, remainder),
        }

    def _remember(self, chat_id: str, remainder: str) -> str:
        key, sep, value = remainder.partition("=")
        if not sep:
            return "Usage: /remember key = value"
        key = key.strip()
        value = value.strip()
        if not key or not value:
            return "Usage: /remember key = value"
        self.db.remember(chat_id, key, value)
        return f"Remembered '{key}'."

    def _forget(self, chat_id: str, remainder: str) -> str:
        key = remainder.strip()
        if not key:
            return "Usage: /forget key"
        deleted = self.db.forget(chat_id, key)
        if deleted:
            return f"Forgot '{key}'."
        return f"I didn't have a memory stored for '{key}'."

    def _memory(self, chat_id: str) -> str:
        items = self.db.list_memories(chat_id)
        if not items:
            return "No memories stored yet."
        lines = [f"{item.key} = {item.value}" for item in items]
        return "Stored memories:\n" + "\n".join(lines)

    def _debug_trace(self, chat_id: str, remainder: str) -> str:
        mode = remainder.strip().lower()
        if mode in {"on", "summary"}:
            self.db.set_trace_mode(chat_id, "summary")
            return "Execution trace is now on for this chat."
        if mode == "off":
            self.db.set_trace_mode(chat_id, "off")
            return "Execution trace is now off for this chat."
        current = self.db.get_trace_mode(chat_id)
        return f"Trace mode is {current}. Usage: /debug on|off"

    def _trace_last(self, chat_id: str, remainder: str) -> str:
        _ = remainder
        rendered = self.render_latest_trace(chat_id)
        if rendered:
            return rendered
        return "No execution trace is available yet."

    def _help_text(self) -> str:
        return (
            "Commands:\n"
            "/remember key = value\n"
            "/memory\n"
            "/forget key\n"
            "/debug on|off\n"
            "/trace [last]\n"
            "/approve req_123\n"
            "/deny req_123\n"
            "/grants\n"
            "/revoke 1\n"
            "/abort req_123"
        )

    def _approve(self, chat_id: str, remainder: str) -> str:
        request_id = remainder.strip()
        if not request_id:
            return "Usage: /approve req_123"
        request = self.db.get_approval_request(request_id)
        if request is None or request.chat_id != chat_id:
            return "Approval request not found."
        if request.status != "pending":
            return f"Approval request {request_id} is already {request.status}."
        if request.kind == "grant":
            grant = self.db.upsert_grant(request.root_path, request.capabilities, chat_id)
            self.db.update_approval_request_status(request_id, "approved")
            grant_message = (
                f"Granted {', '.join(grant.capabilities)} access for {grant.root_path}. "
                f"Grant id: {grant.id}"
            )
            continuation_result = self._dispatch_request_continuation(
                request,
                key="approve_action",
                fallback_key="action",
                chat_id=chat_id,
                user_id="approval",
            )
            if continuation_result is None:
                return grant_message
            return f"{grant_message}\n\n{continuation_result}"
        continuation_result = self._dispatch_request_continuation(
            request,
            key="approve_action",
            fallback_key="action",
            chat_id=chat_id,
            user_id="approval",
        )
        if continuation_result is None:
            return f"Approval request kind '{request.kind}' is not supported."
        return continuation_result

    def _deny(self, chat_id: str, remainder: str) -> str:
        request_id = remainder.strip()
        if not request_id:
            return "Usage: /deny req_123"
        request = self.db.get_approval_request(request_id)
        if request is None or request.chat_id != chat_id:
            return "Approval request not found."
        if request.status != "pending":
            return f"Approval request {request_id} is already {request.status}."
        self.db.update_approval_request_status(request_id, "denied")
        return f"Denied request {request_id}."

    def _grants(self) -> str:
        grants = self.db.list_grants(active_only=True)
        if not grants:
            return "No active grants."
        lines = [f"{grant.id}. {grant.root_path} [{', '.join(grant.capabilities)}]" for grant in grants]
        return "Active grants:\n" + "\n".join(lines)

    def _revoke(self, remainder: str) -> str:
        token = remainder.strip()
        if not token.isdigit():
            return "Usage: /revoke 1"
        revoked = self.db.revoke_grant(int(token))
        if revoked:
            return "Grant revoked."
        return "Grant not found."

    def _abort(self, chat_id: str, remainder: str) -> str:
        request_id = remainder.strip()
        if not request_id:
            return "Usage: /abort req_123"
        request = self.db.get_approval_request(request_id)
        if request is None or request.chat_id != chat_id:
            return "Request not found."
        if request.status != "pending":
            return f"Request {request_id} is already {request.status}."
        continuation_result = self._dispatch_request_continuation(
            request,
            key="abort_action",
            chat_id=chat_id,
            user_id="abort",
        )
        if continuation_result is not None:
            return continuation_result
        self.db.update_approval_request_status(request_id, "aborted")
        return f"Aborted request {request_id}."

    def _dispatch_request_continuation(
        self,
        request: Any,
        *,
        key: str,
        chat_id: str,
        user_id: str,
        fallback_key: str = "",
    ) -> str | None:
        continuation = request.payload.get("continuation", {})
        if not isinstance(continuation, dict):
            return None
        tool = str(continuation.get("tool", "")).strip()
        action = str(continuation.get(key, "")).strip()
        if not action and fallback_key:
            action = str(continuation.get(fallback_key, "")).strip()
        params = continuation.get("params", {})
        if not tool or not action or not isinstance(params, dict):
            return None
        continuation_params = dict(params)
        continuation_params.setdefault("request_id", request.request_id)
        result = self.tools.invoke(
            tool,
            action,
            continuation_params,
            ToolContext(chat_id=chat_id, user_id=user_id),
        )
        return self.tools.get(tool).format_result(action, result)
