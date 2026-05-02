from __future__ import annotations

import html

from jclaw.tools.base import ToolResult


class WorkspaceFormattingMixin:
    def format_result(self, action: str, result: ToolResult) -> str:
        lines = [result.summary]
        data = result.data
        if data.get("root_path"):
            lines.append(f"Root: {data['root_path']}")
        if data.get("target_path"):
            lines.append(f"Target: {data['target_path']}")
        if "exists" in data:
            lines.append(f"Exists: {data['exists']}")
        if data.get("kind"):
            lines.append(f"Kind: {data['kind']}")
        if data.get("entry_count") and data.get("kind") == "directory":
            lines.append(f"Total entries: {data['entry_count']}")
        if data.get("request_id"):
            lines.append(f"Request: {data['request_id']}")
        if data.get("request_kind"):
            lines.append(f"Request kind: {data['request_kind']}")
        if data.get("capabilities"):
            lines.append(f"Capabilities: {', '.join(str(item) for item in data['capabilities'])}")
        if data.get("entries"):
            lines.append("Entries:")
            for entry in data["entries"][:10]:
                lines.append(f"- {entry['kind']}: {entry['name']}")
            if data.get("entries_truncated"):
                shown = len(data["entries"])
                total = data.get("entry_count", shown)
                lines.append(f"Shown {shown} of {total} entries.")
        elif data.get("kind") == "directory":
            lines.append("Entries: none")
        if data.get("touched_files"):
            lines.append("Touched files:")
            for file_path in data["touched_files"][:10]:
                lines.append(f"- {file_path}")
        if data.get("source_path"):
            lines.append(f"Source: {data['source_path']}")
        if data.get("destination_path"):
            lines.append(f"Destination: {data['destination_path']}")
        if data.get("metadata"):
            lines.append("Metadata:")
            for key in ("size_bytes", "modified_at", "created_at", "suffix", "mode"):
                if key in data["metadata"]:
                    lines.append(f"- {key}: {data['metadata'][key]}")
        if "start_line" in data and "end_line" in data:
            lines.append(f"Lines: {data['start_line']}-{data['end_line']}")
        if "line_count" in data:
            lines.append(f"Line count: {data['line_count']}")
        if "char_count" in data:
            lines.append(f"Characters: {data['char_count']}")
        if "bytes_read" in data:
            lines.append(f"Bytes read: {data['bytes_read']}")
        if "truncated" in data:
            lines.append(f"Truncated: {data['truncated']}")
        if data.get("content"):
            lines.append(f"Content:\n{self._code_block(str(data['content'])[:3200])}")
        if data.get("symbols"):
            lines.append("Symbols:")
            for item in data["symbols"][:10]:
                lines.append(
                    f"- {item['kind']} {item['name']} ({item['path']}:{item['line_number']}-{item['end_line']})"
                )
            if data.get("match_count", 0) > len(data["symbols"]):
                lines.append(f"Shown {len(data['symbols'])} of {data['match_count']} symbols.")
        if data.get("matches"):
            lines.append("Matches:")
            for item in data["matches"][:10]:
                if "kind" in item and "name" in item and "line_number" in item:
                    lines.append(
                        f"- {item['kind']} {item['name']} ({item['path']}:{item['line_number']}-{item['end_line']})"
                    )
                elif "line_number" in item:
                    lines.append(f"- {item['path']}:{item['line_number']}: {item['line']}")
                else:
                    lines.append(f"- {item['path']}")
            if data.get("match_count", 0) > len(data["matches"]):
                lines.append(f"Shown {len(data['matches'])} of {data['match_count']} matches.")
        if data.get("diff_preview"):
            lines.append(f"Diff preview:\n{self._code_block(str(data['diff_preview'])[:1500])}")
        if "diff" in data:
            diff_text = str(data["diff"])
            lines.append(f"Diff:\n{self._code_block(diff_text[:3200])}")
        if data.get("command"):
            lines.append(f"Command: {data['command']}")
        if data.get("preview"):
            lines.append(f"Preview: {data['preview']}")
        if data.get("status"):
            lines.append(f"Git status:\n{self._code_block(str(data['status'])[:1200])}")
        if data.get("diff_stat"):
            lines.append(f"Git diff:\n{self._code_block(str(data['diff_stat'])[:1200])}")
        if data.get("stdout"):
            lines.append(f"Stdout:\n{self._code_block(str(data['stdout'])[:1200])}")
        if data.get("stderr"):
            lines.append(f"Stderr:\n{self._code_block(str(data['stderr'])[:1200])}")
        if data.get("output"):
            lines.append(f"Output:\n{self._code_block(str(data['output'])[:1200])}")
        return "\n".join(lines)

    def _code_block(self, text: str) -> str:
        return f"<pre>{html.escape(text)}</pre>"
