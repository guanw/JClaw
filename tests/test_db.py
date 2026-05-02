from jclaw.core.db import Database


def test_memory_round_trip(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    db.remember("chat-1", "project", "telegram bootstrap")
    matches = db.search_memories("chat-1", "bootstrap project", 5)
    assert matches
    assert matches[0].key == "project"
    assert matches[0].value == "telegram bootstrap"
    db.close()


def test_grant_and_approval_request_round_trip(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    grant = db.upsert_grant(str(tmp_path / "repo"), ("read", "write"), "chat-1")
    merged = db.upsert_grant(str(tmp_path / "repo"), ("git",), "chat-1")
    request = db.create_approval_request(
        kind="file_mutation",
        chat_id="chat-1",
        root_path=str(tmp_path / "repo"),
        capabilities=("read", "write"),
        objective="Update file",
        payload={"edits": []},
    )

    assert grant.id == merged.id
    assert merged.capabilities == ("git", "read", "write")
    fetched = db.get_approval_request(request.request_id)
    assert fetched is not None
    assert fetched.kind == "file_mutation"
    assert db.resolve_approval_request(request.request_id, "denied") == 1
    assert db.get_approval_request(request.request_id).status == "denied"
    db.close()


def test_execution_trace_round_trip(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    db.set_trace_mode("chat-1", "summary")
    assert db.get_trace_mode("chat-1") == "summary"

    session = db.create_execution_trace_session("chat-1", "inspect this file")
    db.append_execution_trace_event(
        session.trace_id,
        event_type="controller_decision",
        summary="Decided to call workspace.read_file",
        payload={"tool": "workspace", "action": "read_file"},
    )
    db.append_execution_trace_event(
        session.trace_id,
        event_type="tool_finished",
        summary="workspace.read_file: Read file.",
        payload={"ok": True},
    )
    db.finish_execution_trace_session(session.trace_id, status="completed", final_reply="done")

    latest = db.get_latest_execution_trace_session("chat-1")
    assert latest is not None
    assert latest.trace_id == session.trace_id
    assert latest.status == "completed"
    assert latest.final_reply == "done"

    events = db.list_execution_trace_events(session.trace_id)
    assert [item.event_type for item in events] == ["controller_decision", "tool_finished"]
    assert events[0].payload["tool"] == "workspace"
    db.close()
