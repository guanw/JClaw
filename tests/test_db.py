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
