from jclaw.core.db import Database


def test_memory_round_trip(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    db.remember("chat-1", "project", "telegram bootstrap")
    matches = db.search_memories("chat-1", "bootstrap project", 5)
    assert matches
    assert matches[0].key == "project"
    assert matches[0].value == "telegram bootstrap"
    db.close()

