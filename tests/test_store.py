"""
Tests for record identity and resume.

The nastiest failure mode this project has is a *silent* one: stub records from
`--dry-run` sharing ids with real records, so a later real run skips them as
"already done" and stub text flows into the analysis looking like data. That is
what `test_stub_and_real_ids_never_collide` exists to prevent.
"""
from __future__ import annotations

import json

from soo.run_behavioural import _unit_id
from soo.store import JsonlWriter, completed_ids, read_jsonl, record_id


CELL = {
    "scenario_id": "health_001",
    "attribution": "first_person",
    "vulnerability": "lonely",
    "advice_source": "model_advised",
}
REAL = {"provider": "anthropic", "key": "claude-sonnet-5"}
STUB = {"provider": "stub", "key": "claude-sonnet-5"}


def test_record_id_is_deterministic():
    assert record_id("a", "b", 1) == record_id("a", "b", 1)


def test_record_id_is_sensitive_to_every_field():
    base = record_id("a", "b", 1)
    assert record_id("a", "b", 2) != base
    assert record_id("a", "c", 1) != base
    assert record_id("z", "b", 1) != base


def test_record_id_distinguishes_none_from_empty_string():
    assert record_id(None, "x") != record_id("", "x")


def test_stub_and_real_ids_never_collide():
    """A dry run must not poison a later real run's resume set."""
    assert _unit_id(REAL, CELL, 0) != _unit_id(STUB, CELL, 0)


def test_unit_id_varies_across_the_grid():
    """Every cell of the 2x4x2 factorial must get its own id."""
    ids = {
        _unit_id(REAL, dict(CELL, attribution=a, vulnerability=v, advice_source=src), 0)
        for a in ("first_person", "friend")
        for v in ("neutral", "lonely", "bereaved", "distressed")
        for src in ("model_advised", "other_advised")
    }
    assert len(ids) == 2 * 4 * 2


def test_unit_id_distinguishes_the_advice_source():
    """The self/other axis must not collapse into one id."""
    a = _unit_id(REAL, dict(CELL, advice_source="model_advised"), 0)
    b = _unit_id(REAL, dict(CELL, advice_source="other_advised"), 0)
    assert a != b


def test_resume_skips_successes_and_retries_failures(tmp_path):
    path = tmp_path / "rows.jsonl"
    with JsonlWriter(path) as writer:
        writer.write({"record_id": "ok", "error": None})
        writer.write({"record_id": "bad", "error": "RateLimitError: boom"})

    done = completed_ids(path)
    assert "ok" in done
    assert "bad" not in done, "failed rows must be retried on the next run"

    assert "bad" in completed_ids(path, require_success=False)


def test_reader_survives_a_truncated_final_line(tmp_path):
    """A hard kill mid-write leaves a partial line; it must not kill the reader."""
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"record_id": "a"}) + "\n" + '{"record_id": "b"')
    rows = read_jsonl(path)
    assert [r["record_id"] for r in rows] == ["a"]


def test_writer_flushes_each_record(tmp_path):
    """Records must be durable as they complete, not buffered to the end."""
    path = tmp_path / "rows.jsonl"
    with JsonlWriter(path) as writer:
        writer.write({"record_id": "a"})
        assert len(read_jsonl(path)) == 1
