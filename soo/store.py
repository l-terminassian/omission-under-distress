"""
soo/store.py — append-only JSONL with deterministic ids and free resume.

Every record carries a `record_id` derived from the fields that identify it, so
a re-run skips what is already done. This is what makes a 4,000-call grid
survivable: an expired credential, a rate-limit storm or an interrupted machine
costs only the in-flight calls, and the next run resumes where it stopped.

Records are flushed as they complete rather than at the end, so a hard kill
never loses more than the calls currently in flight.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator


def record_id(*parts: Any) -> str:
    """Stable short id from the fields that identify a row.

    None is encoded distinctly from the empty string. They are different states
    (an unresolved model id versus a blank one) and collapsing them would let
    two genuinely different rows share an id, which resume would then treat as
    one — a silent data-loss bug rather than a loud one.
    """
    joined = "\x1f".join("\x00none" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping unparseable lines rather than dying on them."""
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A partial final line is the normal signature of a hard kill
                # mid-write; losing it is correct, dying on it is not.
                print(f"[store] skipping unparseable line {path.name}:{line_no}", file=sys.stderr)
    return rows


def completed_ids(path: Path, require_success: bool = True) -> set[str]:
    """Ids already present. With require_success, failed rows are retried."""
    ids: set[str] = set()
    for row in read_jsonl(path):
        rid = row.get("record_id")
        if not rid:
            continue
        if require_success and row.get("error"):
            continue
        ids.add(rid)
    return ids


class JsonlWriter:
    """Append-only writer that flushes every record immediately."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None

    def __enter__(self) -> JsonlWriter:
        self._handle = self.path.open("a")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None

    def write(self, record: dict) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlWriter used outside its context manager")
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()


def iter_jsonl(path: Path) -> Iterator[dict]:
    yield from read_jsonl(path)
