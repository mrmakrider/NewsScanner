"""Cross-day memory: what the brief learned, kept so tomorrow can use it.

The per-story analysis cache answers "what did we say about this story".
This answers a different question: "what has this publication been tracking,
and did today's news change the picture?" A model has no state between runs,
so continuity has to be written down and handed back.

The store is an append-only JSONL file, one record per day::

    state/daily_memory.jsonl
    {"date": "2026-10-02", "editor_note_en": "...", "links": [...], ...}

Append-only because a crashed run must never corrupt earlier days, and
because the file is the artefact a reader can audit. Bounded because an
unbounded file in a git repository is a liability, not an asset — only the
most recent ``KEEP_DAYS`` records are retained.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# How many days of memory to keep. A month is enough to see a trend and small
# enough that the file stays readable in a diff.
KEEP_DAYS = 30

# The fields worth carrying forward. Everything else (raw summaries, per-story
# subtext) is reconstructible from the digests already in the repository, so
# duplicating it here would bloat the file for no gain.
_MEMORY_FIELDS = (
    "editor_note_en",
    "editor_note_ar",
    "story_of_the_day",
    "top_themes",
    "not_being_said_en",
    "not_being_said_ar",
    "watchlist_en",
    "watchlist_ar",
    "links",
    "tactical_en",
    "tactical_ar",
    "strategic_en",
    "strategic_ar",
    # Entities cross-reference the day's cross-story graph. Without them a
    # recurring actor is invisible from one day to the next, which is the
    # whole reason for keeping the file.
    "entities",
)


def default_path() -> Path:
    """Where the memory lives, overridable for tests."""
    raw = os.getenv("NEWSCANNER_MEMORY_PATH", "").strip()
    if raw:
        return Path(raw)
    root = Path(__file__).resolve().parent.parent
    return root / "state" / "daily_memory.jsonl"


def _project(record: dict[str, Any]) -> dict[str, Any]:
    return {k: record[k] for k in _MEMORY_FIELDS if record.get(k)}


def append(path: Path | None, date_str: str, synthesis: dict[str, Any]) -> bool:
    """Record one day. Rewrites the file, dropping the oldest beyond KEEP_DAYS.

    Returns False on any I/O problem: a brief that cannot remember is still a
    brief, and losing memory must never fail the run.
    """
    target = Path(path) if path else default_path()
    if not synthesis:
        return False
    record = {"date": date_str, **_project(synthesis)}
    if not any(v for k, v in record.items() if k != "date"):
        return False
    try:
        existing = load_all(target)
        # One record per day: a manual re-run replaces the day's own entry
        # rather than appending a second, contradictory version of it.
        kept = [r for r in existing if r.get("date") != date_str]
        kept.append(record)
        kept.sort(key=lambda r: str(r.get("date", "")))
        if len(kept) > KEEP_DAYS:
            kept = kept[-KEEP_DAYS:]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
            "utf-8",
        )
        return True
    except (OSError, ValueError) as exc:
        # ValueError covers the embedded-NUL and other invalid-path cases that
        # are not OSError subclasses; a path this broken is the caller's
        # mistake and must not take the whole run down with it.
        log.warning("could not write daily memory: %s", exc)
        return False


def load_all(path: Path | None = None) -> list[dict[str, Any]]:
    """Every stored day, oldest first. A malformed line is skipped, not fatal."""
    target = Path(path) if path else default_path()
    try:
        text = target.read_text("utf-8")
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("date"):
            records.append(item)
    return records


def load_recent(
    days: int = 7, *, path: Path | None = None, before: str | None = None
) -> list[dict[str, Any]]:
    """The last ``days`` records, optionally excluding ``before``'s own date.

    ``before`` matters when the 08:30 retry runs: today's record may already be
    written, and feeding it back as "previous days" would let the retry grade
    itself against its own first attempt.
    """
    records = load_all(path)
    if before:
        records = [r for r in records if r.get("date") != before]
    return records[-days:] if days > 0 else records


def recurring_entities(
    records: list[dict[str, Any]], *, min_days: int = 2, limit: int = 12
) -> list[str]:
    """Entities that keep appearing across remembered days.

    This is what makes the memory worth carrying: an entity that shows up in
    five of the last seven days is a thread, and telling the model so is more
    useful than making it re-derive the pattern from one day's headlines.
    """
    counts: dict[str, set[str]] = {}
    for record in records:
        day = str(record.get("date", ""))
        for entity in record.get("entities") or []:
            name = str(entity).strip()
            if name:
                counts.setdefault(name, set()).add(day)
    ranked = sorted(
        ((name, len(days)) for name, days in counts.items() if len(days) >= min_days),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return [name for name, _ in ranked[:limit]]


def build_context(
    records: list[dict[str, Any]], *, max_chars: int = 2400
) -> str:
    """Render remembered days as prompt context.

    Bounded by characters, not records: a long-running brief would otherwise
    build a prompt that grows without limit, which is exactly the call that
    times out and loses the whole synthesis.
    """
    if not records:
        return ""
    lines: list[str] = []
    for record in records:
        note = str(record.get("editor_note_en", "")).strip()
        if not note:
            continue
        entry = f"- {record.get('date')}: {note}"
        themes = [
            str(t.get("theme_en", "")).strip()
            for t in (record.get("top_themes") or [])
            if isinstance(t, dict) and str(t.get("theme_en", "")).strip()
        ]
        if themes:
            entry += f" | recurring themes: {'; '.join(themes[:4])}"
        strategic = str(record.get("strategic_en", "")).strip()
        if strategic:
            entry += f" | pattern under way: {strategic}"
        lines.append(entry)

    if not lines:
        return ""
    # Newest days matter most; drop the oldest until the block fits.
    while lines and len("\n".join(lines)) > max_chars:
        lines.pop(0)
    return "\n".join(lines)
