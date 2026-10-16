#!/usr/bin/env python3
"""Render an existing digest JSON through the web edition, for previewing.

The published page is the main way anyone reads this brief, but a change to the
renderer is only visible once a real digest has been through it. This rebuilds
`digests/<date>.html` from the JSON record of the same day, without touching
the network or the analysis pipeline.

    python3 scripts/preview_web.py                      # newest dated digest
    python3 scripts/preview_web.py 2026-09-30 --stdout
    python3 scripts/preview_web.py --out preview.html

Writes to the scratch directory by default; `--stdout` prints to the terminal.
It never overwrites a dated digest, because that file is committed history.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from news_scanner.models import Article, Digest, Story  # noqa: E402
from news_scanner.web import render_web  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DIGESTS = REPO_ROOT / "digests"


def load_digest(path: Path) -> Digest:
    """Rebuild a Digest from the JSON written on the day it was generated."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    stories: list[Story] = []
    analyses: list[dict] = []
    for item in raw.get("stories") or []:
        # sources and outlet_count are derived from the articles, so only the
        # articles are restored here — passing the recorded values back in would
        # be rejected by the dataclass and would be redundant anyway.
        stories.append(
            Story(
                key=item.get("key", ""),
                score=float(item.get("score") or 0.0),
                articles=[Article.from_dict(a) for a in item.get("articles") or []],
            )
        )
        analyses.append(item.get("analysis") or {})

    generated = raw.get("generated_at")
    try:
        generated_at = datetime.datetime.fromisoformat(generated)
    except (TypeError, ValueError):
        generated_at = datetime.datetime.now()

    return Digest(
        date=raw.get("date", ""),
        generated_at=generated_at,
        window_hours=int(raw.get("window_hours") or 24),
        stories=stories,
        analyses=analyses,
        synthesis=raw.get("synthesis") or {},
        source_health=raw.get("source_health") or [],
        stats=raw.get("stats") or {},
        llm_provider=raw.get("provider", "none"),
        errors=raw.get("errors") or [],
    )


def newest_digest() -> Path | None:
    if not DIGESTS.is_dir():
        return None
    dated = sorted(
        (p for p in DIGESTS.glob("*.json") if p.stem[:2].isdigit()),
        reverse=True,
    )
    return dated[0] if dated else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("date", nargs="?", help="digest date, YYYY-MM-DD")
    parser.add_argument("--out", help="write the page here instead of stdout")
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args(argv)

    if args.date:
        source = DIGESTS / f"{args.date}.json"
        if not source.exists():
            print(f"No digest JSON at {source}", file=sys.stderr)
            return 1
    else:
        source = newest_digest()
        if source is None:
            print(f"No dated digest JSON under {DIGESTS}", file=sys.stderr)
            return 1

    digest = load_digest(source)
    page = render_web(digest)

    if args.stdout or not args.out:
        print(page)
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(
        f"Wrote {out} ({len(page):,} bytes, {len(digest.stories)} stories "
        f"from {source.name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
