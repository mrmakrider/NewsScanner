#!/usr/bin/env python3
"""Build the static GitHub Pages site under docs/ from digests/.

Every dated digest (digests/YYYY-MM-DD.html) is copied into docs/digests/,
and three artifacts keep the archive browsable:

  docs/index.html            landing page: latest edition + archive list
  docs/digests/index.json    machine-readable manifest of all editions
  docs/digests/*.html        the editions themselves, served as-is

The script is deterministic and idempotent: files are only rewritten when
their content actually changes, so running it every day never dirties the
git tree on a quiet day. Orphaned files in docs/digests/ (editions deleted
from digests/) are pruned so the published site cannot drift.

Stdlib only, like the rest of NewsScanner.

Usage:
  python scripts/sync_docs.py            # write docs/ (used by workflows)
  python scripts/sync_docs.py --check    # exit 1 if docs/ is stale, no writes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from html import escape
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIGESTS = REPO_ROOT / "digests"
DEFAULT_DOCS = REPO_ROOT / "docs"

# Only dated files are published; latest.* / scratch files are ignored.
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.html$")

# The stats line ("42 articles · 7 outlets · 23 stories · 5 major · generated
# 00:52 Kuwait") sits in a known meta div of every rendered digest. If a
# future digest format changes, extract_stats() returns "" and cards simply
# lose their stats line — nothing breaks.
STATS_RE = re.compile(r'font-size:12px;margin-top:6px;">([^<]+)</div>')
TAG_RE = re.compile(r"<[^>]+>")

AR_WEEKDAY = {
    0: "الاثنين",
    1: "الثلاثاء",
    2: "الأربعاء",
    3: "الخميس",
    4: "الجمعة",
    5: "السبت",
    6: "الأحد",
}


def extract_stats(html: str) -> str:
    """Pull the one-line stats summary out of a rendered digest."""
    match = STATS_RE.search(html)
    if not match:
        return ""
    return TAG_RE.sub("", match.group(1)).strip()


def find_editions(digests_dir: Path) -> list[dict]:
    """Collect metadata for every dated digest, newest first."""
    editions = []
    for path in sorted(digests_dir.glob("*.html")):
        match = DATE_RE.match(path.name)
        if not match:
            continue
        from datetime import date as _date

        day = _date.fromisoformat(match.group(1))
        html = path.read_text(encoding="utf-8")
        editions.append(
            {
                "date": match.group(1),
                "weekday_en": day.strftime("%A"),
                "weekday_ar": AR_WEEKDAY[day.weekday()],
                "long_en": day.strftime("%d %B %Y"),
                "href": f"digests/{path.name}",
                "stats": extract_stats(html),
            }
        )
    editions.sort(key=lambda e: e["date"], reverse=True)
    return editions


INDEX_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kuwait Morning Brief — NewsScanner</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0; background: #f6f7f8; color: #1c1e21;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Tahoma, Arial, sans-serif;
    font-size: 15px; line-height: 1.65;
  }
  .wrap { max-width: 720px; margin: 0 auto; padding: 28px 18px 40px; }
  .masthead { border-bottom: 3px solid #0b6b4f; padding-bottom: 14px; margin-bottom: 22px; }
  .masthead h1 { margin: 0; font-size: 26px; color: #0b6b4f; }
  .masthead .tag { color: #65676b; font-size: 13px; margin-top: 6px; }
  .hero {
    display: block; text-decoration: none; color: inherit;
    background: #ffffff; border: 1px solid #e4e6eb; border-left: 5px solid #0b6b4f;
    border-radius: 8px; padding: 18px 20px; margin-bottom: 26px;
  }
  .hero:hover { border-color: #0b6b4f; }
  .hero .date { font-size: 19px; font-weight: 700; color: #0b6b4f; }
  .hero .date small { font-weight: 400; color: #65676b; font-size: 13px; }
  .hero .stats { color: #65676b; font-size: 13px; margin-top: 4px; }
  .hero .cta {
    display: inline-block; margin-top: 12px; padding: 8px 16px;
    background: #0b6b4f; color: #ffffff; border-radius: 6px;
    font-size: 14px; font-weight: 600; text-decoration: none;
  }
  h2.section { font-size: 15px; text-transform: uppercase; letter-spacing: .06em; color: #65676b; margin: 0 0 10px; }
  ul.archive { list-style: none; margin: 0; padding: 0; }
  ul.archive li { margin: 0 0 8px; }
  ul.archive a {
    display: flex; justify-content: space-between; gap: 14px; flex-wrap: wrap;
    text-decoration: none; color: inherit;
    background: #ffffff; border: 1px solid #e4e6eb; border-radius: 6px;
    padding: 10px 14px;
  }
  ul.archive a:hover { border-color: #0b6b4f; }
  .a-date { font-weight: 600; }
  .a-date small { font-weight: 400; color: #65676b; }
  .a-stats { color: #65676b; font-size: 12.5px; align-self: center; text-align: right; }
  .empty { background: #fff4e5; border-left: 4px solid #f0a04b; padding: 10px 12px; font-size: 13px; }
  footer { margin-top: 30px; color: #65676b; font-size: 12px; border-top: 1px solid #e4e6eb; padding-top: 12px; }
  footer a { color: #0b6b4f; }
</style>
</head>
<body>
<div class="wrap">
  <div class="masthead">
    <h1>🇰🇼 Kuwait Morning Brief</h1>
    <div class="tag">Every morning at 08:00 Kuwait time, NewsScanner reads every Kuwaiti news outlet it can reach and merges the coverage into one briefed digest.</div>
  </div>
$hero
  <h2 class="section">Archive</h2>
$list
  <footer>
    Generated automatically by <code>scripts/sync_docs.py</code> — do not edit.
    Source: <a href="https://github.com/$repo">github.com/$repo</a>
  </footer>
</div>
</body>
</html>
""")

HERO_TEMPLATE = Template(
    """<a class="hero" href="$href">
  <div class="date">$weekday_en, $long_en <small>· $weekday_ar</small></div>
  <div class="stats">$stats</div>
  <span class="cta">Read the latest edition →</span>
</a>"""
)

ROW_TEMPLATE = Template(
    """<li><a href="$href"><span class="a-date">$weekday_en $long_en <small>· $weekday_ar</small></span><span class="a-stats">$stats</span></a></li>"""
)


def render_index(editions: list[dict], repo: str) -> str:
    if editions:
        hero = HERO_TEMPLATE.substitute(editions[0])
        rows = "\n".join(ROW_TEMPLATE.substitute(e) for e in editions)
        listing = f'<ul class="archive">\n{rows}\n</ul>'
    else:
        hero = ""
        listing = (
            '<div class="empty">📭 No editions published yet. '
            "The archive fills in after the first morning brief runs.</div>"
        )
    return INDEX_TEMPLATE.substitute(hero=hero, list=listing, repo=repo)


def _write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def _copy_if_changed(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return True


def sync(digests_dir: Path = DEFAULT_DIGESTS, docs_dir: Path = DEFAULT_DOCS,
         repo: str = "mrmakrider/NewsScanner") -> list[str]:
    """Bring docs/ in line with digests/. Returns the files it touched."""
    editions = find_editions(digests_dir)
    changed: list[str] = []

    index_json = json.dumps(
        {
            "editions": [
                {"date": e["date"], "href": e["href"], "stats": e["stats"]}
                for e in editions
            ]
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    if _write_if_changed(docs_dir / "digests" / "index.json", index_json):
        changed.append("docs/digests/index.json")

    if _write_if_changed(docs_dir / "index.html", render_index(editions, repo)):
        changed.append("docs/index.html")

    keep = set()
    for edition in editions:
        name = Path(edition["href"]).name
        keep.add(name)
        if _copy_if_changed(digests_dir / name, docs_dir / "digests" / name):
            changed.append(f"docs/digests/{name}")

    for stale in sorted((docs_dir / "digests").glob("*.html")):
        if stale.name not in keep:
            stale.unlink()
            changed.append(f"-docs/digests/{stale.name}")

    return changed


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    """Byte map of every file under root, keyed by relative path."""
    files: dict[str, bytes] = {}
    if not root.is_dir():
        return files
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root))] = path.read_bytes()
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true",
        help="exit 1 if docs/ is out of date, without writing anything",
    )
    args = parser.parse_args()

    if not DEFAULT_DIGESTS.is_dir():
        print(f"No digests directory at {DEFAULT_DIGESTS}; nothing to publish.")
        return 0

    if args.check:
        # Build into a scratch tree, then diff it against docs/.
        probe = REPO_ROOT / ".sync-docs-probe"
        try:
            sync(DEFAULT_DIGESTS, probe)
            have = _tree_snapshot(DEFAULT_DOCS)
            want = _tree_snapshot(probe)
        finally:
            if probe.exists():
                import shutil

                shutil.rmtree(probe)

        stale = [f" docs/{name}" for name in want if have.get(name) != want[name]]
        stale += [f"-docs/{name}" for name in have if name not in want]
        if stale:
            print("docs/ is stale; run: python scripts/sync_docs.py")
            for item in stale:
                print(f"  {item}")
            return 1
        print("docs/ is up to date.")
        return 0

    touched = sync()
    if touched:
        print("Updated docs/:")
        for item in touched:
            print(f"  {item}")
    else:
        print("docs/ already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
