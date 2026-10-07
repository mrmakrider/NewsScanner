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
<meta name="theme-color" content="#f5f6f1">
<meta name="description" content="A thoughtful, bilingual scan of Kuwait's daily news — all the coverage, one clear morning brief.">
<title>Kuwait Morning Brief — NewsScanner</title>
<style>
  :root {
    color-scheme: light;
    --ink: #14251f;
    --muted: #65736d;
    --green: #0b6b4f;
    --green-dark: #084a3b;
    --line: #dfe6df;
    --paper: #fff;
    --canvas: #f5f6f1;
    --gold: #f0c875;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    background: radial-gradient(ellipse at 50% -20%, #e3eee4 0, transparent 54%), var(--canvas);
    color: var(--ink);
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Tahoma, Arial, sans-serif;
    font-size: 15px;
    line-height: 1.65;
    -webkit-font-smoothing: antialiased;
  }
  a { color: inherit; }
  a:focus-visible { outline: 3px solid #d28a16; outline-offset: 4px; }
  .skip-link {
    position: absolute; left: 16px; top: 12px; z-index: 2;
    padding: 9px 13px; transform: translateY(-160%);
    border-radius: 8px; background: var(--ink); color: #fff; font-weight: 700;
  }
  .skip-link:focus { transform: translateY(0); }
  .page-shell { width: min(100% - 40px, 920px); margin: 0 auto; }
  .masthead {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    min-height: 82px; border-bottom: 1px solid rgba(20, 37, 31, .12);
  }
  .brand { display: inline-flex; align-items: center; gap: 11px; text-decoration: none; }
  .brand-mark {
    display: grid; place-items: center; width: 42px; height: 42px;
    border: 1px solid #d7e2d7; border-radius: 13px; background: #fff;
    box-shadow: 0 5px 15px rgba(20, 37, 31, .06); font-size: 21px;
  }
  .brand-copy { display: grid; line-height: 1.2; }
  .brand-copy strong { font-size: 15px; letter-spacing: -.02em; }
  .brand-copy small { margin-top: 5px; color: var(--muted); font-size: 9px; font-weight: 800; letter-spacing: .16em; }
  .edition-time { display: inline-flex; align-items: center; gap: 8px; color: var(--muted); font-size: 12px; font-weight: 650; }
  .live-dot { width: 7px; height: 7px; border-radius: 50%; background: #1c9a69; box-shadow: 0 0 0 4px #dcefe3; }
  main { padding: 58px 0 16px; }
  .intro { max-width: 620px; margin: 0 auto 32px; text-align: center; }
  .eyebrow { margin: 0 0 10px; color: var(--green); font-size: 10px; font-weight: 850; letter-spacing: .18em; }
  .intro h1 { margin: 0; font-size: clamp(34px, 6vw, 54px); line-height: 1.05; letter-spacing: -.055em; }
  .intro h1 em { color: var(--green); font-style: normal; }
  .intro-copy { max-width: 500px; margin: 16px auto 0; color: var(--muted); font-size: 15px; }
  .hero {
    position: relative; display: block; overflow: hidden; isolation: isolate;
    padding: clamp(22px, 5vw, 38px); border: 1px solid rgba(255, 255, 255, .18);
    border-radius: 22px; background: linear-gradient(125deg, #0d352c 0%, #0b6b4f 68%, #16785c 100%);
    box-shadow: 0 18px 42px rgba(7, 55, 42, .17); color: #fff; text-decoration: none;
    transition: transform .2s ease, box-shadow .2s ease;
  }
  .hero::before, .hero::after { position: absolute; z-index: -1; content: ''; pointer-events: none; }
  .hero::before { width: 310px; height: 310px; right: -90px; top: -145px; border: 1px solid rgba(255,255,255,.15); border-radius: 50%; box-shadow: 0 0 0 34px rgba(255,255,255,.035), 0 0 0 70px rgba(255,255,255,.025); }
  .hero::after { width: 180px; height: 180px; right: 68px; bottom: -130px; border-radius: 50%; background: rgba(240,200,117,.12); filter: blur(2px); }
  .hero:hover { transform: translateY(-3px); box-shadow: 0 24px 52px rgba(7, 55, 42, .24); }
  .hero-topline { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; }
  .latest-badge { display: inline-flex; align-items: center; gap: 7px; padding: 5px 10px; border: 1px solid rgba(255,255,255,.22); border-radius: 999px; background: rgba(255,255,255,.09); color: #f6e6bc; font-size: 10px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
  .hero-date { color: rgba(255,255,255,.77); font-size: 12px; font-weight: 600; }
  .hero-title { margin: 30px 0 0; font-size: clamp(23px, 4vw, 32px); font-weight: 750; letter-spacing: -.035em; line-height: 1.18; }
  .hero-title [lang="ar"] { margin-left: 6px; color: #f6e6bc; font-size: .74em; font-weight: 600; }
  .hero-stats { max-width: 620px; margin: 14px 0 0; color: rgba(255,255,255,.78); font-size: 13px; }
  .hero-bottom { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-top: 27px; padding-top: 18px; border-top: 1px solid rgba(255,255,255,.18); }
  .hero-cta { color: #fff; font-size: 13px; font-weight: 750; }
  .hero-arrow { display: grid; place-items: center; width: 34px; height: 34px; border-radius: 50%; background: var(--gold); color: #19372b; font-size: 17px; }
  .archive-section { margin-top: 56px; }
  .section-heading { display: flex; align-items: end; justify-content: space-between; gap: 18px; margin-bottom: 17px; }
  .section-heading .eyebrow { margin-bottom: 4px; }
  .section-heading h2 { margin: 0; font-size: 24px; line-height: 1.25; letter-spacing: -.035em; }
  .edition-count { padding: 5px 10px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: 11px; font-weight: 700; white-space: nowrap; }
  .archive-list { display: grid; gap: 9px; margin: 0; padding: 0; list-style: none; }
  .archive-link {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 15px 17px; border: 1px solid var(--line); border-radius: 13px;
    background: rgba(255,255,255,.78); text-decoration: none;
    transition: border-color .18s ease, background .18s ease, transform .18s ease;
  }
  .archive-link:hover { transform: translateX(3px); border-color: #a9c9b6; background: #fff; }
  .archive-date { display: grid; gap: 2px; font-weight: 750; }
  .archive-date small { color: var(--muted); font-size: 11px; font-weight: 550; }
  .archive-meta { display: flex; align-items: center; gap: 13px; color: var(--muted); font-size: 11px; text-align: right; }
  .archive-arrow { color: var(--green); font-size: 15px; font-weight: 800; }
  .empty { padding: 18px; border: 1px dashed #d7b975; border-radius: 13px; background: #fffaf0; color: #725829; font-size: 13px; }
  footer { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px 20px; margin-top: 48px; padding: 18px 0 28px; border-top: 1px solid rgba(20,37,31,.12); color: var(--muted); font-size: 11px; }
  footer a { color: var(--green); font-weight: 700; text-decoration-thickness: 1px; text-underline-offset: 3px; }
  @media (max-width: 600px) {
    .page-shell { width: min(100% - 28px, 920px); }
    .masthead { min-height: 70px; }
    .edition-time { gap: 7px; font-size: 10px; }
    main { padding-top: 42px; }
    .intro { margin-bottom: 26px; }
    .intro-copy { max-width: 360px; font-size: 13px; }
    .hero { border-radius: 17px; }
    .hero-date { font-size: 11px; }
    .hero-title { margin-top: 24px; }
    .archive-section { margin-top: 42px; }
    .archive-link { align-items: flex-start; padding: 13px 14px; }
    .archive-meta { max-width: 48%; justify-content: flex-end; font-size: 10px; }
  }
  @media (max-width: 390px) {
    .edition-time { max-width: 106px; text-align: right; }
    .archive-link { display: grid; gap: 6px; }
    .archive-meta { max-width: none; justify-content: space-between; text-align: left; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
  }
</style>
</head>
<body>
<a class="skip-link" href="#archive">Skip to previous editions</a>
<div class="page-shell">
  <header class="masthead">
    <a class="brand" href="./" aria-label="Kuwait Morning Brief home">
      <span class="brand-mark" aria-hidden="true">🇰🇼</span>
      <span class="brand-copy"><strong>NewsScanner</strong><small>KUWAIT MORNING BRIEF</small></span>
    </a>
    <span class="edition-time"><span class="live-dot" aria-hidden="true"></span> Daily · 08:00 Kuwait</span>
  </header>
  <main>
    <section class="intro" aria-labelledby="page-title">
      <p class="eyebrow">THE MORNING, MADE CLEARER</p>
      <h1 id="page-title">One brief. <em>Every story.</em></h1>
      <p class="intro-copy">A thoughtful, bilingual scan of Kuwait's daily news — bringing every outlet's coverage together in one clear morning read.</p>
    </section>
    $hero
    <section class="archive-section" id="archive" aria-labelledby="archive-title">
      <div class="section-heading">
        <div><p class="eyebrow">THE BACK CATALOGUE</p><h2 id="archive-title">Previous editions</h2></div>
        <span class="edition-count">$edition_count editions</span>
      </div>
      $list
    </section>
  </main>
  <footer>
    <span>Generated automatically by <code>scripts/sync_docs.py</code>.</span>
    <span>Source: <a href="https://github.com/$repo">github.com/$repo</a></span>
  </footer>
</div>
</body>
</html>
""")

HERO_TEMPLATE = Template(
    """<a class="hero" href="$href" aria-label="Read the latest edition, $weekday_en $long_en">
  <div class="hero-topline">
    <span class="latest-badge">Latest edition</span>
    <time class="hero-date" datetime="$date">$weekday_en, $long_en · $weekday_ar</time>
  </div>
  <div class="hero-title">Kuwait Morning Brief <span lang="ar" dir="rtl">موجز الصباح</span></div>
  <p class="hero-stats">$stats</p>
  <div class="hero-bottom"><span class="hero-cta">Open today's briefing</span><span class="hero-arrow" aria-hidden="true">↗</span></div>
</a>"""
)

ROW_TEMPLATE = Template(
    """<li><a class="archive-link" href="$href" aria-label="Read the edition from $weekday_en $long_en"><span class="archive-date"><time datetime="$date">$weekday_en · $long_en</time><small lang="ar" dir="rtl">$weekday_ar</small></span><span class="archive-meta">$stats <span class="archive-arrow" aria-hidden="true">↗</span></span></a></li>"""
)


def render_index(editions: list[dict], repo: str) -> str:
    if editions:
        hero = HERO_TEMPLATE.substitute(editions[0])
        rows = "\n".join(ROW_TEMPLATE.substitute(e) for e in editions)
        listing = f'<ul class="archive-list">\n{rows}\n</ul>'
    else:
        hero = ""
        listing = (
            '<div class="empty">No editions published yet. '
            "The archive fills in after the first morning brief runs.</div>"
        )
    return INDEX_TEMPLATE.substitute(
        hero=hero, list=listing, repo=repo, edition_count=len(editions)
    )


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
