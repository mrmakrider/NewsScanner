"""The published web edition.

``render.render_html`` builds the email: one inline-styled, table-free blob
that every mail client can be trusted with. That same blob is what GitHub
Pages has been serving, which is why the site is a wall of stories with no
navigation, no way to narrow 224 of them, and no way back to the top once you
are 20 stories down.

An email client and a browser are different targets, so this module is a
separate renderer rather than a restyle of the email. It emits a real
document with a stylesheet, a small amount of progressive-enhancement
JavaScript, and real anchors between stories. Everything still renders with
the JavaScript disabled and even with CSS stripped — the filters and search
are conveniences over markup that is already complete, never the only way to
reach a story.

Stdlib only, like the rest of NewsScanner.
"""

from __future__ import annotations

import html as html_mod
import re
from typing import Any

from .dedupe import CATEGORY_ORDER, categorize
from .models import Digest, Story
from .render import (
    _LINK_STYLES,
    _bilingual,
    _date_heading,
    _has_arabic,
    _renderable_links,
)
from .util import humanize_age, normalize_text

# Ink and accent, shared with the email so the two read as one publication.
INK = "#1c1e21"
MUTED = "#65676b"
FAINT = "#8a8d91"
ACCENT = "#0b6b4f"
RULE = "#e4e6eb"

# Confidence colours, matched to the email's.
_CONF_COLOUR = {"high": "#2e7d32", "medium": "#b26a00", "low": "#b00020"}


def _h(text: Any) -> str:
    # Counts arrive as ints from the stats block; escape takes text only, so
    # coerce rather than making every call site remember to stringify.
    return html_mod.escape(str(text) if text is not None else "", quote=True)


def _dir_attrs(text: str) -> str:
    """dir/align pair for a block whose script is known only at render time."""
    if _has_arabic(text):
        return ' dir="rtl"'
    return ' dir="ltr"'


def _para_html(text: str, css_class: str = "") -> str:
    """Escaped paragraphs, RTL-aware, as real <p> elements.

    Unlike the email version this does not fight the stylesheet with inline
    styles: the class carries the type, the attribute carries the direction.
    """
    if not text:
        return ""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    cls = f' class="{css_class}"' if css_class else ""
    return "".join(
        f"<p{cls}{_dir_attrs(block)}>{_h(block).replace(chr(10), '<br>')}</p>"
        for block in blocks
    )


def _story_alt_headline(story: Story, analysis: dict[str, Any]) -> str:
    """The other language's headline, when it is not a copy of the original."""
    primary = story.primary
    alt = (
        analysis.get("headline_en", "")
        if primary.lang == "ar"
        else analysis.get("headline_ar", "")
    )
    alt = str(alt).strip()
    if not alt or normalize_text(alt) == normalize_text(primary.title):
        return ""
    return alt


def _category_of_story(story: Story) -> str:
    return categorize(story)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:40]


# --------------------------------------------------------------------------
# Story cards
# --------------------------------------------------------------------------


def _refs_html(story: Story) -> str:
    rows = []
    for art in sorted(story.articles, key=lambda a: (a.tier, a.source_name, a.title)):
        when = art.published.strftime("%H:%M") if art.published else "--:--"
        rows.append(
            f'<li><a href="{_h(art.url)}" rel="noopener" target="_blank">'
            f"{_h(art.title)}</a>"
            f'<span class="ref-meta">{_h(art.source_name)} · {when}</span></li>'
        )
    if not rows:
        return ""
    # <details> keeps 200 collapsed lists cheap; the JS expands them on demand.
    return (
        f'<details class="refs"><summary>References ({len(rows)})</summary>'
        f'<ul>{"".join(rows)}</ul></details>'
    )


def _card_html(
    idx: int, story: Story, analysis: dict[str, Any], links_in: list[dict[str, str]]
) -> str:
    """One story as a card.

    The ``data-`` attributes are what the filter and search read. They carry
    the category, the outlets and the searchable text, so filtering needs no
    second index and cannot drift out of step with what is on screen.
    """
    sid = f"S{idx + 1}"
    primary = story.primary
    category = _category_of_story(story)
    alt = _story_alt_headline(story, analysis)

    searchable = " ".join(
        [
            primary.title,
            str(analysis.get("summary_en", "")),
            str(analysis.get("summary_ar", "")),
            " ".join(story.sources),
            category,
        ]
    ).lower()

    tags = " ".join(
        [category.lower()] + [s.lower() for s in story.sources]
    )

    major = story.outlet_count >= 3
    conf = str(analysis.get("confidence", "")).strip().lower()
    conf_colour = _CONF_COLOUR.get(conf, MUTED)

    head_bits = [f'<span class="num">{idx + 1}</span>']
    if major:
        head_bits.append('<span class="tag tag-major" title="carried by 3+ outlets">major</span>')
    head_bits.append(f'<span class="tag cat" data-slug="{_h(_slug(category))}">{_h(category)}</span>')

    summary = _bilingual(analysis.get("summary_en", ""), analysis.get("summary_ar", ""))
    why = _bilingual(analysis.get("why_it_matters_en", ""), analysis.get("why_it_matters_ar", ""))
    btl = _bilingual(
        analysis.get("between_the_lines_en", ""), analysis.get("between_the_lines_ar", "")
    )
    watch = _bilingual(analysis.get("watch_next_en", ""), analysis.get("watch_next_ar", ""))

    parts: list[str] = [
        f'<article class="story{" major" if major else ""}" id="story-{sid}"'
        f' data-category="{_h(_slug(category))}" data-category-label="{_h(category)}"'
        f' data-outlets="{_h(tags)}"'
        f' data-search="{_h(searchable)}">',
        f'<h3 class="story-head">{"".join(head_bits)}'
        f'<span class="story-title" dir="auto">{_h(primary.title)}</span></h3>',
    ]

    meta = [f"{category} · carried by {story.outlet_count} outlet"
            f"{'s' if story.outlet_count != 1 else ''}: {', '.join(story.sources)}"]
    if story.latest:
        meta.append(humanize_age(story.latest))
    parts.append(f'<p class="story-meta">{_h(" · ".join(meta))}</p>')

    if alt:
        parts.append(
            f'<p class="alt-headline"{_dir_attrs(alt)}>{_h(alt)}</p>'
        )

    if summary:
        parts.append(
            f'<div class="block"><span class="label">Summary</span>{_para_html(summary)}</div>'
        )
    if why:
        parts.append(
            f'<div class="block"><span class="label">Why it matters</span>{_para_html(why)}</div>'
        )

    if btl:
        from_llm = str(analysis.get("_source", "")) == "llm"
        heading = (
            '<span class="label">Between the lines</span>'
            '<span class="caveat">inference, not reported fact</span>'
            if from_llm
            else '<span class="label">Between the lines</span>'
        )
        parts.append(f'<div class="subtext">{heading}{_para_html(btl)}</div>')

    if conf:
        reason = str(analysis.get("confidence_reason", "")).strip()
        parts.append(
            f'<p class="confidence" style="color:{conf_colour}">● Confidence: {_h(conf)}'
            + (f' — {_h(reason)}' if reason else "")
            + "</p>"
        )

    if watch:
        parts.append(
            f'<div class="watch"{_dir_attrs(watch)}>'
            f'<span class="label">Watch next</span>{_para_html(watch)}</div>'
        )

    if links_in:
        chips = "".join(
            f'<a class="edge" href="#story-{_h(e["to"])}" style="--edge:{_LINK_STYLES.get(e["type"], ("", MUTED))[1]}"'
            f' title="{_h(e["evidence"])}">{_h(e["from"])} → {_h(e["to"])}</a>'
            for e in links_in
        )
        parts.append(f'<div class="edges">{chips}</div>')

    parts.append(_refs_html(story))
    parts.append("</article>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Editorial sections
# --------------------------------------------------------------------------


def _section(title: str, body: str, *, cls: str = "", anchor: str = "") -> str:
    if not body:
        return ""
    idattr = f' id="{_h(anchor)}"' if anchor else ""
    heading = f'<h2 class="section-title"{idattr}>{_h(title)}</h2>'
    return f'<section class="brief{(" " + cls) if cls else ""}">{heading}{body}</section>'


def _links_section(links: list[dict[str, str]], story_titles: dict[str, str]) -> str:
    if not links:
        return ""
    by_kind: dict[str, list[dict[str, str]]] = {}
    for link in links:
        by_kind.setdefault(link["type"], []).append(link)

    total = len(links)
    kinds = len(by_kind)
    intro = (
        f'<p class="section-intro">{total} connection{"" if total == 1 else "s"} '
        f'across {kinds} relationship type{"" if kinds == 1 else "s"}. '
        "Follow a pair to jump to those stories.</p>"
    )

    groups: list[str] = []
    for kind in sorted(by_kind, key=lambda k: (-len(by_kind[k]), k)):
        label, colour = _LINK_STYLES.get(kind, (kind.replace("_", " "), MUTED))
        edges = by_kind[kind]
        items = []
        for edge in edges:
            src = story_titles.get(edge["from"], edge["from"])
            dst = story_titles.get(edge["to"], edge["to"])
            items.append(
                f'<li class="edge-row" style="--edge:{colour}">'
                f'<a href="#story-{_h(edge["from"])}">{_h(src)}</a>'
                f'<span class="arrow">→</span>'
                f'<a href="#story-{_h(edge["to"])}">{_h(dst)}</a>'
                f'<span class="evidence">{_h(edge["evidence"])}</span></li>'
            )
        groups.append(
            f'<div class="edge-group"><h3 class="edge-head" style="color:{colour}">'
            f'{_h(label)} <span class="count">({len(edges)})</span></h3>'
            f'<ul>{"".join(items)}</ul></div>'
        )
    return _section("How today's stories connect", intro + "".join(groups), cls="links")


def _themes_html(themes: list[Any]) -> str:
    items = []
    for theme in themes:
        if not isinstance(theme, dict):
            continue
        label = str(theme.get("theme_en", "")).strip()
        label_ar = str(theme.get("theme_ar", "")).strip()
        ids = ", ".join(str(i) for i in (theme.get("story_ids") or []))
        if not label and not label_ar:
            continue
        items.append(
            "<li>"
            f"<b>{_h(label)}</b>"
            + (f' <span class="ar" dir="rtl">{_h(label_ar)}</span>' if label_ar else "")
            + (
                ' <span class="ids">'
                + " ".join(
                    # story_ids arrive as "S5", not 5 — prefixing again
                    # produced #story-SS5 and every theme link was dead.
                    f'<a href="#story-{_h(i)}">{_h(i)}</a>'
                    for i in (theme.get("story_ids") or [])
                )
                + "</span>"
                if ids
                else ""
            )
            + "</li>"
        )
    if not items:
        return ""
    return _section("Themes", f'<ul class="themes">{"".join(items)}</ul>')


def _glance_html(digest: Digest, listed: list[Story]) -> str:
    """At-a-glance tiles: the day's shape before any prose.

    ``listed`` is the stories actually rendered, which is a subset — a run
    analyses the top 28 of 224. The category counts are taken from it rather
    than from the whole digest, because these tiles double as the filter
    control and a tile that reads 19 while clicking it reveals 2 is worse than
    no count at all.
    """
    categories: dict[str, int] = {}
    for story in listed:
        name = _category_of_story(story)
        categories[name] = categories.get(name, 0) + 1

    busiest: dict[str, int] = {}
    for story in digest.stories:
        for art in story.articles:
            busiest[art.source_name] = busiest.get(art.source_name, 0) + 1

    total_stories = len(digest.stories)
    top = max(busiest.values()) if busiest else 0

    tiles = [
        ("Articles", digest.stats.get("total_articles", 0)),
        ("Outlets", digest.stats.get("outlets_reporting", 0)),
        ("Stories", total_stories),
        ("Listed here", len(listed)),
    ]
    tile_html = "".join(
        f'<div class="tile"><span class="tile-n">{_h(v)}</span>'
        f'<span class="tile-l">{_h(k)}</span></div>'
        for k, v in tiles
    )

    bars = []
    for name in sorted(busiest, key=lambda n: (-busiest[n], n))[:8]:
        count = busiest[name]
        pct = round(100 * count / top) if top else 0
        bars.append(
            f'<li><span class="bar-label">{_h(name)}</span>'
            f'<span class="bar"><span class="bar-fill" style="width:{pct}%"></span></span>'
            f'<span class="bar-n">{count}</span></li>'
        )

    # Every tile here is a button, so a reader can go straight from "how many"
    # to "which ones" without scrolling back to the control at the top.
    cat_tiles = "".join(
        f'<button type="button" class="cat-tile" data-filter="{_h(_slug(name))}">'
        f'<span class="tile-n">{categories.get(name, 0)}</span>'
        f'<span class="tile-l">{_h(name)}</span></button>'
        for name in CATEGORY_ORDER
        if categories.get(name)
    )

    return _section(
        "Today at a glance",
        f'<div class="tiles">{tile_html}</div>'
        + (f'<div class="cat-tiles">{cat_tiles}</div>' if cat_tiles else "")
        + (f'<ul class="bars">{"".join(bars)}</ul>' if bars else ""),
        anchor="glance",
    )


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------

_CSS = """
:root {
  --ink: __INK__; --muted: __MUTED__; --faint: __FAINT__;
  --accent: __ACCENT__; --rule: __RULE__;
  --bg: #f6f7f8; --card: #ffffff;
  /* Opaque rather than a hardcoded light rgba: the sticky bar must follow the
     colour scheme, and a literal #f6f7f8 there is a white band on dark. */
  --scrim: rgba(246, 247, 248, .94);
  --radius: 10px;
  --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Tahoma, Arial, sans-serif;
  --serif: 'Iowan Old Style', 'Palatino Linotype', Georgia, 'Times New Roman', serif;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--sans); font-size: 16px; line-height: 1.65;
  -webkit-text-size-adjust: 100%;
}
a { color: var(--accent); }
.wrap { max-width: 780px; margin: 0 auto; padding: 0 18px 80px; }

/* masthead */
.masthead { background: var(--card); border-bottom: 3px solid var(--accent); }
.masthead-in { max-width: 780px; margin: 0 auto; padding: 22px 18px 18px; }
.masthead h1 { margin: 0; font-size: 25px; color: var(--accent); letter-spacing: -.01em; }
.masthead .date { margin-top: 4px; color: var(--muted); font-size: 14px; }
.masthead .stats { margin-top: 6px; color: var(--faint); font-size: 12.5px; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.badge {
  font-size: 11.5px; padding: 2px 8px; border-radius: 999px;
  background: #eef4f1; color: var(--accent); border: 1px solid #d6e7e0;
}
.badge.warn { background: #fff4e5; color: #a35c00; border-color: #f2ddbe; }

/* controls */
.controls {
  position: sticky; top: 0; z-index: 10; background: var(--scrim);
  backdrop-filter: saturate(140%) blur(8px);
  border-bottom: 1px solid var(--rule); padding: 10px 0;
}
.controls-in { max-width: 780px; margin: 0 auto; padding: 0 18px; }
.search-row { display: flex; gap: 8px; align-items: center; }
.search {
  flex: 1; min-width: 0; font: inherit; font-size: 14px; padding: 8px 12px;
  border: 1px solid var(--rule); border-radius: 8px; background: var(--card);
  color: var(--ink);
}
.search:focus { outline: 2px solid var(--accent); outline-offset: -1px; border-color: var(--accent); }
.chip {
  font: inherit; font-size: 13px; padding: 7px 12px; cursor: pointer;
  border: 1px solid var(--rule); border-radius: 8px; background: var(--card);
  color: var(--muted); white-space: nowrap;
}
.chip[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
.chip:disabled { opacity: .45; cursor: default; }
.filters { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.filters[hidden] { display: none; }
/* Empty until a filter is active. Reserving height here kept a blank band
   under the search box on every clean page load. */
.status { margin-top: 6px; font-size: 12.5px; color: var(--muted); }
.status:empty { display: none; }
.status b { color: var(--ink); }
mark { background: #fff1b8; color: inherit; padding: 0 1px; border-radius: 2px; }

/* brief sections */
.brief { margin: 26px 0 0; }
.section-title {
  font-size: 14px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); margin: 0 0 10px; font-weight: 700;
}
.section-intro { color: var(--muted); font-size: 13px; margin: 0 0 10px; }
.brief p { margin: 0 0 10px; }
.brief p[dir="rtl"] { text-align: right; }
.callout {
  background: var(--card); border-left: 4px solid var(--accent);
  border-radius: 0 var(--radius) var(--radius) 0; padding: 14px 16px;
}
.callout.plain { border-left-color: var(--rule); }
.themes { list-style: none; margin: 0; padding: 0; }
.themes li { background: var(--card); border: 1px solid var(--rule); border-radius: 8px; padding: 10px 12px; margin-bottom: 6px; }
.themes .ar { color: var(--muted); }
.themes .ids { margin-left: 4px; }
.themes .ids a {
  display: inline-block; min-width: 20px; text-align: center; font-size: 12px;
  text-decoration: none; background: #eef4f1; border-radius: 4px; margin: 0 1px;
}

/* links graph */
.edge-group { margin-bottom: 14px; }
.edge-head { font-size: 13px; margin: 0 0 6px; font-weight: 700; }
.edge-head .count { color: var(--muted); font-weight: 400; }
.edge-group ul { list-style: none; margin: 0; padding: 0; }
.edge-row {
  background: var(--card); border: 1px solid var(--rule);
  border-left: 4px solid var(--edge, var(--accent));
  border-radius: 0 8px 8px 0; padding: 9px 12px; margin-bottom: 6px; font-size: 14px;
}
.edge-row a { text-decoration: none; color: var(--edge, var(--accent)); font-weight: 600; }
.edge-row a:hover { text-decoration: underline; }
.edge-row .arrow { color: var(--muted); margin: 0 5px; }
.edge-row .evidence { display: block; color: var(--muted); font-size: 13px; margin-top: 2px; }

/* tiles */
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
.tile {
  background: var(--card); border: 1px solid var(--rule); border-radius: var(--radius);
  padding: 10px 6px; text-align: center;
}
.tile-n { display: block; font-size: 20px; font-weight: 700; color: var(--accent); }
.tile-l { display: block; font-size: 11.5px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.cat-tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; margin-top: 8px; }
.cat-tile {
  font: inherit; cursor: pointer; text-align: left; display: block;
  background: var(--card); border: 1px solid var(--rule); border-radius: var(--radius);
  padding: 9px 11px;
}
.cat-tile:hover { border-color: var(--accent); }
.cat-tile[aria-pressed="true"] { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.cat-tile .tile-n { text-align: left; font-size: 17px; }
.bars { list-style: none; margin: 12px 0 0; padding: 0; }
.bars li { display: flex; align-items: center; gap: 8px; font-size: 13px; margin-bottom: 5px; }
.bar-label { flex: 0 0 34%; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar { flex: 1; height: 8px; background: #e9eaec; border-radius: 999px; overflow: hidden; }
.bar-fill { display: block; height: 100%; background: var(--accent); border-radius: 999px; }
.bar-n { flex: 0 0 30px; text-align: right; color: var(--muted); }

/* stories */
.stories-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
.story {
  background: var(--card); border: 1px solid var(--rule); border-radius: var(--radius);
  padding: 16px 18px; margin-bottom: 12px; scroll-margin-top: 92px;
}
.story.major { border-left: 4px solid var(--accent); }
.story[hidden] { display: none; }
.story:target { box-shadow: 0 0 0 2px var(--accent); }
.story-head { margin: 0 0 6px; font-size: 17px; line-height: 1.4; display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.story-head .num { color: var(--faint); font-size: 13px; font-weight: 700; min-width: 22px; }
.story-title { font-family: var(--serif); }
.tag {
  font-family: var(--sans); font-size: 10.5px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .05em; padding: 2px 7px; border-radius: 999px;
  background: #eef4f1; color: var(--accent);
}
.tag-major { background: #ffe9d6; color: #a35c00; }
.story-meta { color: var(--muted); font-size: 12.5px; margin: 0 0 10px; }
.alt-headline { color: var(--muted); font-size: 13.5px; font-style: italic; margin: 0 0 10px; }
.block { margin-bottom: 10px; }
.block .label, .watch .label, .subtext .label {
  display: inline-block; font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .05em; color: var(--accent); margin-bottom: 4px;
}
.subtext { background: #f0f7f4; border-left: 4px solid var(--accent); border-radius: 0 8px 8px 0; padding: 12px 14px; margin: 10px 0; }
.subtext .caveat { display: inline-block; font-size: 11.5px; color: #5b7a6d; margin: 0 0 4px 8px; }
.confidence { font-size: 12px; margin: 0 0 8px; }
.watch { margin: 8px 0 0; }
.watch p[dir="rtl"] { text-align: right; }
.edges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.edge {
  font-size: 11.5px; text-decoration: none; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--edge, var(--accent)); color: var(--edge, var(--accent));
  white-space: nowrap;
}
.refs { margin-top: 10px; font-size: 12.5px; color: var(--muted); }
.refs summary { cursor: pointer; }
.refs ul { margin: 8px 0 0; padding-left: 18px; }
.refs li { margin-bottom: 5px; }
.refs a { text-decoration: none; }
.refs a:hover { text-decoration: underline; }
.ref-meta { color: var(--faint); font-size: 11px; }

footer.site {
  margin-top: 34px; padding-top: 14px; border-top: 2px solid var(--rule);
  color: var(--faint); font-size: 12px; line-height: 1.5;
}
footer.site a { color: var(--accent); }
.to-top {
  position: fixed; right: 16px; bottom: 16px; z-index: 20; display: none;
  background: var(--card); color: var(--accent); border: 1px solid var(--accent);
  border-radius: 999px; padding: 9px 14px; font: inherit; font-size: 13px; cursor: pointer;
}
.to-top.on { display: block; }
.empty-note { background: #fff4e5; border-left: 4px solid #f0a04b; padding: 12px 14px; border-radius: 0 8px 8px 0; color: #7a4a06; }

@media (max-width: 560px) {
  body { font-size: 15px; }
  .tiles { grid-template-columns: repeat(2, 1fr); }
  .story { padding: 14px; }
  .bar-label { flex-basis: 30%; }
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #e8e8e6; --muted: #a5a8ac; --faint: #83878c;
    --bg: #16181a; --card: #1e2124; --rule: #2c3034; --accent: #3fae87;
    --scrim: rgba(22, 24, 26, .92);
  }
  .badge { background: #16261f; border-color: #24463a; color: #6fcaa0; }
  .badge.warn { background: #2e2415; border-color: #4a3a1d; color: #e0a860; }
  .subtext { background: #16261f; }
  .subtext .caveat { color: #7fb69f; }
  .tag { background: #16261f; }
  .tag-major { background: #2e2415; color: #e0a860; }
  mark { background: #4a3f14; color: #f5e6a8; }
  .callout, .story, .tile, .edge-row, .themes li { background: var(--card); }
  .bar { background: #2c3034; }
  .empty-note { background: #2e2415; color: #e8c79a; }
}
@media print {
  .controls, .to-top { display: none !important; }
  .story { break-inside: avoid; border: 1px solid #ccc; }
  body { background: #fff; }
}
"""

# The sheet carries literal ``%`` (widths, media queries), so it is tokenised
# rather than %-formatted — a brace-collision CSS formatter would break it too.
_CSS = _CSS.replace("__INK__", INK).replace("__MUTED__", MUTED).replace(
    "__FAINT__", FAINT
).replace("__ACCENT__", ACCENT).replace("__RULE__", RULE)


_JS = """
(function () {
  var stories = [].slice.call(document.querySelectorAll('.story'));
  if (!stories.length) return;
  var search = document.getElementById('q');
  var clear = document.getElementById('clear');
  var status = document.getElementById('status');
  var toTop = document.getElementById('to-top');
  var category = '';
  var query = '';

  function catButtons() {
    return [].slice.call(document.querySelectorAll('.cat-tile'));
  }

  function apply() {
    var shown = 0;
    stories.forEach(function (el) {
      var show = (!category || el.getAttribute('data-category') === category)
        && (!query || el.getAttribute('data-search').indexOf(query) > -1);
      el.hidden = !show;
      if (show) shown++;
    });
    if (status) {
      status.textContent = (shown === stories.length)
        ? ''
        : shown + ' of ' + stories.length + ' stories match'
          + (category ? ' · ' + category.replace(/-/g, ' ') : '')
          + (query ? ' · "' + query + '"' : '');
    }
    if (clear) clear.hidden = !query && !category;
    catButtons().forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.getAttribute('data-filter') === category));
    });
  }

  if (search) search.addEventListener('input', function () {
    query = search.value.trim().toLowerCase();
    apply();
  });
  if (clear) clear.addEventListener('click', function () {
    if (search) search.value = '';
    query = ''; category = '';
    apply();
  });
  catButtons().forEach(function (b) {
    b.addEventListener('click', function () {
      var next = b.getAttribute('data-filter');
      category = (category === next) ? '' : next;
      apply();
      if (category) {
        document.getElementById('glance').scrollIntoView({ block: 'start' });
      }
    });
  });

  // Expand every story's reference list at once — reading the brief as a
  // document is the common case; scrolling past <details> markers is not.
  var expand = document.getElementById('expand-refs');
  if (expand) expand.addEventListener('click', function () {
    var open = expand.getAttribute('aria-pressed') !== 'true';
    document.querySelectorAll('.refs').forEach(function (d) { d.open = open; });
    expand.setAttribute('aria-pressed', String(open));
    expand.textContent = open ? 'Collapse references' : 'Expand references';
  });

  window.addEventListener('scroll', function () {
    if (toTop) toTop.classList.toggle('on', window.scrollY > 900);
  }, { passive: true });
  if (toTop) toTop.addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === '/' && document.activeElement !== search) {
      e.preventDefault();
      if (search) search.focus();
    } else if (e.key === 'Escape' && document.activeElement === search) {
      search.value = ''; query = '';
      apply();
    }
  });
})();
"""


def render_web(digest: Digest) -> str:
    """The edition as a standalone page: masthead, editorial, graph, stories."""
    stories = digest.stories
    analyses = digest.analyses
    synth = digest.synthesis or {}

    majors = digest.stats.get("major_stories", 0)
    analysed_count = min(
        digest.stats.get("analysed_stories", len(stories)), len(stories)
    )
    # Declared up front because the glance section counts what is on the page,
    # not what the clustering produced.
    shown_stories = stories[:analysed_count]
    parts: list[str] = []

    title = f"Kuwait Morning Brief — {_date_heading(digest.date).split(' · ')[0]}"
    parts.append(
        "<!DOCTYPE html>\n"
        f'<html lang="en" dir="ltr">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>{_h(title)}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        # The measure wrapper: without it every block sits flush against the
        # viewport edge on a phone, because the masthead and controls carry
        # their own padding but the body content does not.
        '<div class="wrap">'
    )

    # -- masthead ---------------------------------------------------------
    badges = [f'<span class="badge">last {digest.window_hours}h window</span>']
    if digest.llm_provider and digest.llm_provider != "none":
        badges.append(f'<span class="badge">analysed by {_h(digest.llm_provider)}</span>')
    else:
        badges.append(
            '<span class="badge warn">extractive mode — no AI provider</span>'
        )
    if digest.errors:
        badges.append(
            f'<span class="badge warn">{len(digest.errors)} call(s) fell back</span>'
        )

    parts.append(
        '<header class="masthead"><div class="masthead-in">'
        "<h1>🇰🇼 Kuwait Morning Brief</h1>"
        f'<div class="date">{_h(_date_heading(digest.date))}</div>'
        f'<div class="stats">'
        f"{digest.stats.get('total_articles', 0)} articles · "
        f"{digest.stats.get('outlets_reporting', 0)} outlets · "
        f"{len(stories)} stories · {majors} major · generated "
        f"{digest.generated_at.strftime('%H:%M')} Kuwait</div>"
        f'<div class="badges">{"".join(badges)}</div>'
        "</div></header>"
    )

    # -- controls ---------------------------------------------------------
    parts.append(
        '<div class="controls"><div class="controls-in">'
        '<div class="search-row">'
        '<label class="visually-hidden" for="q" hidden>Search stories</label>'
        '<input class="search" id="q" type="search" autocomplete="off" '
        'placeholder="Search headlines, outlets, summaries  (press /)">'
        '<button type="button" class="chip" id="clear" hidden>Clear</button>'
        '</div>'
        '<div class="status" id="status" role="status" aria-live="polite"></div>'
        "</div></div>"
    )

    # -- editorial --------------------------------------------------------
    note = _bilingual(synth.get("editor_note_en", ""), synth.get("editor_note_ar", ""))
    if note:
        parts.append(_section("Editor's note", f'<div class="callout">{_para_html(note)}</div>'))

    not_said = _bilingual(
        synth.get("not_being_said_en", ""), synth.get("not_being_said_ar", "")
    )
    if not_said:
        parts.append(
            _section(
                "What today's coverage is not saying",
                f'<div class="callout plain">{_para_html(not_said)}</div>',
            )
        )

    parts.append(_glance_html(digest, shown_stories))
    parts.append(_themes_html(synth.get("top_themes") or []))

    links = _renderable_links(synth, len(stories))
    story_titles = {
        f"S{i + 1}": story.primary.title for i, story in enumerate(stories)
    }
    parts.append(_links_section(links, story_titles))

    tactical = _bilingual(synth.get("tactical_en", ""), synth.get("tactical_ar", ""))
    strategic = _bilingual(synth.get("strategic_en", ""), synth.get("strategic_ar", ""))
    if tactical or strategic:
        blocks = ""
        if tactical:
            blocks += (
                '<div class="callout"><p class="label">This week</p>'
                f"{_para_html(tactical)}</div>"
            )
        if strategic:
            blocks += (
                '<div class="callout plain" style="margin-top:10px;">'
                '<p class="label">The longer arc</p>'
                f"{_para_html(strategic)}</div>"
            )
        parts.append(_section("Tactical and strategic", blocks))

    # -- stories ----------------------------------------------------------
    parts.append('<hr style="border:none;border-top:2px solid var(--rule);margin:30px 0 16px;">')
    parts.append(
        '<div class="stories-head"><h2 class="section-title" style="margin:0;">The stories</h2>'
        '<button type="button" class="chip" id="expand-refs" aria-pressed="false">'
        "Expand references</button></div>"
    )
    parts.append(
        f'<p class="section-intro">The {analysed_count} most widely carried stories in the '
        "window, each with the outlets that carried it and a link to every original.</p>"
    )

    if not shown_stories:
        parts.append('<div class="empty-note">📭 No stories were carried in this window.</div>')

    links_by_story: dict[str, list[dict[str, str]]] = {}
    for link in links:
        for sid in (link["from"], link["to"]):
            links_by_story.setdefault(sid, []).append(link)

    for idx, story in enumerate(shown_stories):
        analysis = analyses[idx] if idx < len(analyses) else {}
        parts.append(
            _card_html(
                idx, story, analysis, links_by_story.get(f"S{idx + 1}", [])
            )
        )

    parts.append(
        '<footer class="site">Compiled automatically by NewsScanner from public RSS feeds. '
        "Headlines and links belong to the original publishers. Anything framed as "
        "inference is machine-generated and should be verified before being acted on."
        "</footer></div>"
        '<button type="button" class="to-top" id="to-top">↑ Top</button>'
        f"<script>{_JS}</script></body></html>"
    )
    return "".join(parts)
