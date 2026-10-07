"""Render a Digest into Markdown, HTML email and machine-readable JSON."""

from __future__ import annotations

import html as html_mod
import re
from collections import Counter, defaultdict
from typing import Any

from .dedupe import CATEGORY_ORDER, categorize
from .models import Article, Digest, Story
from .util import humanize_age, normalize_text, truncate

WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
]

_AR_WEEKDAYS = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
_AR_MONTHS = [
    "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس",
    "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _md_escape(text: str) -> str:
    """Keep titles from breaking Markdown link syntax."""
    text = (text or "").replace("[", "(").replace("]", ")")
    return re.sub(r"\s+", " ", text).strip()


def _renderable_links(synth: dict[str, Any], story_count: int) -> list[dict[str, str]]:
    """Links safe to show, re-checked at the point of display.

    The engine validates links before they reach a Digest, but a renderer also
    reads digests written days ago and by older versions of the pipeline. A
    link pointing at a story that is not on the page renders as a dead
    reference, so the same check is applied again here — cheaply, and without
    trusting that the data came through the current code path.
    """
    raw = synth.get("links")
    if not isinstance(raw, list):
        return []
    valid = {f"S{i + 1}" for i in range(story_count)}
    out: list[dict[str, str]] = []
    for link in raw:
        if not isinstance(link, dict):
            continue
        kind = str(link.get("type", "")).strip()
        src = str(link.get("from", "")).strip()
        dst = str(link.get("to", "")).strip()
        evidence = str(link.get("evidence", "")).strip()
        if not (kind and src and dst and evidence) or src not in valid or dst not in valid:
            continue
        out.append({"type": kind, "from": src, "to": dst, "evidence": evidence})
    return out


def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06ff]", text or ""))


def _bilingual(en: str, ar: str, *, prefer_ar: bool = False) -> str:
    """Join an English and an Arabic field into one readable block."""
    en, ar = (en or "").strip(), (ar or "").strip()
    if en and ar:
        return f"{ar}\n\n{en}" if prefer_ar else f"{en}\n\n{ar}"
    return en or ar


def _date_heading(date_str: str) -> str:
    from datetime import date

    try:
        y, m, d = (int(p) for p in date_str.split("-"))
        dt = date(y, m, d)
    except (ValueError, AttributeError):
        return date_str
    return (
        f"{WEEKDAYS[dt.weekday()]}, {dt.day} {MONTHS[dt.month - 1]} {dt.year} · "
        f"{_AR_WEEKDAYS[dt.weekday()]} {dt.day} {_AR_MONTHS[dt.month - 1]} {dt.year}"
    )


def _refs_lines(story: Story) -> list[str]:
    """Every outlet that carried a story, with a link and a timestamp."""
    lines = []
    for art in sorted(
        story.articles, key=lambda a: (a.tier, a.source_name, a.title)
    ):
        when = art.published.strftime("%H:%M") if art.published else "--:--"
        lines.append(
            f"- **{_md_escape(art.source_name)}** ({art.lang}) — "
            f"[{_md_escape(art.title)}]({art.url}) · {when}"
        )
    return lines


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def render_markdown(digest: Digest) -> str:
    stories = digest.stories
    analyses = digest.analyses
    synth = digest.synthesis or {}
    out: list[str] = []

    total_articles = digest.stats.get("total_articles", 0)
    outlet_count = digest.stats.get("outlets_reporting", 0)
    majors = digest.stats.get("major_stories", 0)
    analysed_count = min(
        digest.stats.get("analysed_stories", len(stories)), len(stories)
    )
    listed_limit = digest.stats.get("listed_stories", 60)

    # -- masthead ---------------------------------------------------------
    out.append("# 🇰🇼 Kuwait Morning Brief")
    out.append(f"### {_date_heading(digest.date)}")
    out.append("")
    out.append(
        f"> Generated **{digest.generated_at.strftime('%H:%M')} Kuwait time** · "
        f"window: last {digest.window_hours}h · "
        f"**{total_articles} articles** from **{outlet_count} outlets** "
        f"→ **{len(stories)} stories** ({majors} major)"
    )
    out.append("")

    if digest.llm_provider == "none":
        out.append(
            "> ⚠️ **Evidence-only mode.** No LLM analysis was available for this run; "
            "the digest reports source text and observable coverage signals without "
            "inferring subtext."
        )
        out.append("")

    # -- editor's note ----------------------------------------------------
    note = _bilingual(
        synth.get("editor_note_en", ""), synth.get("editor_note_ar", "")
    )
    if note:
        out.append("## ✍️ Editor's note")
        out.append("")
        out.append(note)
        out.append("")

    not_said = _bilingual(
        synth.get("not_being_said_en", ""), synth.get("not_being_said_ar", "")
    )
    if not_said:
        out.append("## 🔍 What today's coverage is *not* saying")
        out.append("")
        out.append(not_said)
        out.append("")

    # -- themes -----------------------------------------------------------
    themes = synth.get("top_themes") or []
    if themes:
        out.append("## 🧭 Themes of the day")
        out.append("")
        for theme in themes:
            if not isinstance(theme, dict):
                continue
            label_en = str(theme.get("theme_en", "")).strip()
            label_ar = str(theme.get("theme_ar", "")).strip()
            label = f"{label_en} — {label_ar}" if (label_en and label_ar) else (label_en or label_ar)
            ids = ", ".join(str(i) for i in (theme.get("story_ids") or []))
            note_txt = _bilingual(
                str(theme.get("note_en", "")), str(theme.get("note_ar", ""))
            )
            out.append(f"- **{_md_escape(label)}**" + (f" ({ids})" if ids else ""))
            if note_txt:
                out.append(f"  {note_txt.splitlines()[0] if note_txt else ''}")
        out.append("")

    watch = _bilingual(synth.get("watchlist_en", ""), synth.get("watchlist_ar", ""))
    if watch:
        out.append("## 👀 Watch next")
        out.append("")
        out.append(watch)
        out.append("")

    # -- connections ------------------------------------------------------
    # The point of the whole link layer: today's stories shown against each
    # other rather than as a list. Rendered immediately after the editorial
    # material because it is what explains why the editor chose what they did.
    links = _renderable_links(synth, len(stories))
    if links:
        out.append("## 🕸️ How today's stories connect")
        out.append("")
        by_kind: dict[str, list[dict[str, str]]] = {}
        for link in links:
            by_kind.setdefault(link["type"], []).append(link)
        out.append(
            f"*{len(links)} connections across {len(by_kind)} relationship "
            "types.*"
        )
        out.append("")
        for kind in sorted(by_kind, key=lambda k: (-len(by_kind[k]), k)):
            label, _color = _link_label(kind)
            out.append(f"**{_md_escape(label)}** ({len(by_kind[kind])})")
            out.append("")
            for edge in by_kind[kind]:
                out.append(
                    f"- **{_md_escape(edge['from'])} → {edge['to']}** — "
                    f"{_md_escape(edge['evidence'])}"
                )
            out.append("")

    # -- tactical / strategic ---------------------------------------------
    tactical = _bilingual(synth.get("tactical_en", ""), synth.get("tactical_ar", ""))
    strategic = _bilingual(synth.get("strategic_en", ""), synth.get("strategic_ar", ""))
    if tactical or strategic:
        out.append("## 🎯 Tactical and strategic")
        out.append("")
        if tactical:
            out.append("**This week**")
            out.append("")
            out.append(tactical)
            out.append("")
        if strategic:
            out.append("**The longer arc**")
            out.append("")
            out.append(strategic)
            out.append("")

    # -- at a glance ------------------------------------------------------
    by_category = _group_by_category(stories)
    out.append("## 📊 Today at a glance")
    out.append("")
    out.append(f"- **{len(stories)}** distinct stories, **{majors}** carried by 3+ outlets")
    busiest = Counter()
    for story in stories:
        for art in story.articles:
            busiest[art.source_name] += 1
    if busiest:
        top = ", ".join(f"{name} ({n})" for name, n in busiest.most_common(6))
        out.append(f"- **Busiest outlets:** {top}")
    for category in CATEGORY_ORDER:
        group = by_category.get(category)
        if group:
            out.append(f"- **{category}:** {len(group)}")
    out.append("")

    # -- the stories ------------------------------------------------------
    out.append("---")
    out.append("")
    out.append("## 📰 The stories")
    out.append("")
    out.append(
        f"The {analysed_count} most widely carried stories this window, each with "
        "its sources. Full analysis where a model was available."
    )
    out.append("")

    for idx, story in enumerate(stories[:analysed_count]):
        analysis = analyses[idx] if idx < len(analyses) else {}
        category = categorize(story)
        primary = story.primary

        badge = "🔥 " if story.outlet_count >= 3 else ""
        out.append(f"### {idx + 1}. {badge}{_md_escape(primary.title)}")
        out.append("")
        out.append(
            f"`{category}` · **{story.outlet_count} outlet"
            f"{'s' if story.outlet_count != 1 else ''}** · "
            f"{', '.join(_md_escape(s) for s in story.sources)} · "
            f"{humanize_age(story.latest)}"
        )
        out.append("")
        if analysis.get("_source") == "hybrid":
            out.append(
                "*Hybrid recovery: usable model text was retained; missing fields "
                "were completed from observable evidence.*"
            )
            out.append("")

        headline_ar = str(analysis.get("headline_ar", "")).strip()
        headline_en = str(analysis.get("headline_en", "")).strip()
        if primary.lang == "ar" and headline_en and normalize_text(headline_en) != normalize_text(primary.title):
            out.append(f"*EN: {_md_escape(headline_en)}*")
            out.append("")
        elif primary.lang == "en" and headline_ar:
            out.append(f"*AR: {_md_escape(headline_ar)}*")
            out.append("")

        summary = _bilingual(
            analysis.get("summary_en", ""), analysis.get("summary_ar", "")
        )
        if summary:
            out.append(f"**Summary.** {summary}")
            out.append("")

        why = _bilingual(
            analysis.get("why_it_matters_en", ""), analysis.get("why_it_matters_ar", "")
        )
        if why:
            out.append(f"**Why it matters.** {why}")
            out.append("")

        btl = _bilingual(
            analysis.get("between_the_lines_en", ""),
            analysis.get("between_the_lines_ar", ""),
        )
        if btl:
            # Mark the epistemic status at the point of use: a machine reading
            # is not the same thing as a reported fact.
            llm_fields = set(analysis.get("_llm_fields") or [])
            from_llm = (
                str(analysis.get("_source", "")) == "llm"
                or bool(llm_fields & {"between_the_lines_en", "between_the_lines_ar"})
            )
            label = (
                "**🔎 Between the lines** *(inference)*"
                if from_llm
                else "**🔎 Between the lines** *(no inference; observable evidence only)*"
            )
            out.append(label)
            out.append("")
            for paragraph in btl.split("\n\n"):
                paragraph = paragraph.strip()
                if paragraph:
                    out.append(f"> {paragraph}" if _has_arabic(paragraph) else paragraph)
                    out.append("")

        coverage = _bilingual(
            analysis.get("coverage_evidence_en", ""),
            analysis.get("coverage_evidence_ar", ""),
        )
        if coverage:
            out.append(f"**Coverage signals.** {coverage}")
            out.append("")

        confidence = str(analysis.get("confidence", "")).strip()
        reason = str(analysis.get("confidence_reason", "")).strip()
        if confidence:
            icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence.lower(), "⚪")
            out.append(f"{icon} *Confidence: {confidence}*" + (f" — {reason}" if reason else ""))
            out.append("")

        nxt = _bilingual(
            analysis.get("watch_next_en", ""), analysis.get("watch_next_ar", "")
        )
        if nxt:
            out.append(f"**Watch next.** {nxt}")
            out.append("")

        entities = analysis.get("entities") or []
        if entities:
            out.append("*Entities: " + ", ".join(_md_escape(str(e)) for e in entities[:12]) + "*")
            out.append("")

        out.append("<details><summary><b>References (" + str(len(story.articles)) + ")</b></summary>")
        out.append("")
        out.extend(_refs_lines(story))
        out.append("")
        out.append("</details>")
        out.append("")

    # -- compact list for everything else --------------------------------
    tail = stories[analysed_count : analysed_count + listed_limit]
    if tail:
        out.append("---")
        out.append("")
        out.append(f"## 🗞️ Also today ({len(tail)} more stories)")
        out.append("")
        out.append("Carried by fewer outlets, ranked by reach. Links go to the primary source.")
        out.append("")
        for idx, story in enumerate(tail, start=analysed_count + 1):
            primary = story.primary
            others = [s for s in story.sources if s != primary.source_name]
            more = f" *(+{len(others)}: {', '.join(_md_escape(o) for o in others[:3])})*" if others else ""
            out.append(
                f"{idx}. {_md_escape(primary.title)} — "
                f"[{_md_escape(primary.source_name)}]({primary.url}) "
                f"`{categorize(story)}`{more}"
            )
        out.append("")

    # -- complete index ---------------------------------------------------
    out.append("---")
    out.append("")
    out.append("## 📚 Appendix — complete coverage index")
    out.append("")
    out.append(
        "Every article collected in this window, grouped by outlet. "
        "This is the raw record behind the briefing above."
    )
    out.append("")
    for source_name, articles in _group_by_source(digest):
        label = "article" if len(articles) == 1 else "articles"
        out.append(f"### {_md_escape(source_name)} — {len(articles)} {label}")
        out.append("")
        for art in articles:
            when = art.published.strftime("%H:%M") if art.published else "--:--"
            section = f" `{art.section}`" if art.section else ""
            out.append(f"- {when}{section} [{_md_escape(art.title)}]({art.url})")
        out.append("")

    # -- footer -----------------------------------------------------------
    out.append("---")
    out.append("")
    out.append("### Source health")
    out.append("")
    out.append("| Outlet | Tier | Articles | Feeds OK | Status |")
    out.append("|---|---|---|---|---|")
    for entry in digest.source_health:
        status = "✅" if entry.get("ok") else ("⚪ optional" if entry.get("optional") else "❌ failing")
        out.append(
            f"| {_md_escape(entry['name'])} | {entry['tier']} | {entry['articles']} "
            f"| {entry['targets']} | {status} |"
        )
    out.append("")

    if digest.errors:
        out.append("<details><summary>Run warnings</summary>")
        out.append("")
        for err in digest.errors[:40]:
            out.append(f"- `{_md_escape(err)}`")
        out.append("")
        out.append("</details>")
        out.append("")

    out.append(
        "*Compiled automatically by NewsScanner from public RSS feeds and pages. "
        "Headlines and links belong to the original publishers. Analysis marked as "
        "inference is machine-generated and should be verified before being acted on.*"
    )
    return "\n".join(out)


def _group_by_category(stories: list[Story]) -> dict[str, list[Story]]:
    grouped: dict[str, list[Story]] = defaultdict(list)
    for story in stories:
        grouped[categorize(story)].append(story)
    return grouped


def _group_by_source(digest: Digest) -> list[tuple[str, list[Article]]]:
    grouped: dict[str, list[Article]] = defaultdict(list)
    for story in digest.stories:
        for art in story.articles:
            grouped[art.source_name].append(art)
    return sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))


# --------------------------------------------------------------------------
# HTML email
# --------------------------------------------------------------------------

_CSS_WRAP = (
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Tahoma,Arial,sans-serif;"
    "font-size:15px;line-height:1.65;color:#1c1e21;"
)


# Relationship types, as the reader sees them. Each gets a colour so a graph
# can be scanned rather than read: a page of same_actor edges in one colour
# reads differently from a page of contradictions.
_LINK_STYLES = {
    "consequence": ("caused", "#0b6b4f"),
    "reaction": ("in response to", "#8a5a00"),
    "contradiction": ("outlets disagree", "#b00020"),
    "same_actor": ("same actor", "#3b4fa0"),
    "same_source_angle": ("same source", "#6b4fa0"),
    "escalation": ("escalating", "#a03b00"),
}


def _link_label(kind: str) -> tuple[str, str]:
    """(human label, colour) for a relationship type, with a safe default."""
    return _LINK_STYLES.get(kind, (kind.replace("_", " "), "#65676b"))


def _h(text: str) -> str:
    return html_mod.escape(text or "", quote=True)


def _para(text: str) -> str:
    """Escape and turn blank lines into paragraphs, RTL-aware."""
    if not text:
        return ""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    html_parts = []
    for block in blocks:
        direction = "rtl" if _has_arabic(block) else "ltr"
        align = "right" if direction == "rtl" else "left"
        html_parts.append(
            f'<p dir="{direction}" style="margin:0 0 10px;text-align:{align};">'
            f"{_h(block).replace(chr(10), '<br>')}</p>"
        )
    return "".join(html_parts)


def render_html(digest: Digest) -> str:
    stories = digest.stories
    analyses = digest.analyses
    synth = digest.synthesis or {}

    majors = digest.stats.get("major_stories", 0)
    analysed_count = min(
        digest.stats.get("analysed_stories", len(stories)), len(stories)
    )
    parts: list[str] = []

    parts.append(
        '<div style="max-width:720px;margin:0 auto;padding:22px 18px;'
        f'{_CSS_WRAP}">'
    )
    parts.append(
        '<div style="border-bottom:3px solid #0b6b4f;padding-bottom:12px;margin-bottom:20px;">'
        '<div style="font-size:24px;font-weight:700;color:#0b6b4f;">🇰🇼 Kuwait Morning Brief</div>'
        f'<div style="color:#65676b;font-size:13px;margin-top:4px;">{_h(_date_heading(digest.date))}</div>'
        f'<div style="color:#65676b;font-size:12px;margin-top:6px;">'
        f"{digest.stats.get('total_articles', 0)} articles · "
        f"{digest.stats.get('outlets_reporting', 0)} outlets · "
        f"{len(stories)} stories · {majors} major · generated "
        f"{digest.generated_at.strftime('%H:%M')} Kuwait</div></div>"
    )

    if digest.llm_provider == "none":
        parts.append(
            '<div style="background:#fff4e5;border-left:4px solid #f0a04b;padding:10px 12px;'
            'margin-bottom:18px;font-size:13px;">⚠️ <b>Evidence-only mode</b> — no LLM '
            "analysis was available for this run; observable coverage is reported without inferred subtext.</div>"
        )

    note = _bilingual(synth.get("editor_note_en", ""), synth.get("editor_note_ar", ""))
    if note:
        parts.append('<h2 style="font-size:17px;color:#0b6b4f;margin:24px 0 8px;">✍️ Editor\'s note</h2>')
        parts.append(_para(note))

    not_said = _bilingual(
        synth.get("not_being_said_en", ""), synth.get("not_being_said_ar", "")
    )
    if not_said:
        parts.append(
            '<h2 style="font-size:17px;color:#0b6b4f;margin:24px 0 8px;">'
            "🔍 What today's coverage is <i>not</i> saying</h2>"
        )
        parts.append(
            '<div style="background:#f7f7f8;border-radius:8px;padding:12px 14px;">'
            + _para(not_said)
            + "</div>"
        )

    themes = synth.get("top_themes") or []
    if themes:
        parts.append('<h2 style="font-size:17px;color:#0b6b4f;margin:24px 0 8px;">🧭 Themes</h2><ul style="margin:0;padding-left:20px;">')
        for theme in themes:
            if not isinstance(theme, dict):
                continue
            label = str(theme.get("theme_en", "")).strip()
            label_ar = str(theme.get("theme_ar", "")).strip()
            ids = ", ".join(str(i) for i in (theme.get("story_ids") or []))
            parts.append(
                f'<li style="margin-bottom:6px;"><b>{_h(label)}</b>'
                + (f' <span dir="rtl" style="color:#65676b;">{_h(label_ar)}</span>' if label_ar else "")
                + (f' <span style="color:#65676b;font-size:12px;">({_h(ids)})</span>' if ids else "")
                + "</li>"
            )
        parts.append("</ul>")

    links = _renderable_links(synth, len(stories))
    if links:
        # Group by relationship type so the shape of the day is legible at a
        # glance: several contradictions in one edition is itself a finding.
        by_kind: dict[str, list[dict[str, str]]] = {}
        for link in links:
            by_kind.setdefault(link["type"], []).append(link)

        parts.append(
            '<h2 style="font-size:17px;color:#0b6b4f;margin:24px 0 4px;">'
            "🕸️ How today's stories connect</h2>"
        )
        total = len(links)
        kinds = len(by_kind)
        parts.append(
            '<div style="font-size:12px;color:#65676b;margin-bottom:10px;">'
            f"{total} connection{'' if total == 1 else 's'} across {kinds} "
            f"relationship{'s' if kinds != 1 else ''}. Follow a number to jump "
            "to that story.</div>"
        )

        for kind in sorted(by_kind, key=lambda k: (-len(by_kind[k]), k)):
            label, color = _link_label(kind)
            edges = by_kind[kind]
            parts.append(
                f'<div style="margin:12px 0 4px;font-size:13px;font-weight:700;'
                f'color:{color};">{_h(label)} '
                f'<span style="color:#65676b;font-weight:400;">({len(edges)})</span></div>'
                '<ul style="margin:0;padding-left:20px;">'
            )
            for edge in edges:
                parts.append(
                    '<li style="margin-bottom:6px;">'
                    f'<a href="#story-{_h(edge["from"])}" style="color:{color};'
                    f'text-decoration:none;font-weight:700;">{_h(edge["from"])}</a>'
                    '<span style="color:#65676b;"> → </span>'
                    f'<a href="#story-{_h(edge["to"])}" style="color:{color};'
                    f'text-decoration:none;font-weight:700;">{_h(edge["to"])}</a>'
                    f' — {_h(edge["evidence"])}</li>'
                )
            parts.append("</ul>")

    tactical = _bilingual(synth.get("tactical_en", ""), synth.get("tactical_ar", ""))
    strategic = _bilingual(synth.get("strategic_en", ""), synth.get("strategic_ar", ""))
    if tactical or strategic:
        parts.append(
            '<h2 style="font-size:17px;color:#0b6b4f;margin:24px 0 8px;">'
            "🎯 Tactical and strategic</h2>"
        )
        if tactical:
            parts.append(
                '<div style="font-size:12px;font-weight:700;color:#65676b;'
                f'margin:8px 0 4px;">THIS WEEK</div><div style="background:#f7f7f8;'
                f'border-radius:8px;padding:12px 14px;">{_para(tactical)}</div>'
            )
        if strategic:
            parts.append(
                '<div style="font-size:12px;font-weight:700;color:#65676b;'
                f'margin:12px 0 4px;">THE LONGER ARC</div>'
                f'<div style="background:#eef4ff;border-radius:8px;padding:12px 14px;">'
                f"{_para(strategic)}</div>"
            )

    parts.append('<hr style="border:none;border-top:2px solid #e4e6eb;margin:26px 0 18px;">')
    parts.append('<h2 style="font-size:19px;color:#0b6b4f;margin:0 0 4px;">📰 The stories</h2>')

    for idx, story in enumerate(stories[:analysed_count]):
        analysis = analyses[idx] if idx < len(analyses) else {}
        primary = story.primary
        confidence = str(analysis.get("confidence", "")).lower()
        badge_color = {"high": "#2e7d32", "medium": "#b26a00", "low": "#b00020"}.get(confidence, "#65676b")

        parts.append(
            f'<div id="story-S{idx + 1}" style="margin:22px 0 0;padding-top:16px;'
            'border-top:1px solid #e4e6eb;">'
        )
        parts.append(
            f'<div style="font-size:16px;font-weight:700;margin-bottom:6px;" dir="auto">'
            f"{idx + 1}. {_h(primary.title)}</div>"
        )
        parts.append(
            '<div style="font-size:12px;color:#65676b;margin-bottom:10px;">'
            f"{_h(categorize(story))} · carried by {story.outlet_count} outlet"
            f"{'s' if story.outlet_count != 1 else ''}: {_h(', '.join(story.sources))}</div>"
        )
        if analysis.get("_source") == "hybrid":
            parts.append(
                '<div style="font-size:12px;color:#6b4e16;margin:6px 0 10px;">'
                "Hybrid recovery: usable model text was retained; missing fields "
                "were completed from observable evidence.</div>"
            )

        headline_alt = (
            analysis.get("headline_en", "")
            if primary.lang == "ar"
            else analysis.get("headline_ar", "")
        )
        if headline_alt and str(headline_alt).strip():
            alt = str(headline_alt).strip()
            if normalize_text(alt) != normalize_text(primary.title):
                direction = "rtl" if _has_arabic(alt) else "ltr"
                parts.append(
                    f'<div dir="{direction}" style="font-size:13px;color:#65676b;'
                    f'font-style:italic;margin-bottom:8px;">{_h(alt)}</div>'
                )

        for label, base, color in (
            ("Summary", "summary", "#1c1e21"),
            ("Why it matters", "why_it_matters", "#1c1e21"),
        ):
            value = _bilingual(analysis.get(f"{base}_en", ""), analysis.get(f"{base}_ar", ""))
            if value:
                parts.append(
                    f'<div style="margin-bottom:8px;color:{color};"><b>{label}.</b> '
                    + _para(value).replace("<p ", "<p style=\"margin:0 0 6px;\" ")
                    + "</div>"
                )

        btl = _bilingual(
            analysis.get("between_the_lines_en", ""), analysis.get("between_the_lines_ar", "")
        )
        if btl:
            # Same epistemic labelling as the Markdown version.
            llm_fields = set(analysis.get("_llm_fields") or [])
            from_llm = (
                str(analysis.get("_source", "")) == "llm"
                or bool(llm_fields & {"between_the_lines_en", "between_the_lines_ar"})
            )
            heading = "🔎 BETWEEN THE LINES" if from_llm else "🔎 OBSERVABLE EVIDENCE — NO INFERENCE"
            if from_llm:
                heading += (
                    '<span style="font-weight:400;color:#5b7a6d;">'
                    " &mdash; inference, not reported fact</span>"
                )
            parts.append(
                '<div style="background:#f0f7f4;border-left:4px solid #0b6b4f;'
                'padding:10px 12px;border-radius:0 6px 6px 0;margin:10px 0;">'
                '<div style="font-weight:700;font-size:13px;color:#0b6b4f;margin-bottom:6px;">'
                + heading
                + "</div>"
                + _para(btl)
                + "</div>"
            )

        coverage = _bilingual(
            analysis.get("coverage_evidence_en", ""),
            analysis.get("coverage_evidence_ar", ""),
        )
        if coverage:
            parts.append(
                '<div style="font-size:12px;color:#5b6573;margin:8px 0;">'
                f"<b>Coverage signals.</b> {_para(coverage)}</div>"
            )

        if confidence:
            reason = str(analysis.get("confidence_reason", "")).strip()
            parts.append(
                f'<div style="font-size:12px;color:{badge_color};margin-bottom:6px;">'
                f"● Confidence: {_h(confidence)}" + (f" — {_h(reason)}" if reason else "") + "</div>"
            )

        nxt = _bilingual(analysis.get("watch_next_en", ""), analysis.get("watch_next_ar", ""))
        if nxt:
            parts.append(
                '<div style="font-size:13px;margin-bottom:6px;">'
                f"<b>Watch next.</b> {_h(nxt)}</div>"
            )

        refs = "".join(
            f'<li style="margin-bottom:4px;">'
            f'<a href="{_h(a.url)}" style="color:#0b6b4f;text-decoration:none;">{_h(a.title)}</a>'
            f' <span style="color:#8a8d91;font-size:11px;">— {_h(a.source_name)}'
            f'{", " + a.published.strftime("%H:%M") if a.published else ""}</span></li>'
            for a in sorted(story.articles, key=lambda a: a.tier)
        )
        parts.append(
            '<details style="font-size:12px;color:#65676b;margin-top:6px;">'
            f"<summary style=\"cursor:pointer;\">References ({len(story.articles)})</summary>"
            f'<ul style="margin:6px 0 0;padding-left:18px;">{refs}</ul></details>'
        )
        parts.append("</div>")

    parts.append(
        '<div style="margin-top:28px;padding-top:14px;border-top:2px solid #e4e6eb;'
        'font-size:11px;color:#8a8d91;line-height:1.5;">'
        "Compiled automatically by NewsScanner from public RSS feeds. Headlines and links "
        "belong to the original publishers. Anything framed as inference is machine-generated "
        "and should be verified before being acted on."
        "</div></div>"
    )
    return "<!DOCTYPE html><html><head><meta charset=\"utf-8\">" \
           f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">" \
           f"</head><body style=\"margin:0;padding:0;background:#ffffff;\">{''.join(parts)}</body></html>"


def render_json(digest: Digest) -> dict[str, Any]:
    return {
        "date": digest.date,
        "generated_at": digest.generated_at.isoformat(),
        "window_hours": digest.window_hours,
        "provider": digest.llm_provider,
        "stats": digest.stats,
        "synthesis": digest.synthesis,
        "source_health": digest.source_health,
        "errors": digest.errors,
        "stories": [
            {
                **story.to_dict(),
                "category": categorize(story),
                "analysis": (
                    digest.analyses[idx] if idx < len(digest.analyses) else {}
                ),
            }
            for idx, story in enumerate(digest.stories)
        ],
    }
