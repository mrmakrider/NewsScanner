"""NewsScanner command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from .analyze import AnalysisConfig, AnalysisEngine, detect_provider
from .dedupe import build_stories, categorize, major_count
from .deliver import email_configured, recipients, send_email, write_outputs
from .extract import extract_body
from .models import Article, Digest
from .render import render_html, render_markdown, render_json
from .sources import collect, load_config
from .util import fmt_local, now_kuwait, to_kuwait, truncate

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "digests"
DEFAULT_CACHE = REPO_ROOT / "state" / "analysis_cache.json"
DEFAULT_CONFIG = REPO_ROOT / "config" / "sources.toml"

log = logging.getLogger("newsscanner")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def within_window(articles: list[Article], hours: int) -> list[Article]:
    """Keep articles published inside the window; keep undated ones.

    Undated items come from pages that are current by construction (a
    section front page), so dropping them would silently lose coverage.
    """
    cutoff = now_kuwait() - timedelta(hours=hours)
    kept: list[Article] = []
    for art in articles:
        if art.published is None:
            kept.append(art)
            continue
        if to_kuwait(art.published) >= cutoff:
            kept.append(art)
    return kept


def enrich_bodies(stories, *, limit: int, workers: int = 8) -> None:
    """Fetch full text for the stories we are about to analyse.

    Headlines alone produce shallow analysis; a few paragraphs of real body
    text is what makes the "between the lines" pass worth reading.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    targets: list[Article] = []
    for story in stories[:limit]:
        primary = story.primary
        if not (primary.body or "").strip() and len(story.articles) >= 1:
            targets.append(primary)

    if not targets:
        return

    log.info("fetching full text for %d stories…", len(targets))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_body, art.url): art for art in targets}
        for future in as_completed(futures):
            art = futures[future]
            try:
                body = future.result()
            except Exception:
                body = ""
            if body:
                art.body = body
                # Also harvest bodies from the other outlets for better framing
                # comparison on the strongest stories.
            else:
                log.debug("no body extracted for %s", art.url)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_check_sources(args: argparse.Namespace) -> int:
    sources, _ = load_config(args.config)
    if args.sources:
        wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
        sources = [s for s in sources if s.id in wanted]

    print(f"Checking {len(sources)} sources…\n")
    articles, health = collect(sources, max_workers=args.workers)

    width = max((len(h.name) for h in health), default=20)
    ok_count = 0
    for entry in health:
        if entry.ok:
            ok_count += 1
            flag = "OK "
        elif entry.optional:
            flag = "OPT"
        else:
            flag = "FAIL"
        line = (
            f"[{flag}] {entry.name.ljust(width)}  tier{entry.tier}  "
            f"feeds {entry.targets_ok}/{entry.targets_total}  "
            f"articles {entry.article_count}"
        )
        print(line)
        if entry.error:
            print(f"        └─ {truncate(entry.error, 160)}")

    print(f"\n{ok_count}/{len(health)} sources healthy · {len(articles)} articles fetched")
    return 0 if ok_count else 1


def cmd_check_llm(args: argparse.Namespace) -> int:
    """Ask the configured provider to analyse one real story end to end.

    This is the dry run that matters: it proves the credential, the model name
    and the network path all work together before 08:00 does.
    """
    from .analyze import LLMClient

    stories = build_stories(_probe_articles())
    story = stories[0]

    print("Resolving provider…")
    config = AnalysisConfig.from_env(
        language=args.language, cache_path=None, use_cache=False
    )
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.llm_off:
        config.provider = "none"

    engine = AnalysisEngine(config)
    detected = "explicitly disabled" if engine.provider == "none" else engine.provider
    print(f"  provider : {detected}")
    print(f"  model    : {engine.model or '(none)'}")
    if engine.provider == "llm7":
        from .analyze import OPENAI_COMPATIBLE_BASES

        base = os.getenv("NEWSCANNER_BASE_URL", "").strip() or OPENAI_COMPATIBLE_BASES["llm7"]
        print(f"  endpoint : {base}/chat/completions")
        print(f"  auth     : {'token supplied' if engine.api_key else 'anonymous (free tier)'}")

    if engine.provider == "none":
        print("\nNo provider available — the daily run would use the extractive fallback.")
        print("Set GEMINI_API_KEY (or another provider key), or unset NEWSCANNER_NO_LLM7.")
        return 1

    print(f"\nSending one real story to {engine.provider}…")
    if engine.client is None:
        print("  client was not constructed — cannot test")
        return 1
    try:
        payload = engine._analyse_batch([story], 0)  # noqa: SLF001 — a diagnostic
    except Exception as exc:  # noqa: BLE001 — the point is to report the failure
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("\nThe scheduled run would still complete, using the extractive fallback.")
        return 1

    items = payload.get("stories") or []
    print(f"  OK — the model returned {len(items)} analysed story(ies)")
    if items:
        first = items[0]
        for field in ("headline_en", "summary_en", "why_it_matters_en", "between_the_lines_en"):
            value = str(first.get(field, "")).strip()
            if value:
                print(f"  {field}: {truncate(value, 150)}")

    print("\nAnalysis is configured correctly.")
    return 0


def _probe_articles() -> list[Article]:
    """A small, real headline pair used to exercise the analysis path."""
    now = now_kuwait()
    rows = [
        (
            "alrai", "الرأي", "ar",
            "مجلس الوزراء يوافق على مشروع مرسوم بإلغاء مرسوم إنشاء جهاز تطوير مدينة الحرير وجزيرة بوبيان",
            "وافق مجلس الوزراء على مشروع مرسوم بإلغاء مرسوم إنشاء جهاز تطوير مدينة الحرير "
            "وجزيرة بوبيان، على أن تتولى جهات أخرى استكمال الملف.",
        ),
        (
            "kuwaittimes", "Kuwait Times", "en",
            "Cabinet clears abolition of Silk City development body",
            "The Cabinet approved a draft decree abolishing the authority set up to develop "
            "Silk City and Bubiyan Island.",
        ),
    ]
    return [
        Article(
            source_id=sid, source_name=name, lang=lang, title=title, summary=summary,
            url=f"https://example.invalid/{sid}", published=now - timedelta(hours=i + 1),
            tier=1,
        )
        for i, (sid, name, lang, title, summary) in enumerate(rows)
    ]


def _run_marker_path() -> Path:
    return DEFAULT_CACHE.parent / "last_run.json"


def _read_run_marker() -> dict:
    path = _run_marker_path()
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_run_marker(date_str: str, *, emailed: bool, stories: int) -> None:
    """Record that today's brief went out.

    The retry schedule reads this so a second attempt at 08:30 cannot send a
    duplicate digest when the 08:00 run already succeeded.
    """
    path = _run_marker_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "date": date_str,
                    "delivered_at": now_kuwait().isoformat(),
                    "emailed": emailed,
                    "stories": stories,
                    "attempts": int(_read_run_marker().get("attempts", 0)) + 1,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            "utf-8",
        )
    except OSError as exc:
        log.warning("could not write run marker: %s", exc)


def cmd_check_email(args: argparse.Namespace) -> int:
    """Verify SMTP credentials and show exactly where the digest will go."""
    from .deliver import DEFAULT_MAIL_TO, email_enabled, recipients, verify_email

    to_list = recipients()
    print("Delivery configuration")
    print(f"  recipients : {', '.join(to_list)}")
    if to_list == [DEFAULT_MAIL_TO]:
        print(f"               (built-in default — set MAIL_TO to change it)")
    print(f"  from       : {os.getenv('MAIL_FROM', '').strip() or os.getenv('SMTP_USER', '').strip() or '(SMTP_USER)'}")
    print(f"  smtp host  : {os.getenv('SMTP_HOST', '').strip() or '(not set)'}")
    print(f"  smtp port  : {os.getenv('SMTP_PORT', '').strip() or '587'}")
    print(f"  security   : {os.getenv('SMTP_SECURITY', '').strip() or 'starttls'}")
    print(f"  enabled    : {'no (NEWSCANNER_NO_EMAIL=1)' if not email_enabled() else 'yes'}")

    smtp_user = os.getenv("SMTP_USER", "").strip()
    if not smtp_user or not os.getenv("SMTP_PASSWORD", "").strip():
        print("\nSMTP_USER / SMTP_PASSWORD are not both set.")
        print("For Gmail you must use an App Password, not your account password:")
        print("  https://myaccount.google.com/apppasswords")
        print("Then set SMTP_HOST=smtp.gmail.com, SMTP_PORT=587, SMTP_SECURITY=starttls,")
        print(f"SMTP_USER=<you@gmail.com>, SMTP_PASSWORD=<app password>.")

    print("\nContacting the mail server…")
    ok, detail = verify_email()
    print(f"  {'OK' if ok else 'FAILED'} — {detail}")
    if not ok:
        print("\nThe digest would still be built and committed, but not emailed.")
        return 1

    if args.send:
        from .deliver import send_email

        subject = "🇰🇼 Kuwait Morning Brief — test message"
        body = (
            "<p>This is a test message from NewsScanner.</p>"
            f"<p>If you are reading it, the daily 08:00 Kuwait brief will arrive "
            f"here: <b>{', '.join(to_list)}</b>.</p>"
        )
        print("\nSending a test message…")
        if send_email(subject, body, "NewsScanner test message — delivery is working."):
            print(f"  OK — sent to {', '.join(to_list)}")
            return 0
        print("  FAILED — see the log above")
        return 1

    print("\nDelivery is configured. Re-run with --send to send a test message.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    started = now_kuwait()

    # -- 0. has today's brief already gone out? ---------------------------
    today = args.date or started.strftime("%Y-%m-%d")
    if args.skip_if_done:
        marker = _read_run_marker()
        if marker.get("date") == today and marker.get("delivered_at"):
            stamp = marker.get("delivered_at")
            try:
                label = fmt_local(datetime.fromisoformat(stamp))
            except (TypeError, ValueError):
                label = stamp
            log.info("today's brief already went out at %s — nothing to do", label)
            return 0

    if args.mail_to:
        os.environ["MAIL_TO"] = args.mail_to

    sources, settings = load_config(args.config)

    if args.sources:
        wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
        sources = [s for s in sources if s.id in wanted]
        log.info("restricted to %d source(s)", len(sources))

    window_hours = args.hours or settings.window_hours
    max_analyze = args.max_analyze or settings.max_analyze
    major_threshold = args.major_outlets or settings.major_outlet_threshold

    # -- 1. collect -------------------------------------------------------
    log.info("collecting from %d sources (window %dh)…", len(sources), window_hours)
    articles, health = collect(sources, max_workers=args.workers)
    articles = within_window(articles, window_hours)
    log.info("collected %d articles in window", len(articles))

    if not articles:
        log.error("no articles collected — aborting (check --check-sources)")
        return 2

    # -- 2. cluster into stories -----------------------------------------
    stories = build_stories(articles, major_threshold=major_threshold)
    majors = major_count(stories, major_threshold)
    log.info("→ %d stories (%d major, carried by %d+ outlets)", len(stories), majors, major_threshold)

    # -- 3. enrich + analyse the top stories ------------------------------
    top = stories[:max_analyze]
    if not args.no_fetch_bodies:
        enrich_bodies(top, limit=len(top), workers=args.workers)

    config = AnalysisConfig.from_env(
        language=None if args.language is None else args.language,
        cache_path=DEFAULT_CACHE,
        use_cache=not args.no_cache,
    )
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.llm_off:
        config.provider = "none"

    engine = AnalysisEngine(config)
    if engine.provider == "none":
        log.warning(
            "no LLM provider configured — running in extractive mode "
            "(set GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)"
        )
    else:
        log.info("analysing %d stories with %s/%s…", len(top), engine.provider, engine.model)

    result = engine.analyse(top)
    engine.save_cache()

    analyses = [result.stories.get(story.key, {}) for story in top]
    # Stories beyond the analysis budget still appear in the appendix.
    tail = stories[max_analyze:]
    if tail:
        log.info("%d lower-ranked stories listed without analysis", len(tail))

    # -- 4. assemble ------------------------------------------------------
    digest_date = today
    digest = Digest(
        date=digest_date,
        generated_at=started,
        window_hours=window_hours,
        stories=top + tail,
        analyses=analyses + [{} for _ in tail],
        synthesis=result.synthesis,
        source_health=[h.to_dict() for h in health],
        llm_provider=result.provider,
        errors=result.errors,
        stats={
            "total_articles": len(articles),
            "outlets_reporting": sum(1 for h in health if h.article_count > 0),
            "total_stories": len(stories),
            "major_stories": majors,
            "analysed_stories": len(top),
            "window_start": (started - timedelta(hours=window_hours)).isoformat(),
            "window_end": started.isoformat(),
            "major_threshold": major_threshold,
            "categories": {},
        },
    )
    from collections import Counter

    digest.stats["categories"] = dict(
        Counter(categorize(s) for s in digest.stories)
    )

    markdown = render_markdown(digest)
    html = render_html(digest)
    payload = render_json(digest)

    # -- 5. output --------------------------------------------------------
    out_dir = Path(args.output) if args.output else DEFAULT_OUTPUT
    paths = write_outputs(
        out_dir,
        date_str=digest_date,
        markdown=markdown,
        html=html,
        payload=payload,
        keep_html=not args.no_html,
    )
    for name, path in paths.items():
        log.info("wrote %s → %s", name, path)

    # -- 6. email ---------------------------------------------------------
    delivered = False
    if args.no_email or args.dry_run:
        log.info("email skipped (dry run / --no-email)")
    elif not email_configured():
        log.warning(
            "email not configured — the brief was written and committed but not sent. "
            "Set SMTP_HOST, SMTP_USER and SMTP_PASSWORD, then run "
            "`python -m news_scanner check-email --send` to verify. "
            "The recipient defaults to %s.",
            ", ".join(recipients()),
        )
    else:
        date_label = datetime.strptime(digest_date, "%Y-%m-%d").strftime("%a %d %b %Y")
        subject = f"🇰🇼 Kuwait Morning Brief — {date_label} — {majors} major stories"
        delivered = send_email(subject, html, _plain_text_fallback(digest))
        if not delivered:
            log.error("the brief was built and committed, but email delivery failed")

    # Record the completed delivery so the retry schedule can tell a finished
    # run from one that died part-way through. A dry run, a --no-email run and
    # a failed send must all leave the marker alone, or a later attempt would
    # decide there was nothing left to do and the day's brief would never send.
    if not (args.dry_run or args.no_email or args.no_marker):
        if delivered or not email_configured():
            _write_run_marker(digest_date, emailed=delivered, stories=len(stories))

    elapsed = (now_kuwait() - started).total_seconds()
    log.info(
        "done in %.1fs · %d articles → %d stories · %d major · %d analysed (%s)",
        elapsed, len(articles), len(stories), majors, len(top), result.provider,
    )
    if args.dry_run or args.no_email or delivered or not email_configured():
        return 0
    # Email was configured and did not go out: surface it so the workflow
    # fails visibly and the retry schedule runs again.
    return 1


def _plain_text_fallback(digest: Digest) -> str:
    """Short plain-text alternative for mail clients that block HTML."""
    lines = [
        f"Kuwait Morning Brief — {digest.date}",
        f"{digest.stats.get('total_articles', 0)} articles from "
        f"{digest.stats.get('outlets_reporting', 0)} outlets → "
        f"{digest.stats.get('total_stories', 0)} stories",
        "",
    ]
    note = digest.synthesis.get("editor_note_en") or digest.synthesis.get("editor_note_ar")
    if note:
        lines += ["EDITOR'S NOTE", str(note), ""]
    for idx, story in enumerate(digest.stories[:15]):
        analysis = digest.analyses[idx] if idx < len(digest.analyses) else {}
        lines.append(f"{idx + 1}. {story.primary.title}")
        summary = analysis.get("summary_en") or analysis.get("summary_ar")
        if summary:
            lines.append(f"   {truncate(str(summary), 300)}")
        btl = analysis.get("between_the_lines_en") or analysis.get("between_the_lines_ar")
        if btl:
            lines.append(f"   [between the lines] {truncate(str(btl), 300)}")
        lines.append(f"   sources: {', '.join(story.sources)}")
        lines.append(f"   {story.primary.url}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newsscanner",
        description="Aggregate Kuwait's news into one analysed morning brief.",
    )
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "check-sources", "check-llm", "check-email"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="path to sources.toml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output directory (default: digests/)")
    parser.add_argument("--date", help="digest date label, YYYY-MM-DD (default: today in Kuwait)")
    parser.add_argument("--hours", type=int, help="look-back window in hours")
    parser.add_argument("--max-analyze", type=int, dest="max_analyze",
                        help="how many top stories get full analysis")
    parser.add_argument("--major-outlets", type=int, dest="major_outlets",
                        help="outlets needed for a story to count as major")
    parser.add_argument("--sources", help="comma-separated source ids to use (testing)")
    parser.add_argument("--workers", type=int, default=12, help="parallel fetch workers")
    parser.add_argument("--provider", help="gemini|openai|anthropic|openrouter|groq|ollama|auto")
    parser.add_argument("--model", help="override the model name")
    parser.add_argument("--language", choices=["bilingual", "en", "ar"],
                        help="analysis language (default: bilingual)")
    parser.add_argument("--no-llm", action="store_true", dest="llm_off",
                        help="skip AI analysis, use the extractive fallback")
    parser.add_argument("--no-email", action="store_true", help="do not send email")
    parser.add_argument("--no-cache", action="store_true", help="ignore the analysis cache")
    parser.add_argument("--no-html", action="store_true", help="do not write the .html file")
    parser.add_argument("--no-fetch-bodies", action="store_true",
                        help="do not fetch full article text (faster, shallower analysis)")
    parser.add_argument("--dry-run", action="store_true", help="run everything except email")
    parser.add_argument("--to", dest="mail_to", help="override the recipient address(es)")
    parser.add_argument("--skip-if-done", action="store_true", dest="skip_if_done",
                        help="exit without doing anything if today's brief already went out")
    parser.add_argument("--no-marker", action="store_true", dest="no_marker",
                        help="do not write state/last_run.json (diagnostics, demos)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    if not args.date:
        args.date = now_kuwait().strftime("%Y-%m-%d")
    if args.language is None:
        import os

        args.language = os.getenv("NEWSCANNER_LANGUAGE", "bilingual").lower()

    try:
        if args.command == "check-sources":
            return cmd_check_sources(args)
        if args.command == "check-llm":
            return cmd_check_llm(args)
        if args.command == "check-email":
            return cmd_check_email(args)
        return cmd_run(args)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 — top-level guard for unattended runs
        log.exception("fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
