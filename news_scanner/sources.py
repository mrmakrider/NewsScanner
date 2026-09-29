"""Load the source registry and pull articles from every outlet."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

from .http import FetchError
from .models import Article
from .parse import load_feed, scrape_html_list
from .util import now_kuwait

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "sources.toml"


@dataclass
class FeedTarget:
    url: str
    section: str = ""


@dataclass
class Source:
    id: str
    name_en: str
    name_ar: str
    lang: str
    type: str
    tier: int
    homepage: str = ""
    feeds: list[FeedTarget] = field(default_factory=list)
    pages: list[FeedTarget] = field(default_factory=list)
    include_pattern: str | None = None
    exclude_pattern: str | None = None
    optional: bool = False
    enabled: bool = True

    @property
    def name(self) -> str:
        return self.name_en if self.lang == "en" else (self.name_ar or self.name_en)


@dataclass
class Settings:
    timezone: str = "Asia/Kuwait"
    window_hours: int = 26
    max_analyze: int = 28
    major_outlet_threshold: int = 3


@dataclass
class SourceHealth:
    source_id: str
    name: str
    tier: int
    ok: bool
    article_count: int = 0
    targets_total: int = 0
    targets_ok: int = 0
    error: str = ""
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "tier": self.tier,
            "ok": self.ok,
            "articles": self.article_count,
            "targets": f"{self.targets_ok}/{self.targets_total}",
            "optional": self.optional,
            "error": self.error,
        }


def load_config(path: Path | str | None = None) -> tuple[list[Source], Settings]:
    """Read sources.toml into Source objects plus global settings."""
    if tomllib is None:  # pragma: no cover
        raise RuntimeError(
            "Python 3.11+ is required (tomllib). "
            "On older versions run: pip install tomli"
        )
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"source config not found: {cfg_path}")

    with cfg_path.open("rb") as fh:
        data = tomllib.load(fh)

    settings = Settings(**(data.get("settings") or {}))

    sources: list[Source] = []
    for raw in data.get("source", []):
        feeds = [FeedTarget(**f) for f in raw.get("feeds", [])]
        pages = [FeedTarget(**p) for p in raw.get("pages", [])]
        sources.append(
            Source(
                id=raw["id"],
                name_en=raw.get("name_en", raw["id"]),
                name_ar=raw.get("name_ar", ""),
                lang=raw.get("lang", "ar"),
                type=raw.get("type", "rss"),
                tier=int(raw.get("tier", 2)),
                homepage=raw.get("homepage", ""),
                feeds=feeds,
                pages=pages,
                include_pattern=raw.get("include_pattern"),
                exclude_pattern=raw.get("exclude_pattern"),
                optional=bool(raw.get("optional", False)),
                enabled=bool(raw.get("enabled", True)),
            )
        )
    return sources, settings


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def _collect_from_target(source: Source, target: FeedTarget) -> list[Article]:
    if source.type == "rss":
        return load_feed(
            target.url,
            source_id=source.id,
            source_name=source.name,
            lang=source.lang,
            section=target.section,
            tier=source.tier,
        )
    return scrape_html_list(
        target.url,
        source_id=source.id,
        source_name=source.name,
        lang=source.lang,
        section=target.section,
        tier=source.tier,
        include_pattern=source.include_pattern,
        exclude_pattern=source.exclude_pattern,
    )


def collect(
    sources: list[Source],
    *,
    max_workers: int = 12,
    per_source_limit: int = 120,
) -> tuple[list[Article], list[SourceHealth]]:
    """Fetch every enabled source concurrently.

    A failing outlet never aborts the run — it is recorded in the health
    report and the pipeline continues with whatever succeeded.
    """
    jobs: list[tuple[Source, FeedTarget]] = []
    for source in sources:
        if not source.enabled:
            continue
        targets = source.feeds if source.type == "rss" else source.pages
        for target in targets:
            jobs.append((source, target))

    results: dict[str, list[Article]] = {s.id: [] for s in sources}
    health: dict[str, SourceHealth] = {}
    for source in sources:
        targets = source.feeds if source.type == "rss" else source.pages
        health[source.id] = SourceHealth(
            source_id=source.id,
            name=source.name,
            tier=source.tier,
            ok=False,
            targets_total=len(targets),
            optional=source.optional,
        )

    errors: dict[str, list[str]] = {}

    def run(job: tuple[Source, FeedTarget]) -> tuple[Source, FeedTarget, list[Article], str]:
        source, target = job
        try:
            return source, target, _collect_from_target(source, target), ""
        except FetchError as exc:
            return source, target, [], str(exc)
        except Exception as exc:  # defensive: never let one feed kill the run
            return source, target, [], f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            source, target, articles, error = future.result()
            entry = health[source.id]
            if error:
                errors.setdefault(source.id, []).append(error)
                log.warning("[%s] %s -> %s", source.id, target.url, error)
            else:
                entry.targets_ok += 1
                results[source.id].extend(articles)

    all_articles: list[Article] = []
    for source in sources:
        items = results.get(source.id, [])
        # De-duplicate within a source: the same story often appears in both
        # the "latest" feed and its section feed.
        seen: set[str] = set()
        unique: list[Article] = []
        for art in items:
            key = art.url.split("?")[0]
            if key in seen:
                continue
            seen.add(key)
            unique.append(art)

        unique.sort(key=lambda a: a.published or now_kuwait(), reverse=True)
        unique = unique[:per_source_limit]

        entry = health[source.id]
        entry.article_count = len(unique)
        entry.ok = entry.targets_ok > 0
        all_articles.extend(unique)

    ordered_health = [health[s.id] for s in sources if s.enabled]
    return all_articles, ordered_health
