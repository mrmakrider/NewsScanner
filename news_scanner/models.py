"""Data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .util import containment, jaccard, title_fingerprint, tokenize


@dataclass
class Article:
    """One item as published by one outlet."""

    source_id: str
    source_name: str
    lang: str            # "ar" | "en"
    title: str
    url: str
    section: str = ""
    summary: str = ""
    body: str = ""       # filled in only for stories we analyse deeply
    published: datetime | None = None
    fetched_at: datetime | None = None
    tier: int = 2

    @property
    def text(self) -> str:
        return self.body or self.summary or ""

    @property
    def best_text(self) -> str:
        parts = [self.title, self.body or self.summary]
        return "\n".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "lang": self.lang,
            "title": self.title,
            "url": self.url,
            "section": self.section,
            "summary": self.summary,
            "body": self.body,
            "published": self.published.isoformat() if self.published else None,
            "tier": self.tier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Article":
        published = data.get("published")
        return cls(
            source_id=data.get("source_id", ""),
            source_name=data.get("source_name", ""),
            lang=data.get("lang", "en"),
            title=data.get("title", ""),
            url=data.get("url", ""),
            section=data.get("section", ""),
            summary=data.get("summary", ""),
            body=data.get("body", ""),
            published=datetime.fromisoformat(published) if published else None,
            tier=data.get("tier", 2),
        )


@dataclass
class Story:
    """A cluster of articles from different outlets about the same event."""

    articles: list[Article] = field(default_factory=list)
    key: str = ""
    score: float = 0.0

    def add(self, article: Article) -> None:
        # Guard against duplicates: cluster merging can legitimately offer the
        # same article twice, and a duplicate would double-count an outlet.
        if any(a.url == article.url for a in self.articles):
            return
        self.articles.append(article)
        if not self.key:
            self.key = title_fingerprint(article.title)

    # -- aggregates -------------------------------------------------------
    @property
    def primary(self) -> Article:
        """Best single article to represent the cluster (tier, then detail)."""
        return sorted(
            self.articles,
            key=lambda a: (a.tier, -(len(a.body or a.summary or "")), a.title),
        )[0]

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for a in self.articles:
            if a.source_name not in seen:
                seen.append(a.source_name)
        return seen

    @property
    def outlet_count(self) -> int:
        return len(self.sources)

    @property
    def langs(self) -> set[str]:
        return {a.lang for a in self.articles}

    @property
    def latest(self) -> datetime | None:
        dates = [a.published for a in self.articles if a.published]
        return max(dates) if dates else None

    @property
    def earliest(self) -> datetime | None:
        dates = [a.published for a in self.articles if a.published]
        return min(dates) if dates else None

    @property
    def sections(self) -> list[str]:
        seen: list[str] = []
        for a in self.articles:
            if a.section and a.section not in seen:
                seen.append(a.section)
        return seen

    def similarity(self, other: "Story") -> float:
        """How likely two clusters describe the same underlying event."""
        best = 0.0
        for a in self.articles:
            ta = tokenize(a.title)
            if not ta:
                continue
            for b in other.articles:
                tb = tokenize(b.title)
                if not tb:
                    continue
                best = max(best, jaccard(ta, tb), containment(ta, tb) * 0.9)
        return best

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "score": round(self.score, 3),
            "sources": self.sources,
            "outlet_count": self.outlet_count,
            "articles": [a.to_dict() for a in self.articles],
        }


@dataclass
class Digest:
    """Everything needed to render one morning report."""

    date: str                       # YYYY-MM-DD in Kuwait local time
    generated_at: datetime
    window_hours: int
    stories: list[Story] = field(default_factory=list)
    analyses: list[dict[str, Any]] = field(default_factory=list)
    synthesis: dict[str, Any] = field(default_factory=dict)
    source_health: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    llm_provider: str = "none"
    errors: list[str] = field(default_factory=list)
