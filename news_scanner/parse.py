"""Feed (RSS / Atom / RDF) parsing and generic HTML list scraping.

Uses xml.etree with a DOCTYPE guard; no lxml or feedparser required.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse

from .http import FetchError, decode, fetch, guess_encoding_from_meta
from .models import Article
from .util import extract_links, html_to_text, normalize_text, strip_cdata, truncate

log = logging.getLogger(__name__)

_DOCTYPE_RE = re.compile(br"<!DOCTYPE[^>[]*(\[[^]]*\])?[^>]*>", re.IGNORECASE | re.DOTALL)
_ENTITY_RE = re.compile(br"<!ENTITY[^>]*>", re.IGNORECASE)
_UNKNOWN_ENTITY_RE = re.compile(br"&([A-Za-z_][A-Za-z0-9_.-]*);")

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "rss": "http://purl.org/rss/1.0/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "media": "http://search.yahoo.com/mrss/",
}

_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S %Z",
    "%A, %B %d, %Y",
)


def parse_date(value: str | None) -> datetime | None:
    """Best-effort date parsing across the formats these outlets emit."""
    if not value:
        return None
    value = strip_cdata(value).strip()
    if not value:
        return None

    try:
        dt = parsedate_to_datetime(value)
        if dt is not None:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass

    cleaned = value.replace("Z", "+0000") if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(cleaned)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _first_text(node: ET.Element, names: tuple[str, ...]) -> str:
    """Find the first child element whose local name matches, return its text."""
    for child in node:
        if _localname(child.tag) in names:
            text = "".join(child.itertext())
            if text and text.strip():
                return strip_cdata(text)
    return ""


def _link_from_entry(node: ET.Element) -> str:
    """RSS <link>, Atom <link href>, <guid isPermaLink>, or dc:identifier."""
    for child in node:
        name = _localname(child.tag)
        if name == "link":
            href = child.attrib.get("href")
            rel = child.attrib.get("rel", "alternate")
            if href and rel in ("alternate", ""):
                return strip_cdata(href)
            if child.text and child.text.strip():
                return strip_cdata(child.text)
        elif name == "guid":
            text = strip_cdata(child.text or "")
            if text.startswith("http"):
                return text
        elif name == "identifier":
            text = strip_cdata(child.text or "")
            if text.startswith("http"):
                return text
    return ""


def parse_feed(
    xml_bytes: bytes,
    *,
    source_id: str,
    source_name: str,
    lang: str,
    section: str,
    tier: int,
    base_url: str = "",
) -> list[Article]:
    """Parse an RSS/Atom/RDF document into Articles."""
    # Defuse XXE / billion-laughs before handing anything to ElementTree.
    head = xml_bytes[:8192]
    if _DOCTYPE_RE.search(head) or _ENTITY_RE.search(head):
        xml_bytes = _DOCTYPE_RE.sub(b"", xml_bytes)
        xml_bytes = _ENTITY_RE.sub(b"", xml_bytes)

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        # A feed may reference entities we stripped with the DOCTYPE. Drop any
        # non-standard entity reference and try once more before giving up.
        kept_entities = (b"amp", b"lt", b"gt", b"quot", b"apos")
        cleaned = _UNKNOWN_ENTITY_RE.sub(
            lambda m: m.group(0) if m.group(1) in kept_entities else b"",
            xml_bytes,
        )
        root = ET.fromstring(cleaned)

    entries: list[ET.Element] = []
    for child in root.iter():
        if _localname(child.tag) in ("item", "entry"):
            entries.append(child)

    articles: list[Article] = []
    for node in entries:
        title = html_to_text(_first_text(node, ("title",)))
        url = _link_from_entry(node)
        if base_url and url and not url.startswith("http"):
            url = urljoin(base_url, url)
        if not title or not url:
            continue

        summary_raw = _first_text(
            node, ("description", "summary", "encoded", "content", "subtitle")
        )
        summary = truncate(html_to_text(summary_raw), 1200)

        published = parse_date(
            _first_text(node, ("pubdate", "published", "updated", "date", "created"))
        )

        articles.append(
            Article(
                source_id=source_id,
                source_name=source_name,
                lang=lang,
                title=truncate(title, 400),
                url=url.strip(),
                section=section,
                summary=summary,
                published=published,
                tier=tier,
            )
        )
    return articles


# --------------------------------------------------------------------------
# HTML list scraping for outlets with no usable feed (e.g. KUNA, Al-Qabas)
# --------------------------------------------------------------------------

# URL shapes that identify an article page rather than a section index.
_ARTICLE_HINTS = (
    r"/article/", r"/articles/", r"/news/", r"/story/", r"/post/", r"/\d{5,}",
    r"/\d{4}/\d{2}/", r"ArticleDetails", r"NewsDetails", r"/2026/", r"/2025/",
)
_ARTICLE_RE = re.compile("|".join(_ARTICLE_HINTS), re.IGNORECASE)


def scrape_html_list(
    url: str,
    *,
    source_id: str,
    source_name: str,
    lang: str,
    section: str,
    tier: int,
    include_pattern: str | None = None,
    exclude_pattern: str | None = None,
    min_title_len: int = 18,
    max_items: int = 60,
) -> list[Article]:
    """Pull article links out of a section/home page.

    Used only for outlets that block or do not publish feeds. Deliberately
    conservative: it prefers links whose URL looks like an article, whose
    anchor text is long enough to be a headline, and that stay on-site.
    """
    raw, _ = fetch(url, headers={"Accept": "text/html,*/*"})
    enc = guess_encoding_from_meta(raw)
    markup = decode(raw, enc)

    base_host = urlparse(url).netloc
    inc = re.compile(include_pattern, re.IGNORECASE) if include_pattern else None
    exc = re.compile(exclude_pattern, re.IGNORECASE) if exclude_pattern else None

    seen: set[str] = set()
    articles: list[Article] = []
    for abs_url, text in extract_links(markup, url):
        if len(articles) >= max_items:
            break
        parsed = urlparse(abs_url)
        if parsed.netloc and parsed.netloc != base_host:
            continue
        if abs_url.rstrip("/") == url.rstrip("/"):
            continue
        if inc and not inc.search(abs_url):
            continue
        if exc and exc.search(abs_url):
            continue
        if not _ARTICLE_RE.search(abs_url):
            continue
        title = html_to_text(text)
        if len(title) < min_title_len or len(normalize_text(title)) < 8:
            continue
        key = abs_url.split("?")[0]
        if key in seen:
            continue
        seen.add(key)
        articles.append(
            Article(
                source_id=source_id,
                source_name=source_name,
                lang=lang,
                title=truncate(title, 400),
                url=abs_url,
                section=section,
                tier=tier,
            )
        )
    return articles


def load_feed(
    url: str,
    *,
    source_id: str,
    source_name: str,
    lang: str,
    section: str,
    tier: int,
) -> list[Article]:
    raw, ctype = fetch(url)
    if b"<html" in raw[:1000].lower() and b"<item" not in raw[:20000].lower():
        raise FetchError(f"{url}: response is HTML, not a feed")
    return parse_feed(
        raw,
        source_id=source_id,
        source_name=source_name,
        lang=lang,
        section=section,
        tier=tier,
        base_url=url,
    )
