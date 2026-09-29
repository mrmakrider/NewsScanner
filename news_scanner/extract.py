"""Pull the readable body text out of a news article page.

Deliberately dependency-free and conservative: we would rather return a
short accurate excerpt than a page of navigation chrome, because this text is
what the analysis step reasons over.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser

from .http import FetchError, decode, fetch
from .util import html_to_text, normalize_text, truncate

log = logging.getLogger(__name__)

MAX_BYTES = 700_000
MIN_PARAGRAPH = 55

# Boilerplate we never want in the body.
_NOISE_PATTERNS = [
    r"اقرأ (أيضا|أيضاً)", r"تابعونا", r"جميع الحقوق محفوظة", r"حقوق النشر",
    r"read (also|more)\s*:", r"related (articles|stories)", r"all rights reserved",
    r"follow us", r"subscribe", r"advertisement", r"share this",
    r"المزيد من", r"مواضيع ذات صلة", r"الأكثر قراءة", r"most read",
    r"comments?$", r"التعليقات", r"newsletter",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)

_META_RE = {
    "description": re.compile(
        r'<meta[^>]+(?:name|property)=["\'](?:og:)?description["\'][^>]+content=["\']([^"\']{40,})["\']',
        re.IGNORECASE,
    ),
    "og_title": re.compile(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
}

_SKIP_TAGS = {"script", "style", "noscript", "iframe", "svg", "form", "nav",
              "footer", "header", "aside", "figure", "button", "select"}


class _ParagraphParser(HTMLParser):
    """Collect block text plus its link density, so we can drop nav blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, float]] = []
        self._skip = 0
        self._capture = False
        self._text: list[str] = []
        self._link_chars = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in ("p", "h2", "h3", "li", "blockquote") and not self._skip:
            self._capture = True
            self._text = []
            self._link_chars = 0
        elif tag == "a" and self._capture:
            self._link_chars += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag in ("p", "h2", "h3", "li", "blockquote") and self._capture:
            text = re.sub(r"\s+", " ", "".join(self._text)).strip()
            if text:
                # Approximate link density: anchors seen per character.
                density = self._link_chars / max(len(text), 1) * 40
                self.blocks.append((text, density))
            self._capture = False
            self._text = []

    def handle_data(self, data):
        if self._capture and not self._skip:
            self._text.append(data)


def extract_body(url: str, *, timeout: int = 20) -> str:
    """Fetch a URL and return clean body text ('' if it cannot be read)."""
    try:
        raw, _ = fetch(url, timeout=timeout, retries=1)
    except FetchError as exc:
        log.debug("body fetch failed %s: %s", url, exc)
        return ""
    if len(raw) > MAX_BYTES:
        raw = raw[:MAX_BYTES]

    markup = decode(raw)

    # Prefer an explicit <article> element when the page provides one.
    article_match = re.search(
        r"<article[^>]*>(.*?)</article>", markup, re.IGNORECASE | re.DOTALL
    )
    scope = article_match.group(1) if article_match else markup

    parser = _ParagraphParser()
    try:
        parser.feed(scope)
        parser.close()
    except Exception:
        pass

    kept: list[str] = []
    seen: set[str] = set()
    for text, density in parser.blocks:
        if len(text) < MIN_PARAGRAPH and len(kept) < 3:
            continue
        if len(text) < 25:
            continue
        if density > 1.2:            # mostly a link farm
            continue
        if _NOISE_RE.search(text):
            continue
        norm = normalize_text(text)[:120]
        if norm in seen:
            continue
        seen.add(norm)
        kept.append(text)

    body = "\n\n".join(kept)

    if len(body) < 200:
        meta = _META_RE["description"].search(markup)
        if meta:
            body = html_to_text(meta.group(1))
    return truncate(body, 12_000, suffix="")
