"""Pluggable LLM analysis: per-story briefing plus a daily synthesis.

Providers are selected automatically from whichever API key is present, so
the same workflow file works on Gemini, OpenAI, Anthropic, OpenRouter, Groq
or a local Ollama. If no key is configured the pipeline still runs and
degrades to an extractive summary that is clearly labelled as such.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .http import FetchError, fetch_json
from .models import Story
from .util import fmt_local, truncate

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Provider plumbing
# --------------------------------------------------------------------------

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5-20250929",
    "openrouter": "google/gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1",
}

KEY_ENV = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "groq": ("GROQ_API_KEY",),
}

PROVIDER_ORDER = ["gemini", "openai", "anthropic", "openrouter", "groq", "ollama"]


@dataclass
class AnalysisConfig:
    provider: str = "auto"
    model: str = ""
    language: str = "bilingual"       # bilingual | en | ar
    temperature: float = 0.25
    batch_size: int = 5
    max_story_chars: int = 3200
    timeout: int = 150
    cache_path: Path | None = None
    use_cache: bool = True

    @classmethod
    def from_env(cls, **overrides) -> "AnalysisConfig":
        cfg = cls(
            provider=os.getenv("NEWSCANNER_PROVIDER", "auto").strip().lower() or "auto",
            model=os.getenv("NEWSCANNER_MODEL", "").strip(),
            language=os.getenv("NEWSCANNER_LANGUAGE", "bilingual").strip().lower(),
            temperature=float(os.getenv("NEWSCANNER_TEMPERATURE", "0.25")),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg


def detect_provider(requested: str = "auto") -> tuple[str, str, str]:
    """Return ``(provider, model, api_key)`` or ``("none", "", "")``."""
    if requested == "none":
        return "none", "", ""

    if requested and requested != "auto":
        key = _key_for(requested)
        if requested == "ollama" or key:
            return requested, os.getenv("NEWSCANNER_MODEL", "") or DEFAULT_MODELS.get(requested, ""), key
        log.warning("provider %r requested but no API key found", requested)
        return "none", "", ""

    for provider in PROVIDER_ORDER:
        key = _key_for(provider)
        if provider == "ollama":
            if os.getenv("OLLAMA_HOST") or os.getenv("NEWSCANNER_USE_OLLAMA") == "1":
                return provider, DEFAULT_MODELS[provider], ""
            continue
        if key:
            return provider, DEFAULT_MODELS.get(provider, ""), key
    return "none", "", ""


def _key_for(provider: str) -> str:
    for env_name in KEY_ENV.get(provider, ()):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


class LLMClient:
    """Minimal chat client covering the OpenAI, Gemini and Anthropic shapes."""

    def __init__(self, provider: str, model: str, api_key: str, timeout: int = 150):
        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider, "")
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, system: str, user: str, *, max_tokens: int = 8000) -> str:
        if self.provider == "gemini":
            return self._gemini(system, user, max_tokens)
        if self.provider == "anthropic":
            return self._anthropic(system, user, max_tokens)
        if self.provider == "ollama":
            return self._ollama(system, user)
        return self._openai_compatible(system, user, max_tokens)

    # -- providers ---------------------------------------------------------
    def _openai_compatible(self, system: str, user: str, max_tokens: int) -> str:
        bases = {
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "groq": "https://api.groq.com/openai/v1",
        }
        base = bases.get(self.provider, "https://api.openai.com/v1")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/mrmakrider/NewsScanner"
            headers["X-Title"] = "NewsScanner"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.25,
            "response_format": {"type": "json_object"},
        }
        # Reasoning-era OpenAI models renamed the cap parameter.
        if re.match(r"^(o\d|gpt-5)", self.model):
            payload["max_completion_tokens"] = max_tokens
            payload.pop("temperature", None)
            payload.pop("response_format", None)
        else:
            payload["max_tokens"] = max_tokens

        data = fetch_json(
            f"{base}/chat/completions", payload, headers=headers, timeout=self.timeout
        )
        return data["choices"][0]["message"]["content"]

    def _gemini(self, system: str, user: str, max_tokens: int) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        data = fetch_json(url, payload, timeout=self.timeout)
        candidates = data.get("candidates") or []
        if not candidates:
            raise FetchError(f"gemini returned no candidates: {str(data)[:300]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    def _anthropic(self, system: str, user: str, max_tokens: int) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.25,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = fetch_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            headers=headers,
            timeout=self.timeout,
        )
        blocks = data.get("content") or []
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    def _ollama(self, system: str, user: str) -> str:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.25},
        }
        data = fetch_json(f"{host}/api/chat", payload, timeout=self.timeout)
        return data.get("message", {}).get("content", "")


# --------------------------------------------------------------------------
# JSON recovery
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(text: str) -> dict:
    """Best-effort JSON object extraction from an LLM response."""
    if not text:
        return {}
    text = text.strip()

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"stories": data}
    except json.JSONDecodeError:
        pass

    # Walk the string and grab the first balanced object.
    start = text.find("{")
    while start != -1:
        depth, in_str, escape = 0, False, False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : idx + 1]
                    try:
                        data = json.loads(chunk)
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return {}


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the night editor of a Kuwaiti news desk. Each morning you write \
a private briefing for one reader: an extremely well-informed person who has already seen \
the headlines and wants the part that is not written down.

Any wire service can summarise. Your value is telling this reader what a careful Kuwaiti \
newsroom notices but cannot print.

NON-NEGOTIABLE RULES
1. Work ONLY from the material supplied below. Never invent a fact, name, number, date, \
quotation, or event. If something is not in the material, it does not exist for you.
2. Always keep three things visibly separate in your writing:
   - what the reporting states outright,
   - what it implies,
   - what is your own hypothesis.
   Never dress up inference as fact.
3. If the material genuinely contains no subtext worth reporting, say so plainly \
("No significant subtext — this reads as a routine administrative announcement"). \
Do NOT manufacture depth. An honest "nothing here" is more useful than invented intrigue.
4. Kuwaiti outlets operate under real legal and political constraints, and nearly all of \
them reprint the state wire (KUNA). When you interpret, ground every claim in a concrete \
signal you can point to:
   - attribution: a named official vs "informed sources" vs an unattributed KUNA reprint,
   - emphasis: what the headline stresses against what the body quietly concedes,
   - omission: what a story of this size would normally include but does not,
   - timing and sequence: what was released late, what was bundled with other news,
   - divergence: where outlets covering the same event frame it differently.
5. Never accuse an outlet, official or person of lying or of bad faith. Describe framing, \
emphasis, sequencing and omission neutrally. Speculate about incentives, not about deceit.
6. Be concrete. No filler, no "it remains to be seen", no restating the headline.
7. Write for someone who is busy: lead with the point."""

SYNTHESIS_SYSTEM = """You are the night editor of a Kuwaiti news desk, writing the opening \
page of a private morning briefing.

You will receive the day's analysed stories. Your job is to say what the day as a whole \
means: the through-line, the themes, the contradictions between outlets, and — most \
valuable of all — what the shape of today's coverage conspicuously avoids.

NON-NEGOTIABLE RULES
1. Only reference stories present in the material, and cite them by their id.
2. Keep stated fact, implication, and hypothesis visibly separate.
3. Do not manufacture a narrative. If the day is quiet, say the day is quiet and why that \
itself is unremarkable.
4. Never accuse anyone of lying. Describe emphasis and omission neutrally.
5. Be specific and brief. No throat-clearing."""


def _field_spec(language: str) -> tuple[str, str]:
    """Return (json schema fragment, instruction sentence) for the language mode."""
    if language == "en":
        return (
            '"summary_en", "why_it_matters_en", "between_the_lines_en", "watch_next_en"',
            "Write all analysis fields in English only.",
        )
    if language == "ar":
        return (
            '"summary_ar", "why_it_matters_ar", "between_the_lines_ar", "watch_next_ar"',
            "اكتب جميع حقول التحليل باللغة العربية فقط.",
        )
    return (
        '"summary_en", "summary_ar", "why_it_matters_en", "why_it_matters_ar", '
        '"between_the_lines_en", "between_the_lines_ar", "watch_next_en", "watch_next_ar"',
        "Write every field twice: the _en field in English, the _ar field in natural "
        "journalistic Arabic (not a literal translation — write it as a Kuwaiti editor "
        "would).",
    )


def build_story_schema(language: str) -> str:
    fields, _ = _field_spec(language)
    return f"""{{
  "stories": [
    {{
      "id": "S1",
      "headline_en": "the headline in English (translate if the source is Arabic)",
      "headline_ar": "العنوان بالعربية",
      {fields},
      "confidence": "high | medium | low",
      "confidence_reason": "one short sentence in English on why confidence is what it is",
      "entities": ["named people, ministries, companies, places"],
      "tags": ["3-6 short topical tags"]
    }}
  ]
}}"""


def build_story_user_prompt(
    stories: list[Story],
    *,
    language: str,
    max_story_chars: int,
    index: int,
) -> str:
    _, lang_instruction = _field_spec(language)
    blocks = []
    for offset, story in enumerate(stories):
        sid = f"S{index + offset + 1}"
        primary = story.primary

        headline_lines = []
        for art in sorted(story.articles, key=lambda a: a.tier):
            headline_lines.append(
                f'  - [{art.source_name} | {art.lang} | tier{art.tier}'
                f'{" | " + art.section if art.section else ""}] {art.title}'
            )

        body = (primary.body or primary.summary or "").strip()
        if not body:
            # Fall back to the other outlets' summaries so the model is not blind.
            alt = [a.summary for a in story.articles if a.summary and a is not primary]
            body = truncate(" ".join(alt), max_story_chars)

        blocks.append(
            f"=== {sid} ===\n"
            f"Category: {_category_of(story)}\n"
            f"Latest report: {fmt_local(story.latest)} (Kuwait time)\n"
            f"Carried by {story.outlet_count} outlet(s): {', '.join(story.sources)}\n"
            f"Headlines as each outlet published them:\n" + "\n".join(headline_lines) + "\n"
            f"Body text from the primary source ({primary.source_name}):\n"
            f"{truncate(body, max_story_chars)}\n"
        )

    return (
        "Analyse the following Kuwait news stories for this morning's briefing.\n\n"
        "For each story produce a JSON object.\n\n"
        "MEANING OF EACH FIELD\n"
        "- summary: what actually happened, 2-3 sentences, plain and factual.\n"
        "- why_it_matters: the concrete consequence for Kuwait, its people, its money or "
        "its politics. Not 'this is significant' — say what changes.\n"
        "- between_the_lines: the reading a well-sourced Kuwaiti editor would have but "
        "would not print. Cover, where the material supports it: the attribution pattern "
        "and what it avoids saying on the record; what the headline emphasises against "
        "what the body concedes; what is missing that would normally be present; the "
        "timing and sequencing; which outlets diverge and how. If there is genuinely "
        "nothing to read between the lines, write exactly that and stop.\n"
        "- confidence: how strongly the supplied material supports your "
        "between_the_lines reading.\n"
        "- watch_next: the specific, checkable thing to watch in the coming days.\n\n"
        f"{lang_instruction}\n\n"
        "Return ONLY a JSON object matching this schema exactly:\n"
        f"{build_story_schema(language)}\n\n"
        "--- MATERIAL ---\n\n" + "\n".join(blocks)
    )


def _category_of(story: Story) -> str:
    from .dedupe import categorize

    return categorize(story)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    stories: dict[str, dict] = field(default_factory=dict)   # story.key -> analysis
    synthesis: dict[str, Any] = field(default_factory=dict)
    provider: str = "none"
    model: str = ""
    analysed: int = 0
    errors: list[str] = field(default_factory=list)


class AnalysisEngine:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.provider, self.model, self.api_key = detect_provider(config.provider)
        if config.model:
            self.model = config.model
        self.client = (
            LLMClient(self.provider, self.model, self.api_key, config.timeout)
            if self.provider != "none"
            else None
        )
        self.cache: dict[str, Any] = {}
        self._cache_dirty = False
        if config.use_cache and config.cache_path:
            self._load_cache()

    # -- cache -------------------------------------------------------------
    def _load_cache(self) -> None:
        path = self.config.cache_path
        if not path or not path.exists():
            return
        try:
            self.cache = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            self.cache = {}

    def save_cache(self) -> None:
        path = self.config.cache_path
        if not path or not self._cache_dirty:
            return
        # Keep the cache bounded.
        if len(self.cache) > 4000:
            for key in list(self.cache)[: len(self.cache) - 3000]:
                self.cache.pop(key, None)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.cache, ensure_ascii=False), "utf-8")
        except OSError as exc:
            log.warning("could not write analysis cache: %s", exc)

    def _cache_key(self, story: Story) -> str:
        raw = "|".join(
            [
                story.key,
                self.provider,
                self.model,
                self.config.language,
                ",".join(sorted(a.url for a in story.articles)),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    # -- public API --------------------------------------------------------
    def analyse(self, stories: list[Story]) -> AnalysisResult:
        result = AnalysisResult(provider=self.provider, model=self.model)
        if not stories:
            return result

        if self.provider == "none":
            result.stories = {s.key: extractive_analysis(s, self.config.language) for s in stories}
            result.synthesis = extractive_synthesis(stories, self.config.language)
            result.provider = "none"
            result.analysed = len(stories)
            return result

        pending: list[Story] = []
        for story in stories:
            key = self._cache_key(story)
            cached = self.cache.get(key)
            if isinstance(cached, dict) and cached.get("summary_en") or (
                isinstance(cached, dict) and cached.get("summary_ar")
            ):
                result.stories[story.key] = cached
            else:
                pending.append(story)

        log.info(
            "analysis: %d cached, %d to analyse via %s/%s",
            len(result.stories), len(pending), self.provider, self.model,
        )

        size = max(1, self.config.batch_size)
        for start in range(0, len(pending), size):
            batch = pending[start : start + size]
            index = start
            try:
                payload = self._analyse_batch(batch, index)
            except Exception as exc:
                result.errors.append(f"batch {start // size + 1}: {exc}")
                log.warning("analysis batch failed (%s); using extractive fallback", exc)
                for story in batch:
                    result.stories[story.key] = extractive_analysis(
                        story, self.config.language
                    )
                continue

            by_id = {}
            for item in payload.get("stories", []) or []:
                if isinstance(item, dict) and item.get("id"):
                    by_id[str(item["id"]).strip().upper()] = item

            for offset, story in enumerate(batch):
                sid = f"S{index + offset + 1}"
                item = by_id.get(sid)
                if not item:
                    result.stories[story.key] = extractive_analysis(
                        story, self.config.language
                    )
                    continue
                item.pop("id", None)
                item["_source"] = "llm"
                result.stories[story.key] = item
                self.cache[self._cache_key(story)] = item
                self._cache_dirty = True

        result.analysed = len(result.stories)

        try:
            result.synthesis = self._synthesise(stories, result.stories)
        except Exception as exc:
            result.errors.append(f"synthesis: {exc}")
            log.warning("synthesis failed: %s", exc)
            result.synthesis = extractive_synthesis(stories, self.config.language)

        return result

    # -- calls -------------------------------------------------------------
    def _analyse_batch(self, batch: list[Story], index: int) -> dict:
        assert self.client is not None
        user = build_story_user_prompt(
            batch,
            language=self.config.language,
            max_story_chars=self.config.max_story_chars,
            index=index,
        )
        raw = self.client.complete(SYSTEM_PROMPT, user)
        payload = parse_json_object(raw)
        if not payload.get("stories"):
            raise FetchError("model returned no usable stories array")
        return payload

    def _synthesise(self, stories: list[Story], analyses: dict[str, dict]) -> dict:
        assert self.client is not None
        _, lang_instruction = _field_spec(self.config.language)

        lines = []
        for idx, story in enumerate(stories):
            analysis = analyses.get(story.key, {})
            sid = f"S{idx + 1}"
            summary = (
                analysis.get("summary_en")
                or analysis.get("summary_ar")
                or truncate(story.primary.summary or story.primary.title, 300)
            )
            btl = analysis.get("between_the_lines_en") or analysis.get("between_the_lines_ar") or ""
            lines.append(
                f"- {sid} [{_category_of(story)}] {story.primary.title} "
                f"(carried by {story.outlet_count} outlets: {', '.join(story.sources)})\n"
                f"    summary: {truncate(str(summary), 400)}\n"
                f"    subtext: {truncate(str(btl), 400)}"
            )

        schema = """{
  "editor_note_en": "one tight paragraph (4-6 sentences) - the single most important thing about today and why",
  "editor_note_ar": "الملاحظة التحريرية بالعربية",
  "story_of_the_day": "S3",
  "top_themes": [
    {"theme_en": "short label", "theme_ar": "التسمية بالعربية", "story_ids": ["S1", "S5"], "note_en": "one sentence", "note_ar": "جملة واحدة"}
  ],
  "not_being_said_en": "what the shape of today's coverage avoids or under-reports, and what that pattern suggests",
  "not_being_said_ar": "ما يتجنبه التغطية اليوم",
  "watchlist_en": "2-4 comma-separated concrete things to watch next",
  "watchlist_ar": "ما يجب متابعته"
}
If the requested language is English only, you may leave the _ar fields empty; if Arabic \
only, leave the _en fields empty."""

        user = (
            "Here is the full set of analysed stories for today's Kuwait briefing.\n\n"
            f"{lang_instruction}\n\n"
            f"Return ONLY a JSON object matching this schema:\n{schema}\n\n"
            "--- STORIES ---\n\n" + "\n".join(lines)
        )
        raw = self.client.complete(SYNTHESIS_SYSTEM, user, max_tokens=4000)
        payload = parse_json_object(raw)
        if not payload:
            raise FetchError("model returned no usable synthesis")
        payload["_source"] = "llm"
        return payload


# --------------------------------------------------------------------------
# No-API-key fallback
# --------------------------------------------------------------------------


def _first_sentences(text: str, count: int = 3, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?؟])\s+", text)
    out = " ".join(parts[:count])
    return truncate(out or text, limit)


def extractive_analysis(story: Story, language: str) -> dict[str, Any]:
    """Rule-based stand-in used when no LLM is configured or a call fails."""
    primary = story.primary
    text = primary.body or primary.summary or ""
    summary = _first_sentences(text)
    others = [a for a in story.articles if a is not primary]
    note = (
        f"Analysed without an AI provider. This story was carried by "
        f"{story.outlet_count} outlet(s): {', '.join(story.sources)}."
    )

    item: dict[str, Any] = {
        "headline_en": primary.title if primary.lang == "en" else "",
        "headline_ar": primary.title if primary.lang == "ar" else "",
        "confidence": "low",
        "confidence_reason": "Extractive mode — no model was used, so no subtext is inferred.",
        "entities": [],
        "tags": story.sections[:4],
        "_source": "extractive",
    }
    if language in ("bilingual", "en"):
        item["summary_en"] = summary or primary.title
        item["why_it_matters_en"] = note
        item["between_the_lines_en"] = (
            "Not available — no AI provider is configured for this run. "
            + (
                "Outlets carrying it: " + "; ".join(a.title for a in others[:4])
                if others
                else ""
            )
        )
        item["watch_next_en"] = ""
    if language in ("bilingual", "ar"):
        arabic_note = (
            "تم التحليل دون مزود ذكاء اصطناعي. نشر هذا الخبر "
            f"{story.outlet_count} وسيلة إعلامية: {', '.join(story.sources)}."
        )
        item["summary_ar"] = (summary or primary.title) if primary.lang == "ar" else ""
        item["why_it_matters_ar"] = arabic_note
        item["between_the_lines_ar"] = (
            "غير متاح — لم يتم إعداد مزود ذكاء اصطناعي لهذه الجولة."
        )
        item["watch_next_ar"] = ""
    return item


def extractive_synthesis(stories: list[Story], language: str) -> dict[str, Any]:
    top = stories[0] if stories else None
    majors = [s for s in stories if s.outlet_count >= 3]
    note_en = (
        f"{len(stories)} distinct stories were carried across the Kuwaiti press in this "
        f"window, {len(majors)} of them by three or more outlets."
        + (f" The most widely carried was: {top.primary.title}." if top else "")
        + " Configure an AI provider (GEMINI_API_KEY, OPENAI_API_KEY or "
        "ANTHROPIC_API_KEY) to enable per-story subtext analysis and the daily synthesis."
    )
    payload: dict[str, Any] = {
        "editor_note_en": note_en,
        "editor_note_ar": "تم إعداد هذا الملخص دون مزود ذكاء اصطناعي.",
        "story_of_the_day": "S1" if top else "",
        "top_themes": [],
        "not_being_said_en": "Available only with an AI provider configured.",
        "watchlist_en": "",
        "_source": "extractive",
    }
    if language == "ar":
        payload["editor_note_en"] = ""
    return payload
