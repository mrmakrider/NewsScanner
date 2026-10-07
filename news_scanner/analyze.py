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
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .http import FetchError, fetch_json
from .models import Story
from .resilience import (
    CircuitBreaker,
    Deadline,
    Pacer,
    classify_failure,
    retry_after_of,
    sleep_for,
)
from .util import fmt_local, normalize_text, truncate

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
    # Free-tier rotation partners. Both speak the OpenAI chat protocol, so
    # adding them costs one line each and buys a second and third chance when
    # the primary refuses the day's traffic.
    "cerebras": "qwen-3.8-27b",
    "nvidia": "meta/llama-3.3-70b-instruct",
    # LLM7's "default" routing mode balances quality against latency by
    # picking the first available model. "fast" and "pro" also work.
    "llm7": "default",
    "ollama": "llama3.1",
}

KEY_ENV = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY", "NGC_API_KEY"),
    # Optional: a free token from https://token.llm7.io/ raises the rate
    # limit. Anonymous access works without one.
    "llm7": ("LLM7_API_KEY", "NEWSCANNER_LLM7_KEY"),
}

# OpenAI-compatible gateways and the base URL each one lives behind.
OPENAI_COMPATIBLE_BASES = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "llm7": "https://api.llm7.io/v1",
}

# Providers that honour the OpenAI `response_format: json_object` parameter.
# Cerebras documents JSON mode for both shared-inference models, so it is used.
# NIM is deliberately absent: its catalog spans dozens of models behind one
# URL, and whether a given one accepts the parameter varies, so sending it
# blind risks a 400 that benches a provider which was working. LLM7 fans a
# request out across heterogeneous backends for the same reason. Neither needs
# it — parse_json_object() recovers JSON from prose or fenced blocks anyway.
JSON_MODE_PROVIDERS = {"openai", "openrouter", "groq", "cerebras"}

# Providers whose models reason before answering and count those reasoning
# tokens against max_tokens. The briefing needs a JSON object, not a
# derivation, so reasoning is turned off for it entirely.
REASONING_EFFORT_PROVIDERS = {"cerebras"}

# Providers that work with no API key at all.
KEYLESS_PROVIDERS = {"llm7", "ollama"}

# Auto-detection order. Explicit configuration always wins: an API key, an
# OLLAMA_* opt-in or a NEWSCANNER_PROVIDER setting all take precedence over
# the keyless LLM7 default below.
PROVIDER_ORDER = [
    "gemini", "openai", "anthropic", "openrouter", "groq",
    "cerebras", "nvidia", "ollama", "llm7",
]

# The keyless LLM7 tier rate-limits by IP and reacts badly to a burst, so a
# run without a token is paced from the start. Keyed providers get no floor:
# they bill per token and are expected to absorb a full-speed run.
KEYLESS_FLOOR_SECONDS = 2.5

# A free gateway gives up on a slow generation well before the 150s default,
# and a request that is going to time out is worth abandoning early so the
# batch can be split into something that will actually finish.
KEYLESS_TIMEOUT_CAP = 60

# Longest single pause when every provider is cooling down. The breaker's own
# cooldowns grow to 15 minutes; waiting that long only makes sense if there is
# a budget to spend, so an unbudgeted run re-probes sooner instead.
MAX_RECOVERY_PAUSE = 120.0


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def _env_float_optional(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; ignoring", name, raw)
        return None


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

    # -- resilience --------------------------------------------------------
    # A wall-clock budget for the whole analysis step. The GitHub job is
    # killed at 30 minutes; stopping early means the brief is still written,
    # committed and emailed, with the un-analysed stories labelled as such.
    max_llm_seconds: float = 0.0        # 0 = no budget (tests, ad-hoc runs)
    call_attempts: int = 2              # rounds through the provider chain per call
    retry_passes: int = 1               # deferred retries of what failed
    pass_cooldown: float = 12.0         # seconds to let a throttled provider recover
    min_interval: float | None = None   # override the per-provider pacer floor
    max_interval: float = 20.0          # ceiling the pacer backs off to
    breaker_threshold: int = 3          # failures before a provider is benched
    breaker_cooldown: float = 45.0      # first cooldown, doubling per trip
    incremental_cache: bool = True      # persist each batch, survive a crash
    cache_flush_seconds: float = 30.0

    @classmethod
    def from_env(cls, **overrides) -> "AnalysisConfig":
        cfg = cls(
            provider=os.getenv("NEWSCANNER_PROVIDER", "auto").strip().lower() or "auto",
            model=os.getenv("NEWSCANNER_MODEL", "").strip(),
            language=os.getenv("NEWSCANNER_LANGUAGE", "bilingual").strip().lower(),
            temperature=float(os.getenv("NEWSCANNER_TEMPERATURE", "0.25")),
            max_llm_seconds=_env_float("NEWSCANNER_LLM_BUDGET", 0.0),
            call_attempts=_env_int("NEWSCANNER_LLM_ATTEMPTS", 2),
            retry_passes=_env_int("NEWSCANNER_LLM_RETRY_PASSES", 1),
            min_interval=_env_float_optional("NEWSCANNER_LLM_MIN_INTERVAL"),
            max_interval=_env_float("NEWSCANNER_LLM_MAX_INTERVAL", 20.0),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg


def detect_provider(requested: str = "auto") -> tuple[str, str, str]:
    """Return ``(provider, model, api_key)`` or ``("none", "", "")``.

    Precedence, most deliberate first:

    1. an explicit ``NEWSCANNER_PROVIDER``
    2. any configured API key (Gemini, OpenAI, Anthropic, OpenRouter, Groq)
    3. ``OLLAMA_HOST`` / ``NEWSCANNER_USE_OLLAMA=1``
    4. **LLM7** — the keyless default, so an unconfigured checkout still gets
       real analysis instead of the extractive fallback

    Set ``NEWSCANNER_NO_LLM7=1`` to skip step 4.
    """
    if requested == "none":
        return "none", "", ""

    if requested and requested != "auto":
        key = _key_for(requested)
        if requested in KEYLESS_PROVIDERS or key:
            model = os.getenv("NEWSCANNER_MODEL", "") or DEFAULT_MODELS.get(requested, "")
            return requested, model, key
        log.warning("provider %r requested but no API key found", requested)
        return "none", "", ""

    for provider in PROVIDER_ORDER:
        if provider == "ollama":
            if os.getenv("OLLAMA_HOST") or os.getenv("NEWSCANNER_USE_OLLAMA") == "1":
                return provider, DEFAULT_MODELS[provider], ""
            continue
        if provider == "llm7":
            if os.getenv("NEWSCANNER_NO_LLM7", "").strip() == "1":
                continue
            return provider, os.getenv("NEWSCANNER_MODEL", "") or DEFAULT_MODELS["llm7"], _key_for("llm7")
        key = _key_for(provider)
        if key:
            return provider, DEFAULT_MODELS.get(provider, ""), key
    return "none", "", ""


def _key_for(provider: str) -> str:
    for env_name in KEY_ENV.get(provider, ()):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True)
class ProviderCandidate:
    """One provider the engine may call, and the model to ask it for."""

    provider: str
    model: str
    api_key: str

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}" if self.model else self.provider


def detect_candidates(requested: str = "auto") -> list[ProviderCandidate]:
    """Every provider this run may use, best first.

    ``detect_provider`` answers "who should do the work". This answers the
    follow-up question a scheduled job has to care about: "and who else, if
    that one starts refusing?" The primary is always first, so an ordinary
    run behaves exactly as before; the rest only matter on failure.

    Two fallbacks are appended for the keyless default:

    * the same gateway on its ``fast`` route, which is the documented answer
      to the gateway timeouts (HTTP 524) the ``default`` route produces on
      long prompts;
    * nothing else, unless other API keys are configured.

    A run with a real key therefore falls back to the keyless gateway rather
    than falling all the way to extractive summaries. Set
    ``NEWSCANNER_NO_LLM7=1`` to opt out of that, or ``NEWSCANNER_MODEL`` to
    pin one model and suppress the fast-route variant.

    An explicitly requested provider is the whole chain: pinning ``openai``
    must not quietly spend someone else's quota, so the extra candidates are
    only gathered in ``auto`` mode.
    """
    candidates: list[ProviderCandidate] = []
    seen: set[tuple[str, str]] = set()
    explicit_model = os.getenv("NEWSCANNER_MODEL", "").strip()

    def add(provider: str, model: str, key: str) -> None:
        if not provider or provider == "none":
            return
        model = model or DEFAULT_MODELS.get(provider, "")
        signature = (provider, model)
        if signature in seen:
            return
        seen.add(signature)
        candidates.append(ProviderCandidate(provider, model, key))

    primary = detect_provider(requested)
    if primary[0] != "none":
        add(*primary)

    auto = requested in ("", "auto")
    for provider in PROVIDER_ORDER:
        if not auto and provider != primary[0]:
            continue
        if provider == "ollama":
            if os.getenv("OLLAMA_HOST") or os.getenv("NEWSCANNER_USE_OLLAMA") == "1":
                add(provider, DEFAULT_MODELS[provider], "")
            continue
        if provider == "llm7":
            if os.getenv("NEWSCANNER_NO_LLM7", "").strip() == "1":
                continue
            add("llm7", explicit_model or DEFAULT_MODELS["llm7"], _key_for("llm7"))
            if not explicit_model:
                add("llm7", "fast", _key_for("llm7"))
            continue
        key = _key_for(provider)
        if key:
            add(provider, explicit_model or DEFAULT_MODELS.get(provider, ""), key)

    return candidates


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
        base = OPENAI_COMPATIBLE_BASES.get(
            self.provider, OPENAI_COMPATIBLE_BASES["openai"]
        )
        base = (
            os.getenv("NEWSCANNER_BASE_URL", "").strip().rstrip("/")
            if self.provider == "llm7"
            else ""
        ) or base

        headers: dict[str, str] = {}
        # LLM7's free anonymous tier works with no credential at all, so the
        # header is only sent when there is genuinely something to send.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
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
        }
        if self.provider in JSON_MODE_PROVIDERS:
            payload["response_format"] = {"type": "json_object"}

        # Reasoning-era OpenAI models renamed the cap parameter.
        if re.match(r"^(o\d|gpt-5)", self.model):
            payload["max_completion_tokens"] = max_tokens
            payload.pop("temperature", None)
            payload.pop("response_format", None)
        else:
            if self.provider == "llm7":
                # The gateway routes to whatever free backend is available,
                # so ask for less than we would from a first-party API.
                max_tokens = min(max_tokens, 4000)
            payload["max_tokens"] = max_tokens

        # Cerebras reasoning models (qwen-3.8-27b defaults to high effort) spend
        # their reasoning tokens against max_tokens before writing any answer:
        # observed 2249 reasoning tokens against a 4000 cap, which left too
        # little room for the JSON and the response came back with no content
        # at all — surfacing as KeyError('content') and killing the whole
        # synthesis. These calls want a JSON object, not a derivation, so
        # reasoning is switched off rather than merely budgeted for.
        if self.provider in REASONING_EFFORT_PROVIDERS:
            payload["reasoning_effort"] = "none"

        data = fetch_json(
            f"{base}/chat/completions",
            payload,
            headers=headers,
            timeout=self.timeout,
            retry_connect_errors=False,
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
        data = fetch_json(url, payload, timeout=self.timeout, retry_connect_errors=False)
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
            retry_connect_errors=False,
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
        data = fetch_json(
            f"{host}/api/chat", payload, timeout=self.timeout, retry_connect_errors=False
        )
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


_STORY_RESPONSE_FIELDS = {
    "headline_en", "headline_ar", "summary_en", "summary_ar",
    "why_it_matters_en", "why_it_matters_ar", "between_the_lines_en",
    "between_the_lines_ar", "watch_next_en", "watch_next_ar", "confidence",
    "confidence_reason", "entities", "tags", "headline", "summary",
    "why_it_matters", "between_the_lines", "watch_next",
}


def _normalise_story_payload(
    payload: dict[str, Any], batch: list[tuple[str, Story]]
) -> dict[str, Any]:
    """Recover common, safe-to-map variants of a story response.

    Models sometimes return one story object instead of the documented wrapper,
    a mapping keyed by story id instead of an array, or omit ids when asked for
    one item. These forms can be repaired without guessing when there is exactly
    one requested story, or when the response count exactly matches the request.
    Unknown and duplicate identifiers are never reassigned to another story.
    """
    raw = payload.get("stories")
    rows: list[dict[str, Any]] = []
    if isinstance(raw, list):
        rows = [dict(row) for row in raw if isinstance(row, dict)]
    elif isinstance(raw, dict):
        if _STORY_RESPONSE_FIELDS.intersection(raw):
            rows = [dict(raw)]
        else:
            for sid, row in raw.items():
                if isinstance(row, dict):
                    item = dict(row)
                    item.setdefault("id", sid)
                    rows.append(item)
    elif _STORY_RESPONSE_FIELDS.intersection(payload):
        rows = [dict(payload)]

    requested_ids = [sid.strip().upper() for sid, _story in batch]
    allowed = set(requested_ids)
    recovered: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sid = str(row.get("id", "")).strip().upper()
        if not sid and len(rows) == len(requested_ids):
            sid = requested_ids[index]
        if not sid and len(requested_ids) == 1:
            sid = requested_ids[0]
        if sid not in allowed:
            continue
        row["id"] = sid
        recovered.append(row)
    return {**payload, "stories": recovered}


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
    index: int = 0,
    ids: list[str] | None = None,
) -> str:
    """Build the per-batch prompt.

    ``ids`` names each story explicitly. A batch that gets split in half after
    a failure must keep the identifiers the model was given the first time
    round, or the answers come back attached to the wrong stories.
    """
    _, lang_instruction = _field_spec(language)
    blocks = []
    for offset, story in enumerate(stories):
        sid = ids[offset] if ids and offset < len(ids) else f"S{index + offset + 1}"
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


# How today's stories may relate. Closed set on purpose: an open one lets the
# model invent a relationship type per link, which makes the graph unreadable
# and impossible to count or compare across days.
LINK_TYPES = {
    "consequence", "reaction", "contradiction", "same_actor",
    "same_source_angle", "escalation",
}


# --------------------------------------------------------------------------
# The entity graph
# --------------------------------------------------------------------------
#
# The model is bad at one thing and good at most others: knowing that
# "مجلس الوزراء", "the cabinet" and "Cabinet of Ministers" are the same actor.
# Resolving that in Python and handing the model a finished index is cheaper
# than letting it guess, and a wrong guess here quietly corrupts every link
# the model draws downstream.

# Arabic and English names for bodies that appear in Kuwaiti coverage almost
# every day. Matching on these is exact, not fuzzy: a partial match would link
# "the health ministry" to "a health report", which is worse than no link.
_ENTITY_ALIASES = {
    "the cabinet": "Kuwait Cabinet",
    "cabinet of ministers": "Kuwait Cabinet",
    "المجلس": "Kuwait Cabinet",
    "مجلس الوزراء": "Kuwait Cabinet",
    "المجلس الوزاري": "Kuwait Cabinet",
    "the emir": "Kuwait Emir",
    "emir of kuwait": "Kuwait Emir",
    "صاحب السمو الأمير": "Kuwait Emir",
    "the parliament": "Kuwait Parliament",
    "البرلمان": "Kuwait Parliament",
    "مجلس الوزراء": "Kuwait Cabinet",
    "the ministry of finance": "Ministry of Finance",
    "وزارة المالية": "Ministry of Finance",
    "the supreme court": "Supreme Court",
    "المحكمة الدستورية": "Constitutional Court",
    "kuwait fund": "Kuwait Investment Authority",
    "kia": "Kuwait Investment Authority",
    "صندوق الكويت": "Kuwait Investment Authority",
    "the amir": "Kuwait Emir",
}

# Entities too generic to carry meaning in a link graph.
_STOP_ENTITIES = {
    "kuwait", "news", "today", "the", "government", "official", "officials",
    "source", "sources", "report", "reports", "ministry", "council", "committee",
    "company", "sector", "week", "year", "day", "reuters", "agency",
}

def _outlet_names() -> set[str]:
    """Source names from config/sources.toml, read once and cached.

    Outlets are not actors. The model lists whichever papers it happens to be
    looking at as "entities", and persisting those as a recurring thread would
    tell tomorrow's run that "Al-Rai is running across several days" — which is
    always true and so always useless. Reading the real source names means the
    filter cannot drift from the configured outlets.
    """
    global _OUTLET_CACHE
    if _OUTLET_CACHE is not None:
        return _OUTLET_CACHE
    names: set[str] = set()
    try:
        from .sources import load_config

        sources, _settings = load_config()
        for source in sources:
            for value in (getattr(source, "name", ""), getattr(source, "id", "")):
                cleaned = str(value or "").strip()
                if cleaned:
                    names.add(_squash(cleaned))
    except Exception as exc:  # noqa: BLE001 — a filter must never break analysis
        log.debug("could not read outlet names for the entity filter: %s", exc)
    _OUTLET_CACHE = names
    return names


_OUTLET_CACHE: set[str] | None = None


def _squash(name: str) -> str:
    """Collapse a name to letters and digits so spelling variants still match.

    Punctuation is dropped rather than replaced, so "Al-Ra'y", "Al-Rai" and
    "alray" all squash to the same letters and match the same outlet.
    """
    return re.sub(r"[^a-z0-9\u0600-\u06ff]", "", (name or "").lower())


def _looks_like_outlet(name: str) -> bool:
    """True when an entity is a newspaper rather than an actor in the story.

    The model transliterates Arabic paper names inconsistently — config calls
    one "alrai" and the model writes "alray" — so an exact prefix test alone
    misses variants. Relaxing the letters Arabic transliteration actually
    confuses, on both sides of the comparison, catches those without loosening
    the test generally: "Iran" and "OPEC" still do not match any outlet.
    """
    candidate = _relax(_squash(name))
    if len(candidate) < 4:
        return False
    for outlet in _outlet_names():
        relaxed = _relax(outlet)
        if len(relaxed) < 4:
            continue
        if candidate.startswith(relaxed) or relaxed.startswith(candidate):
            return True
    return False


def _relax(value: str) -> str:
    """Collapse the letter pairs Arabic transliteration routinely swaps.

    'ai'/'ay'/'ei' and doubled letters all reduce to the same form, which is
    what makes 'alray', 'alrai' and 'alrayi' compare equal. Applied to both
    sides so an exact spelling also matches its relaxed neighbours.
    """
    value = re.sub(r"([ae])y", r"\1i", value)
    value = re.sub(r"ei", "i", value)
    return re.sub(r"(.)\1+", r"\1", value)


def normalise_entity(name: str) -> str:
    """Map a raw entity string to a canonical name, or '' if unusable."""
    raw = re.sub(r"\s+", " ", (name or "").strip())
    if not raw:
        return ""
    lowered = raw.lower().strip(" .،,")
    if lowered in _STOP_ENTITIES:
        return ""
    # Checked before canonicalisation so that a name the model spelled with an
    # odd apostrophe still matches: "Al-Ra'y" and "Al-Rai" are one outlet.
    if _looks_like_outlet(raw):
        return ""
    canonical = _ENTITY_ALIASES.get(lowered)
    if canonical:
        return canonical
    # Title-case the rest so casing variants collapse: the model writes
    # "Houthi Militia" and "Houthi militia" for the same actor, and two spellings
    # of one entity would read as two separate threads.
    return raw.strip(" .،,").title() if raw.isascii() else raw.strip(" .،,")


def build_entity_graph(
    stories: list[Story], analyses: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Map each resolved entity to the story ids that mention it.

    Built here rather than asked for, because it is a lookup: the model
    should be told who is who, not asked to work it out from scratch on every
    run. Entities that appear in exactly one story carry no cross-story
    signal, so they are left out to keep the prompt small.
    """
    index: dict[str, set[str]] = {}
    for idx, analysis in enumerate(analyses):
        sid = f"S{idx + 1}"
        for raw in analysis.get("entities") or []:
            name = normalise_entity(str(raw))
            if name:
                index.setdefault(name, set()).add(sid)

    # Only entities spanning two or more stories tell the model something it
    # could not have inferred from one batch of five.
    return {name: sorted(ids) for name, ids in sorted(index.items()) if len(ids) >= 2}


def render_entity_graph(graph: dict[str, list[str]]) -> str:
    """The graph as prompt lines: ``- Entity: S1, S4, S9``."""
    if not graph:
        return ""
    lines = [f"- {name}: {', '.join(ids)}" for name, ids in graph.items()]
    return "\n".join(lines)


def build_link_context(
    stories: list[Story], analyses: list[dict[str, Any]]
) -> tuple[str, dict[str, list[str]]]:
    """Return ``(prompt_block, graph)`` for the synthesis call."""
    graph = build_entity_graph(stories, analyses)
    rendered = render_entity_graph(graph)
    if not rendered:
        return "", graph
    return (
        "ENTITIES SPANNING MORE THAN ONE STORY (resolved for you; treat each as "
        "the same actor wherever it appears):\n" + rendered + "\n",
        graph,
    )


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
    # -- how hard the run had to work, for the log and the run report ------
    providers_used: list[str] = field(default_factory=list)
    retries: int = 0          # failed calls retried, at batch or story level
    splits: int = 0           # batches halved to dodge a gateway timeout
    missing: int = 0          # stories the model silently skipped
    repaired: int = 0         # partial model/cache answers completed deterministically


class _ChainEntry:
    """One callable provider: its client, its pacing, its breaker.

    State lives here rather than in the engine so that a provider which is
    throttled stays throttled across batches — the whole point of pacing is
    that the next batch inherits what the last one learned.
    """

    def __init__(
        self,
        candidate: ProviderCandidate,
        *,
        timeout: int,
        floor: float,
        ceiling: float,
        breaker_threshold: int,
        breaker_cooldown: float,
    ):
        self.provider = candidate.provider
        self.model = candidate.model
        self.api_key = candidate.api_key
        self.label = candidate.label
        self.client = LLMClient(candidate.provider, candidate.model, candidate.api_key, timeout)
        self.pacer = Pacer(floor=floor, ceiling=ceiling)
        self.breaker = CircuitBreaker(
            threshold=breaker_threshold, cooldown=breaker_cooldown
        )
        self.used = 0


class AnalysisEngine:
    def __init__(
        self,
        config: AnalysisConfig,
        *,
        memory_context: str = "",
        recurring_entities: list[str] | None = None,
    ):
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
        self._last_flush = 0.0
        # Whether anything this run was throttled (as opposed to unreachable or
        # just wrong). Only throttles are worth waiting out; see _recovery_wait.
        self._saw_throttle = False
        # What earlier editions concluded, handed to the synthesis call so the
        # day's analysis can be read against yesterday's rather than in
        # isolation. Set by cmd_run from the memory store; empty is correct
        # for a first run and for tests.
        self.memory_context = memory_context
        self.recurring_entities = list(recurring_entities or [])
        if config.use_cache and config.cache_path:
            self._load_cache()

    # -- provider chain ----------------------------------------------------
    def _build_chain(self) -> list[_ChainEntry]:
        """The providers this run may use, each with its own pacing and breaker."""
        entries: list[_ChainEntry] = []
        for candidate in detect_candidates(self.config.provider):
            keyless = candidate.provider in KEYLESS_PROVIDERS and not candidate.api_key
            floor = self.config.min_interval
            if floor is None:
                floor = KEYLESS_FLOOR_SECONDS if keyless else 0.0
            timeout = self.config.timeout
            if keyless and candidate.provider == "llm7":
                timeout = min(timeout, KEYLESS_TIMEOUT_CAP)
            entries.append(
                _ChainEntry(
                    candidate,
                    timeout=timeout,
                    floor=floor,
                    ceiling=max(floor, self.config.max_interval),
                    breaker_threshold=self.config.breaker_threshold,
                    breaker_cooldown=self.config.breaker_cooldown,
                )
            )
        return entries

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
        # `use_cache=False` must mean "do not touch the cache at all". Writing
        # anyway would let a --no-cache diagnostic run (or a test with a stub
        # model) persist its results and serve them back to real runs later.
        if not path or not self.config.use_cache or not self._cache_dirty:
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

        self._saw_throttle = False
        pending = self._pending(stories, result)

        entries = self._build_chain()
        if not entries:
            log.warning("no usable LLM provider — falling back to extractive analysis")
            for story in pending:
                result.stories[story.key] = extractive_analysis(
                    story, self.config.language
                )
            result.synthesis = extractive_synthesis(stories, self.config.language)
            result.analysed = len(result.stories)
            return result

        size = max(1, self.config.batch_size)
        log.info(
            "analysis: %d cached, %d to analyse in ~%d call(s) via %s",
            len(result.stories), len(pending), -(-len(pending) // size),
            " → ".join(e.label for e in entries),
        )

        deadline = Deadline(self.config.max_llm_seconds)
        numbered = [(f"S{i + 1}", story) for i, story in enumerate(pending)]
        queue = deque(numbered[i : i + size] for i in range(0, len(numbered), size))
        unresolved = self._work(queue, entries, deadline, result)

        # Deferred passes. What failed is retried after a cooldown rather than
        # written off: throttles expire, and the retry carries trimmed prompts,
        # so a gateway that timed out on a large payload gets a smaller one.
        for pass_no in range(2, self.config.retry_passes + 2):
            if not unresolved or deadline.expired():
                break
            wait = self._recovery_wait(pass_no, deadline)
            if wait > 0:
                log.info(
                    "analysis: waiting %.0fs before retrying %d unresolved batch(es)",
                    wait, len(unresolved),
                )
                time.sleep(wait)
            if deadline.expired():
                break
            result.retries += len(unresolved)
            unresolved = self._work(
                deque(unresolved),
                entries,
                deadline,
                result,
                trim=self._trim(pass_no),
                attempts=1,
            )

        for batch in unresolved:
            for _sid, story in batch:
                result.stories[story.key] = extractive_analysis(
                    story, self.config.language
                )

        result.analysed = len(result.stories)
        self._flush_cache(force=True)

        try:
            result.synthesis = self._synthesise(
                stories, result.stories, entries, deadline, result,
                self.memory_context, self.recurring_entities,
            )
        except Exception as exc:
            result.errors.append(f"synthesis: {exc}")
            log.warning("synthesis failed: %s", exc)
            result.synthesis = extractive_synthesis(stories, self.config.language)

        if result.providers_used:
            # Name the provider that actually did the work rather than the one
            # that was asked first: a run that failed over should say so.
            result.provider = result.providers_used[0].split("/")[0]
        elif any(
            item.get("_source") in {"llm", "hybrid"}
            for item in result.stories.values()
        ):
            # Cached model analyses still contribute to this digest even when
            # no provider call was needed (or the synthesis call failed).
            result.provider = self.provider
        else:
            # A configured endpoint is not the same thing as a provider that
            # actually answered. Downstream rendering uses this to label a
            # fully degraded run honestly.
            result.provider = "none"

        if result.splits or result.retries or result.missing or result.errors:
            log.info(
                "analysis: %d with a model · %d retried · %d batch(es) split · "
                "%d skipped by the model · %d error(s)",
                sum(1 for a in result.stories.values() if a.get("_source") == "llm"),
                result.retries, result.splits, result.missing, len(result.errors),
            )
        return result

    # -- work queue --------------------------------------------------------
    def _pending(self, stories: list[Story], result: AnalysisResult) -> list[Story]:
        """Split the input into cache hits and stories that still need a call."""
        pending: list[Story] = []
        for story in stories:
            key = self._cache_key(story)
            cached = self.cache.get(key)
            if isinstance(cached, dict) and (cached.get("summary_en") or cached.get("summary_ar")):
                repaired, changed = _merge_model_analysis(
                    cached, story, self.config.language
                )
                result.stories[story.key] = repaired
                if changed or repaired != cached:
                    self.cache[key] = repaired
                    self._cache_dirty = True
                if changed:
                    result.repaired += 1
            else:
                pending.append(story)
        return pending

    def _trim(self, pass_no: int) -> float:
        """How much of each story to keep on retry pass ``pass_no``.

        A Cloudflare 524 from a gateway means the generation ran out of time,
        and prompt size is the largest lever on that. Each pass keeps less of
        the body text — enough for the model to work with, small enough to
        actually come back. With the default ``retry_passes=1`` the retry keeps
        75% of the body; the 35% floor is only reached if several passes are
        configured.
        """
        return max(0.35, 0.75 ** (max(1, pass_no) - 1))

    def _recovery_wait(self, pass_no: int, deadline: Deadline) -> float:
        """How long to pause before a deferred retry pass.

        Only a throttle is worth waiting out. An unreachable gateway or a bad
        key will not improve by sitting still — the retry is about sending a
        smaller request, not about patience — so those get a token pause.
        """
        base = self.config.pass_cooldown if self._saw_throttle else min(1.0, self.config.pass_cooldown)
        return deadline.clamp(base * (pass_no - 1))

    def _work(
        self,
        queue: deque,
        entries: list[_ChainEntry],
        deadline: Deadline,
        result: AnalysisResult,
        trim: float = 1.0,
        attempts: int | None = None,
    ) -> list[list[tuple[str, Story]]]:
        """Drain a queue of batches, returning the ones that could not be done."""
        unresolved: list[list[tuple[str, Story]]] = []
        while queue:
            batch = queue.popleft()
            if deadline.expired():
                log.warning(
                    "analysis: budget of %.0fs spent — %d batch(es) left un-analysed",
                    self.config.max_llm_seconds, len(unresolved) + 1 + len(queue),
                )
                unresolved.append(batch)
                unresolved.extend(queue)
                break

            # If every provider is benched there is nothing to send: pause for
            # the earliest cooldown instead of burning through the queue on
            # calls that cannot be made. Worth doing only when the trouble was
            # a throttle — a host that is down stays down.
            if self._saw_throttle and entries and all(not e.breaker.allow() for e in entries):
                wait = deadline.clamp(
                    min(min(e.breaker.remaining for e in entries), MAX_RECOVERY_PAUSE)
                )
                if wait > 0:
                    log.info("analysis: every provider is cooling down — pausing %.0fs", wait)
                    time.sleep(wait)
                if deadline.expired():
                    unresolved.append(batch)
                    unresolved.extend(queue)
                    break

            leftover = self._attempt_batch(batch, entries, deadline, result, trim, attempts)
            if leftover:
                unresolved.append(leftover)
        return unresolved

    def _attempt_batch(
        self,
        batch: list[tuple[str, Story]],
        entries: list[_ChainEntry],
        deadline: Deadline,
        result: AnalysisResult,
        trim: float = 1.0,
        attempts: int | None = None,
    ) -> list[tuple[str, Story]]:
        """Analyse a batch, halving it on failure before giving up on a story.

        Splitting is the one thing that reliably fixes a gateway timeout: the
        prompt that timed out will time out again, but half of it usually will
        not. Only a single story that still fails is handed to the extractive
        fallback — and the halves are tried breadth-first, so the run gets as
        far as it can with the time it has.

        The batch as it arrived gets the full retry budget; its halves get one
        round each, because a split *is* a retry — re-running the same unhappy
        provider on the same large prompt is what the split exists to avoid.
        """
        pending: list[list[tuple[str, Story]]] = [list(batch)]
        failed: list[tuple[str, Story]] = []
        first_round = True

        while pending:
            current = pending.pop(0)
            rounds = (attempts if attempts is not None else self.config.call_attempts) \
                if first_round else 1
            first_round = False
            max_chars = max(400, int(self.config.max_story_chars * trim))
            try:
                payload = self._analyse_batch(
                    current,
                    0,
                    entries=entries,
                    deadline=deadline,
                    max_chars=max_chars,
                    result=result,
                    attempts=rounds,
                )
            except Exception as exc:
                if len(current) > 1:
                    mid = len(current) // 2
                    result.splits += 1
                    log.warning(
                        "analysis: batch of %d failed (%s) — splitting into %d + %d",
                        len(current), exc, mid, len(current) - mid,
                    )
                    pending.insert(0, current[:mid])
                    pending.insert(1, current[mid:])
                    continue
                sid, story = current[0]
                result.errors.append(f"{sid} {truncate(story.primary.title, 70)}: {exc}")
                log.warning("analysis: giving up on %s — %s", sid, exc)
                failed.append(current[0])
                continue

            self._store(current, payload, result)
            self._flush_cache()

        return failed

    def _store(
        self,
        batch: list[tuple[str, Story]],
        payload: dict,
        result: AnalysisResult,
    ) -> None:
        """Attach a model's answers to the stories they belong to."""
        by_id: dict[str, dict] = {}
        for item in payload.get("stories", []) or []:
            if isinstance(item, dict) and item.get("id"):
                by_id[str(item["id"]).strip().upper()] = item

        for sid, story in batch:
            item = by_id.get(sid.strip().upper())
            if not item:
                # The model answered but skipped this one. That is a quality
                # failure, not an outage: fall back quietly, and count it.
                result.missing += 1
                log.warning("analysis: no answer returned for %s", sid)
                result.stories[story.key] = extractive_analysis(
                    story, self.config.language
                )
                continue
            repaired, was_repaired = _merge_model_analysis(
                item, story, self.config.language
            )
            if was_repaired:
                result.repaired += 1
            result.stories[story.key] = repaired
            self.cache[self._cache_key(story)] = repaired
            self._cache_dirty = True

    def _flush_cache(self, *, force: bool = False) -> None:
        """Persist progress mid-run.

        A run that dies at the step timeout keeps everything it managed to
        analyse, and the 08:30 retry — or tomorrow's run — picks up from
        there instead of paying for the same calls twice.
        """
        if not (self.config.incremental_cache and self.config.use_cache and self.config.cache_path):
            return
        now = time.monotonic()
        if not force and now - self._last_flush < self.config.cache_flush_seconds:
            return
        self._last_flush = now
        self.save_cache()

    # -- calls -------------------------------------------------------------
    def _complete(
        self,
        entries: list[_ChainEntry],
        *,
        system: str,
        user: str,
        max_tokens: int,
        kind: str,
        deadline: Deadline,
        result: AnalysisResult,
        attempts: int | None = None,
    ) -> str:
        """One model call, paced, failed over and circuit-broken.

        Each round walks the provider chain in order: a provider whose breaker
        is open is skipped, the pacer sleeps its interval before the call, and
        a failure is fed back into both the pacer and the breaker. A round
        that fails everywhere backs off before the next one — and if *every*
        provider is benched, the shortest cooldown is waited out instead of
        returning empty-handed, because a job that runs once a day can afford
        to wait for its only gateway to forgive it.
        """
        last_exc: Exception | None = None
        rounds = max(1, attempts if attempts is not None else self.config.call_attempts)

        for attempt in range(1, rounds + 1):
            if deadline.expired():
                raise last_exc or FetchError(
                    f"{kind}: the analysis budget ran out before the call was made"
                )
            blocked: list[float] = []
            down: set[str] = set()
            for entry in entries:
                if entry.provider in down:
                    # Same provider, different model: the host has already
                    # refused this round, so asking it again is just noise.
                    continue
                if not entry.breaker.allow():
                    blocked.append(entry.breaker.remaining)
                    continue
                entry.pacer.wait(deadline)
                try:
                    out = entry.client.complete(system, user, max_tokens=max_tokens)
                except Exception as exc:  # noqa: BLE001 — every failure is the policy's business
                    last_exc = exc
                    if classify_failure(exc) == "unreachable":
                        down.add(entry.provider)
                    self._record_failure(entry, exc)
                    continue
                entry.breaker.record_success()
                entry.pacer.on_success()
                entry.used += 1
                if entry.label not in result.providers_used:
                    result.providers_used.append(entry.label)
                    if entry.provider != self.provider:
                        log.info("analysis: failing over to %s", entry.label)
                return out

            if attempt >= rounds:
                break

            # How long the next round is worth waiting. A throttle clears with
            # time and earns a real backoff; an unreachable host or a rejected
            # key does not, and waiting on those only delays the fallback that
            # actually helps — to another provider, or to the summary.
            failure_kind = classify_failure(last_exc) if last_exc else "unknown"
            wait = (
                sleep_for(attempt, retry_after=retry_after_of(last_exc))
                if failure_kind == "throttled"
                else 0.5
            )
            if blocked:
                # Everything is benched; the earliest recovery is the ceiling
                # on how long it is worth waiting.
                wait = min(wait, min(blocked))
            wait = deadline.clamp(wait)
            if wait <= 0:
                break
            if wait >= 1:
                log.info(
                    "%s: nothing callable (%s), waiting %.0fs before attempt %d/%d",
                    kind, failure_kind, wait, attempt + 1, rounds,
                )
            time.sleep(wait)

        raise last_exc or FetchError(f"{kind}: no provider available")

    def _record_failure(self, entry: _ChainEntry, exc: Exception) -> None:
        """Feed a failure into the provider's pacing and breaker."""
        cooldown = entry.breaker.record_failure()
        kind = classify_failure(exc)
        if kind == "throttled":
            self._saw_throttle = True
        elif kind == "invalid":
            # A rejected key or a malformed request will not fix itself on the
            # next batch. Park the provider for good rather than paying for it
            # on every batch of the run.
            entry.breaker.record_failure()
            entry.breaker.record_failure()
        if kind == "throttled":
            entry.pacer.on_throttle(retry_after_of(exc))
        else:
            # Nothing is reaching the provider, so there is nothing to pace.
            # Spacing these calls out would only slow the run on its way to
            # the breaker that is about to bench the provider anyway.
            entry.pacer.reset()
        if cooldown:
            log.warning("%s: %s — circuit open for %.0fs (%s)", entry.label, exc, cooldown, kind)
        else:
            log.warning("%s: %s (%s)", entry.label, exc, kind)

    def _analyse_batch(
        self,
        batch: list[Story] | list[tuple[str, Story]],
        index: int = 0,
        *,
        entries: list[_ChainEntry] | None = None,
        deadline: Deadline | None = None,
        max_chars: int | None = None,
        result: AnalysisResult | None = None,
        attempts: int | None = None,
    ) -> dict:
        """One batch to the model.

        Accepts either plain stories (the ``check-llm`` diagnostic calls it
        that way) or ``(id, story)`` pairs, which is how the queue tracks a
        batch across a split.
        """
        pairs: list[tuple[str, Story]] = []
        for offset, item in enumerate(batch):
            if isinstance(item, tuple):
                pairs.append((str(item[0]), item[1]))
            else:
                pairs.append((f"S{index + offset + 1}", item))
        if not pairs:
            raise FetchError("empty batch")

        if entries is None:
            entries = self._build_chain()
        if not entries:
            raise FetchError("no LLM provider configured")
        if deadline is None:
            deadline = Deadline(0)
        if result is None:
            result = AnalysisResult()

        user = build_story_user_prompt(
            [story for _sid, story in pairs],
            language=self.config.language,
            max_story_chars=max_chars or self.config.max_story_chars,
            ids=[sid for sid, _story in pairs],
        )
        raw = self._complete(
            entries,
            system=SYSTEM_PROMPT,
            user=user,
            max_tokens=8000,
            kind=f"batch {pairs[0][0]}",
            deadline=deadline,
            result=result,
            attempts=attempts,
        )
        payload = parse_json_object(raw)
        payload = _normalise_story_payload(payload, pairs)
        if not payload.get("stories"):
            raise FetchError("model returned no usable stories array")
        return payload


    def _synthesise(
        self,
        stories: list[Story],
        analyses: dict[str, dict],
        entries: list[_ChainEntry] | None = None,
        deadline: Deadline | None = None,
        result: AnalysisResult | None = None,
        memory_context: str = "",
        recurring: list[str] | None = None,
    ) -> dict:
        """The day's editorial synthesis.

        This is one call over every analysed story, so it carries the largest
        prompt of the run and is the first thing a gateway times out on. On
        failure it is retried over the top stories only — a synthesis of the
        day's twelve biggest stories is a better briefing than none.
        """
        try:
            out = self._synthesise_once(
                stories, analyses, entries, deadline, result, memory_context, recurring
            )
        except Exception as exc:
            reduced = stories[:12]
            if len(reduced) == len(stories):
                raise
            log.warning(
                "synthesis failed over %d stories (%s) — retrying with the top %d",
                len(stories), exc, len(reduced),
            )
            if result is not None:
                result.retries += 1
            out = self._synthesise_once(
                reduced, analyses, entries, deadline, result, memory_context, recurring
            )
        return _validate_links(out, len(stories))

    def _synthesise_once(
        self,
        stories: list[Story],
        analyses: dict[str, dict],
        entries: list[_ChainEntry] | None = None,
        deadline: Deadline | None = None,
        result: AnalysisResult | None = None,
        memory_context: str = "",
        recurring: list[str] | None = None,
    ) -> dict:
        if self.client is None:
            raise FetchError("no LLM provider configured")
        _, lang_instruction = _field_spec(self.config.language)

        # The entity graph is derived here, not by the model: it is a lookup
        # over data we already hold, and getting an identity wrong would
        # silently corrupt every link drawn from it.
        analysis_list = [analyses.get(s.key, {}) for s in stories]
        graph_block, _graph = build_link_context(stories, analysis_list)
        memory_block = (
            f"\nWHAT EARLIER EDITIONS SAID (use it to judge whether today's news "
            f"continues, reverses or outgrows these; say so explicitly if it does):\n"
            f"{memory_context}\n"
            if memory_context else ""
        )
        recurring_block = (
            "\nENTITIES STILL RUNNING ACROSS SEVERAL DAYS (a thread worth "
            f"checking today):\n- " + ", ".join(recurring) + "\n"
            if recurring else ""
        )

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
  "tactical_en": "what matters THIS WEEK: which decisions are pending, who has to act, on what date",
  "tactical_ar": "ما يهم هذا الأسبوع",
  "strategic_en": "what this week's pattern means over months: what is being normalised, what trajectory the coverage reveals",
  "strategic_ar": "ما الذي يكشفه هذا النمط على المدىmonths",
  "links": [
    {"type": "consequence|reaction|contradiction|same_actor|same_source_angle|escalation",
     "from": "S4", "to": "S9", "evidence": "one clause naming the specific connection"}
  ],
  "top_themes": [
    {"theme_en": "short label", "theme_ar": "التسمية بالعربية", "story_ids": ["S1", "S5"], "note_en": "one sentence", "note_ar": "جملة واحدة"}
  ],
  "not_being_said_en": "what the shape of today's coverage avoids or under-reports, and what that pattern suggests",
  "not_being_said_ar": "ما يتجنبه التغطية اليوم",
  "watchlist_en": "2-4 comma-separated concrete things to watch next",
  "watchlist_ar": "ما يجب متابعته"
}
LINKS ARE THE POINT OF THIS CALL. Two stories that merely share a topic are
not linked. Link them only when one of these holds, and say which:
- consequence: one follows from or causes the other
- reaction: the second is coverage responding to the first
- contradiction: the two outlets tell incompatible accounts of one situation
- same_actor: the same resolved entity appears in both
- same_source_angle: the same underlying source is being cited from two directions
- escalation: a conflict or process has measurably advanced
If today's stories genuinely do not connect, return an empty links array. Do not \
invent connections to fill the field.
If the requested language is English only, you may leave the _ar fields empty; if Arabic \
only, leave the _en fields empty."""

        user = (
            "Here is the full set of analysed stories for today's Kuwait briefing.\n\n"
            f"{lang_instruction}\n\n"
            f"Return ONLY a JSON object matching this schema:\n{schema}\n\n"
            f"{graph_block}"
            f"{memory_block}"
            f"{recurring_block}"
            "--- STORIES ---\n\n" + "\n".join(lines)
        )
        raw = self._complete(
            entries if entries is not None else self._build_chain(),
            system=SYNTHESIS_SYSTEM,
            user=user,
            max_tokens=4000,
            kind="synthesis",
            deadline=deadline if deadline is not None else Deadline(0),
            result=result if result is not None else AnalysisResult(),
        )
        payload = parse_json_object(raw)
        payload = _normalise_synthesis(payload, stories, self.config.language)
        if payload.get("_source") != "llm" and result is not None:
            result.repaired += 1
        return payload


def _validate_links(payload: dict, story_count: int) -> dict:
    """Drop links the model invented, and links that point nowhere.

    A model asked to draw a graph will cheerfully emit ``S99`` or invent a
    "same_actor" between two stories with nothing in common. Rendering a
    dangling or nonsensical edge is worse than rendering no graph, because a
    reader cannot tell invented structure from real structure — so anything
    that does not check out is removed here, before it reaches the page.
    """
    links = payload.get("links")
    if not isinstance(links, list):
        payload["links"] = []
        return payload

    valid_ids = {f"S{i + 1}" for i in range(story_count)}
    kept: list[dict[str, str]] = []
    for link in links:
        if not isinstance(link, dict):
            continue
        src, dst = str(link.get("from", "")).strip(), str(link.get("to", "")).strip()
        kind = str(link.get("type", "")).strip().lower()
        if src not in valid_ids or dst not in valid_ids or src == dst:
            continue
        if kind not in LINK_TYPES:
            continue
        evidence = str(link.get("evidence", "")).strip()
        if not evidence:
            continue
        kept.append({"type": kind, "from": src, "to": dst, "evidence": evidence})
    payload["links"] = kept
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


def _coverage_evidence(story: Story) -> tuple[str, str]:
    """Summarise observable coverage differences without inferring motives."""
    articles = story.articles
    ar_count = sum(1 for article in articles if article.lang == "ar")
    en_count = sum(1 for article in articles if article.lang == "en")
    bodies = sum(1 for article in articles if (article.body or "").strip())
    english = [
        f"Coverage: {story.outlet_count} outlet(s), {ar_count} Arabic and "
        f"{en_count} English article(s); full text was fetched for {bodies} of "
        f"{len(articles)} article(s)."
    ]
    arabic = [
        f"التغطية: {story.outlet_count} وسيلة، منها {ar_count} مادة بالعربية "
        f"و{en_count} بالإنجليزية؛ وتوفر النص الكامل لـ{bodies} من أصل "
        f"{len(articles)} مادة."
    ]

    distinct_titles = {
        normalize_text(article.title) for article in articles if article.title
    }
    if len(distinct_titles) > 1:
        english.append(
            "Headline wording differs across outlets; the original versions are "
            "preserved in the references."
        )
        arabic.append("تختلف صياغة العناوين بين الوسائل؛ والعناوين الأصلية محفوظة في المراجع.")

    digit_map = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"
    )
    figures: list[tuple[str, list[str]]] = []
    for article in articles:
        title = article.title.translate(digit_map)
        values = list(dict.fromkeys(re.findall(r"(?<![\w])\d[\d,]*(?:\.\d+)?\s*%?", title)))
        if values:
            figures.append((article.source_name, values))
    if len(figures) > 1 and len({tuple(values) for _source, values in figures}) > 1:
        rendered = "; ".join(
            f"{source}: {', '.join(values)}" for source, values in figures[:4]
        )
        english.append(
            "Figures in headlines differ (" + rendered + "); this text comparison "
            "does not establish whether the figures refer to the same measure."
        )
        arabic.append(
            "تختلف الأرقام الواردة في العناوين (" + rendered + ")؛ ولا يثبت هذا "
            "المقارنة النصية وحدها أن الأرقام تتعلق بالمقياس نفسه."
        )

    dated = [article for article in articles if article.published is not None]
    if len(dated) > 1:
        try:
            earliest = min(dated, key=lambda article: article.published)
            latest = max(dated, key=lambda article: article.published)
            span_hours = (latest.published - earliest.published).total_seconds() / 3600
        except (TypeError, AttributeError):
            span_hours = 0.0
            earliest = latest = None
        if span_hours >= 2 and earliest is not None and latest is not None:
            english.append(
                f"Outlet publication timestamps span about {span_hours:.1f} hours "
                f"({earliest.source_name} {earliest.published:%H:%M} to "
                f"{latest.source_name} {latest.published:%H:%M}); timestamps alone "
                "do not establish an update or a causal sequence."
            )
            arabic.append(
                f"تتباعد أوقات النشر المسجلة بنحو {span_hours:.1f} ساعة "
                f"(من {earliest.source_name} {earliest.published:%H:%M} إلى "
                f"{latest.source_name} {latest.published:%H:%M})؛ ولا تثبت الأوقات "
                "وحدها وجود تحديث أو تسلسل سببي."
            )

    return " ".join(english), " ".join(arabic)


def _merge_model_analysis(
    raw: dict[str, Any], story: Story, language: str
) -> tuple[dict[str, Any], bool]:
    """Keep usable model fields and deterministically fill everything else."""
    fallback = extractive_analysis(story, language)
    if not isinstance(raw, dict) or raw.get("_source") == "extractive":
        return fallback, True

    primary_lang = story.primary.lang if story.articles else "en"
    target_lang = language if language in ("en", "ar") else (
        primary_lang if primary_lang in ("en", "ar") else "en"
    )
    aliases = {
        "summary": f"summary_{target_lang}",
        "why_it_matters": f"why_it_matters_{target_lang}",
        "between_the_lines": f"between_the_lines_{target_lang}",
        "watch_next": f"watch_next_{target_lang}",
        "headline": f"headline_{target_lang}",
    }
    candidate = dict(raw)
    for alias, canonical in aliases.items():
        if canonical not in candidate and isinstance(candidate.get(alias), str):
            candidate[canonical] = candidate[alias]

    claimed = raw.get("_llm_fields")
    allowed_claims = set(claimed) if isinstance(claimed, list) else None
    string_fields = {
        "headline_en", "headline_ar", "summary_en", "summary_ar",
        "why_it_matters_en", "why_it_matters_ar", "between_the_lines_en",
        "between_the_lines_ar", "watch_next_en", "watch_next_ar",
        "confidence_reason",
    }
    model_fields: dict[str, Any] = {}
    for key in string_fields:
        value = candidate.get(key)
        if (
            isinstance(value, str)
            and value.strip()
            and (allowed_claims is None or key in allowed_claims)
        ):
            model_fields[key] = truncate(value.strip(), 1800)

    confidence = candidate.get("confidence")
    if (
        isinstance(confidence, str)
        and confidence.strip().lower() in {"high", "medium", "low"}
        and (allowed_claims is None or "confidence" in allowed_claims)
    ):
        model_fields["confidence"] = confidence.strip().lower()

    for key in ("entities", "tags"):
        values = candidate.get(key)
        if allowed_claims is not None and key not in allowed_claims:
            continue
        if isinstance(values, str):
            values = re.split(r"[,;\n]", values)
        if isinstance(values, list):
            cleaned = [truncate(value.strip(), 100) for value in values
                       if isinstance(value, str) and value.strip()]
            if cleaned:
                model_fields[key] = cleaned[:20]

    substantive = set(model_fields) - {"entities", "tags"}
    if not substantive:
        return fallback, True

    merged = dict(fallback)
    merged.update(model_fields)
    merged["_llm_fields"] = sorted(model_fields)
    for locale, why_text, evidence_text in (
        (
            "en",
            "Evidence-only supplement: the model did not provide a complete rationale in English. This records coverage and category without inferring motives or subtext.",
            "No model subtext was available in English; observable coverage signals are listed separately, without inference.",
        ),
        (
            "ar",
            "استكمال قائم على الأدلة فقط: لم يقدم النموذج تعليلاً كاملاً بالعربية. يقتصر هذا النص على التغطية والتصنيف دون استنتاج الدوافع أو ما بين السطور.",
            "لم يتوفر تحليل استدلالي من النموذج بالعربية؛ وتُعرض إشارات التغطية المرصودة منفصلة دون استنتاج.",
        ),
    ):
        why_key = f"why_it_matters_{locale}"
        evidence_key = f"between_the_lines_{locale}"
        if why_key in merged and why_key not in model_fields:
            merged[why_key] = why_text
        if evidence_key in merged and evidence_key not in model_fields:
            merged[evidence_key] = evidence_text
    if "confidence_reason" not in model_fields:
        merged["confidence_reason"] = (
            "Confidence kept low; the deterministic completion does not estimate "
            "model confidence or infer interpretation."
        )
    required = (
        f"summary_{target_lang}",
        f"why_it_matters_{target_lang}",
        f"between_the_lines_{target_lang}",
        "confidence",
        "confidence_reason",
    )
    complete = all(field in model_fields for field in required)
    merged["_source"] = "llm" if complete else "hybrid"
    return merged, not complete


def _normalise_synthesis(
    raw: dict[str, Any], stories: list[Story], language: str
) -> dict[str, Any]:
    """Validate a partial synthesis and fill omissions with safe coverage facts."""
    fields = (
        "editor_note_en", "editor_note_ar", "tactical_en", "tactical_ar",
        "strategic_en", "strategic_ar", "not_being_said_en",
        "not_being_said_ar", "watchlist_en", "watchlist_ar",
    )
    model_fields = {
        field: truncate(value.strip(), 2400)
        for field in fields
        if isinstance((value := raw.get(field)), str) and value.strip()
    }
    if not model_fields:
        fallback = extractive_synthesis(stories, language)
        fallback["_fallback_reason"] = "model response did not contain synthesis fields"
        return fallback

    merged = dict(extractive_synthesis(stories, language))
    merged.update(model_fields)
    valid_ids = {f"S{i + 1}" for i in range(len(stories))}
    story_of_day = str(raw.get("story_of_the_day", "")).strip()
    if story_of_day in valid_ids:
        merged["story_of_the_day"] = story_of_day

    themes = raw.get("top_themes")
    if isinstance(themes, list):
        valid_themes = []
        for theme in themes:
            if not isinstance(theme, dict):
                continue
            safe_theme = {
                key: truncate(value.strip(), 500)
                for key, value in theme.items()
                if key in {"theme_en", "theme_ar", "note_en", "note_ar"}
                and isinstance(value, str) and value.strip()
            }
            ids = theme.get("story_ids")
            if isinstance(ids, list):
                safe_theme["story_ids"] = [
                    str(sid).strip() for sid in ids if str(sid).strip() in valid_ids
                ]
            if safe_theme.get("theme_en") or safe_theme.get("theme_ar"):
                valid_themes.append(safe_theme)
        if valid_themes:
            merged["top_themes"] = valid_themes[:8]

    merged["links"] = raw.get("links") if isinstance(raw.get("links"), list) else []
    requested_lang = language if language in ("en", "ar") else "en"
    required = (
        f"editor_note_{requested_lang}",
        f"not_being_said_{requested_lang}",
        f"watchlist_{requested_lang}",
    )
    complete = all(field in model_fields for field in required)
    merged["_source"] = "llm" if complete else "hybrid"
    merged["_llm_fields"] = sorted(model_fields)
    return merged


def extractive_analysis(story: Story, language: str) -> dict[str, Any]:
    """Evidence-only story analysis for outages, malformed answers, and no-key runs."""
    primary = story.primary
    best_copy = next(
        (article for article in story.articles if (article.body or article.summary).strip()),
        primary,
    )
    text = best_copy.body or best_copy.summary or ""
    summary = _first_sentences(text)
    summary = summary or primary.title
    evidence_en, evidence_ar = _coverage_evidence(story)
    category = _category_of(story)
    note = (
        "Analysed without an AI provider. This evidence-only fallback records "
        f"the story's category ({category}) and coverage across "
        f"{', '.join(story.sources)}, but does not infer causes, motives, or subtext."
    )

    item: dict[str, Any] = {
        "headline_en": primary.title if primary.lang == "en" else "",
        "headline_ar": primary.title if primary.lang == "ar" else "",
        "confidence": "low",
        "confidence_reason": "Evidence-only fallback; no model evaluated interpretation.",
        "entities": [],
        "tags": list(dict.fromkeys([category, *story.sections]))[:5],
        "coverage_evidence_en": evidence_en,
        "coverage_evidence_ar": evidence_ar,
        "_source": "extractive",
    }
    summary_lang = best_copy.lang if best_copy.lang in ("en", "ar") else "en"
    item[f"summary_{summary_lang}"] = summary
    if language in ("bilingual", "en"):
        item["why_it_matters_en"] = note
        item["between_the_lines_en"] = (
            "No AI provider was available; no subtext is inferred in deterministic mode. The separate evidence "
            "signals below describe only observable outlet coverage, headline "
            "variation, figures, and publication timestamps."
        )
        item["watch_next_en"] = "No predictive watchlist is generated in evidence-only mode."
    if language in ("bilingual", "ar"):
        item["why_it_matters_ar"] = (
            "تم التحليل دون مزود ذكاء اصطناعي. يحدد هذا البديل فئة الخبر ونطاق "
            "تغطيته، ولا يستنتج الأسباب أو الدوافع أو ما بين السطور."
        )
        item["between_the_lines_ar"] = (
            "لا تُستنتج دلالات ضمن هذا الوضع؛ وتعرض إشارات التغطية المنفصلة "
            "اختلافات العناوين والأرقام وأوقات النشر المرصودة فقط."
        )
        item["watch_next_ar"] = "لا تُنشأ قائمة تنبؤية للمتابعة في وضع الأدلة فقط."
    return item


def extractive_synthesis(stories: list[Story], language: str) -> dict[str, Any]:
    indexed = list(enumerate(stories))
    top_index, top = max(
        indexed,
        key=lambda row: (row[1].outlet_count, row[1].score, -row[0]),
        default=(-1, None),
    )
    majors = sum(1 for story in stories if story.outlet_count >= 3)
    outlet_names = {name for story in stories for name in story.sources}
    category_ar = {
        "Government & Politics": "الحكومة والسياسة",
        "Economy & Business": "الاقتصاد والأعمال",
        "Security & Courts": "الأمن والمحاكم",
        "Society & Services": "المجتمع والخدمات",
        "Sport": "الرياضة",
        "World & Region": "العالم والمنطقة",
        "Culture & Entertainment": "الثقافة والترفيه",
        "Opinion & Analysis": "الرأي والتحليل",
        "Other": "أخرى",
    }
    categories: dict[str, list[int]] = {}
    for index, story in enumerate(stories, start=1):
        categories.setdefault(_category_of(story), []).append(index)
    top_themes = []
    for category, ids in sorted(categories.items(), key=lambda row: (-len(row[1]), row[0])):
        if len(ids) < 2 or len(top_themes) >= 4:
            continue
        top_themes.append({
            "theme_en": category,
            "theme_ar": category_ar.get(category, "أخرى"),
            "story_ids": [f"S{i}" for i in ids],
            "note_en": f"{len(ids)} stories share this desk classification; it is a topic grouping, not a causal link.",
            "note_ar": f"تندرج {len(ids)} أخبار ضمن هذا التصنيف التحريري؛ وهذا تجميع موضوعي لا يثبت علاقة سببية.",
        })

    title = top.primary.title if top else ""
    note_en = (
        f"Coverage map: {len(stories)} distinct story cluster(s) across "
        f"{len(outlet_names)} outlet(s); {majors} cluster(s) were carried by "
        f"three or more outlets. "
        + (f"The broadest coverage was for: “{title}”. " if top else "")
        + f"The collected material includes {sum(1 for story in stories for a in story.articles if a.lang == 'ar')} Arabic "
        f"and {sum(1 for story in stories for a in story.articles if a.lang == 'en')} English article(s). "
        "This deterministic view reports observed coverage only; it does not infer motives, omissions, or future outcomes."
    )
    note_ar = (
        f"خريطة التغطية: {len(stories)} مجموعة خبرية عبر {len(outlet_names)} وسيلة؛ "
        f"غطّت ثلاث وسائل أو أكثر {majors} مجموعة. "
        + (f"وكانت أوسع تغطية للخبر: «{title}». " if top else "")
        + f"تتضمن المواد {sum(1 for story in stories for a in story.articles if a.lang == 'ar')} مادة بالعربية "
        f"و{sum(1 for story in stories for a in story.articles if a.lang == 'en')} بالإنجليزية. "
        "تعرض هذه الخريطة التغطية المرصودة فقط ولا تستنتج دوافع أو omissions أو نتائج مستقبلية."
    )
    payload: dict[str, Any] = {
        "editor_note_en": note_en if language != "ar" else "",
        "editor_note_ar": note_ar if language != "en" else "",
        "story_of_the_day": f"S{top_index + 1}" if top else "",
        "top_themes": top_themes,
        "not_being_said_en": (
            "No claims about omissions are made in evidence-only mode. The themes above are counts of collected stories, not conclusions about what was deliberately left out."
            if language != "ar" else ""
        ),
        "not_being_said_ar": (
            "لا تُطرح ادعاءات بشأن المحذوفات في وضع الأدلة فقط؛ والموضوعات أعلاه أعداد للمواد المجموعة وليست استنتاجات عمّا جرى إغفاله عمداً."
            if language != "en" else ""
        ),
        "watchlist_en": "No predictive watchlist is generated in evidence-only mode." if language != "ar" else "",
        "watchlist_ar": "لا تُنشأ قائمة تنبؤية للمتابعة في وضع الأدلة فقط." if language != "en" else "",
        "links": [],
        "_source": "extractive",
    }
    return payload
