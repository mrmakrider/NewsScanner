"""Resilience policy for the unattended LLM calls.

The morning brief runs on a schedule with nobody watching, so every failure
mode has to be handled by policy rather than by someone noticing. These are
the levers the analysis engine pulls, kept free of network code so the policy
can be reasoned about — and tested — in isolation.

``Pacer``
    Spacing between requests to one provider, adjusted the way TCP adjusts its
    window: multiplicative decrease when the provider pushes back, gentle
    recovery when it does not. A free gateway that rate-limits per IP is the
    motivating case — hammering it is what turns a soft throttle into a hard
    failure.

``CircuitBreaker``
    A per-provider "stop hitting this" latch. Once a provider fails
    repeatedly it is skipped for a cooldown that doubles each time it trips,
    so a provider that is genuinely down cannot eat the run's whole budget.

``Deadline``
    A wall-clock budget for the analysis step. The GitHub job is killed at 30
    minutes; a budget means the digest still gets written and committed when
    that happens, instead of dying mid-call.

``parse_retry_after`` / ``retry_after_of`` / ``sleep_for``
    Backoff that reads the delay the server actually asked for, rather than
    assuming a fixed curve. Gateways are usually explicit about this and it is
    rude — and counter-productive — to ignore them.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Retry-After
# --------------------------------------------------------------------------


def parse_retry_after(value: str | None) -> float | None:
    """Seconds to wait, from either form RFC 9110 allows.

    ``Retry-After`` is either a delta-seconds count or an HTTP-date. Both show
    up in the wild: Cloudflare sends seconds, some APIs send a date.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def retry_after_of(exc: BaseException) -> float | None:
    """The delay a provider asked for, if the transport captured one."""
    value = getattr(exc, "retry_after", None)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, str):
        return parse_retry_after(value)
    return None


def classify_failure(exc: BaseException) -> str:
    """What kind of trouble a provider is in.

    The distinction decides whether patience is the right response. A throttle
    clears on its own, so waiting works. A refused connection or a bad key will
    not be fixed by sitting still — those want a *different* request (smaller,
    or to another provider) rather than a longer wait, and a run that waits
    anyway just burns its budget.

    Returns one of ``"throttled"``, ``"unreachable"``, ``"invalid"`` or
    ``"unknown"``.
    """
    if retry_after_of(exc) is not None:
        return "throttled"
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        if code == 429 or 500 <= code <= 599 or code in (408, 425):
            return "throttled"
        return "invalid"
    text = str(exc).lower()
    for marker in (
        "urlopen error",
        "connection refused",
        "connection reset",
        "timed out",
        "timeout",
        "name or service not known",
        "temporary failure in name resolution",
        "remote end closed",
        "ssl",
    ):
        if marker in text:
            return "unreachable"
    return "unknown"


def sleep_for(
    attempt: int,
    *,
    base: float = 1.5,
    cap: float = 30.0,
    jitter: float = 0.4,
    retry_after: float | None = None,
) -> float:
    """Seconds to wait before retry ``attempt`` (1-based).

    An explicit ``Retry-After`` wins over the curve — the server knows its own
    limits — but is still capped, because a gateway advertising an hour of
    silence must not stall the run.
    """
    if retry_after is not None:
        return max(0.0, min(float(retry_after), cap))
    window = min(cap, base * (2 ** (max(1, attempt) - 1)))
    return window * (1.0 + random.uniform(0, max(0.0, jitter)))


# --------------------------------------------------------------------------
# Wall-clock budget
# --------------------------------------------------------------------------


class Deadline:
    """A shared wall-clock budget, in seconds. ``0`` means unlimited."""

    def __init__(self, seconds: float = 0.0, clock=time.monotonic):
        self.budget = max(0.0, float(seconds or 0.0))
        self._clock = clock
        self._start = clock()

    @property
    def elapsed(self) -> float:
        return self._clock() - self._start

    @property
    def remaining(self) -> float:
        if not self.budget:
            return float("inf")
        return max(0.0, self.budget - self.elapsed)

    @property
    def unlimited(self) -> bool:
        return not self.budget

    def expired(self) -> bool:
        return not self.unlimited and self.remaining <= 0.0

    def clamp(self, wait: float) -> float:
        """Trim a proposed sleep so it cannot outlive the budget."""
        if self.unlimited:
            return wait
        return max(0.0, min(wait, self.remaining))


# --------------------------------------------------------------------------
# Pacing
# --------------------------------------------------------------------------


class Pacer:
    """Minimum spacing between calls to one provider, with AIMD adjustment.

    ``on_throttle`` doubles the interval; every success decays it back toward
    the floor. The floor is where a provider is known to be comfortable (zero
    for a paid API, a few seconds for a keyless free tier), so a healthy run
    pays nothing and a throttled one slows down before it gets blocked.

    The interval is jittered at dispatch time so that two runs starting
    together — the daily job and a manual retry, say — do not stay in lockstep
    and hit the gateway as one burst.
    """

    def __init__(
        self,
        floor: float = 0.0,
        ceiling: float = 20.0,
        decay: float = 0.7,
        jitter: float = 0.25,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.floor = max(0.0, float(floor))
        self.ceiling = max(self.floor, float(ceiling))
        self.decay = min(1.0, max(0.0, float(decay)))
        self.jitter = max(0.0, float(jitter))
        self._sleep = sleep
        self._clock = clock
        self._interval = self.floor
        self._last: float | None = None
        self.requests = 0
        self.throttles = 0

    @property
    def interval(self) -> float:
        return self._interval

    def wait(self, deadline: Deadline | None = None) -> float:
        """Block until this provider may be called again. Returns the delay."""
        delay = 0.0
        if self._last is not None and self._interval > 0:
            window = self._interval * (1.0 + random.uniform(0, self.jitter))
            delay = self._last + window - self._clock()
            if delay > 0:
                if deadline is not None:
                    delay = deadline.clamp(delay)
                if delay > 0:
                    self._sleep(delay)
        self._last = self._clock()
        self.requests += 1
        return max(0.0, delay)

    def on_success(self) -> None:
        """Ease back toward the floor. Proportional, so full speed returns
        after a handful of good calls instead of a hundred additive steps."""
        if self._interval > self.floor:
            self._interval = max(self.floor, self._interval * self.decay)

    def on_throttle(self, retry_after: float | None = None) -> None:
        """Back off. An explicit server-side wait sets the new interval."""
        self.throttles += 1
        doubled = (self._interval * 2) if self._interval > 0 else 1.0
        if retry_after is not None:
            doubled = max(doubled, float(retry_after))
        self._interval = min(self.ceiling, max(self.floor, doubled))

    def reset(self) -> None:
        self._interval = self.floor
        self._last = None


# --------------------------------------------------------------------------
# Circuit breaking
# --------------------------------------------------------------------------


class CircuitBreaker:
    """Skip a provider that keeps failing, and probe it again later.

    ``allow()`` is the gate; a caller that gets ``True`` after the breaker has
    tripped is running the half-open probe. A success closes the circuit, a
    failure re-opens it for twice as long, up to ``max_cooldown``.
    """

    def __init__(
        self,
        threshold: int = 3,
        cooldown: float = 45.0,
        factor: float = 2.0,
        max_cooldown: float = 900.0,
        clock=time.monotonic,
    ):
        self.threshold = max(1, int(threshold))
        self.cooldown = max(0.0, float(cooldown))
        self.factor = max(1.0, float(factor))
        self.max_cooldown = max(self.cooldown, float(max_cooldown))
        self._clock = clock
        self.failures = 0
        self.trips = 0
        self._open_until = 0.0

    @property
    def open(self) -> bool:
        return self._clock() < self._open_until

    @property
    def remaining(self) -> float:
        return max(0.0, self._open_until - self._clock())

    @property
    def state(self) -> str:
        if self.open:
            return "open"
        return "half-open" if self.trips else "closed"

    def allow(self) -> bool:
        return not self.open

    def record_success(self) -> None:
        self.failures = 0
        self.trips = 0
        self._open_until = 0.0

    def record_failure(self) -> float:
        """Count a failure. Returns the cooldown if this tripped the breaker."""
        self.failures += 1
        if self.failures < self.threshold:
            return 0.0
        self.failures = 0
        self.trips += 1
        wait = min(self.max_cooldown, self.cooldown * (self.factor ** (self.trips - 1)))
        self._open_until = self._clock() + wait
        return wait
