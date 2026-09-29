"""Tests for the unattended-run resilience policy.

Run with:  python3 -m unittest discover -s tests -v

Everything here is offline. The primitives are tested against fake clocks, so
the backoff curves and cooldowns can be asserted exactly, and the engine is
tested against a stubbed model, so the behaviour that matters — failing over,
splitting a batch, refusing to retry what cannot be fixed — is deterministic
rather than timing-dependent.

The scenario these protect is the one that actually happened: the keyless
gateway answered the first two batches and then returned 429 for every call
after that, and the run quietly produced a brief in which 188 of 216 stories
carried no analysis at all.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from news_scanner.analyze import (  # noqa: E402
    AnalysisConfig,
    AnalysisEngine,
    detect_candidates,
    detect_provider,
)
from news_scanner.http import FetchError  # noqa: E402
from news_scanner.models import Article, Story  # noqa: E402
from news_scanner.resilience import (  # noqa: E402
    CircuitBreaker,
    Deadline,
    Pacer,
    classify_failure,
    parse_retry_after,
    retry_after_of,
    sleep_for,
)

NOW = datetime(2026, 9, 30, 8, 0, tzinfo=timezone(timedelta(hours=3)))

# Every provider key, so the chain under test is the one the test describes
# rather than whatever the surrounding environment happens to have set.
PROVIDER_ENV = (
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY", "GROQ_API_KEY", "LLM7_API_KEY", "NEWSCANNER_LLM7_KEY",
    "OLLAMA_HOST", "NEWSCANNER_USE_OLLAMA", "NEWSCANNER_NO_LLM7",
    "NEWSCANNER_PROVIDER", "NEWSCANNER_MODEL", "NEWSCANNER_BASE_URL",
)

STORY_FIELDS = {
    "headline_en": "Cabinet approves the budget",
    "summary_en": "The Cabinet approved the draft budget.",
    "why_it_matters_en": "Spending plans are now public.",
    "between_the_lines_en": "No minister is named in the statement.",
    "confidence": "medium",
    "confidence_reason": "Official statement only.",
}

CANNED = {
    "stories": [dict(STORY_FIELDS, id=f"S{i}") for i in range(1, 9)],
    "editor_note_en": "The budget led the day.",
    "not_being_said_en": "Nothing on the cost of the new spending.",
    "watchlist_en": "Where the budget goes next",
}


TITLES = [
    "Cabinet approves the draft budget for the coming year",
    "Crown Prince returns from an official visit to Switzerland",
    "Oil prices hold above seventy dollars a barrel",
    "Municipality sets a date for the local elections",
    "Health ministry opens a new clinic in Jahra",
    "Airport passenger numbers rise over the summer",
    "Court issues its ruling in the contractor dispute",
    "National football team signs a new goalkeeper",
]


def make_story(index: int) -> Story:
    """A one-article story, which is all the analysis path needs.

    The titles are deliberately unlike each other: ``Story.key`` is a title
    fingerprint, so near-identical titles would silently merge into one key
    and hide exactly the per-story bugs these tests are looking for.
    """
    story = Story()
    story.add(
        Article(
            source_id="alrai",
            source_name="الرأي",
            lang="en",
            title=TITLES[index % len(TITLES)],
            url=f"https://example.com/{index}",
            summary="A summary of what happened.",
            body="A body paragraph long enough to be sent to a model.",
            published=NOW - timedelta(hours=index),
            tier=1,
        )
    )
    return story


def make_stories(count: int) -> list[Story]:
    return [make_story(i) for i in range(count)]


def answer(prompt: str) -> str:
    """Answer a prompt the way a well-behaved model would.

    Batch prompts name their stories ``=== S3 ===``; the synthesis prompt asks
    for the schema fields instead. Returning exactly the stories asked about is
    what makes the split test meaningful — a lazy stub that always answers with
    everything would pass whether or not the split preserved identifiers.
    """
    if "editor_note_en" in prompt:
        return json.dumps(CANNED, ensure_ascii=False)
    ids = re.findall(r"^=== (S\d+) ===$", prompt, re.MULTILINE)
    payload = dict(CANNED)
    payload["stories"] = [s for s in CANNED["stories"] if s["id"] in ids]
    return json.dumps(payload, ensure_ascii=False)


# ------------------------------------------------------------------ primitives


class TestRetryAfter(unittest.TestCase):
    def test_delta_seconds(self):
        self.assertEqual(parse_retry_after("12"), 12.0)
        self.assertEqual(parse_retry_after(" 0 "), 0.0)

    def test_http_date(self):
        future = datetime.now(timezone.utc) + timedelta(seconds=30)
        parsed = parse_retry_after(future.strftime("%a, %d %b %Y %H:%M:%S GMT"))
        self.assertIsNotNone(parsed)
        self.assertGreater(parsed, 25)
        self.assertLessEqual(parsed, 31)

    def test_past_date_and_rubbish(self):
        self.assertIsNone(parse_retry_after(None))
        self.assertIsNone(parse_retry_after(""))
        self.assertIsNone(parse_retry_after("not a date"))
        past = datetime.now(timezone.utc) - timedelta(seconds=30)
        self.assertEqual(parse_retry_after(past.strftime("%a, %d %b %Y %H:%M:%S GMT")), 0.0)

    def test_read_from_an_exception(self):
        self.assertEqual(retry_after_of(FetchError("x", retry_after=7.5)), 7.5)
        self.assertEqual(retry_after_of(FetchError("x", retry_after="3")), 3.0)
        self.assertIsNone(retry_after_of(FetchError("x")))
        self.assertIsNone(retry_after_of(FetchError("x", retry_after=True)))

    def test_explicit_retry_after_beats_the_curve_but_is_capped(self):
        self.assertEqual(sleep_for(5, retry_after=4.0), 4.0)
        self.assertEqual(sleep_for(1, retry_after=9999.0, cap=30.0), 30.0)

    def test_backoff_grows(self):
        self.assertEqual(sleep_for(1, base=1.0, jitter=0.0), 1.0)
        self.assertEqual(sleep_for(2, base=1.0, jitter=0.0), 2.0)
        self.assertEqual(sleep_for(3, base=1.0, jitter=0.0), 4.0)


class TestClassifyFailure(unittest.TestCase):
    def test_rate_limit_is_throttling(self):
        self.assertEqual(classify_failure(FetchError("429", code=429)), "throttled")
        self.assertEqual(classify_failure(FetchError("x", retry_after=5.0)), "throttled")

    def test_server_faults_count_as_throttling_because_they_clear(self):
        for code in (500, 502, 503, 504, 522, 524):
            self.assertEqual(classify_failure(FetchError("x", code=code)), "throttled")

    def test_a_rejected_key_is_not_worth_waiting_for(self):
        self.assertEqual(classify_failure(FetchError("401", code=401)), "invalid")
        self.assertEqual(classify_failure(FetchError("404", code=404)), "invalid")

    def test_a_refused_connection_is_unreachable(self):
        exc = FetchError("http://127.0.0.1:9/v1: <urlopen error [Errno 111] Connection refused>")
        self.assertEqual(classify_failure(exc), "unreachable")


class TestPacer(unittest.TestCase):
    def _clocked(self, floor: float = 0.0):
        ticks = {"now": 0.0}
        slept: list[float] = []

        def sleep(seconds):
            slept.append(seconds)
            ticks["now"] += seconds

        return Pacer(floor=floor, jitter=0.0, sleep=sleep, clock=lambda: ticks["now"]), slept

    def test_the_first_call_is_immediate(self):
        pacer, slept = self._clocked(floor=5.0)
        pacer.wait()
        self.assertEqual(slept, [], "the first call should not wait on a floor")

    def test_the_floor_is_respected_between_calls(self):
        pacer, slept = self._clocked(floor=5.0)
        pacer.wait()
        pacer.wait()
        self.assertEqual(len(slept), 1)
        self.assertAlmostEqual(slept[0], 5.0, places=6)

    def test_a_throttle_doubles_the_interval(self):
        pacer, _slept = self._clocked(floor=1.0)
        pacer.on_success()
        self.assertEqual(pacer.interval, 1.0)
        pacer.on_throttle()
        self.assertEqual(pacer.interval, 2.0)
        pacer.on_throttle()
        self.assertEqual(pacer.interval, 4.0)

    def test_an_explicit_retry_after_sets_the_interval_within_the_ceiling(self):
        pacer = Pacer(floor=1.0, ceiling=60.0, jitter=0.0)
        pacer.on_throttle(retry_after=30.0)
        self.assertEqual(pacer.interval, 30.0)
        pacer.on_throttle(retry_after=9999.0)
        self.assertEqual(pacer.interval, 60.0, "the ceiling still applies")

    def test_the_ceiling_caps_an_ordinary_backoff(self):
        pacer = Pacer(floor=1.0, ceiling=20.0, jitter=0.0)
        for _ in range(6):
            pacer.on_throttle()
        self.assertEqual(pacer.interval, 20.0)

    def test_the_wait_is_jittered(self):
        ticks = {"now": 0.0}
        slept: list[float] = []
        pacer = Pacer(
            floor=10.0,
            jitter=0.25,
            sleep=lambda seconds: (slept.append(seconds), ticks.update(now=ticks["now"] + seconds)),
            clock=lambda: ticks["now"],
        )
        pacer.wait()
        pacer.wait()
        self.assertGreaterEqual(slept[0], 10.0)
        self.assertLessEqual(slept[0], 12.5)

    def test_success_eases_back_toward_the_floor(self):
        pacer, _slept = self._clocked(floor=1.0)
        pacer.on_throttle(retry_after=20.0)
        for _ in range(10):
            pacer.on_success()
        self.assertAlmostEqual(pacer.interval, 1.0, places=6)

    def test_reset_makes_the_next_call_immediate(self):
        pacer, slept = self._clocked(floor=5.0)
        pacer.wait()
        pacer.reset()
        pacer.wait()
        self.assertEqual(slept, [], "reset should clear the pending wait")


class TestCircuitBreaker(unittest.TestCase):
    def _breaker(self, **kwargs):
        ticks = {"now": 0.0}
        return CircuitBreaker(clock=lambda: ticks["now"], **kwargs), ticks

    def test_it_opens_only_after_the_threshold(self):
        breaker, _ticks = self._breaker(threshold=3, cooldown=60.0)
        breaker.record_failure()
        breaker.record_failure()
        self.assertTrue(breaker.allow(), "two failures should not bench a provider")
        breaker.record_failure()
        self.assertFalse(breaker.allow())
        self.assertEqual(breaker.state, "open")

    def test_it_reopens_for_a_probe_then_closes_on_success(self):
        breaker, ticks = self._breaker(threshold=1, cooldown=30.0)
        breaker.record_failure()
        self.assertFalse(breaker.allow())
        ticks["now"] = 31.0
        self.assertTrue(breaker.allow(), "the cooldown should have elapsed")
        self.assertEqual(breaker.state, "half-open")
        breaker.record_success()
        self.assertEqual(breaker.state, "closed")

    def test_each_trip_doubles_the_cooldown(self):
        breaker, ticks = self._breaker(threshold=1, cooldown=10.0, factor=2.0)
        self.assertEqual(breaker.record_failure(), 10.0)
        ticks["now"] = 11.0
        self.assertEqual(breaker.record_failure(), 20.0)
        ticks["now"] = 32.0
        self.assertEqual(breaker.record_failure(), 40.0)

    def test_the_cooldown_is_capped(self):
        breaker, ticks = self._breaker(threshold=1, cooldown=10.0, max_cooldown=25.0)
        breaker.record_failure()
        ticks["now"] = 11.0
        breaker.record_failure()
        ticks["now"] = 40.0
        self.assertEqual(breaker.record_failure(), 25.0)


class TestDeadline(unittest.TestCase):
    def test_unlimited_by_default(self):
        deadline = Deadline(0)
        self.assertTrue(deadline.unlimited)
        self.assertFalse(deadline.expired())
        self.assertEqual(deadline.clamp(9999.0), 9999.0)

    def test_it_expires(self):
        ticks = {"now": 0.0}
        deadline = Deadline(10.0, clock=lambda: ticks["now"])
        self.assertFalse(deadline.expired())
        ticks["now"] = 11.0
        self.assertTrue(deadline.expired())
        self.assertEqual(deadline.remaining, 0.0)

    def test_clamp_never_sleeps_past_the_budget(self):
        ticks = {"now": 0.0}
        deadline = Deadline(10.0, clock=lambda: ticks["now"])
        self.assertEqual(deadline.clamp(9999.0), 10.0)
        ticks["now"] = 6.0
        self.assertAlmostEqual(deadline.clamp(9999.0), 4.0, places=6)


# ---------------------------------------------------------------------- chain


class TestProviderChain(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in PROVIDER_ENV}

    def tearDown(self):
        for key in PROVIDER_ENV:
            os.environ.pop(key, None)
        for key, value in self._saved.items():
            if value is not None:
                os.environ[key] = value

    def test_the_keyless_default_offers_its_fast_route_too(self):
        chain = detect_candidates("auto")
        self.assertEqual([(c.provider, c.model) for c in chain],
                         [("llm7", "default"), ("llm7", "fast")])
        self.assertEqual(detect_provider("auto"), ("llm7", "default", ""))

    def test_a_key_comes_first_and_the_gateway_is_the_safety_net(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        chain = detect_candidates("auto")
        self.assertEqual(chain[0].provider, "gemini")
        self.assertIn("llm7", [c.provider for c in chain])
        self.assertEqual(detect_provider("auto")[0], "gemini")

    def test_the_safety_net_can_be_switched_off(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["NEWSCANNER_NO_LLM7"] = "1"
        self.assertEqual([c.provider for c in detect_candidates("auto")], ["gemini"])

    def test_a_pinned_model_suppresses_the_fast_variant(self):
        os.environ["NEWSCANNER_MODEL"] = "pro"
        self.assertEqual([c.model for c in detect_candidates("auto")], ["pro"])

    def test_no_providers_at_all_is_an_empty_chain(self):
        os.environ["NEWSCANNER_NO_LLM7"] = "1"
        self.assertEqual(detect_candidates("auto"), [])

    def test_a_pinned_provider_does_not_pull_in_another_key(self):
        """Pinning a provider must not quietly spend a different quota."""
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["GROQ_API_KEY"] = "test-key"
        self.assertEqual(
            [c.provider for c in detect_candidates("groq")], ["groq"],
            "the pinned provider is the whole chain",
        )


# --------------------------------------------------------------------- engine


class EngineCase(unittest.TestCase):
    """Keeps the environment, the network and the cache out of the way.

    ``NEWSCANNER_BASE_URL`` points at a closed port, so a test that forgets to
    stub the model fails immediately instead of calling the real gateway.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_path = Path(self._tmp.name) / "cache.json"

        self._saved = {k: os.environ.pop(k, None) for k in PROVIDER_ENV}
        os.environ["NEWSCANNER_BASE_URL"] = "http://127.0.0.1:9/v1"
        self.addCleanup(self._restore)

    def _restore(self):
        for key in PROVIDER_ENV:
            os.environ.pop(key, None)
        for key, value in self._saved.items():
            if value is not None:
                os.environ[key] = value

    def engine(self, **overrides) -> AnalysisEngine:
        config = AnalysisConfig(
            provider="llm7",
            use_cache=False,
            cache_path=None,
            # Tests must not sit through production backoff.
            pass_cooldown=0.0,
            breaker_cooldown=0.0,
            min_interval=0.0,
            max_interval=0.0,
            incremental_cache=False,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return AnalysisEngine(config)


class TestEngineFailover(EngineCase):
    def test_a_throttled_primary_fails_over_to_the_next_provider(self):
        """What the chain is for: a 429 stops costing the day's analysis."""
        calls: list[str] = []

        def fake_complete(self, system, user, *, max_tokens=8000):
            calls.append(self.provider)
            if self.provider == "gemini":
                raise FetchError("HTTP Error 429", code=429, retry_after=0.0)
            return answer(user)

        os.environ["GEMINI_API_KEY"] = "test-key"
        with mock.patch("news_scanner.analyze.LLMClient.complete", fake_complete):
            engine = self.engine(provider="auto")
            self.assertEqual(engine.provider, "gemini")
            result = engine.analyse(make_stories(2))

        self.assertIn("gemini", calls)
        self.assertIn("llm7", calls, "the run should have fallen back to the gateway")
        self.assertEqual(result.provider, "llm7", "the report names who did the work")
        self.assertEqual(result.providers_used, ["llm7/default"])
        self.assertTrue(all(a.get("_source") == "llm" for a in result.stories.values()))
        self.assertFalse(result.errors, result.errors)

    def test_a_story_the_model_skips_falls_back_without_an_outage_report(self):
        def fake_complete(self, system, user, *, max_tokens=8000):
            if "editor_note_en" in user:
                return json.dumps(CANNED, ensure_ascii=False)
            # Answer for the first story only, as a distracted model would.
            return json.dumps({"stories": [dict(STORY_FIELDS, id="S1")]})

        with mock.patch("news_scanner.analyze.LLMClient.complete", fake_complete):
            result = self.engine().analyse(make_stories(2))

        self.assertEqual(result.missing, 1)
        self.assertFalse(result.errors, "a skipped story is not a provider outage")
        sources = sorted(str(a.get("_source")) for a in result.stories.values())
        self.assertEqual(sources, ["extractive", "llm"])


class TestEngineSplitting(EngineCase):
    def test_a_batch_too_big_for_the_gateway_is_split_not_abandoned(self):
        """A 524 on a four-story prompt is fixed by asking about two at a time."""
        sizes: list[int] = []

        def fake_complete(self, system, user, *, max_tokens=8000):
            size = user.count("=== S")
            if size:
                sizes.append(size)
            if size > 2:
                raise FetchError("HTTP Error 524", code=524)
            return answer(user)

        stories = make_stories(4)
        with mock.patch("news_scanner.analyze.LLMClient.complete", fake_complete):
            result = self.engine(batch_size=4).analyse(stories)

        self.assertEqual(sizes[0], 4, "the run should start at the configured batch size")
        self.assertEqual(sizes[-2:], [2, 2], "the final attempts should be the halves")
        self.assertGreaterEqual(result.splits, 1)
        self.assertEqual(len(result.stories), len(stories))
        self.assertTrue(
            all(a.get("_source") == "llm" for a in result.stories.values()),
            "every story should be analysed once the batch has been split",
        )
        self.assertFalse(result.errors, result.errors)

    def test_a_split_keeps_each_story_attached_to_its_own_answer(self):
        """Identifiers must survive the split, or answers land on wrong stories."""
        seen: dict[str, str] = {}

        def fake_complete(self, system, user, *, max_tokens=8000):
            ids = re.findall(r"^=== (S\d+) ===$", user, re.MULTILINE)
            if len(ids) > 1:
                raise FetchError("HTTP Error 524", code=524)
            if not ids:
                return answer(user)  # the synthesis call
            block = user.split(f"=== {ids[0]} ===", 1)[1]
            headline = re.search(r"^\s+- \[[^\]]+\] (.+)$", block, re.MULTILINE).group(1)
            seen[ids[0]] = headline
            return json.dumps({"stories": [dict(STORY_FIELDS, id=ids[0], headline_en=headline)]})

        stories = make_stories(2)
        with mock.patch("news_scanner.analyze.LLMClient.complete", fake_complete):
            result = self.engine(batch_size=2).analyse(stories)

        self.assertEqual(sorted(seen), ["S1", "S2"])
        for story in stories:
            self.assertEqual(
                result.stories[story.key]["headline_en"], story.primary.title,
                "each story must carry back the headline that was sent for it",
            )


class TestEngineDegradation(EngineCase):
    def test_an_unreachable_provider_gives_up_instead_of_waiting(self):
        """Retrying a host that is not listening buys nothing."""
        result = self.engine(retry_passes=1, call_attempts=2).analyse(make_stories(2))
        self.assertTrue(result.errors, "the failure must be reported")
        self.assertEqual(len(result.stories), 2)
        self.assertTrue(
            all(a.get("_source") != "llm" for a in result.stories.values()),
            "with no provider, every story needs the extractive fallback",
        )
        self.assertFalse(result.providers_used)

    def test_a_spent_budget_degrades_instead_of_overrunning(self):
        result = self.engine(max_llm_seconds=0.0001).analyse(make_stories(3))
        self.assertEqual(len(result.stories), 3)
        self.assertTrue(all(a.get("_source") != "llm" for a in result.stories.values()))

    def test_a_failing_synthesis_still_leaves_a_readable_digest(self):
        """The synthesis is one big call; losing it must not cost the rest."""
        def fake_complete(self, system, user, *, max_tokens=8000):
            if "editor_note_en" in user:
                raise FetchError("HTTP Error 524", code=524)
            return answer(user)

        with mock.patch("news_scanner.analyze.LLMClient.complete", fake_complete):
            result = self.engine(call_attempts=1).analyse(make_stories(2))

        self.assertTrue(result.synthesis.get("editor_note_en"))
        self.assertTrue(any("synthesis" in e for e in result.errors), result.errors)
        self.assertTrue(
            all(a.get("_source") == "llm" for a in result.stories.values()),
            "the per-story analysis should have survived",
        )


class TestEngineCache(EngineCase):
    def test_progress_is_persisted_as_it_happens(self):
        """A run killed at the 30-minute timeout keeps what it finished."""
        with mock.patch("news_scanner.analyze.LLMClient.complete",
                        lambda self, system, user, **kw: answer(user)):
            engine = self.engine(
                use_cache=True,
                cache_path=self.cache_path,
                incremental_cache=True,
                cache_flush_seconds=0.0,
            )
            engine.analyse(make_stories(2))

        self.assertTrue(self.cache_path.exists(), "the cache should have been written")
        self.assertEqual(len(json.loads(self.cache_path.read_text("utf-8"))), 2)

    def test_a_second_run_does_not_pay_for_the_analysis_twice(self):
        prompts: list[str] = []

        def fake_complete(self, system, user, *, max_tokens=8000):
            prompts.append(user)
            return answer(user)

        with mock.patch("news_scanner.analyze.LLMClient.complete", fake_complete):
            first = self.engine(use_cache=True, cache_path=self.cache_path)
            first.analyse(make_stories(2))
            first.save_cache()
            batches_before = sum(1 for p in prompts if "=== S" in p)

            second = self.engine(use_cache=True, cache_path=self.cache_path)
            result = second.analyse(make_stories(2))

        batches_after = sum(1 for p in prompts if "=== S" in p)
        self.assertEqual(
            batches_after, batches_before,
            "every story was cached, so no batch call should have been made",
        )
        self.assertEqual(len(result.stories), 2)
        self.assertTrue(all(a.get("_source") == "llm" for a in result.stories.values()))


class TestTransportContract(unittest.TestCase):
    """The transport's own retry rules, against a scripted server."""

    @classmethod
    def setUpClass(cls):
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            script: list = []
            requests = 0

            def do_POST(self):  # noqa: N802
                Handler.requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                code, headers, body = Handler.script.pop(0)
                raw = body.encode()
                self.send_response(code)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        cls.handler = Handler
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1/chat/completions"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.handler.requests = 0
        self.handler.script = []
        self._proxies = {
            k: os.environ.pop(k, None)
            for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                      "no_proxy", "NO_PROXY")
        }
        self.addCleanup(self._restore_proxies)

    def _restore_proxies(self):
        for key, value in self._proxies.items():
            if value is not None:
                os.environ[key] = value

    def test_retry_after_is_honoured_and_the_call_then_succeeds(self):
        from news_scanner.http import fetch_json

        self.handler.script = [
            (429, {"Retry-After": "0"}, '{"error": "slow down"}'),
            (200, {}, '{"ok": true}'),
        ]
        self.assertTrue(fetch_json(self.url, {"a": 1}, retries=2).get("ok"))
        self.assertEqual(self.handler.requests, 2, "the 429 should have been retried")

    def test_an_absurd_retry_after_is_not_slept_through(self):
        """A gateway advertising an hour of silence must not stall the run."""
        from news_scanner.http import fetch_json

        self.handler.script = [(429, {"Retry-After": "3600"}, "{}")] * 3
        with self.assertRaises(FetchError) as caught:
            fetch_json(self.url, {"a": 1}, retries=2)
        self.assertEqual(caught.exception.code, 429)
        self.assertEqual(caught.exception.retry_after, 3600.0)
        self.assertEqual(self.handler.requests, 1, "it should not have slept an hour")

    def test_a_fatal_status_is_not_retried(self):
        from news_scanner.http import fetch_json

        self.handler.script = [(401, {}, '{"error": "bad key"}')]
        with self.assertRaises(FetchError) as caught:
            fetch_json(self.url, {"a": 1}, retries=2)
        self.assertEqual(caught.exception.code, 401)
        self.assertEqual(self.handler.requests, 1, "a 401 will not fix itself")

    def test_connect_errors_can_skip_the_transport_retry(self):
        """The engine retries instead, with pacing and failover behind it."""
        import time

        from news_scanner.http import fetch_json

        started = time.monotonic()
        with self.assertRaises(FetchError):
            fetch_json(
                "http://127.0.0.1:9/v1/chat/completions",
                {"a": 1},
                retry_connect_errors=False,
            )
        self.assertLess(time.monotonic() - started, 1.5,
                        "a refused connection should not be retried on the transport")


if __name__ == "__main__":
    unittest.main(verbosity=2)
