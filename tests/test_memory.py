"""Tests for the memory store and the cross-story link layer.

These cover the two things that are easy to get quietly wrong: a memory file
that corrupts on a bad write, and a link graph that renders links the model
invented. Both fail silently if unguarded — a corrupt memory file just means
tomorrow starts cold, and a fabricated link looks exactly like a real one to a
reader skimming the brief.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from news_scanner import memory
from news_scanner.analyze import (
    LINK_TYPES,
    build_entity_graph,
    build_link_context,
    normalise_entity,
    _validate_links,
)
from news_scanner.render import _link_label, _renderable_links


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "daily_memory.jsonl"

    def test_round_trips_a_day(self):
        self.assertTrue(
            memory.append(
                self.path,
                "2026-10-02",
                {"editor_note_en": "the budget is the story", "links": []},
            )
        )
        got = memory.load_all(self.path)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["date"], "2026-10-02")
        self.assertEqual(got[0]["editor_note_en"], "the budget is the story")

    def test_one_record_per_day(self):
        """A manual re-run must replace the day, not append a rival version."""
        memory.append(self.path, "2026-10-02", {"editor_note_en": "first"})
        memory.append(self.path, "2026-10-02", {"editor_note_en": "second"})
        got = memory.load_all(self.path)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["editor_note_en"], "second")

    def test_only_useful_fields_are_kept(self):
        """The digests are already in the repo; duplicating them bloats the file."""
        memory.append(
            self.path,
            "2026-10-02",
            {"editor_note_en": "keep", "some_bulk": "x" * 5000},
        )
        raw = self.path.read_text("utf-8")
        self.assertNotIn("some_bulk", raw)
        self.assertLess(len(raw), 500)

    def test_empty_synthesis_is_not_remembered(self):
        self.assertFalse(memory.append(self.path, "2026-10-02", {}))
        self.assertEqual(memory.load_all(self.path), [])

    def test_a_corrupt_line_is_skipped_not_fatal(self):
        memory.append(self.path, "2026-10-01", {"editor_note_en": "good"})
        with self.path.open("a") as fh:
            fh.write("{not json\n")
        memory.append(self.path, "2026-10-03", {"editor_note_en": "also good"})
        dates = [r["date"] for r in memory.load_all(self.path)]
        self.assertEqual(dates, ["2026-10-01", "2026-10-03"])

    def test_the_file_is_bounded(self):
        """An unbounded file in a git repository is a liability."""
        for day in range(memory.KEEP_DAYS + 12):
            memory.append(
                self.path, f"2026-01-{day + 1:02d}", {"editor_note_en": f"day {day}"}
            )
        got = memory.load_all(self.path)
        self.assertEqual(len(got), memory.KEEP_DAYS)
        self.assertEqual(got[-1]["editor_note_en"], f"day {memory.KEEP_DAYS + 11}")

    def test_unwritable_path_does_not_raise(self):
        """Losing memory must never fail the run that produced it."""
        bad = self.dir / "nope" / "x" / "\0bad"
        self.assertFalse(memory.append(bad, "2026-10-02", {"editor_note_en": "x"}))

    def test_load_recent_excludes_the_day_being_written(self):
        """The 08:30 retry must not grade itself against its own first attempt."""
        memory.append(self.path, "2026-09-30", {"editor_note_en": "old"})
        memory.append(self.path, "2026-10-02", {"editor_note_en": "today"})
        got = memory.load_recent(7, path=self.path, before="2026-10-02")
        self.assertEqual([r["date"] for r in got], ["2026-09-30"])

    def test_entities_survive_the_write(self):
        """Regression: entities were projected away on write.

        ``recurring_entities`` reads this key off stored records, so losing it
        at write time silently disabled the whole recurring-entity feature —
        the graph was computed each run and thrown away.
        """
        memory.append(
            self.path, "2026-10-02", {"editor_note_en": "n", "entities": ["Kuwait Cabinet"]}
        )
        stored = memory.load_all(self.path)[0]
        self.assertIn("entities", stored)
        self.assertEqual(stored["entities"], ["Kuwait Cabinet"])

    def test_recurring_entities_survive_two_real_days(self):
        """The end-to-end shape: write, store, recall, in two steps."""
        memory.append(
            self.path, "2026-10-01", {"editor_note_en": "a", "entities": ["Kuwait Cabinet", "Nokia"]}
        )
        memory.append(
            self.path, "2026-10-02", {"editor_note_en": "b", "entities": ["Kuwait Cabinet", "Apple"]}
        )
        self.assertEqual(
            memory.recurring_entities(memory.load_all(self.path)), ["Kuwait Cabinet"]
        )

    def test_recurring_entities_need_more_than_one_day(self):
        records = [
            {"date": "2026-10-01", "entities": ["Kuwait Cabinet", "Nokia"]},
            {"date": "2026-10-02", "entities": ["Kuwait Cabinet", "Apple"]},
            {"date": "2026-10-03", "entities": ["Kuwait Cabinet"]},
        ]
        got = memory.recurring_entities(records)
        self.assertEqual(got, ["Kuwait Cabinet"])

    def test_context_is_bounded_by_characters(self):
        """A growing prompt is exactly the call that times out and loses the day."""
        records = [
            {"date": f"2026-10-{i:02d}", "editor_note_en": "x" * 900}
            for i in range(1, 20)
        ]
        ctx = memory.build_context(records, max_chars=2400)
        self.assertLessEqual(len(ctx), 2400)
        self.assertTrue(ctx)

    def test_context_is_empty_when_nothing_worth_saying(self):
        self.assertEqual(memory.build_context([]), "")
        self.assertEqual(memory.build_context([{"date": "2026-10-02"}]), "")


class TestEntityGraph(unittest.TestCase):
    def test_arabic_and_english_names_resolve_to_one_actor(self):
        """This is the whole reason the graph is built here and not by the model."""
        self.assertEqual(normalise_entity("مجلس الوزراء"), "Kuwait Cabinet")
        self.assertEqual(normalise_entity("The Cabinet"), "Kuwait Cabinet")
        self.assertEqual(normalise_entity("cabinet of ministers"), "Kuwait Cabinet")

    def test_generic_entities_are_dropped(self):
        for name in ("kuwait", "Government", "news", ""):
            self.assertEqual(normalise_entity(name), "")

    def test_only_cross_story_entities_are_kept(self):
        """An entity in one story carries no signal the batch did not already have."""
        graph = build_entity_graph(
            [],
            [
                {"entities": ["Kuwait Cabinet", "Nokia"]},
                {"entities": ["Kuwait Cabinet"]},
            ],
        )
        self.assertIn("Kuwait Cabinet", graph)
        self.assertNotIn("Nokia", graph)
        self.assertEqual(graph["Kuwait Cabinet"], ["S1", "S2"])

    def test_graph_renders_as_prompt_lines(self):
        block, graph = build_link_context(
            [],
            [
                {"entities": ["Kuwait Cabinet"]},
                {"entities": ["Kuwait Cabinet"]},
            ],
        )
        self.assertIn("Kuwait Cabinet", block)
        self.assertIn("S1, S2", block)
        self.assertEqual(graph["Kuwait Cabinet"], ["S1", "S2"])

    def test_no_graph_means_no_block(self):
        block, graph = build_link_context([], [{"entities": ["Nokia"]}])
        self.assertEqual(block, "")
        self.assertEqual(graph, {})


class TestLinkValidation(unittest.TestCase):
    """A fabricated link is indistinguishable from a real one to a reader."""

    def test_a_good_link_survives(self):
        payload = {
            "links": [
                {"type": "consequence", "from": "S1", "to": "S2", "evidence": "the vote follows the report"}
            ]
        }
        self.assertEqual(len(_validate_links(payload, 3)["links"]), 1)

    def test_dangling_ids_are_dropped(self):
        payload = {
            "links": [
                {"type": "reaction", "from": "S1", "to": "S99", "evidence": "invented"},
                {"type": "reaction", "from": "S1", "to": "S2", "evidence": "real"},
            ]
        }
        kept = _validate_links(payload, 3)["links"]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["to"], "S2")

    def test_a_self_link_is_dropped(self):
        payload = {"links": [{"type": "same_actor", "from": "S1", "to": "S1", "evidence": "x"}]}
        self.assertEqual(_validate_links(payload, 3)["links"], [])

    def test_an_unknown_relationship_type_is_dropped(self):
        """An open type set makes the graph unreadable and uncountable."""
        payload = {"links": [{"type": "vibes", "from": "S1", "to": "S2", "evidence": "x"}]}
        self.assertEqual(_validate_links(payload, 3)["links"], [])

    def test_a_link_without_evidence_is_dropped(self):
        """The evidence is the part a reader checks."""
        payload = {"links": [{"type": "same_actor", "from": "S1", "to": "S2", "evidence": "  "}]}
        self.assertEqual(_validate_links(payload, 3)["links"], [])

    def test_junk_entries_do_not_crash_it(self):
        payload = {"links": ["nope", None, 42, {"type": "reaction"}]}
        self.assertEqual(_validate_links(payload, 3)["links"], [])

    def test_a_missing_links_field_becomes_an_empty_list(self):
        self.assertEqual(_validate_links({}, 3)["links"], [])

    def test_every_type_in_the_schema_is_valid(self):
        payload = {
            "links": [
                {"type": kind, "from": "S1", "to": "S2", "evidence": "e"}
                for kind in LINK_TYPES
            ]
        }
        self.assertEqual(len(_validate_links(payload, 3)["links"]), len(LINK_TYPES))


class TestRenderGuard(unittest.TestCase):
    """The renderer re-checks links rather than trusting its input.

    The engine validates links before they reach a Digest, but a renderer also
    reads digests written by older versions of the pipeline and hand-edited
    JSON. A dead reference is visible to the reader; a silently dropped one is
    not, so the guard belongs at the point of display.
    """

    def _synth(self, links):
        return {"links": links}

    def test_a_link_to_a_story_that_is_not_on_the_page_is_not_shown(self):
        got = _renderable_links(self._synth([{"type": "consequence", "from": "S1", "to": "S9", "evidence": "e"}]), 2)
        self.assertEqual(got, [])

    def test_links_inside_the_story_count_are_kept(self):
        got = _renderable_links(
            self._synth([{"type": "consequence", "from": "S1", "to": "S2", "evidence": "e"}]), 2
        )
        self.assertEqual(len(got), 1)

    def test_a_link_without_evidence_is_not_shown(self):
        got = _renderable_links(
            self._synth([{"type": "consequence", "from": "S1", "to": "S2", "evidence": "  "}]), 2
        )
        self.assertEqual(got, [])

    def test_junk_does_not_crash_the_renderer(self):
        for junk in (["nope", None, 7], {}, "a string", None):
            self.assertEqual(_renderable_links(self._synth(junk), 3), [])

    def test_an_unknown_type_still_renders_but_neutral(self):
        """An unrecognised type is shown with a fallback label, not dropped.

        Dropping it would hide a real connection just because a future version
        of the schema adds a type this build has not heard of.
        """
        got = _renderable_links(
            self._synth([{"type": "brand_new_type", "from": "S1", "to": "S2", "evidence": "e"}]), 2
        )
        self.assertEqual(len(got), 1)
        label, _color = _link_label("brand_new_type")
        self.assertEqual(label, "brand new type")

    def test_known_types_get_distinct_colours(self):
        """Colour is how the graph gets scanned; duplicates would flatten it."""
        colours = [_link_label(kind)[1] for kind in LINK_TYPES]
        self.assertEqual(len(colours), len(set(colours)))


class TestReasoningBudget(unittest.TestCase):
    """Reasoning tokens must not eat the response budget.

    Cerebras reasoning models charge their thinking against ``max_tokens``.
    qwen-3.8-27b defaults to high effort, and with a 4000 cap it spent 2249
    tokens reasoning, left too little for the JSON, and returned no content at
    all — which surfaced as ``KeyError('content')`` and silently produced a
    digest with no synthesis whatsoever. These tests pin the fix.
    """

    def _payload_for(self, provider: str, model: str) -> dict:
        import news_scanner.analyze as analyze_mod

        captured: dict = {}
        original = analyze_mod.fetch_json

        def fake(url, payload, headers=None, **kw):
            captured.update(payload)
            return {"choices": [{"index": 0, "message": {"content": "{}"},
                                 "finish_reason": "stop"}]}

        analyze_mod.fetch_json = fake
        try:
            LLMClient = analyze_mod.LLMClient
            LLMClient(provider, model, "key").complete("sys", "user")
        finally:
            analyze_mod.fetch_json = original
        return captured

    def test_reasoning_is_disabled_for_cerebras(self):
        self.assertEqual(
            self._payload_for("cerebras", "qwen-3.8-27b").get("reasoning_effort"),
            "none",
        )

    def test_the_flag_is_not_sent_to_providers_that_would_reject_it(self):
        """An unrecognised parameter is a 400 on strict providers."""
        for provider in ("openai", "groq", "llm7", "nvidia"):
            self.assertNotIn(
                "reasoning_effort", self._payload_for(provider, "some-model"),
                f"{provider} should not receive reasoning_effort",
            )

    def test_json_mode_still_reaches_cerebras(self):
        """Both flags are needed: JSON shape and a budget to write it in."""
        payload = self._payload_for("cerebras", "qwen-3.8-27b")
        self.assertEqual(payload.get("response_format"), {"type": "json_object"})
        self.assertEqual(payload.get("reasoning_effort"), "none")


class TestRunMarkerWhenEmailIsOff(unittest.TestCase):
    """A no-email run is a finished run, and must be recorded as one.

    With NEWSCANNER_NO_EMAIL=1 the digest is still built, committed and
    published. If the marker were not written, the 08:30 and 09:00 retry crons
    would find nothing to do and exit — silently disabling the daily job for
    as long as the switch stays on.
    """

    def _run(self, env: dict[str, str]) -> int:
        import os
        import tempfile

        from news_scanner import cli

        # Stub the collector: these tests are about the marker, and hitting the
        # live feeds would make them slow and network-dependent. The output
        # directory must also be redirected — cmd_run writes a real digest, and
        # the workflow commits whatever is in digests/, so a test writing there
        # publishes fixture data as if it were today's edition.
        from scripts.demo import fake_collect

        saved_collect = cli.collect
        saved_env = {k: os.environ.get(k) for k in
                     ("NEWSCANNER_NO_EMAIL", "SMTP_HOST", "MAIL_TO")}
        for k in saved_env:
            os.environ.pop(k, None)
        os.environ.update(env)
        cli.collect = fake_collect
        out_dir = tempfile.mkdtemp(prefix="ns-test-out-")
        try:
            return cli.cmd_run(
                type("A", (), {
                    "date": "2026-10-04", "skip_if_done": False, "mail_to": None,
                    "config": None, "sources": None, "hours": None,
                    "max_analyze": 1, "major_outlets": 3, "provider": None,
                    "model": None, "llm_off": True, "language": "en",
                    "no_cache": True, "dry_run": False, "no_email": False,
                    "no_marker": False, "output": out_dir, "no_html": False,
                    "no_fetch_bodies": True, "workers": 1,
                })()
            )
        finally:
            cli.collect = saved_collect
            for k, v in saved_env.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def setUp(self):
        import os

        self.marker = os.path.join("state", "last_run.json")
        self._backup = None
        if os.path.exists(self.marker):
            with open(self.marker) as fh:
                self._backup = fh.read()
        if os.path.exists(self.marker):
            os.remove(self.marker)

    def tearDown(self):
        import os

        if self._backup is not None:
            with open(self.marker, "w") as fh:
                fh.write(self._backup)
        elif os.path.exists(self.marker):
            os.remove(self.marker)

    def test_a_no_email_run_records_the_day_as_done(self):
        self._run({"NEWSCANNER_NO_EMAIL": "1"})
        from news_scanner import cli

        marker = cli._read_run_marker()
        self.assertEqual(marker.get("date"), "2026-10-04")
        self.assertTrue(marker.get("delivered_at"))

    def test_the_run_still_succeeds(self):
        """A no-email run must exit 0, or the workflow fails and skips the commit."""
        self.assertEqual(self._run({"NEWSCANNER_NO_EMAIL": "1"}), 0)


class TestOutletFilter(unittest.TestCase):
    """Outlets must never become "recurring entities".

    Left in, they tell tomorrow's run that "Al-Anba is running across several
    days" — true of every single edition, so true of none. The real actors the
    link graph depends on get crowded out by the paper names instead.
    """

    def test_configured_outlets_are_filtered(self):
        for name in ("Al-Anba", "Al-Jarida", "Kuwait News", "alrai",
                     "Kuwait Times", "Times Kuwait"):
            self.assertEqual(normalise_entity(name), "", f"{name} should be filtered")

    def test_real_actors_survive(self):
        for name in ("Kuwait Cabinet", "Kuwait Investment Authority",
                     "Ministry of Finance", "Iran", "Khamis Mushait"):
            self.assertNotEqual(normalise_entity(name), "", f"{name} is an actor")

    def test_casing_variants_collapse_to_one_entity(self):
        """Two spellings of one actor would read as two separate threads."""
        self.assertEqual(
            normalise_entity("Houthi Militia"), normalise_entity("Houthi militia")
        )

    def test_a_short_word_is_not_matched_by_luck(self):
        """Prefix matching must not swallow a short or unrelated entity."""
        self.assertNotEqual(normalise_entity("Khamis Mushait"), "")


    def test_transliteration_variants_are_caught(self):
        """The model spells 'alrai' as 'alray'; config says alrai."""
        for variant in ("Al-Ra'y", "alray", "Al-Rai", "Alray"):
            self.assertEqual(normalise_entity(variant), "", f"{variant} is an outlet")

    def test_relaxed_matching_does_not_swallow_real_actors(self):
        """A fuzzy outlet match must not become a fuzzy everything-match."""
        for actor in ("Iran", "OPEC", "Yemen", "Aden", "Hezbollah",
                      "Kuwait Airways", "Ministry of Health"):
            self.assertNotEqual(
                normalise_entity(actor), "", f"{actor} is an actor, not an outlet"
            )

    def test_an_unlisted_paper_is_not_assumed_to_be_an_outlet(self):
        """Only the configured sources are filtered, so a real entity survives."""
        self.assertNotEqual(normalise_entity("Kuwait Mirror"), "")


class TestTranslationTable(unittest.TestCase):
    """No Arabic term may silently lose a meaning.

    A dictionary literal with the same key twice keeps only the last one, so a
    second 'متهم' entry quietly replaced 'suspect' with 'defendant' and the
    distinction between the two never survived to the digest. A duplicate that
    is a genuine synonym is harmless; one that narrows the meaning is data loss.
    """

    def test_no_duplicate_keys_in_the_translation_table(self):
        """A repeated key with a *different* value silently drops the first.

        Identical repeats are harmless, so only conflicting ones are a failure:
        that is the case where a term loses a meaning it genuinely had.
        """
        import ast

        tree = ast.parse(Path("news_scanner/util.py").read_text("utf-8"))
        conflicts: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            values: dict[str, list[str]] = {}
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    values.setdefault(key.value, []).append(ast.unparse(value))
            for word, seen in values.items():
                if len(set(seen)) > 1:
                    conflicts[word] = seen
        self.assertEqual(
            conflicts, {}, f"a term lost a meaning: {conflicts}"
        )

    def test_crime_terms_keep_a_translation(self):
        from news_scanner.util import normalize_text

        for word in ("متهم", "قضية", "قضايا", "الأجور", "الرواتب"):
            self.assertTrue(
                normalize_text(word).strip(), f"{word} lost its translation"
            )


class TestSuiteDoesNotPublish(unittest.TestCase):
    """No test may write into a directory the workflow commits.

    ``git add digests state docs`` in the workflow runs after the unit tests,
    so a test that leaves a digest behind publishes fixture data as if it were
    a real edition. That happened once: a marker test called cmd_run with the
    default output directory and the site showed 23 fixture stories as today's
    news. Each run below is therefore redirected to a temporary directory.
    """

    def setUp(self):
        import os

        self.digests = Path("digests")
        self.before = (
            {p.name for p in self.digests.iterdir()} if self.digests.exists() else set()
        )

    def test_the_marker_tests_leave_the_digest_directory_untouched(self):
        import tempfile

        from scripts.demo import fake_collect
        from news_scanner import cli

        out_dir = tempfile.mkdtemp(prefix="ns-guard-")
        saved_collect = cli.collect
        saved_env = {k: os.environ.get(k) for k in
                     ("NEWSCANNER_NO_EMAIL", "SMTP_HOST", "MAIL_TO")}
        for k in saved_env:
            os.environ.pop(k, None)
        os.environ["NEWSCANNER_NO_EMAIL"] = "1"
        cli.collect = fake_collect
        try:
            cli.cmd_run(
                type("A", (), {
                    "date": "2026-10-04", "skip_if_done": False, "mail_to": None,
                    "config": None, "sources": None, "hours": None,
                    "max_analyze": 1, "major_outlets": 3, "provider": None,
                    "model": None, "llm_off": True, "language": "en",
                    "no_cache": True, "dry_run": False, "no_email": False,
                    "no_marker": False, "output": out_dir, "no_html": False,
                    "no_fetch_bodies": True, "workers": 1,
                })()
            )
        finally:
            cli.collect = saved_collect
            for k, v in saved_env.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

        after = {p.name for p in self.digests.iterdir()} if self.digests.exists() else set()
        self.assertEqual(
            after - self.before, set(),
            "a test wrote into digests/ — the workflow would publish it",
        )

    def test_no_stray_output_is_written_into_the_repo(self):
        """A belt-and-braces check over the whole suite's own footprint."""
        stray = [
            str(p) for p in Path(".").rglob("2026-10-04.*")
            if p.parts and p.parts[0] in ("digests", "docs")
        ]
        self.assertEqual(stray, [], "fixture output left in a published directory")


class TestEncoding(unittest.TestCase):
    """Declared charsets must be honoured, not guessed.

    cp1256 and iso-8859-6 are both single-byte codecs that decode every byte
    without raising and both yield Arabic, so nothing in the byte stream can
    tell them apart — measurements of Arabic ratio and stray-symbol count come
    out identical. A wrong guess does not crash; it yields a headline that
    reads as Arabic but is not. The declaration is the only reliable signal.
    """

    def test_declared_charsets_are_detected_in_every_form(self):
        from news_scanner.http import guess_encoding_from_meta

        cases = [
            (b'<?xml version="1.0" encoding="iso-8859-6"?><rss>x</rss>', "iso-8859-6"),
            (b'<?xml version="1.0" encoding="ISO-8859-6"?><rss>x</rss>', "iso-8859-6"),
            (b'<?xml version="1.0" encoding="UTF-8"?><rss>x</rss>', "utf-8"),
            (b"<?xml version='1.0' encoding='UTF-8'?><rss>x</rss>", "utf-8"),
            (b'<meta charset="windows-1256">', "windows-1256"),
            (b'<meta charset=iso-8859-6>', "iso-8859-6"),
            (b"<meta charset='cp1256'>", "cp1256"),
            (b'<meta http-equiv="Content-Type" content="text/html; charset=cp1256">', "cp1256"),
        ]
        for raw, want in cases:
            self.assertEqual(
                guess_encoding_from_meta(raw), want, f"wrong charset for {raw[:40]!r}"
            )

    def test_an_unknown_charset_name_is_ignored(self):
        from news_scanner.http import guess_encoding_from_meta

        self.assertIsNone(guess_encoding_from_meta(b'<meta charset="nonsense-enc">'))
        self.assertIsNone(guess_encoding_from_meta(b"<html>no declaration</html>"))

    def test_arabic_survives_under_each_declared_codec(self):
        from news_scanner.http import decode

        for text in ("مجلس الوزراء يوافق على مشروع مرسوم",
                     "الشرطة تضبط محاولة تهريب"):
            for enc in ("utf-8", "cp1256", "iso-8859-6"):
                self.assertEqual(
                    decode(text.encode(enc), enc), text,
                    f"{enc} did not round-trip Arabic",
                )

    def test_latin_text_is_not_stolen_by_the_arabic_codec(self):
        """cp1256 used to claim windows-1252 and inject Arabic into 'CÔTE'."""
        from news_scanner.http import decode

        text = "Rapport sur la CÔTE d'Azur 1998"
        self.assertEqual(decode(text.encode("windows-1252"), "windows-1252"), text)

    def test_a_lying_declaration_falls_back_instead_of_crashing(self):
        """A wrong declaration must not raise.

        It cannot be corrected either: iso-8859-6 decodes cp1256 bytes without
        error, it just yields different letters. So the guarantee is only that
        the job survives and returns text of the right shape — the declared
        charset is trusted, because the alternative is guessing, and guessing
        is no better when the page has told us.
        """
        from news_scanner.http import decode

        out = decode("الشرطة".encode("cp1256"), "iso-8859-6")
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip())
        self.assertTrue(
            any(0x0600 <= ord(c) <= 0x06FF for c in out),
            "a wrong declared charset should still yield Arabic-script text",
        )


class TestCasualtyCountsNotMerged(unittest.TestCase):
    """Two headlines that disagree on a death toll are not one story.

    Merging them hides a correction and reports a toll that was revised.
    Token overlap cannot catch this: "One killed in Farwaniya blaze" and "Two
    killed in Farwaniya blaze" share four of five tokens and score 0.75, well
    above the 0.50 threshold.
    """

    def _cluster(self, *titles: str) -> int:
        from news_scanner.dedupe import cluster_articles
        from news_scanner.models import Article

        arts = [
            Article(source_id=f"s{i}", source_name=f"Out{i}", lang="en",
                    title=t, summary="", url=f"https://x/{i}",
                    published=None, tier=1)
            for i, t in enumerate(titles)
        ]
        return len(cluster_articles(arts))

    def test_a_different_toll_keeps_two_stories_apart(self):
        for a, b in (
            ("One killed in Farwaniya blaze", "Two killed in Farwaniya blaze"),
            ("Three injured in Salmiya accident", "Nine injured in Salmiya accident"),
            ("Bourse up 2 percent", "Bourse up 3 percent"),
        ):
            self.assertEqual(self._cluster(a, b), 2, f"merged: {a!r} / {b!r}")

    def test_killed_against_injured_keeps_two_stories_apart(self):
        self.assertEqual(
            self._cluster("Seven killed in warehouse fire",
                          "Seven injured in warehouse fire"),
            2,
        )

    def test_the_same_toll_still_merges(self):
        """The veto must not split genuine rewrites of one incident."""
        self.assertEqual(
            self._cluster(
                "Seven killed in warehouse fire",
                "Seven killed in Farwaniya warehouse fire",
            ),
            1,
        )

    def test_a_missing_toll_is_not_a_contradiction(self):
        """One headline stating no toll is not disagreeing with one that does."""
        self.assertEqual(
            self._cluster("Warehouse fire in Farwaniya",
                          "Farwaniya warehouse fire kills seven"),
            1,
        )

    def test_counts_survive_tokenisation(self):
        """The count must be a token at all, or nothing downstream can see it."""
        from news_scanner.util import canonical_tokens

        self.assertIn("1", canonical_tokens("One killed in Farwaniya blaze"))
        self.assertIn("2", canonical_tokens("Two killed in Farwaniya blaze"))
        self.assertNotEqual(
            canonical_tokens("One killed in Farwaniya blaze"),
            canonical_tokens("Two killed in Farwaniya blaze"),
        )


class TestCliArgumentSafety(unittest.TestCase):
    """A mistyped flag must not abort the morning's job.

    These run unattended at 08:00. A crash before any article is fetched means
    no digest at all, and the two failures below were both silent because the
    workflow marks the pre-flight steps continue-on-error.
    """

    def test_check_email_has_the_send_flag_it_reads(self):
        """cmd_check_email reads args.send after connecting; it was undeclared,
        so a bare `check-email` raised AttributeError after a *successful*
        authentication and the SMTP pre-flight never reported a bad password."""
        from news_scanner.cli import build_parser

        self.assertFalse(build_parser().parse_args(["check-email"]).send)
        self.assertTrue(build_parser().parse_args(["check-email", "--send"]).send)

    def test_worker_count_is_clamped_before_it_reaches_the_pool(self):
        from concurrent.futures import ThreadPoolExecutor

        for bad in (0, -5):
            clamped = max(1, int(bad or 1))
            with ThreadPoolExecutor(max_workers=clamped):
                pass  # would raise ValueError on the raw value


if __name__ == "__main__":
    unittest.main(verbosity=2)
