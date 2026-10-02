"""Regression tests for the NewsScanner pipeline.

Run with:  python3 -m unittest discover -s tests -v

These run fully offline. They protect the decisions that were expensive to get
right: Arabic↔English story merging, the guards against over-merging, the
per-article citation guarantee, and the "no API key" fallback.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from news_scanner.analyze import (  # noqa: E402
    AnalysisConfig,
    AnalysisEngine,
    detect_provider,
    extractive_analysis,
)
from news_scanner.dedupe import build_stories, categorize, major_count  # noqa: E402
from news_scanner.deliver import email_configured  # noqa: E402
from news_scanner.sources import collect  # noqa: E402
from news_scanner.models import Article, Digest, Story  # noqa: E402
from news_scanner.parse import parse_date, parse_feed  # noqa: E402
from news_scanner.render import render_html, render_markdown  # noqa: E402
from news_scanner.util import (  # noqa: E402
    canonical_tokens,
    distinctive_tokens,
    html_to_text,
    now_kuwait,
    normalize_text,
)

NOW = datetime(2026, 9, 30, 8, 0, tzinfo=timezone(timedelta(hours=3)))


def make_articles(rows) -> list[Article]:
    """rows: (source_id, source_name, lang, title)"""
    out = []
    for i, (sid, name, lang, title) in enumerate(rows):
        out.append(
            Article(
                source_id=sid,
                source_name=name,
                lang=lang,
                title=title,
                url=f"https://news.example/{sid}/{i}",
                summary=f"Body text for {title}",
                published=NOW - timedelta(hours=i % 20),
                tier=1,
            )
        )
    return out


def story_for(stories: list[Story], needle: str) -> Story:
    for story in stories:
        if any(needle in a.title for a in story.articles):
            return story
    raise AssertionError(f"no story containing {needle!r} in {len(stories)} stories")


# ---------------------------------------------------------------- normalisation


class TestTextNormalisation(unittest.TestCase):
    def test_arabic_orthography_is_folded(self):
        self.assertEqual(normalize_text("الأمير"), normalize_text("الامير"))
        self.assertEqual(normalize_text("مباحثاتٍ"), normalize_text("مباحثات"))
        self.assertIn("ه", normalize_text("حديقة"))       # ة -> ه
        self.assertNotIn("ـ", normalize_text("مـبـاحـثـات"))

    def test_arabic_indic_digits_become_ascii(self):
        self.assertIn("12", normalize_text("ضبط ١٢ متهماً"))

    def test_english_aliases_reach_arabic(self):
        self.assertIn("cabinet", canonical_tokens("مجلس الوزراء يوافق"))
        self.assertIn("kuwaitiz", canonical_tokens("تكويت الوظائف"))
        self.assertIn("judicial", canonical_tokens("الوظائف القضائية"))

    def test_distinctive_tokens_drop_only_boilerplate(self):
        tokens = distinctive_tokens("Kuwait news: Cabinet approves Silk City decree")
        self.assertNotIn("kuwait", tokens)      # boilerplate for a Kuwaiti digest
        self.assertNotIn("news", tokens)
        self.assertIn("cabinet", tokens)        # kept: it carries meaning
        self.assertIn("silk", tokens)
        self.assertIn("decre", tokens)

    def test_generic_words_still_count_for_similarity(self):
        # The gate uses distinctive tokens, but scoring still sees everything.
        self.assertIn("kuwait", canonical_tokens("Kuwait oil price"))

    def test_html_to_text_drops_markup(self):
        text = html_to_text("<p>Hello <b>Kuwait</b></p><script>x=1</script>")
        self.assertIn("Kuwait", text)
        self.assertNotIn("<b>", text)
        self.assertNotIn("x=1", text)


# ------------------------------------------------------------------- clustering

SILK_CITY = [
    ("alrai", "الرأي", "ar",
     "مجلس الوزراء يوافق على إلغاء مرسوم إنشاء جهاز تطوير مدينة الحرير «الصبية» وجزيرة بوبيان"),
    ("alanba", "الأنباء", "ar",
     "إلغاء تطوير «الحرير» و«بوبيان» وتعديل «الشركات»"),
    ("aljarida", "الجريدة", "ar",
     "مجلس الوزراء يوافق على مشروع مرسوم بإلغاء مرسوم إنشاء جهاز تطوير مدينة الحرير وجزيرة بوبيان"),
    ("kuwaittimes", "Kuwait Times", "en",
     "Cabinet clears abolition of Silk City development body"),
    ("timeskuwait", "Times Kuwait", "en",
     "Govt approves cancellation of decree establishing Silk City, BIDA"),
]

CROWN_PRINCE = [
    ("alrai", "الرأي", "ar",
     "سمو ولي العهد ورئيس الاتحاد السويسري يترأسان جلسة المباحثات الرسمية بين الكويت وسويسرا"),
    ("kuwaittimes", "Kuwait Times", "en",
     "Crown Prince's visit to Switzerland to boost relations"),
]


class TestClustering(unittest.TestCase):
    def test_silk_city_story_merges_all_five_outlets(self):
        stories = build_stories(make_articles(SILK_CITY))
        story = story_for(stories, "الحرير")
        self.assertEqual(story.outlet_count, 5, [a.title for a in story.articles])
        self.assertEqual(story.langs, {"ar", "en"})

    def test_arabic_and_english_sources_bridge(self):
        stories = build_stories(make_articles(CROWN_PRINCE))
        story = story_for(stories, "السويسري")
        self.assertEqual(story.outlet_count, 2)
        self.assertEqual(story.langs, {"ar", "en"})

    def test_drug_bust_bridges_across_languages(self):
        rows = [
            ("alqabas", "القبس", "ar",
             "الداخلية: ضبط 12 متهماً في 8 قضايا بحوزتهم مواد مخدرة ومؤثرات عقلية وسلاح"),
            ("arabtimes", "Arab Times", "en",
             "Kuwait Arrests 12 Suspects in 8 Cases, Seizes Drugs, Alcohol and Firearm"),
        ]
        stories = build_stories(make_articles(rows))
        self.assertEqual(len(stories), 1)
        self.assertEqual(stories[0].outlet_count, 2)

    def test_unrelated_stories_sharing_boilerplate_do_not_merge(self):
        rows = [
            ("kuwaittimes", "Kuwait Times", "en",
             "Kuwait oil price rises to 106 dollars a barrel"),
            ("arabtimes", "Arab Times", "en",
             "Kuwait gold price falls ahead of US data release"),
        ]
        stories = build_stories(make_articles(rows))
        self.assertEqual(len(stories), 2, "generic market words must not merge these")

    def test_one_shared_entity_is_not_enough(self):
        rows = [
            ("alrai", "الرأي", "ar", "مجلس الوزراء يبحث الميزانية العامة للدولة"),
            ("kuwaittimes", "Kuwait Times", "en", "Cabinet approves a new traffic law"),
        ]
        stories = build_stories(make_articles(rows))
        self.assertEqual(len(stories), 2, "a single shared anchor must not bridge")

    def test_same_article_is_never_counted_twice(self):
        story = Story()
        article = make_articles([SILK_CITY[0]])[0]
        story.add(article)
        story.add(article)
        self.assertEqual(len(story.articles), 1)
        self.assertEqual(story.outlet_count, 1)

    def test_every_article_survives_into_exactly_one_story(self):
        articles = make_articles(SILK_CITY + CROWN_PRINCE)
        stories = build_stories(articles)
        seen = [a.url for s in stories for a in s.articles]
        self.assertEqual(len(seen), len(articles))
        self.assertEqual(len(set(seen)), len(articles))

    def test_major_count_uses_the_outlet_threshold(self):
        stories = build_stories(make_articles(SILK_CITY + CROWN_PRINCE))
        self.assertGreaterEqual(major_count(stories, 3), 1)


# --------------------------------------------------------------- categorisation


class TestCategorisation(unittest.TestCase):
    def _category(self, *titles):
        return categorize(Story(articles=[
            Article(source_id="s", source_name="S", lang="en", title=t, url=f"u{i}")
            for i, t in enumerate(titles)
        ]))

    def test_sport_beats_generic_section(self):
        self.assertEqual(
            self._category("Al-Mudhaf wins double gold as Kuwait shooters light up Asian Games"),
            "Sport",
        )

    def test_cabinet_story_is_politics(self):
        self.assertEqual(
            self._category("Cabinet clears abolition of Silk City development body"),
            "Government & Politics",
        )

    def test_celebrity_story_is_culture_not_politics(self):
        self.assertEqual(
            self._category("Taylor Swift sets new record at MTV VMAs"),
            "Culture & Entertainment",
        )

    def test_hurricane_is_world_not_politics(self):
        self.assertEqual(
            self._category("Hurricane Polo slams Mexico with flooding and power cuts"),
            "World & Region",
        )

    def test_drug_arrest_is_security(self):
        self.assertEqual(
            self._category("Anti-narcotics officers arrest 12 suspects in eight cases"),
            "Security & Courts",
        )


# ---------------------------------------------------------------------- render


class TestRendering(unittest.TestCase):
    def _digest(self) -> Digest:
        articles = make_articles(SILK_CITY + CROWN_PRINCE)
        stories = build_stories(articles)
        return Digest(
            date="2026-09-30",
            generated_at=NOW,
            window_hours=24,
            stories=stories,
            analyses=[
                extractive_analysis(story, "bilingual") for story in stories
            ],
            synthesis={"editor_note_en": "Note", "editor_note_ar": "ملاحظة"},
            source_health=[],
            llm_provider="none",
            errors=[],
            stats={
                "total_articles": len(articles),
                "total_stories": len(stories),
                "major_stories": 1,
                "outlets_reporting": 5,
                "analysed_stories": len(stories),
                "listed_stories": len(stories),
                "categories": {},
            },
        )

    def test_every_article_is_cited_with_its_url(self):
        markdown = render_markdown(self._digest())
        for article in make_articles(SILK_CITY + CROWN_PRINCE):
            self.assertIn(article.url, markdown, f"missing citation for {article.title}")
            self.assertIn(article.source_name, markdown)

    def test_arabic_headlines_are_preserved_verbatim(self):
        markdown = render_markdown(self._digest())
        self.assertIn("الحرير", markdown)
        self.assertIn("السويسري", markdown)

    def test_between_the_lines_section_is_always_present(self):
        markdown = render_markdown(self._digest())
        self.assertIn("Between the lines", markdown)

    def test_extractive_run_is_labelled_honestly(self):
        markdown = render_markdown(self._digest())
        self.assertIn("without an ai provider", markdown.lower())
        self.assertIn("Confidence", markdown)

    def test_markdown_links_survive_brackets_in_titles(self):
        story = Story()
        story.add(
            Article(
                source_id="s", source_name="S", lang="en",
                title="Report [exclusive]: oil rises (again)",
                url="https://news.example/x", summary="body",
            )
        )
        digest = Digest(
            date="2026-09-30", generated_at=NOW, window_hours=24,
            stories=[story],
            analyses=[extractive_analysis(story, "en")],
            synthesis={}, source_health=[], llm_provider="none", errors=[],
            stats={"analysed_stories": 1, "listed_stories": 1},
        )
        markdown = render_markdown(digest)
        self.assertIn("https://news.example/x", markdown)
        # The title must not have punched a hole in the link syntax.
        self.assertNotIn("[exclusive]: oil rises (again)](", markdown)


# ----------------------------------------------------------------------- parse

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Kuwait cabinet approves draft decree</title>
      <link>https://news.example/a</link>
      <description><![CDATA[<p>The cabinet met on Tuesday.</p>]]></description>
      <pubDate>Tue, 29 Sep 2026 21:05:00 +0300</pubDate>
    </item>
    <item>
      <title>الثاني</title>
      <link>https://news.example/b</link>
      <description>وصف الخبر</description>
      <dc:date>2026-09-29T18:00:00+03:00</dc:date>
    </item>
  </channel>
</rss>
"""

ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <entry>
    <title>Kuwait shooters win gold</title>
    <link href="https://news.example/c"/>
    <summary>Two golds at the Asian Games.</summary>
    <updated>2026-09-29T20:00:00+03:00</updated>
  </entry>
</feed>
"""


class TestParsing(unittest.TestCase):
    def _parse(self, xml: str, lang: str = "en"):
        return parse_feed(
            xml.encode("utf-8"),
            source_id="test",
            source_name="Test",
            lang=lang,
            section="local",
            tier=1,
        )

    def test_rss_items_are_parsed(self):
        articles = self._parse(RSS)
        self.assertEqual(len(articles), 2)
        first = articles[0]
        self.assertEqual(first.title, "Kuwait cabinet approves draft decree")
        self.assertEqual(first.url, "https://news.example/a")
        self.assertIsNotNone(first.published)
        self.assertEqual(first.published.utcoffset(), timedelta(hours=3))
        self.assertIn("cabinet met", first.summary)

    def test_atom_entries_are_parsed(self):
        articles = self._parse(ATOM)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].url, "https://news.example/c")

    def test_doctype_is_rejected_not_expanded(self):
        evil = """<?xml version="1.0"?>
        <!DOCTYPE rss [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
        <rss version="2.0"><channel><item>
          <title>&xxe;</title><link>https://news.example/x</link>
        </item></channel></rss>"""
        # Either the document is rejected outright or the entity is stripped —
        # what must never happen is the external entity being resolved.
        for article in self._parse(evil):
            self.assertNotIn("root:", article.title)

    def test_parse_date_handles_both_families(self):
        self.assertIsNotNone(parse_date("Tue, 29 Sep 2026 21:05:00 +0300"))
        self.assertIsNotNone(parse_date("2026-09-29T18:00:00+03:00"))
        self.assertIsNone(parse_date("not a date"))
        self.assertIsNone(parse_date(None))

    def test_empty_feed_is_not_an_error(self):
        self.assertEqual(self._parse("<rss><channel></channel></rss>"), [])


# --------------------------------------------------------------- analysis layer


class TestAnalysis(unittest.TestCase):
    def test_no_keys_still_yields_a_provider(self):
        """With no secrets configured the default is the keyless LLM7 gateway.

        Falling back to `none` here would mean a fresh fork silently produces
        the extractive digest, which is exactly the outcome LLM7 removes.
        """
        import os

        saved = {k: os.environ.pop(k, None) for k in
                 ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                  "OPENROUTER_API_KEY", "GROQ_API_KEY", "LLM7_API_KEY",
                  "OLLAMA_HOST", "NEWSCANNER_USE_OLLAMA", "NEWSCANNER_NO_LLM7",
                  "NEWSCANNER_PROVIDER")}
        try:
            self.assertEqual(detect_provider("auto"), ("llm7", "default", ""))
            self.assertEqual(detect_provider("none"), ("none", "", ""))
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value

    def test_extractive_analysis_is_labeled_and_bilingual(self):
        story = Story()
        for row in SILK_CITY:
            story.add(make_articles([row])[0])
        item = extractive_analysis(story, "bilingual")
        self.assertEqual(item["confidence"], "low")
        self.assertIn("without an ai provider", item["why_it_matters_en"].lower())
        self.assertIn("no ai provider", item["between_the_lines_en"].lower())
        self.assertTrue(item["between_the_lines_ar"])
        self.assertNotIn("between_the_lines_ar", ("", None))
        for source in story.sources:
            self.assertIn(source, item["why_it_matters_en"])

    def test_no_provider_run_never_raises(self):
        engine = AnalysisEngine(
            AnalysisConfig(provider="none", use_cache=False, cache_path=None)
        )
        stories = build_stories(make_articles(SILK_CITY + CROWN_PRINCE))
        result = engine.analyse(stories)
        self.assertEqual(result.provider, "none")
        self.assertEqual(len(result.stories), len(stories))
        self.assertTrue(result.synthesis.get("editor_note_en"))

    def test_llm_path_parses_and_renders(self):
        """The provider code path, exercised with a stubbed client.

        Both the per-story batch and the synthesis call are answered with one
        merged JSON object, so the test does not depend on prompt wording.
        """
        import json
        from unittest import mock

        canned = {
            "stories": [
                {
                    "id": "S1",
                    "headline_en": "Cabinet abolishes the Silk City development body",
                    "headline_ar": "مجلس الوزراء يلغي جهاز تطوير مدينة الحرير",
                    "summary_en": "The Cabinet approved a draft decree abolishing the "
                                  "Silk City and Bubiyan development authority.",
                    "summary_ar": "أقر مجلس الوزراء مشروع مرسوم بإلغاء جهاز تطوير مدينة الحرير.",
                    "why_it_matters_en": "A flagship project file changes hands.",
                    "why_it_matters_ar": "ملف مشروع رئيسي ينتقل إلى جهة أخرى.",
                    "between_the_lines_en": "The decree is attributed to the Cabinet "
                                            "collectively, with no single minister named.",
                    "between_the_lines_ar": "المرسوم منسوب إلى مجلس الوزراء مجتمعاً.",
                    "confidence": "medium",
                    "confidence_reason": "Based on official statements only.",
                    "entities": ["Cabinet", "Silk City"],
                    "tags": ["infrastructure", "cabinet"],
                },
                {
                    "id": "S2",
                    "headline_en": "Crown Prince holds talks in Bern",
                    "summary_en": "Official talks covered trade and investment.",
                    "why_it_matters_en": "Signals continuity in the Swiss file.",
                    "between_the_lines_en": "No agenda detail was released.",
                    "confidence": "low",
                    "confidence_reason": "Wire copy only.",
                },
            ],
            "editor_note_en": "The Cabinet's decision on Silk City was the day's pivot.",
            "editor_note_ar": "قرار مجلس الوزراء كان محور اليوم.",
            "story_of_the_day": "S1",
            "top_themes": [
                {"theme_en": "State project files", "theme_ar": "ملفات المشاريع",
                 "story_ids": ["S1"], "note_en": "One sentence.", "note_ar": "جملة."}
            ],
            "not_being_said_en": "Nothing on the cost of unwinding the authority.",
            "not_being_said_ar": "لا حديث عن الكلفة.",
            "watchlist_en": "Where the file lands next",
            "watchlist_ar": "الجهة التي ستتولى الملف",
        }

        with mock.patch(
            "news_scanner.analyze.LLMClient.complete",
            return_value=json.dumps(canned, ensure_ascii=False),
        ), mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
            engine = AnalysisEngine(
                AnalysisConfig(provider="gemini", use_cache=False, cache_path=None)
            )
            # The real client is constructed; only its network call is stubbed.
            assert engine.provider == "gemini", engine.provider
            assert engine.client is not None
            stories = build_stories(make_articles(SILK_CITY + CROWN_PRINCE))
            result = engine.analyse(stories)

        self.assertEqual(result.provider, "gemini")
        self.assertEqual(len(result.stories), len(stories))
        self.assertFalse(result.errors, result.errors)

        analysis = next(iter(result.stories.values()))
        self.assertEqual(analysis["_source"], "llm")
        self.assertIn("Cabinet", analysis["summary_en"])

        digest = Digest(
            date="2026-09-30", generated_at=NOW, window_hours=24,
            stories=stories,
            analyses=[result.stories[s.key] for s in stories],
            synthesis=result.synthesis,
            source_health=[], llm_provider="gemini", errors=[],
            stats={"analysed_stories": len(stories), "listed_stories": len(stories)},
        )
        markdown = render_markdown(digest)
        # The model's subtext makes it into the briefing, labelled as inference.
        self.assertIn("attributed to the Cabinet", markdown)
        self.assertIn("(inference)", markdown)
        self.assertIn("not reported fact", render_html(digest))
        self.assertIn("Silk City", markdown)
        # And the synthesis is rendered too.
        self.assertIn("the day's pivot", markdown)


# ------------------------------------------------------------------- utilities


class TestUtilities(unittest.TestCase):
    def test_kuwait_is_utc_plus_three_year_round(self):
        stamp = datetime(2026, 9, 30, 8, 0, tzinfo=timezone(timedelta(hours=3)))
        self.assertEqual(stamp.astimezone(timezone.utc).hour, 5)
        winter = datetime(2026, 1, 15, 8, 0, tzinfo=timezone(timedelta(hours=3)))
        self.assertEqual(winter.astimezone(timezone.utc).hour, 5)

    def test_now_kuwait_has_the_right_offset(self):
        self.assertEqual(now_kuwait().utcoffset(), timedelta(hours=3))

    def test_email_is_off_unless_fully_configured(self):
        import os

        saved = {k: os.environ.pop(k, None) for k in
                 ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "MAIL_TO")}
        try:
            self.assertFalse(email_configured())
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value


class TestDelivery(unittest.TestCase):
    """Email is the delivery channel, so its defaults are part of the contract."""

    SMTP_VARS = (
        "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_USERNAME", "SMTP_PASSWORD",
        "SMTP_PASS", "SMTP_SECURITY", "MAIL_FROM", "MAIL_TO",
        "NEWSCANNER_NO_EMAIL",
    )

    def setUp(self):
        import os

        self._saved = {k: os.environ.pop(k, None) for k in self.SMTP_VARS}

    def tearDown(self):
        import os

        for key in self.SMTP_VARS:
            os.environ.pop(key, None)
        for key, value in self._saved.items():
            if value is not None:
                os.environ[key] = value

    def test_default_recipient_is_the_configured_address(self):
        from news_scanner.deliver import DEFAULT_MAIL_TO, recipients

        self.assertEqual(DEFAULT_MAIL_TO, "mrmakrider@gmail.com")
        self.assertEqual(recipients(), ["mrmakrider@gmail.com"])

    def test_mail_to_overrides_the_default(self):
        import os

        from news_scanner.deliver import recipients

        os.environ["MAIL_TO"] = "someone@example.com"
        self.assertEqual(recipients(), ["someone@example.com"])

    def test_multiple_recipients_are_split(self):
        import os

        from news_scanner.deliver import recipients

        os.environ["MAIL_TO"] = "a@example.com; b@example.com, c@example.com"
        self.assertEqual(recipients(), ["a@example.com", "b@example.com", "c@example.com"])

    def test_email_can_be_switched_off(self):
        import os

        from news_scanner.deliver import email_configured, email_enabled

        os.environ["SMTP_HOST"] = "smtp.example.com"
        os.environ["NEWSCANNER_NO_EMAIL"] = "1"
        self.assertFalse(email_enabled())
        self.assertFalse(email_configured())

    def test_missing_smtp_host_still_counts_as_unconfigured(self):
        from news_scanner.deliver import email_configured

        self.assertFalse(email_configured(), "no SMTP_HOST means we cannot send")

    def test_send_email_addresses_the_default_recipient(self):
        """The To: header must carry the default address, not a placeholder."""
        import os
        import smtplib
        from unittest import mock

        os.environ["SMTP_HOST"] = "smtp.example.com"
        os.environ["SMTP_USER"] = "sender@example.com"
        os.environ["SMTP_PASSWORD"] = "app-password"

        sent = {}

        class FakeSMTP:
            def __init__(self, host, port, **kwargs):
                sent["host"] = host
                sent["port"] = port

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def ehlo(self):
                pass

            def starttls(self, context=None):
                sent["starttls"] = True

            def login(self, user, password):
                sent["login"] = (user, password)

            def send_message(self, msg):
                sent["message"] = msg

        with mock.patch.object(smtplib, "SMTP", FakeSMTP):
            from news_scanner.deliver import send_email

            ok = send_email("Subject", "<p>html</p>", "text")

        self.assertTrue(ok)
        message = sent["message"]
        self.assertEqual(message["To"], "mrmakrider@gmail.com")
        self.assertEqual(message["Subject"], "Subject")
        self.assertEqual(sent["host"], "smtp.example.com")
        self.assertEqual(sent["login"], ("sender@example.com", "app-password"))
        self.assertTrue(sent["starttls"])
        # Both a plain-text and an HTML part must be present.
        self.assertTrue(message.is_multipart())
        self.assertIn("text/plain", message.as_string())
        self.assertIn("text/html", message.as_string())

    def test_send_email_is_a_no_op_without_smtp_host(self):
        from news_scanner.deliver import send_email

        self.assertFalse(send_email("Subject", "<p>x</p>", "x"))

    def test_verify_email_explains_what_is_missing(self):
        from news_scanner.deliver import verify_email

        ok, detail = verify_email()
        self.assertFalse(ok)
        self.assertIn("SMTP_HOST", detail)


class TestRunMarker(unittest.TestCase):
    """The 08:30 retry relies on this marker; getting it wrong double-sends."""

    def setUp(self):
        from news_scanner import cli

        self.cli = cli
        self.marker = cli._run_marker_path()
        self._backup = self.marker.read_text("utf-8") if self.marker.exists() else None
        if self.marker.exists():
            self.marker.unlink()

    def tearDown(self):
        if self.marker.exists():
            self.marker.unlink()
        if self._backup is not None:
            self.marker.parent.mkdir(parents=True, exist_ok=True)
            self.marker.write_text(self._backup, "utf-8")

    def test_marker_round_trips(self):
        self.assertEqual(self.cli._read_run_marker(), {})
        self.cli._write_run_marker("2026-09-30", emailed=True, stories=23)
        marker = self.cli._read_run_marker()
        self.assertEqual(marker["date"], "2026-09-30")
        self.assertTrue(marker["emailed"])
        self.assertEqual(marker["stories"], 23)

    def test_attempts_accumulate(self):
        self.cli._write_run_marker("2026-09-30", emailed=False, stories=1)
        self.cli._write_run_marker("2026-09-30", emailed=True, stories=1)
        self.assertEqual(self.cli._read_run_marker()["attempts"], 2)

    def test_corrupt_marker_does_not_crash(self):
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.write_text("{not json", "utf-8")
        self.assertEqual(self.cli._read_run_marker(), {})

    def test_skip_if_done_only_skips_for_today(self):
        from scripts.demo import fake_collect

        self.cli.collect = fake_collect
        try:
            self.cli._write_run_marker("2026-09-29", emailed=True, stories=5)
            rc = self.cli.main(
                ["run", "--skip-if-done", "--date", "2026-09-30", "--no-llm",
                 "--dry-run", "--no-fetch-bodies", "--no-cache", "--no-marker",
                 "--output", "/tmp/newsscanner-marker-test"]
            )
            self.assertEqual(rc, 0)
            # It ran (and rewrote nothing) because the marker was for another day.
            self.assertEqual(self.cli._read_run_marker()["date"], "2026-09-29")
        finally:
            self.cli.collect = collect


class TestFreeProviderRotation(unittest.TestCase):
    """The free tiers the brief rotates through when the primary is throttled.

    A run that loses its only provider degrades to raw headlines instead of
    failing, so the chain has to be right in configuration *and* on the wire.
    """

    PROVIDER_ENV = (
        "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY",
        "NVIDIA_API_KEY", "NGC_API_KEY", "LLM7_API_KEY",
        "OLLAMA_HOST", "NEWSCANNER_USE_OLLAMA", "NEWSCANNER_PROVIDER",
        "NEWSCANNER_MODEL", "NEWSCANNER_NO_LLM7",
    )

    def setUp(self):
        import os

        # Taken out of the environment, not just overridden: these tests run
        # before the ones that assert what the *absence* of a key means, and a
        # leaked CEREBRAS_API_KEY would make the keyless default unreachable.
        self._saved = {k: os.environ.pop(k, None) for k in self.PROVIDER_ENV}

    def tearDown(self):
        import os

        for key in self.PROVIDER_ENV:
            os.environ.pop(key, None)
        for key, value in self._saved.items():
            if value is not None:
                os.environ[key] = value

    def test_a_free_tier_becomes_the_primary(self):
        import os

        from news_scanner.analyze import detect_provider

        os.environ["CEREBRAS_API_KEY"] = "csk-test"
        provider, model, key = detect_provider("auto")
        self.assertEqual(provider, "cerebras")
        self.assertEqual(key, "csk-test")
        self.assertEqual(model, "qwen-3.8-27b")

    def test_the_nvidia_key_is_accepted_under_either_name(self):
        import os

        from news_scanner.analyze import detect_provider

        os.environ["NVIDIA_API_KEY"] = "nvapi-test"
        self.assertEqual(detect_provider("auto")[0], "nvidia")
        del os.environ["NVIDIA_API_KEY"]
        os.environ["NGC_API_KEY"] = "nvapi-test"
        self.assertEqual(detect_provider("auto")[0], "nvidia")

    def test_both_free_tiers_are_in_the_rotation_chain(self):
        import os

        from news_scanner.analyze import detect_candidates

        os.environ["CEREBRAS_API_KEY"] = "csk-test"
        os.environ["NVIDIA_API_KEY"] = "nvapi-test"
        labels = [c.provider for c in detect_candidates("auto")]
        # Every configured provider is callable, and llm7 remains the last
        # resort behind all of them.
        self.assertIn("cerebras", labels)
        self.assertIn("nvidia", labels)
        self.assertLess(labels.index("cerebras"), labels.index("nvidia"))
        self.assertEqual(labels[-1], "llm7")

    def test_json_mode_is_sent_only_where_it_is_documented(self):
        from news_scanner.analyze import JSON_MODE_PROVIDERS

        self.assertIn("cerebras", JSON_MODE_PROVIDERS)
        # NIM's catalog spans many models behind one URL and does not promise
        # the parameter on all of them, so it is left out on purpose.
        self.assertNotIn("nvidia", JSON_MODE_PROVIDERS)
        self.assertNotIn("llm7", JSON_MODE_PROVIDERS)

    def test_the_rotation_is_tried_when_the_primary_refuses(self):
        """The whole point: a 429 on the first provider must not end the run."""
        import json
        import os

        import news_scanner.analyze as analyze_mod
        from news_scanner.analyze import AnalysisConfig, AnalysisEngine
        from news_scanner.http import FetchError

        calls: list[str] = []

        def fake_fetch(url, payload, **kwargs):
            calls.append(url)
            if "cerebras" in url:
                raise FetchError("429 Too Many Requests", code=429)
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"stories": [{"id": "S1", "summary_en": "ok"}]}
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

        os.environ["CEREBRAS_API_KEY"] = "csk-test"
        os.environ["NVIDIA_API_KEY"] = "nvapi-test"
        saved_fetch = analyze_mod.fetch_json
        analyze_mod.fetch_json = fake_fetch
        try:
            engine = AnalysisEngine(
                AnalysisConfig(
                    provider="auto", use_cache=False, cache_path=None,
                    max_llm_seconds=30,
                )
            )
            stories = build_stories(make_articles(SILK_CITY))
            result = engine.analyse(stories)

            self.assertEqual(result.errors, [])
            self.assertTrue(
                any("cerebras" in url for url in calls),
                f"the throttled provider should have been tried first, got {calls}",
            )
            self.assertTrue(
                any("nvidia" in url for url in calls),
                f"the run should have rotated to the next provider, got {calls}",
            )
            self.assertTrue(
                any(label.startswith("nvidia/") for label in result.providers_used),
                result.providers_used,
            )
        finally:
            analyze_mod.fetch_json = saved_fetch


class TestLLM7Provider(unittest.TestCase):
    """LLM7 is the keyless default, so its wire format must be exactly right.

    A local HTTP server speaks LLM7's documented OpenAI-compatible protocol,
    which lets the real client, transport and parser run end to end offline.
    """

    @classmethod
    def setUpClass(cls):
        import http.server
        import json as _json
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            # Real gateways answer over HTTP/1.1 with keep-alive; the mock
            # should not be more forgiving than production.
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length).decode("utf-8")
                Handler.requests.append(
                    {
                        "path": self.path,
                        "body": _json.loads(raw),
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                    }
                )
                if Handler.fail_next:
                    Handler.fail_next = False
                    payload = {"error": {"message": "upstream busy"}}
                    body = _json.dumps(payload).encode()
                    self.send_response(503)
                else:
                    # Exactly the shape api.llm7.io returns.
                    payload = {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "default",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": Handler.reply,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                    body = _json.dumps(payload).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # silence the test output
                pass

        Handler.requests = []
        Handler.reply = "{}"
        Handler.fail_next = False
        cls.handler = Handler
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        import os

        self.handler.requests.clear()
        self.handler.fail_next = False
        self._saved = {
            k: os.environ.pop(k, None)
            for k in (
                "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "OPENROUTER_API_KEY", "GROQ_API_KEY", "LLM7_API_KEY",
                "OLLAMA_HOST", "NEWSCANNER_USE_OLLAMA", "NEWSCANNER_PROVIDER",
                "NEWSCANNER_MODEL", "NEWSCANNER_NO_LLM7",
            )
        }
        os.environ["NEWSCANNER_BASE_URL"] = self.base_url
        # urllib honours proxy variables; a local ephemeral server must not.
        self._proxies = {
            k: os.environ.pop(k, None)
            for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY")
        }

    def tearDown(self):
        import os

        os.environ.pop("NEWSCANNER_BASE_URL", None)
        for key, value in self._saved.items():
            if value is not None:
                os.environ[key] = value
        for key, value in self._proxies.items():
            if value is not None:
                os.environ[key] = value

    # -- tests -------------------------------------------------------------
    def test_llm7_is_the_default_without_any_key(self):
        self.assertEqual(detect_provider("auto"), ("llm7", "default", ""))

    def test_a_real_key_still_wins_over_the_keyless_default(self):
        import os

        os.environ["GEMINI_API_KEY"] = "test-key"
        self.assertEqual(detect_provider("auto")[0], "gemini")

    def test_llm7_can_be_turned_off(self):
        import os

        os.environ["NEWSCANNER_NO_LLM7"] = "1"
        self.assertEqual(detect_provider("auto")[0], "none")

    def test_request_matches_the_documented_llm7_shape(self):
        import json

        from news_scanner.analyze import LLMClient

        self.handler.reply = json.dumps(
            {"stories": [{"id": "S1", "summary_en": "ok"}]}
        )
        client = LLMClient("llm7", "default", "")
        text = client.complete("SYSTEM PROMPT", "USER PROMPT")

        self.assertEqual(len(self.handler.requests), 1)
        sent = self.handler.requests[0]
        self.assertEqual(sent["path"], "/v1/chat/completions")
        self.assertIn("ok", text)

        body = sent["body"]
        self.assertEqual(body["model"], "default")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][0]["content"], "SYSTEM PROMPT")
        self.assertEqual(body["messages"][1]["role"], "user")
        self.assertEqual(body["messages"][1]["content"], "USER PROMPT")
        self.assertIn("temperature", body)
        self.assertLessEqual(body["max_tokens"], 4000)
        # The gateway fans out across backends, so JSON mode is not requested.
        self.assertNotIn("response_format", body)
        # And no credential is leaked when none was configured.
        self.assertNotIn("authorization", sent["headers"])

    def test_optional_token_is_sent_when_present(self):
        import json
        import os

        from news_scanner.analyze import LLMClient

        os.environ["LLM7_API_KEY"] = "sk-free-token"
        self.handler.reply = json.dumps({"stories": []})
        LLMClient("llm7", "default", "sk-free-token").complete("s", "u")
        sent = self.handler.requests[0]
        self.assertEqual(sent["headers"]["authorization"], "Bearer sk-free-token")

    def test_transient_5xx_is_retried_by_the_transport(self):
        import json

        from news_scanner.analyze import LLMClient

        self.handler.fail_next = True
        self.handler.reply = json.dumps({"stories": [{"id": "S1"}]})
        LLMClient("llm7", "default", "").complete("s", "u")
        self.assertEqual(len(self.handler.requests), 2, "the 503 should be retried")

    def test_end_to_end_analysis_and_render_through_the_gateway(self):
        import json

        canned = {
            "stories": [
                {
                    "id": "S1",
                    "headline_en": "Cabinet abolishes the Silk City development body",
                    "summary_en": "A draft decree abolishing the authority was approved.",
                    "why_it_matters_en": "A flagship project file changes hands.",
                    "between_the_lines_en": "The decree names no individual minister.",
                    "confidence": "medium",
                    "confidence_reason": "Official statements only.",
                },
                {
                    "id": "S2",
                    "headline_en": "Crown Prince holds talks in Bern",
                    "summary_en": "Talks covered trade and investment.",
                    "why_it_matters_en": "Continuity in the Swiss file.",
                    "between_the_lines_en": "No agenda detail was released.",
                    "confidence": "low",
                    "confidence_reason": "Wire copy only.",
                },
            ],
            "editor_note_en": "Silk City was the day's pivot.",
        }
        self.handler.reply = json.dumps(canned, ensure_ascii=False)

        engine = AnalysisEngine(
            AnalysisConfig(provider="llm7", use_cache=False, cache_path=None)
        )
        self.assertEqual(engine.provider, "llm7")
        self.assertEqual(engine.model, "default")

        stories = build_stories(make_articles(SILK_CITY + CROWN_PRINCE))
        result = engine.analyse(stories)
        self.assertFalse(result.errors, result.errors)
        self.assertEqual(result.provider, "llm7")

        digest = Digest(
            date="2026-09-30", generated_at=NOW, window_hours=24,
            stories=stories,
            analyses=[result.stories[s.key] for s in stories],
            synthesis=result.synthesis,
            source_health=[], llm_provider="llm7", errors=[],
            stats={"analysed_stories": len(stories), "listed_stories": len(stories)},
        )
        markdown = render_markdown(digest)
        self.assertIn("A draft decree abolishing the authority was approved.", markdown)
        self.assertIn("names no individual minister", markdown)
        self.assertIn("(inference)", markdown)
        self.assertIn("the day's pivot", markdown)
        # Every story still carries its sources.
        self.assertIn("https://news.example/kuwaittimes/3", markdown)

    def test_no_cache_never_writes_to_disk(self):
        """--no-cache must not persist results a later real run would reuse."""
        import json
        import tempfile

        self.handler.reply = json.dumps({"stories": [{"id": "S1", "summary_en": "stub"}]})
        path = Path(tempfile.mkdtemp()) / "analysis_cache.json"
        engine = AnalysisEngine(
            AnalysisConfig(provider="llm7", use_cache=False, cache_path=path)
        )
        stories = build_stories(make_articles(SILK_CITY))
        engine.analyse(stories)
        engine.save_cache()
        self.assertFalse(path.exists(), "a --no-cache run wrote a cache file")

    def test_a_dead_gateway_degrades_instead_of_crashing(self):
        """If LLM7 is unreachable the run must still produce a digest."""
        import os

        os.environ["NEWSCANNER_BASE_URL"] = "http://127.0.0.1:9/v1"  # discard port
        engine = AnalysisEngine(
            AnalysisConfig(provider="llm7", use_cache=False, cache_path=None)
        )
        stories = build_stories(make_articles(SILK_CITY))
        result = engine.analyse(stories)
        self.assertTrue(result.errors, "the failure should be reported")
        self.assertEqual(len(result.stories), len(stories))
        self.assertIn("without an ai provider", result.stories[stories[0].key]["why_it_matters_en"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
