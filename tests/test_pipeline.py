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
    def test_no_provider_detected_without_keys(self):
        import os

        saved = {k: os.environ.pop(k, None) for k in
                 ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                  "GEMINI_KEY", "OPENAI_KEY")}
        try:
            self.assertEqual(detect_provider("auto")[0], "none")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
