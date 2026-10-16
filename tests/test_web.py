"""Tests for the published web edition (news_scanner/web.py).

The page is the main way anyone reads this brief, and it is the one renderer
whose output is *executed*: the search box, the category filter and the
back-to-top button are code in the document. Those three failure modes are
covered here — a filter whose counts disagree with what it filters, an anchor
pointing at a story that is not on the page, and content that disappears
without JavaScript.

Everything here is offline and builds its own Digest, so no digest file, no
network and no LLM is involved.
"""

from __future__ import annotations

import re
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from news_scanner.models import Article, Digest, Story  # noqa: E402
from news_scanner.web import render_web  # noqa: E402

NOW = datetime(2026, 9, 30, 8, 0, tzinfo=timezone(timedelta(hours=3)))


def article(idx: int, name: str, title: str, *, lang: str = "en") -> Article:
    return Article(
        source_id=f"s{idx}",
        source_name=name,
        lang=lang,
        title=title,
        url=f"https://news.example/{name}/{idx}",
        summary=f"Summary for {title}",
        published=NOW - timedelta(hours=idx),
        tier=1,
    )


def story(key: str, articles: list[Article]) -> Story:
    return Story(key=key, score=1.0, articles=articles)


def digest(
    stories: list[Story],
    analyses: list[dict],
    synthesis: dict | None = None,
    *,
    provider: str = "gemini",
    stats: dict | None = None,
) -> Digest:
    base = {"analysed_stories": len(stories)}
    base.update(stats or {})
    return Digest(
        date="2026-09-30",
        generated_at=NOW,
        window_hours=24,
        stories=stories,
        analyses=analyses,
        synthesis=synthesis or {},
        source_health=[],
        stats=base,
        llm_provider=provider,
        errors=[],
    )


class TestDocument(unittest.TestCase):
    """The page must be a real document, not a bag of styled divs."""

    def setUp(self):
        self.stories = [
            story("cabinet", [article(0, "KUNA", "Cabinet approves the new port decree")]),
            story("match", [
                article(1, "Alrai", "Al Ahli wins the league title", lang="ar"),
                article(2, "Alrai", "Al Ahli secure the championship"),
            ]),
        ]
        self.analyses = [
            {
                "summary_en": "The cabinet signed the decree.",
                "between_the_lines_en": "The timing suggests pressure.",
                "_source": "llm",
                "confidence": "medium",
                "confidence_reason": "single outlet",
            },
            {"summary_en": "A goal in the 88th minute."},
        ]
        self.page = render_web(digest(self.stories, self.analyses))

    def test_is_a_complete_html_document(self):
        self.assertTrue(self.page.startswith("<!DOCTYPE html>"))
        self.assertIn("<meta charset=\"utf-8\">", self.page)
        self.assertIn('name="viewport"', self.page)
        self.assertIn("</html>", self.page)

    def test_has_a_stylesheet_not_just_inline_styles(self):
        self.assertIn("<style>", self.page)
        # The email version is inline-only; the page must not be a copy of it.
        self.assertIn(".story {", self.page)
        self.assertIn("@media (max-width: 560px)", self.page)

    def test_declares_colour_scheme_and_carries_a_dark_theme(self):
        self.assertIn('name="color-scheme" content="light dark"', self.page)
        self.assertIn("prefers-color-scheme: dark", self.page)

    def test_has_the_progressive_enhancement_script(self):
        self.assertIn("<script>", self.page)
        self.assertIn("id=\"to-top\"", self.page)

    def test_language_and_direction_attributes_survive(self):
        # An Arabic headline in an LTR document is set dir="auto" on the title;
        # without it the browser guesses per-run and the text jumps.
        self.assertIn('class="story-title" dir="auto"', self.page)

    def test_rtl_blocks_are_marked(self):
        ar_story = story(
            "ar", [article(9, "Alrai", "مجلس الوزراء يوافق على المرسوم", lang="ar")]
        )
        page = render_web(
            digest([ar_story], [{"summary_ar": "وافق المجلس على المرسوم."}])
        )
        self.assertIn('dir="rtl"', page)


class TestAnchors(unittest.TestCase):
    """Every internal link must land on a story that is actually rendered."""

    def _fixture(self, synthesis: dict, count: int = 3) -> str:
        stories = [
            story(f"s{i}", [article(i, "KUNA", f"Story number {i} about Kuwait")])
            for i in range(count)
        ]
        return render_web(digest(stories, [{} for _ in stories], synthesis))

    def test_every_story_anchor_resolves(self):
        synthesis = {
            "links": [
                {"type": "consequence", "from": "S1", "to": "S2",
                 "evidence": "because the decree landed first"},
            ]
        }
        page = self._fixture(synthesis)
        ids = set(re.findall(r'id="story-([^"]+)"', page))
        targets = set(re.findall(r'href="#story-([^"]+)"', page))
        self.assertTrue(targets, "expected at least one cross-story link")
        self.assertEqual(targets - ids, set())

    def test_theme_ids_are_not_double_prefixed(self):
        # story_ids arrive as "S2". Rendering "#story-S{S2}" produced
        # #story-SS2 and every theme chip was dead.
        page = self._fixture(
            {"top_themes": [{"theme_en": "Decrees", "story_ids": ["S2", "S3"]}]}
        )
        self.assertIn('href="#story-S2"', page)
        self.assertNotIn("#story-SS", page)

    def test_links_to_unrendered_stories_are_dropped(self):
        synthesis = {
            "links": [
                {"type": "same_actor", "from": "S1", "to": "S99",
                 "evidence": "invented"},
            ]
        }
        page = self._fixture(synthesis)
        self.assertNotIn('href="#story-S99"', page)
        self.assertNotIn("invented", page)


class TestFilterContract(unittest.TestCase):
    """The category tiles are buttons; their numbers must be the truth.

    The tiles are the filter control, so a count of 19 that reveals 2 stories
    is a lie the reader can check in one click.
    """

    def setUp(self):
        self.stories = [
            story("sport", [article(0, "KUNA", "League title decided in extra time")]),
            story("sport2", [article(1, "Alrai", "Cup final ends in a draw")]),
            story("court", [article(2, "Alrai", "Court sentences the fraud suspect")]),
            # Not analysed, so not on the page — its category must not be
            # counted on a tile the reader can click.
            story("sport3", [article(3, "KUNA", "Another sports story entirely")]),
        ]
        self.analyses = [{}, {}, {}]
        self.page = render_web(
            digest(
                self.stories,
                self.analyses,
                stats={"analysed_stories": 3, "total_stories": 4},
            )
        )

    def test_tiles_count_only_the_stories_on_the_page(self):
        counts = {
            m.group(2): int(m.group(1))
            for m in re.finditer(
                r'class="tile-n">(\d+)</span><span class="tile-l">([^<]+)</span>', self.page
            )
        }
        # The label carries HTML entities ("Security &amp; Courts"); only the
        # Sport tile is needed here, so match on its unambiguous name.
        self.assertEqual(counts.get("Sport"), 2, "the 2 rendered sport stories")
        self.assertNotIn("sport3", " ".join(counts))

    def test_every_tile_matches_a_category_actually_present(self):
        slugs = set(re.findall(r'data-filter="([^"]+)"', self.page))
        rendered = set(
            re.findall(r'class="story[^"]*" id="story-S\d+" data-category="([^"]+)"', self.page)
        )
        # Normalise the story-attribute capture, which includes leading junk.
        rendered = {r for r in rendered if r}
        self.assertTrue(slugs)
        self.assertEqual(slugs - rendered, set())

    def test_each_story_carries_the_attributes_the_filter_reads(self):
        for name in ("data-category", "data-search"):
            self.assertIn(name, self.page)
        # Searchable text must include the headline, or search cannot work.
        block = re.search(r'data-search="([^"]*)"', self.page)
        self.assertIn("league", block.group(1).lower())


class TestNoJavaScriptStillReads(unittest.TestCase):
    """The filters are a convenience over complete markup, never the only path."""

    def setUp(self):
        self.stories = [
            story("a", [article(0, "KUNA", "Cabinet approves decree")]),
            story("b", [article(1, "Alrai", "Cup final ends in a draw")]),
        ]
        self.page = render_web(
            digest(
                self.stories,
                [{"summary_en": "One."}, {"summary_en": "Two."}],
                {"editor_note_en": "The day in one paragraph."},
            )
        )

    def test_no_story_is_hidden_in_the_markup(self):
        # The only `hidden` attributes belong to the controls, never a card.
        for match in re.finditer(r'<article class="story[^>]*>', self.page):
            self.assertNotIn("hidden", match.group(0))

    def test_prose_is_in_the_html_not_built_by_script(self):
        self.assertIn("The day in one paragraph.", self.page)
        self.assertIn("Cabinet approves decree", self.page)
        self.assertIn("Cup final ends in a draw", self.page)

    def test_references_are_details_not_script_generated(self):
        self.assertEqual(self.page.count("<details class=\"refs\">"), 2)


class TestEscaping(unittest.TestCase):
    """Headlines come from feeds; they are untrusted text."""

    def test_markup_in_a_headline_is_escaped(self):
        hostile = story(
            "x", [article(0, "KUNA", '<script>alert("x")</script> & "quotes"')]
        )
        page = render_web(digest([hostile], [{"summary_en": "ok"}]))
        self.assertNotIn("<script>alert", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("&amp;", page)

    def test_a_hostile_url_cannot_inject_an_attribute(self):
        bad = Article(
            source_id="s", source_name="KUNA", lang="en", title="t",
            url='" onmouseover="alert(1)',
            published=NOW,
        )
        page = render_web(digest([story("x", [bad])], [{}]))
        self.assertNotIn('onmouseover="alert', page)

    def test_a_hostile_analysis_string_is_escaped(self):
        hostile = story("x", [article(0, "KUNA", "headline")])
        page = render_web(
            digest(
                [hostile],
                [{"summary_en": "<img src=x onerror=alert(1)>"}],
            )
        )
        # The dangerous part is a live tag, not the characters: the payload is
        # present as text, but never as markup the browser will act on.
        self.assertNotIn("<img", page)
        self.assertIn("&lt;img", page)


class TestGracefulDegradation(unittest.TestCase):
    """A day with nothing in it must still produce a readable page."""

    def test_empty_digest_renders_a_note_not_a_crash(self):
        page = render_web(digest([], [], {}))
        self.assertIn("<!DOCTYPE html>", page)
        self.assertIn("No stories were carried", page)

    def test_extractive_mode_is_labelled(self):
        stories = [story("a", [article(0, "KUNA", "Something happened")])]
        page = render_web(digest(stories, [{}], provider="none"))
        self.assertIn("extractive mode", page)

    def test_malformed_synthesis_entries_are_skipped(self):
        stories = [story("a", [article(0, "KUNA", "Something happened")])]
        page = render_web(
            digest(
                stories,
                [{}],
                {
                    "top_themes": ["not a dict", {"theme_en": "", "story_ids": []}],
                    "links": "not a list",
                },
            )
        )
        self.assertIn("<!DOCTYPE html>", page)


class TestPreviewLoader(unittest.TestCase):
    """scripts/preview_web.py rebuilds a Digest from a committed JSON record.

    It is the only way to see a renderer change against a real edition, so a
    silent failure there means nobody previews again.
    """

    def test_loads_a_digest_round_tripped_through_json(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import json
        import tempfile

        import preview_web
        from news_scanner.render import render_json

        stories = [
            story("a", [article(0, "KUNA", "Cabinet approves decree")]),
            story("b", [article(1, "Alrai", "Cup final ends in a draw")]),
        ]
        original = digest(stories, [{"summary_en": "One."}, {"summary_en": "Two."}])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.json"
            path.write_text(
                json.dumps(render_json(original), ensure_ascii=False), "utf-8"
            )
            reloaded = preview_web.load_digest(path)

        self.assertEqual(len(reloaded.stories), 2)
        self.assertEqual(reloaded.date, original.date)
        # sources/outlet_count are derived, so they must survive the trip.
        self.assertEqual(
            reloaded.stories[0].sources, ["KUNA"]
        )
        self.assertIn("Cabinet approves decree", render_web(reloaded))


if __name__ == "__main__":
    unittest.main()
