"""Tests for the Pages site builder (scripts/sync_docs.py).

Run with:  python3 -m unittest discover -s tests -v

These run fully offline. They protect the contract the Pages workflow
relies on: dated digests are published verbatim, scratch files (latest.*)
are ignored, the archive lists newest first, re-running changes nothing,
and editions removed from digests/ disappear from the published site too.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import sync_docs  # noqa: E402

DIGEST = (
    '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>'
    '<div style="font-size:24px;font-weight:700;color:#0b6b4f;">'
    "\U0001F1F0\U0001F1FC Kuwait Morning Brief</div>"
    '<div style="color:#65676b;font-size:13px;margin-top:4px;">'
    "Tuesday, 29 September 2026</div>"
    '<div style="color:#65676b;font-size:12px;margin-top:6px;">'
    "7 articles \u00b7 3 outlets \u00b7 4 stories \u00b7 1 major \u00b7 "
    "generated 08:00 Kuwait</div>"
    "</body></html>"
)


class SyncDocsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.digests = root / "digests"
        self.docs = root / "docs"
        self.digests.mkdir()

    def write_digest(self, name: str, body: str = DIGEST) -> None:
        (self.digests / name).write_text(body, encoding="utf-8")

    def publish(self) -> list[str]:
        return sync_docs.sync(self.digests, self.docs)

    def test_publishes_dated_digest_and_ignores_scratch_files(self) -> None:
        self.write_digest("2026-09-30.html")
        (self.digests / "latest.html").write_text("scratch", encoding="utf-8")
        (self.digests / "2026-09-30.json").write_text("{}", encoding="utf-8")

        self.publish()

        published = self.docs / "digests" / "2026-09-30.html"
        self.assertTrue(published.exists())
        self.assertEqual(published.read_text(encoding="utf-8"), DIGEST)
        self.assertFalse((self.docs / "digests" / "latest.html").exists())
        self.assertFalse((self.docs / "digests" / "2026-09-30.json").exists())

        manifest = json.loads(
            (self.docs / "digests" / "index.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["editions"]), 1)
        edition = manifest["editions"][0]
        self.assertEqual(edition["date"], "2026-09-30")
        self.assertEqual(edition["href"], "digests/2026-09-30.html")
        self.assertIn("7 articles", edition["stats"])

    def test_index_lists_newest_first(self) -> None:
        self.write_digest("2026-09-29.html")
        self.write_digest("2026-09-30.html")

        self.publish()

        index = (self.docs / "index.html").read_text(encoding="utf-8")
        hero = index.index('class="hero" href="digests/2026-09-30.html"')
        second = index.index("2026-09-29")
        self.assertLess(hero, second)

    def test_landing_page_has_accessible_responsive_navigation(self) -> None:
        self.write_digest("2026-09-30.html")

        self.publish()

        index = (self.docs / "index.html").read_text(encoding="utf-8")
        for feature in (
            'class="skip-link"',
            '<main>',
            'aria-labelledby="archive-title"',
            'class="archive-link"',
            'time datetime="2026-09-30"',
            'lang="ar" dir="rtl"',
            ":focus-visible",
            "@media (max-width: 600px)",
            "prefers-reduced-motion",
        ):
            self.assertIn(feature, index)

    def test_sync_is_idempotent(self) -> None:
        self.write_digest("2026-09-30.html")
        self.publish()

        self.assertEqual(self.publish(), [])

    def test_removed_editions_are_pruned(self) -> None:
        self.write_digest("2026-09-29.html")
        self.write_digest("2026-09-30.html")
        self.publish()

        (self.digests / "2026-09-29.html").unlink()
        changed = self.publish()

        self.assertIn("-docs/digests/2026-09-29.html", changed)
        self.assertFalse((self.docs / "digests" / "2026-09-29.html").exists())
        manifest = json.loads(
            (self.docs / "digests" / "index.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [e["date"] for e in manifest["editions"]], ["2026-09-30"]
        )

    def test_empty_archive_renders_placeholder(self) -> None:
        self.publish()

        index = (self.docs / "index.html").read_text(encoding="utf-8")
        self.assertIn("No editions published yet", index)
        manifest = json.loads(
            (self.docs / "digests" / "index.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["editions"], [])


if __name__ == "__main__":
    unittest.main()
