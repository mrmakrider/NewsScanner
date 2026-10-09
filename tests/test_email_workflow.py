"""Workflow contract tests for mail delivery and scheduled retries."""

from __future__ import annotations

import unittest
from pathlib import Path


class TestDailyEmailWorkflow(unittest.TestCase):
    def test_smtp_preflight_does_not_disable_the_real_send_or_retries(self):
        workflow = (
            Path(__file__).resolve().parent.parent
            / ".github"
            / "workflows"
            / "daily-digest.yml"
        ).read_text("utf-8")

        preflight = workflow.split("- name: Verify email delivery", 1)[1].split(
            "- name:", 1
        )[0]
        self.assertIn("continue-on-error: true", preflight)
        self.assertNotIn("steps.email_check.outcome", workflow)
        self.assertNotIn("NEWSCANNER_NO_EMAIL:", workflow)
        self.assertIn('cron: "0 0 * * *"', workflow)
        self.assertIn('cron: "30 0 * * *"', workflow)
        self.assertIn('cron: "0 1 * * *"', workflow)


if __name__ == "__main__":
    unittest.main()
