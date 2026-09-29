"""Delivery: write files, commit-friendly output, and send email."""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path

log = logging.getLogger(__name__)


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def email_configured() -> bool:
    return bool(_env("SMTP_HOST") and _env("MAIL_TO"))


def send_email(subject: str, html_body: str, text_body: str) -> bool:
    """Send the digest over SMTP. Returns True when actually sent."""
    host = _env("SMTP_HOST")
    if not host:
        log.info("SMTP_HOST not set — skipping email")
        return False
    if not _env("MAIL_TO"):
        log.info("MAIL_TO not set — skipping email")
        return False

    port = int(_env("SMTP_PORT", default="587"))
    user = _env("SMTP_USER", "SMTP_USERNAME")
    password = _env("SMTP_PASSWORD", "SMTP_PASS")
    sender = _env("MAIL_FROM", default=user or "newsscanner@localhost")
    recipients = [a.strip() for a in _env("MAIL_TO").replace(";", ",").split(",") if a.strip()]
    security = _env("SMTP_SECURITY", default="starttls").lower()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr(("NewsScanner", sender))
    msg["To"] = ", ".join(recipients)
    msg.set_content(text_body or "See the HTML version of this email.")
    msg.add_alternative(html_body, subtype="html")

    context = ssl.create_default_context()
    try:
        if security in ("ssl", "tls", "smtps"):
            with smtplib.SMTP_SSL(host, port, context=context, timeout=45) as server:
                if user:
                    server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=45) as server:
                server.ehlo()
                if security == "starttls":
                    server.starttls(context=context)
                    server.ehlo()
                if user:
                    server.login(user, password)
                server.send_message(msg)
    except Exception as exc:
        log.error("email delivery failed: %s", exc)
        return False

    log.info("digest emailed to %s", ", ".join(recipients))
    return True


def write_outputs(
    output_dir: Path,
    *,
    date_str: str,
    markdown: str,
    html: str,
    payload: dict,
    keep_html: bool = True,
) -> dict[str, Path]:
    """Persist the Markdown digest, HTML email and JSON record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": output_dir / f"{date_str}.md",
        "json": output_dir / f"{date_str}.json",
    }
    paths["markdown"].write_text(markdown, "utf-8")
    import json

    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), "utf-8"
    )
    if keep_html:
        paths["html"] = output_dir / f"{date_str}.html"
        paths["html"].write_text(html, "utf-8")

    latest = output_dir / "latest.md"
    latest.write_text(markdown, "utf-8")
    paths["latest"] = latest

    return paths
