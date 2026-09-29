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

# Where the morning brief goes unless MAIL_TO says otherwise. Hard-coded
# rather than empty so that a missing repository secret still delivers.
DEFAULT_MAIL_TO = "mrmakrider@gmail.com"


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def email_enabled() -> bool:
    """False when delivery is explicitly switched off."""
    return os.getenv("NEWSCANNER_NO_EMAIL", "").strip() != "1"


def recipients() -> list[str]:
    """The MAIL_TO list, falling back to the built-in default address."""
    raw = _env("MAIL_TO", default=DEFAULT_MAIL_TO)
    return [a.strip() for a in raw.replace(";", ",").split(",") if a.strip()]


def email_configured() -> bool:
    return bool(email_enabled() and _env("SMTP_HOST") and recipients())


def verify_email() -> tuple[bool, str]:
    """Connect (and authenticate) without sending anything.

    Used by `check-email`, so that a wrong app password or a blocked port is
    discovered on demand rather than at 08:00.
    """
    if not email_enabled():
        return False, "email is switched off (NEWSCANNER_NO_EMAIL=1)"
    host = _env("SMTP_HOST")
    if not host:
        return False, (
            "SMTP_HOST is not set — email cannot be sent. "
            "Set SMTP_HOST, SMTP_PORT, SMTP_USER and SMTP_PASSWORD."
        )

    port = int(_env("SMTP_PORT", default="587"))
    user = _env("SMTP_USER", "SMTP_USERNAME")
    password = _env("SMTP_PASSWORD", "SMTP_PASS")
    security = _env("SMTP_SECURITY", default="starttls").lower()
    context = ssl.create_default_context()

    try:
        if security in ("ssl", "tls", "smtps"):
            with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
                if user:
                    server.login(user, password)
                server.noop()
        else:
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.ehlo()
                if security == "starttls":
                    server.starttls(context=context)
                    server.ehlo()
                if user:
                    server.login(user, password)
                server.noop()
    except Exception as exc:  # noqa: BLE001 — the point is to report it
        return False, f"{type(exc).__name__}: {exc}"

    detail = f"connected to {host}:{port} ({security})"
    detail += " and authenticated" if user else " (no credentials configured)"
    return True, detail


def send_email(subject: str, html_body: str, text_body: str) -> bool:
    """Send the digest over SMTP. Returns True when actually sent."""
    if not email_enabled():
        log.info("email switched off (NEWSCANNER_NO_EMAIL=1)")
        return False

    host = _env("SMTP_HOST")
    if not host:
        log.info("SMTP_HOST not set — skipping email")
        return False
    to_list = recipients()
    if not to_list:
        log.info("no recipients configured — skipping email")
        return False

    port = int(_env("SMTP_PORT", default="587"))
    user = _env("SMTP_USER", "SMTP_USERNAME")
    password = _env("SMTP_PASSWORD", "SMTP_PASS")
    sender = _env("MAIL_FROM", default=user or "newsscanner@localhost")
    security = _env("SMTP_SECURITY", default="starttls").lower()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((sender, sender)) if "@" in sender else formataddr(("NewsScanner", sender))
    msg["To"] = ", ".join(to_list)
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

    log.info("digest emailed to %s", ", ".join(to_list))
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
