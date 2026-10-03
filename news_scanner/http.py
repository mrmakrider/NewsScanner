"""Tiny resilient HTTP client built on urllib (no third-party deps)."""

from __future__ import annotations

import codecs

import gzip
import io
import json
import logging
import random
import socket
import time
import urllib.error
import urllib.request
import zlib

from .resilience import parse_retry_after

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 25
DEFAULT_RETRIES = 2

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 NewsScanner/1.0"
)

BASE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ar,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate",
    "Cache-Control": "no-cache",
    "Connection": "close",
}


# Status codes worth retrying: transient server faults, and the two Cloudflare
# codes a gateway in front of a busy model returns (522 origin down, 524
# origin timed out). A 524 from an LLM gateway means the generation took too
# long, which a smaller prompt can fix — see the batch splitting in analyze.
RETRYABLE_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 522, 524, 599})

# How long a Retry-After we are willing to sit through inside the transport.
# Beyond this the caller's own policy is better placed to decide: it can try a
# different provider or shrink the request instead of sleeping.
MAX_RETRY_AFTER = 30.0


class FetchError(RuntimeError):
    """A request that could not be completed.

    ``code`` and ``retry_after`` carry back whatever the server said, so
    callers can apply their own policy: the analysis engine paces itself and
    fails over to another provider, while the feed fetcher only needs to know
    that this source is unhappy.
    """

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        retry_after: float | None = None,
        url: str = "",
    ):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after
        self.url = url


def _decompress(raw: bytes, encoding: str) -> bytes:
    encoding = (encoding or "").lower()
    try:
        if "gzip" in encoding:
            return gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        if "deflate" in encoding:
            try:
                return zlib.decompress(raw)
            except zlib.error:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
    except Exception:
        return raw
    return raw


def fetch(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    method: str | None = None,
    retry_connect_errors: bool = True,
) -> tuple[bytes, str]:
    """GET a URL and return ``(body_bytes, content_type)``.

    Retries transient failures with exponential backoff plus jitter so a
    daily unattended run survives a flaky outlet.

    ``retry_connect_errors=False`` skips the retry loop for connection-level
    failures (refused, DNS, reset) while still retrying HTTP status codes.
    The analysis engine sets it: it keeps a slower, smarter central retry
    policy of its own, and a host that refuses a connection will refuse it
    again two seconds later — retrying there only delays the failover that
    actually helps.
    """
    hdrs = dict(BASE_HEADERS)
    if headers:
        hdrs.update(headers)

    last_err: Exception | None = None
    last_code: int | None = None
    last_retry_after: float | None = None
    for attempt in range(retries + 1):
        if attempt:
            sleep = (2 ** attempt) * 0.8 + random.uniform(0, 0.6)
            time.sleep(sleep)
        req = urllib.request.Request(
            url, data=data, headers=hdrs, method=method or ("POST" if data else "GET")
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                raw = _decompress(raw, resp.headers.get("Content-Encoding", ""))
                ctype = resp.headers.get("Content-Type", "")
                return raw, ctype
        except urllib.error.HTTPError as exc:
            last_err = exc
            last_code = exc.code
            last_retry_after = parse_retry_after(
                exc.headers.get("Retry-After") if exc.headers else None
            )
            # 4xx other than 429 will not fix themselves; stop early.
            if exc.code not in RETRYABLE_CODES:
                break
            # A server that names its own delay is obeyed rather than guessed
            # at — but only up to a cap, after which the caller's policy (try
            # another provider, shrink the request) beats sleeping here.
            if (
                last_retry_after is not None
                and attempt < retries
                and last_retry_after <= MAX_RETRY_AFTER
            ):
                time.sleep(last_retry_after + random.uniform(0.0, 0.5))
            elif last_retry_after is not None and last_retry_after > MAX_RETRY_AFTER:
                break
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            last_err = exc
            last_code = None
            if not retry_connect_errors:
                break

    raise FetchError(
        f"{url}: {last_err}",
        code=last_code,
        retry_after=last_retry_after,
        url=url,
    )


def fetch_text(url: str, **kwargs) -> str:
    raw, _ = fetch(url, **kwargs)
    # A feed that states its charset is read with it. Skipping this was why an
    # iso-8859-6 or windows-1252 feed could arrive as Arabic-looking nonsense:
    # the guess chain cannot tell those codecs apart from the bytes alone.
    return decode(raw, guess_encoding_from_meta(raw))


def fetch_json(
    url: str,
    payload: dict | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 90,
    retries: int = 2,
    retry_connect_errors: bool = True,
) -> dict:
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    raw, _ = fetch(
        url,
        headers=hdrs,
        data=data,
        timeout=timeout,
        retries=retries,
        retry_connect_errors=retry_connect_errors,
    )
    try:
        return json.loads(decode(raw))
    except json.JSONDecodeError as exc:
        raise FetchError(f"{url}: invalid JSON response ({exc})") from exc


def decode(raw: bytes, declared: str | None = None) -> str:
    """Decode bytes using the encodings Arabic news sites actually use.

    ``declared`` is the charset the document states for itself, when one was
    found, and it is the only reliable answer. cp1256 and iso-8859-6 are both
    single-byte codecs that decode every byte without raising and both yield
    Arabic, so nothing in the byte stream can distinguish them — measurements
    of Arabic-letter ratio and stray-symbol count come out identical for both.
    A wrong guess does not crash; it silently yields a headline that reads as
    Arabic but is not, which is far worse than a visible decode failure.

    Without a declaration, UTF-8 is tried first (strict, so it never guesses)
    and cp1256 second, which is what most Kuwaiti outlets actually serve.
    """
    if declared:
        try:
            return raw.decode(declared)
        except (UnicodeDecodeError, LookupError):
            # The page lies about its encoding, or was served truncated. Fall
            # through to the guess chain rather than trusting the claim.
            log.debug("declared charset %r did not decode; guessing", declared)

    for enc in ("utf-8", "cp1256"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # Undecodable: mark it rather than silently substituting. A visible
    # replacement character tells a reader the text is damaged; a
    # plausible-looking wrong decoding does not.
    return raw.decode("utf-8", errors="replace")


def guess_encoding_from_meta(raw: bytes) -> str | None:
    """The charset the document declares for itself, if it declares one.

    Quoted forms are matched first: ``encoding="utf-8"`` also contains the
    unquoted marker ``encoding=``, and matching that one leaves the opening
    quote at the head of the value, which then looks like an unknown codec and
    gets discarded.
    """
    head = raw[:4096].lower()
    markers = (
        (b'charset="', b'"'),
        (b"charset='", b"'"),
        (b'encoding="', b'"'),
        (b"encoding='", b"'"),
        (b"charset=", b""),
        (b"encoding=", b""),
    )
    for marker, closer in markers:
        idx = head.find(marker)
        if idx == -1:
            continue
        rest = head[idx + len(marker):]
        end = len(rest)
        for i, byte in enumerate(rest):
            # Stop at the closing quote, or at any byte that cannot occur in an
            # encoding name. '"' and "'" terminate the unquoted forms too.
            if byte in (0x22, 0x27, 0x3B, 0x20, 0x3E, 0x2F, 0x0A, 0x0D, 0x09):
                end = i
                break
        name = rest[:end].decode("ascii", "ignore").strip()
        if name and _is_known_encoding(name):
            return name
    return None


def _is_known_encoding(name: str) -> str | None:
    try:
        codecs.lookup(name)
    except (LookupError, TypeError):
        return None
    return name
