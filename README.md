# NewsScanner 🇰🇼

Every morning at **08:00 Kuwait time**, NewsScanner reads every Kuwaiti news
outlet it can reach, merges the coverage into single stories, and emails you a
briefed digest — with a reference to every original article, and an analysis
that tries to read what is *between* the lines.

```
42 articles · 7 outlets  →  23 stories  →  5 major  →  1 email
```

- **Aggregates everything.** Nine outlets, ~55 feed endpoints, plus best-effort
  scrapes for the outlets that publish no feed.
- **Merges coverage.** Five outlets writing about the same event become one
  story, in both Arabic and English, with every headline preserved.
- **Keeps the receipts.** Every article in the appendix, with its outlet, time
  and link. Nothing is summarised away.
- **Reads between the lines.** Per story: what happened, why it matters, and
  what a cautious Kuwaiti newsroom would notice but not print — attribution
  patterns, what the headline stresses, what is missing, how outlets diverge.
- **Works with no API key.** The default provider is the free LLM7 gateway, so
  a fresh fork gets real analysis on its first run. Add your own key and it
  takes over automatically.
- **Says when it doesn't know.** Every inference is labelled as inference, with
  a confidence level. If there is no subtext, it says so instead of inventing
  intrigue.
- **Zero dependencies.** Pure Python standard library. Nothing to install,
  nothing to break at 3am.

---

## Quick start

### 1. Fork or use this repository

### 2. Nothing — AI analysis works out of the box

The default provider is **[LLM7](https://api.llm7.io/v1)**, an
OpenAI-compatible gateway whose free tier needs no API key:

```
base URL : https://api.llm7.io/v1
model    : default          # balance of quality and latency
auth     : none required
```

So a fresh fork produces a fully analysed brief on its first run, with no
secrets configured at all.

**To use your own key instead**, add any one of these as a repository secret
(`Settings → Secrets and variables → Actions`) — an explicitly configured key
always takes precedence over the keyless default:

| Secret | Where to get it | Notes |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) | Most generous free tier for a daily job |
| `OPENAI_API_KEY` | platform.openai.com | Alternative |
| `ANTHROPIC_API_KEY` | console.anthropic.com | Alternative |
| `OPENROUTER_API_KEY` | openrouter.ai | Alternative |
| `GROQ_API_KEY` | console.groq.com | Alternative |
| `LLM7_API_KEY` | [token.llm7.io](https://token.llm7.io/) | Optional — raises LLM7's rate limit |

### 2b. Check it before 08:00 does

```bash
python -m news_scanner check-llm -v
```

This sends one real story through the configured provider and prints the
analysis it gets back, so a bad model name or an expired key surfaces
immediately rather than at 08:00. The workflow runs it too.

### 3. Add the mailbox it sends from

The brief is addressed to **mrmakrider@gmail.com** by default — you do not have
to configure a recipient. Set a `MAIL_TO` secret only to send it somewhere else
(comma-separate for several).

Sending still needs a mailbox to send *from*. Add these four secrets:

| Secret | Value |
|---|---|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | the Gmail account you send from |
| `SMTP_PASSWORD` | a Gmail **App Password** — *not* your login password |

Optional: `SMTP_SECURITY` (`starttls`, the default, or `ssl` for port 465),
`MAIL_FROM` (defaults to `SMTP_USER`), `MAIL_TO`, `NEWSCANNER_NO_EMAIL=1` to
switch delivery off.

**Gmail will reject your normal password.** You need an App Password:
turn on 2-Step Verification, then create one at
<https://myaccount.google.com/apppasswords> and paste the 16-character code
into `SMTP_PASSWORD`.

Check it before 08:00 does:

```bash
python -m news_scanner check-email          # connects and authenticates
python -m news_scanner check-email --send   # sends a real test message
```

### 4. Merge to `main` — this is what starts the clock

**GitHub only fires `schedule` triggers for the default branch.** While this
workflow sits on a feature branch it will never run by itself, no matter what
the cron says.

Once it is on `main`:

| Kuwait time | UTC cron | What happens |
|---|---|---|
| **08:00** | `0 5 * * *` | Collects, analyses, emails the brief |
| 08:30 | `30 5 * * *` | Retry — exits instantly if 08:00 already delivered |
| 09:00 | `0 6 * * *` | Last chance for the day |

The two later crons are a safety net for a dropped or failed run, not a second
edition: they read `state/last_run.json` and do nothing if today's brief
already went out, so you will never get the digest twice.

Trigger it once by hand first:
`Actions → Daily Kuwait News Brief → Run workflow` (tick **dry_run**).

### Where it is sent

`mrmakrider@gmail.com`, daily at 08:00 Kuwait time. Override the address with a
`MAIL_TO` secret; nothing else needs changing.

> GitHub's scheduler can fire a few minutes late under load, and it disables
> crons after 60 days of repository inactivity — which this job prevents by
> committing every day. If the exact minute matters, set the first cron to
> `45 4 * * *` for a comfortable margin.

---

## What the digest looks like

```
digests/
├── 2026-09-30.md      ← the briefing, committed to the repo
├── 2026-09-30.html    ← the email version
├── 2026-09-30.json    ← machine-readable: stories, analyses, source health
└── latest.md          ← always the most recent run
```

See the structure for yourself without any network access:

```bash
python3 scripts/demo.py
```

---

## Read it online

The archive is also published as a small static site at
**https://mrmakrider.github.io/NewsScanner/** — served by GitHub Pages from
the `docs/` folder of `main`, and refreshed automatically after every
morning brief by `.github/workflows/pages.yml`.

Enabling the site is a one-time step that GitHub only allows the repo owner
to do (workflow tokens cannot create Pages sites):

- **In the browser:** Settings → Pages → Source: *Deploy from a branch* →
  branch `main`, folder `/docs` → Save.
- **Or from a terminal logged in as you:**

  ```bash
  gh api -X POST repos/mrmakrider/NewsScanner/pages \
    -f "source[branch]=main" -f "source[path]=/docs"
  ```

That's it — every edition lands in the archive at
`docs/digests/YYYY-MM-DD.html` with a manifest at `docs/digests/index.json`,
regenerated by `scripts/sync_docs.py` (run `--check` in CI to catch a
stale site). Until Pages is enabled the workflows stay green and simply log
a reminder in the "Ask Pages to rebuild" step.

---

## How it works

```
 collect ──► cluster ──► enrich ──► analyse ──► render ──► deliver
   │           │           │          │          │          │
 55 feeds     same story  fetch full  LLM or    MD/HTML/    SMTP +
 in parallel  across 2    body text   extractive JSON        repo commit
              languages
```

**1. Collect.** Every feed is fetched concurrently with retries and a real
browser user-agent. A dead outlet is recorded in a *source health* table and
never aborts the run.

**2. Cluster.** Headlines are normalised (Arabic diacritics, `أ/إ/آ → ا`,
`ة → ه`, `ى → ي`, Eastern Arabic numerals), stemmed, and mapped through a
curated Arabic→English news lexicon — so `مجلس الوزراء` and `Cabinet` become
the same token. Two headlines merge only if they are ≥ 0.50 similar **and**
agree on at least two *distinctive* tokens, which is what stops
"oil prices rise" from swallowing "gold prices rise". A second pass pairs up
clusters that only overlap through that entity vocabulary, restricted to
clusters that share no language — so it can never loosen same-language
precision.

**3. Enrich.** The full article text is fetched for the stories about to be
analysed. Headlines alone produce shallow analysis.

**4. Analyse.** Stories are sent to the model in batches, each with *every*
outlet's version of the headline — the divergence between them is often the
story. The model is instructed to work only from the supplied material,
separate stated / implied / hypothesis, ground every inference in a concrete
textual signal, and say plainly when there is no subtext.

**5. Render & deliver.** Bilingual Markdown, HTML email, and JSON. Results are
cached in `state/analysis_cache.json` so a re-run never pays twice.

---

## Configuration

### Sources — `config/sources.toml`

Currently configured (all verified live):

| Outlet | Language | Tier | Feeds |
|---|---|---|---|
| الرأي — Al-Rai | ar | 1 | 6 |
| الأنباء — Al-Anba | ar | 1 | 19 |
| الجريدة — Al-Jarida | ar | 1 | 7 |
| كويت نيوز — Kuwait News | ar | 2 | 1 |
| كونا — KUNA (state wire) | ar | 1 | scrape* |
| القبس — Al-Qabas | ar | 1 | scrape* |
| Kuwait Times | en | 1 | 9 |
| Arab Times | en | 2 | 8 |
| Times Kuwait | en | 2 | 1 |

\* KUNA publishes no public RSS and Al-Qabas dropped its feed; both are
best-effort scrapes marked `optional`, so a failure is reported but harmless.
In practice almost all KUNA wire copy still reaches you, because Al-Anba and
Al-Jarida reprint it.

Adding an outlet takes three lines:

```toml
[[source]]
id = "example"
name_en = "Example Daily"
name_ar = "مثال"
lang = "ar"
type = "rss"
tier = 2
feeds = [ { url = "https://example.com/feed/", section = "local" } ]
```

Other settings live under `[settings]`:

```toml
window_hours = 26            # how far back the 08:00 run looks
max_analyze  = 28            # stories that get full analysis
major_outlet_threshold = 3   # outlets needed to call a story "major"
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `NEWSCANNER_PROVIDER` | auto-detected | `llm7`, `gemini`, `openai`, `anthropic`, `openrouter`, `groq`, `ollama`, `none` |
| `NEWSCANNER_MODEL` | per provider | For LLM7 also accepts `fast` or `pro` |
| `NEWSCANNER_LANGUAGE` | `bilingual` | `bilingual`, `en`, `ar` |
| `NEWSCANNER_TEMPERATURE` | `0.25` | Lower = more literal |
| `NEWSCANNER_NO_LLM7` | – | `1` to skip the keyless default and get the extractive digest |
| `NEWSCANNER_BASE_URL` | `https://api.llm7.io/v1` | Point LLM7 at a mirror or self-hosted gateway |
| `LLM7_API_KEY` | – | Optional LLM7 token, raises the rate limit |
| `NEWSCANNER_USE_OLLAMA` | – | `1` to allow the local Ollama provider |

**Provider precedence.** Explicit configuration always beats the default:
`NEWSCANNER_PROVIDER` → any configured API key → `OLLAMA_HOST` →
**LLM7 (keyless default)** → extractive fallback.

---

## Command line

```bash
python3 -m news_scanner run                 # the daily job
python3 -m news_scanner check-sources       # is every outlet reachable?
python3 -m news_scanner check-llm           # does the analysis provider work?
python3 -m news_scanner check-email --send  # does delivery actually work?
python3 -m news_scanner run --dry-run       # everything except email
python3 -m news_scanner run --no-llm        # extractive only, no API cost
python3 -m news_scanner run --no-fetch-bodies   # faster, shallower
python3 -m news_scanner run --hours 8       # narrow the window
python3 -m news_scanner run --language en   # English-only analysis
python3 -m news_scanner run --sources alrai,kuwaittimes   # test one pair
python3 -m news_scanner run --max-analyze 10 -v
```

Run it locally around a cron job if you prefer your own machine:

```bash
cd /path/to/NewsScanner
0 5 * * *  cd /path/to/NewsScanner && GEMINI_API_KEY=... python3 -m news_scanner run >> /tmp/newsscanner.log 2>&1
```

Requires **Python 3.11+** (for `tomllib`). Nothing to `pip install`.

---

## Honest limitations

- **Clustering is lexical, not semantic.** Two stories can stay separate when
  outlets paraphrase heavily — e.g. a headline about `ذهبيتان وبرونزيتان`
  (nominative dual) and another about `بذهبيتين وبرونزيتين` (accusative dual)
  will not always merge. Every article is still listed with its link, so the
  cost of a miss is a repeated entry, never a lost one.
- **Over-merging is bounded, not impossible.** The distinctive-token gate makes
  it rare, and when it happens no information is lost: every outlet's own
  headline and link stays in the reference list.
- **"Between the lines" is machine inference.** It is labelled as such, carries
  a confidence level, and is instructed to be honest when there is nothing to
  read. It is a prompt for your own judgement, not a finding.
- **Scraped sources are fragile.** KUNA and Al-Qabas may break when they
  redesign. `check-sources` tells you.
- **The free LLM7 tier is shared and rate-limited** (roughly 100 requests per
  hour anonymously, `default` routing). A run makes about a dozen calls, so
  that is ample — but if the gateway is busy, the affected batch silently
  degrades to the extractive text rather than failing the run. A token from
  [token.llm7.io](https://token.llm7.io/) or your own provider key removes
  the shared-tier variability.
- **The state wire dominates coverage.** Most Kuwaiti outlets reprint KUNA, so
  a wide cluster often means "the wire carried it", not "five newsrooms
  independently confirmed it". The digest shows you each version so you can
  tell the difference.

## Legal

NewsScanner reads publicly available RSS feeds and web pages and stores only
headlines, links and short excerpts for personal briefing use. It does not
republish article bodies. All content belongs to its original publishers.
