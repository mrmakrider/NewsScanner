"""Cluster articles from many outlets into single stories, then rank them.

This is what turns "412 articles" into "86 stories, 12 of them major".

Clustering happens in two deliberate, separate stages:

1. ``cluster_articles`` groups headlines within a language, using a symmetric
   headline-similarity threshold. This is high precision.

2. ``bridge_languages`` then pairs up *monolingual* clusters that describe the
   same event in different languages. Arabic and English headlines share
   almost no surface vocabulary — only the curated entity vocabulary from
   ``util.ALIASES`` — so they need their own, looser rule. Restricting that
   rule to clusters that do not yet share a language means it can never
   loosen same-language matching, which is where the precision matters.
"""

from __future__ import annotations

import re
import logging
from collections import Counter, defaultdict

from .models import Article, Story
from .util import (
    canonical_tokens,
    containment,
    distinctive_tokens,
    jaccard,
    normalize_text,
    now_kuwait,
    title_fingerprint,
    to_kuwait,
)

log = logging.getLogger(__name__)

# Similarity needed to treat two headlines as the same story. Similarity is
# the larger of Jaccard and containment, because a short wire line is often a
# strict subset of another outlet's longer rewrite of the same event.
SAME_STORY_THRESHOLD = 0.50

# However similar two headlines look, they are only merged when they also
# agree on at least this many *distinctive* tokens (names, entities, figures),
# so "oil prices rise" and "gold prices rise" cannot merge on boilerplate.
MIN_SHARED_DISTINCTIVE = 2

# Rule for pairing an Arabic cluster with an English one.
CROSS_LANG_ANCHOR_MIN = 2
CROSS_LANG_ANCHOR_SCORE = 0.30

TIER_WEIGHT = {1: 3.0, 2: 2.0, 3: 1.0}

# Sections that genuinely say what a story is about, versus the catch-all
# buckets ("latest", "kuwait") that outlets use for everything.
CATEGORY_SECTIONS: dict[str, dict[str, set[str]]] = {
    "Government & Politics": {
        "strong": {"official", "parliament", "politics", "cabinet",
                   "front-page", "top-stories"},
        "weak": {"local", "kuwait", "latest", "general", "last-page"},
    },
    "Economy & Business": {
        "strong": {"economy", "business", "markets", "energy", "realestate",
                   "economy-reports"},
        "weak": set(),
    },
    "Security & Courts": {
        "strong": {"security-courts", "crime", "investigations"},
        "weak": set(),
    },
    "Society & Services": {
        "strong": {"regions", "municipal", "cooperatives", "initiatives",
                   "health-science"},
        "weak": set(),
    },
    "Sport": {"strong": {"sport"}, "weak": set()},
    "World & Region": {
        "strong": {"world", "middle-east", "gulf", "un", "world-reports"},
        "weak": set(),
    },
    "Culture & Entertainment": {
        "strong": {"culture", "entertainment"},
        "weak": set(),
    },
    "Opinion & Analysis": {"strong": {"opinion"}, "weak": set()},
}

CATEGORY_ORDER = [
    "Government & Politics",
    "Economy & Business",
    "Security & Courts",
    "Society & Services",
    "Sport",
    "World & Region",
    "Culture & Entertainment",
    "Opinion & Analysis",
    "Other",
]

# Section labels are too coarse on their own ("latest" covers everything from a
# cabinet decision to a drug bust), so headlines are also scored against
# vocabulary and the two signals are combined.
CATEGORY_KEYWORDS_RAW: dict[str, list[str]] = {
    "Security & Courts": [
        "arrest", "drug", "court", "verdict", "prison", "police", "crime",
        "smuggl", "customs", "prosecut", "seiz", "sentence", "jail", "theft",
        "fraud", "murder", "terror", "offence", "offense", "suspect",
        "أمن", "ضبط", "مخدرات", "محكمة", "جنايات", "نيابة", "سجن", "غرامة",
        "قضية", "تهريب", "جمارك", "جريمة", "الشرطة", "الحكم", "اعتقال",
        "متهم", "تحقيق", "قضايا", "أحكام", "استئناف", "مدان", "مستندات",
    ],
    "Economy & Business": [
        "oil", "barrel", "bourse", "market", "stock", "bank", "econom",
        "finance", "invest", "budget", "gold price", "bullion", "ounce",
        "price", "trade", "export",
        "import", "opec", "subsidy", "revenue", "profit", "loan", "tender",
        "insurance", "dinar",
        "نفط", "برميل", "بورصة", "سوق", "بنك", "اقتصاد", "استثمار",
        "ميزانية", "ذهب", "سعر", "تصدير", "استيراد", "تمويل", "أرباح",
        "إيرادات", "عقود", "مناقصة", "دعم", "قرض", "فائدة", "تضخم",
        "دينار", "أسعار", "رواتب", "أجور", "أسواق",
    ],
    "Sport": [
        "match", "league", "goal", "player", "coach", "tournament", "medal",
        "olympic", "asian games", "gulf cup", "fifa", "football", "club",
        "handball", "semifinal", "champion", "sport", "cup",
        "shooter", "shooting", "athlete", "medal", "rifle", "championship",
        "qualifier", "tournament",
        "مباراة", "دوري", "كأس", "هدف", "لاعب", "منتخب", "مدرب", "بطولة",
        "ميدالية", "أولمبياد", "آسياد", "ذهبي", "فضية", "برونز", "رياضة",
        "فريق", "خليجي", "تصفيات", "لاعبين", "رماية",
    ],
    "Society & Services": [
        "school", "student", "hospital", "health", "teacher", "university",
        "electricity", "water", "municipal", "cooperative", "housing",
        "weather", "traffic", "road", "fire", "accident", "wage", "salary",
        "تربية", "طلبة", "مدارس", "مستشفى", "صحة", "معلم", "جامعة",
        "كهرباء", "مياه", "بلدية", "تعاونية", "إسكان", "طقس", "مرور",
        "حريق", "حادث", "حوادث", "مناهج", "امتحان", "اختبار", "طلاب",
        "أجور", "رواتب", "تعليم", "مدارس",
        "حديقة", "حدائق", "منتزه", "park", "garden",
    ],
    "Government & Politics": [
        "cabinet", "parliament", "decree", "draft law", "kuwaitization",
        "crown prince", "reshuffle", "municipal council",
        "مجلس الوزراء", "مجلس الأمة", "مرسوم", "تكويت", "استجواب",
        "ولي العهد", "البلدية", "هيكلة",
    ],
    "World & Region": [
        "hormuz", "gaza", "iran", "israel", "iraq", "saudi", "yemen",
        "syria", "lebanon", "egypt", "qatar", "emirates", "bahrain", "oman",
        "turkey", "trump", "russia", "ukraine", "china", "missile", "strike",
        "mexico", "britain", "british", "europe", "washington", "united nations",
        "hurricane", "flood", "earthquake", "wildfire", "refugee", "sanction",
        "إيران", "إسرائيل", "غزة", "العراق", "السعودية", "اليمن", "سوريا",
        "لبنان", "مصر", "قطر", "الإمارات", "البحرين", "عمان", "تركيا",
        "ترامب", "روسيا", "أوكرانيا", "الصين", "هرمز", "صاروخ", "قصف",
        "المكسيك", "بريطانيا", "البريطانية", "أميركا", "الأمم المتحدة",
        "روبيو", "قاعدة جوية", "أجنبي", "الخارجية الأميركية",
        "إعصار", "فيضانات", "زلزال", "عقوبات",
    ],
    "Culture & Entertainment": [
        "film", "movie", "festival", "concert", "singer", "actor", "actress",
        "artist", "album", "award", "premiere", "theatre", "theater", "museum",
        "heritage", "fashion", "celebrity", "vmas", "mtv", "oscar", "grammy",
        "مهرجان", "مطرب", "مطربة", "فيلم", "مسلسل", "مسرحية",
        "جائزة", "معرض", "فنان", "فنانة", "متحف", "تراث", "أزياء",
        "حفلات", "سينما", "مسرح",
    ],
}

def _compile_keywords(words: list[str]) -> list[re.Pattern[str] | str]:
    """Normalise keywords, then pick the right matching strategy per script.

    Latin keywords are matched on word boundaries — plain substring matching
    would let "oil" fire on "turmoil" and "us" on "business". Arabic keywords
    are matched as substrings on purpose, because Arabic glues its articles,
    prepositions and pronouns onto the front and back of a word.
    """
    compiled: list[re.Pattern[str] | str] = []
    for word in words:
        norm = normalize_text(word)
        if not norm:
            continue
        if norm.isascii():
            compiled.append(
                re.compile(
                    r"(?<![a-z0-9])" + re.escape(norm) + r"(?:e?s)?(?![a-z0-9])"
                )
            )
        else:
            compiled.append(norm)
    return compiled


CATEGORY_KEYWORDS: dict[str, list[re.Pattern[str] | str]] = {
    category: _compile_keywords(words)
    for category, words in CATEGORY_KEYWORDS_RAW.items()
}

STRONG_SECTION_WEIGHT = 1.5
WEAK_SECTION_WEIGHT = 0.4
KEYWORD_WEIGHT = 0.8


# --------------------------------------------------------------------------
# Categorisation
# --------------------------------------------------------------------------


def categorize(story: Story) -> str:
    """Blend the outlet's own section label with headline vocabulary."""
    sections = set(story.sections or [story.primary.section])

    scores: dict[str, float] = {}
    for category, groups in CATEGORY_SECTIONS.items():
        score = 0.0
        if sections & groups["strong"]:
            score += STRONG_SECTION_WEIGHT
        if sections & groups["weak"]:
            score += WEAK_SECTION_WEIGHT
        if score:
            scores[category] = score

    text = normalize_text(" ".join(a.title for a in story.articles[:6]))
    for category, patterns in CATEGORY_KEYWORDS.items():
        hits = 0
        for kw in patterns:
            if isinstance(kw, re.Pattern):
                hits += 1 if kw.search(text) else 0
            elif kw in text:
                hits += 1
        if hits:
            scores[category] = scores.get(category, 0.0) + hits * KEYWORD_WEIGHT

    if not scores:
        return "Other"
    # Deterministic tie-break so output does not depend on dict ordering.
    return max(
        scores.items(),
        key=lambda kv: (kv[1], -CATEGORY_ORDER.index(kv[0])),
    )[0]


# --------------------------------------------------------------------------
# Stage 1 — same-language clustering
# --------------------------------------------------------------------------


def _similarity(ta: set[str], tb: set[str]) -> float:
    """Headline similarity: the better of symmetric and containment overlap."""
    return max(jaccard(ta, tb), containment(ta, tb))


def cluster_articles(
    articles: list[Article],
    *,
    threshold: float = SAME_STORY_THRESHOLD,
) -> list[Story]:
    """Greedy single-pass clustering within a language, with an inverted index."""
    prepared: list[tuple[Article, set[str], set[str]]] = []
    for art in articles:
        toks = canonical_tokens(art.title)
        if toks:
            prepared.append((art, toks, distinctive_tokens(art.title)))

    # Newest first so the freshest framing seeds each cluster.
    prepared.sort(key=lambda p: p[0].published or now_kuwait(), reverse=True)

    clusters: list[Story] = []
    cluster_tokens: list[list[set[str]]] = []
    cluster_distinct: list[set[str]] = []
    token_index: dict[str, set[int]] = defaultdict(set)

    for art, toks, dist in prepared:
        candidates: Counter[int] = Counter()
        for tok in toks:
            for ci in token_index.get(tok, ()):  # type: ignore[arg-type]
                candidates[ci] += 1

        best_ci, best_score = -1, 0.0
        for ci, shared in candidates.items():
            if shared < 2:
                continue
            if len(dist & cluster_distinct[ci]) < MIN_SHARED_DISTINCTIVE:
                continue
            for member in cluster_tokens[ci]:
                score = _similarity(toks, member)
                if score > best_score:
                    best_ci, best_score = ci, score

        if best_ci >= 0 and best_score >= threshold:
            clusters[best_ci].add(art)
            cluster_tokens[best_ci].append(toks)
            cluster_distinct[best_ci] |= dist
            for tok in toks:
                token_index[tok].add(best_ci)
        else:
            story = Story()
            story.add(art)
            ci = len(clusters)
            clusters.append(story)
            cluster_tokens.append([toks])
            cluster_distinct.append(set(dist))
            for tok in toks:
                token_index[tok].add(ci)

    log.debug("stage 1: %d articles -> %d clusters", len(prepared), len(clusters))
    return clusters


# --------------------------------------------------------------------------
# Stage 2 — cross-language bridging
# --------------------------------------------------------------------------


def bridge_languages(
    stories: list[Story],
    *,
    min_shared: int = CROSS_LANG_ANCHOR_MIN,
    threshold: float = CROSS_LANG_ANCHOR_SCORE,
) -> list[Story]:
    """Pair Arabic and English clusters that describe the same event.

    Only clusters with disjoint language sets are considered, so this pass can
    never loosen same-language precision no matter how low the threshold goes.
    """
    infos: list[dict | None] = []
    for story in stories:
        toks: set[str] = set()
        for art in story.articles:
            toks |= distinctive_tokens(art.title)
        infos.append(
            {"toks": toks, "langs": {a.lang for a in story.articles}, "story": story}
        )

    index: dict[str, list[int]] = defaultdict(list)
    for i, info in enumerate(infos):
        assert info is not None
        for tok in info["toks"]:
            index[tok].append(i)

    bridged = 0
    for i, info in enumerate(infos):
        if info is None or not info["toks"]:
            continue

        counts: Counter[int] = Counter()
        for tok in info["toks"]:
            for j in index.get(tok, ()):
                if j > i:
                    counts[j] += 1

        best_j, best_score = -1, 0.0
        qualifies: list[int] = []
        for j, shared in counts.items():
            other = infos[j]
            if other is None or shared < min_shared:
                continue
            # Strictly across languages — this is the whole point of the pass.
            if not info["langs"].isdisjoint(other["langs"]):
                continue
            smaller = min(len(info["toks"]), len(other["toks"])) or 1
            score = shared / float(smaller)
            if score >= threshold:
                qualifies.append(j)
                if score > best_score:
                    best_j, best_score = j, score

        if best_j < 0:
            continue

        target = infos[best_j]
        assert target is not None
        # NOTE: best_j is in `qualifies`; its payload *is* `target`, so it must
        # be skipped or we would append a list to itself while iterating it.
        for j in qualifies:
            if j == best_j:
                continue
            other = infos[j]
            if other is None:
                continue
            for art in other["story"].articles:
                target["story"].add(art)
            target["toks"] |= other["toks"]
            target["langs"] |= other["langs"]
            infos[j] = None
            bridged += 1

        for art in info["story"].articles:
            target["story"].add(art)
        target["toks"] |= info["toks"]
        target["langs"] |= info["langs"]
        infos[i] = None
        bridged += 1

    result = [info["story"] for info in infos if info is not None]
    if bridged:
        log.debug("stage 2: %d cross-language bridges applied", bridged)
    return result


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def _recency_bonus(story: Story) -> float:
    latest = story.latest
    if latest is None:
        return 0.0
    from datetime import timedelta

    age = now_kuwait() - to_kuwait(latest)
    if age <= timedelta(hours=4):
        return 2.0
    if age <= timedelta(hours=10):
        return 1.2
    if age <= timedelta(hours=18):
        return 0.6
    return 0.0


def score_story(story: Story) -> float:
    """Rank by how much of the Kuwaiti press is carrying the story."""
    base = 0.0
    seen_sources: set[str] = set()
    for art in story.articles:
        weight = SECTION_WEIGHT.get(art.section or "general", 1.0)
        tier_w = TIER_WEIGHT.get(art.tier, 1.5)
        if art.source_id in seen_sources:
            # Count each outlet once so one prolific paper cannot dominate.
            base += tier_w * weight * 0.15
        else:
            seen_sources.add(art.source_id)
            base += tier_w * weight

    breadth = (story.outlet_count - 1) * 2.6
    detail = min(len(story.primary.body or story.primary.summary or "") / 2500.0, 1.0)
    score = base + breadth + _recency_bonus(story) + detail * 0.8
    if any(a.tier == 1 for a in story.articles):
        score += 1.5
    return score


# Importance of an outlet's own section label when ranking stories.
SECTION_WEIGHT = {
    "local": 1.5, "kuwait": 1.5, "official": 1.5, "parliament": 1.5,
    "politics": 1.5, "front-page": 1.5, "top-stories": 1.5,
    "security-courts": 1.4, "crime": 1.3, "regions": 1.2,
    "municipal": 1.2, "cooperatives": 1.2, "initiatives": 1.0,
    "economy": 1.3, "economy-reports": 1.1, "business": 1.3,
    "markets": 1.2, "energy": 1.3, "realestate": 1.1,
    "investigations": 1.4,
    "world": 0.8, "middle-east": 0.9, "gulf": 0.9, "un": 0.8,
    "world-reports": 0.7, "last-page": 0.8,
    "opinion": 0.6,
    "sport": 0.5, "health-science": 0.8, "general": 1.0,
}


def rank_stories(stories: list[Story]) -> list[Story]:
    for story in stories:
        story.score = score_story(story)
    stories.sort(key=lambda s: (s.score, s.outlet_count), reverse=True)
    return stories


def build_stories(articles: list[Article], *, major_threshold: int = 3) -> list[Story]:
    """Full pipeline: cluster → bridge languages → rank."""
    stories = cluster_articles(articles)
    stories = bridge_languages(stories)
    stories = rank_stories(stories)
    for story in stories:
        story.key = title_fingerprint(story.primary.title)
    log.info(
        "clustered %d articles into %d stories (%d cross-language)",
        len(articles), len(stories), sum(1 for s in stories if len(s.langs) > 1),
    )
    return stories


def major_count(stories: list[Story], threshold: int) -> int:
    return sum(1 for s in stories if s.outlet_count >= threshold)


def category_counts(stories: list[Story]) -> dict[str, int]:
    counts = Counter(categorize(s) for s in stories)
    return {c: counts.get(c, 0) for c in CATEGORY_ORDER if counts.get(c)}
