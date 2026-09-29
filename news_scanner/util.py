"""Text normalisation, HTML stripping and time helpers.

Everything here is standard library only so the project has zero
third-party dependencies (important for reliable unattended runs).
"""

from __future__ import annotations

import hashlib
import html as html_mod
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Iterable
from zoneinfo import ZoneInfo

KUWAIT_TZ = ZoneInfo("Asia/Kuwait")

# --------------------------------------------------------------------------
# Arabic normalisation
# --------------------------------------------------------------------------

# Harakat / tashkeel, tatweel and Quranic marks.
_DIACRITICS = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]"
)

# Letters that are written differently but compare the same.
_LETTER_MAP = {
    "\u0622": "\u0627",  # آ -> ا
    "\u0623": "\u0627",  # أ -> ا
    "\u0625": "\u0627",  # إ -> ا
    "\u0671": "\u0627",  # ٱ -> ا
    "\u0629": "\u0647",  # ة -> ه
    "\u0649": "\u064a",  # ى -> ي
    "\u0624": "\u0648",  # ؤ -> و
    "\u0626": "\u064a",  # ئ -> ي
}

_ARABIC_INDIC = {chr(0x0660 + i): str(i) for i in range(10)}
_EXT_ARABIC_INDIC = {chr(0x06F0 + i): str(i) for i in range(10)}

_PUNCT_RE = re.compile(r"[^\w\s\u0600-\u06ff]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Words too common to carry meaning when matching headlines across outlets.
STOPWORDS = {
    # Arabic
    "في", "من", "الى", "إلى", "على", "علي", "عن", "مع", "هذا", "هذه", "ذلك", "التي",
    "الذي", "بعد", "قبل", "بين", "خلال", "حول", "او", "أو", "و", "ان", "أن",
    "إن", "كان", "كانت", "يكون", "لم", "لا", "ما", "هو", "هي", "كما", "لكن",
    "حيث", "كل", "بعض", "تم", "يتم", "قد", "عند", "عندما", "حتى", "اذا", "إذا",
    "الا", "إلا", "غير", "نحو", "لدى", "اليوم", "امس", "أمس", "غدا", "غداً",
    "تصوير", "كونا", "الراي", "القبس", "الانباء", "الجريدة",
    # English
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "will", "would", "its", "it", "this", "that",
    "these", "those", "after", "before", "over", "into", "about", "says",
    "said", "amid", "his", "her", "their", "they", "he", "she",
    # Function words that add noise to headline matching but never meaning.
    "more", "most", "also", "than", "out", "off", "not", "no", "nor",
    "only", "such", "some", "any", "all", "may", "might", "can", "could",
    "should", "must", "per", "since", "while", "when", "where", "who",
    "which", "what", "how", "why", "both", "each", "other", "others",
    "between", "during", "under", "against", "without", "within", "across",
    "along", "around", "down", "up", "amid", "onto", "upon", "via",
    # Arabic function words with the same problem
    "ضمن", "بشأن", "وفق", "أي", "أيا", "أيضا", "عبر", "دون", "منذ", "حسب",
    "بحسب", "مثل", "خارج", "داخل", "بينما", "حين", "وقد", "كما", "لكن",
}


def normalize_text(text: str, *, drop_stopwords: bool = False) -> str:
    """Fold Arabic orthographic variation and strip noise."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    for src, dst in _LETTER_MAP.items():
        text = text.replace(src, dst)
    for table in (_ARABIC_INDIC, _EXT_ARABIC_INDIC):
        for src, dst in table.items():
            text = text.replace(src, dst)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip().lower()
    if drop_stopwords:
        words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
        text = " ".join(words)
    return text


def tokenize(text: str) -> set[str]:
    """Meaningful token set used for cross-outlet story matching."""
    return set(normalize_text(text, drop_stopwords=True).split())


def title_fingerprint(title: str) -> str:
    """Stable short hash of a normalised headline."""
    norm = normalize_text(title, drop_stopwords=True)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Cross-language anchors
# --------------------------------------------------------------------------
# Arabic and English outlets describe the same event with entirely different
# words, so a story carried by Al-Rai and by Kuwait Times would otherwise
# become two separate clusters. Mapping recurring Kuwaiti news vocabulary onto
# one canonical English token lets the clusterer see they are the same story.
#
# Keys are written in normal Arabic; they are run through normalize_text() at
# import time so the keys always match the normalised tokens in the text
# (e.g. هيئة -> هييه). Never hand-normalise these.
ALIASES_RAW: dict[str, str] = {
    # --- institutions & roles ---
    "مجلس الوزراء": "cabinet",
    "مجلس الأمة": "parliament",
    "مجلس الامة": "parliament",
    "البرلمان": "parliament",
    "النواب": "parliament",
    "نائب": "mp",
    "لجنة": "committee",
    "جلسة": "session",
    "دورة": "session",
    "الحكومة": "government",
    "وزير": "minister",
    "وزارة": "ministry",
    "وكيل وزارة": "undersecretary",
    "رئيس الوزراء": "prime minister",
    "ولي العهد": "crown prince",
    "الأمير": "amir",
    "سمو": "amir",
    "سفير": "ambassador",
    "سفارة": "embassy",
    "مسؤول": "official",
    "مصدر": "source",
    "متحدث": "spokesman",
    "هيئة": "authority",
    "جهاز": "authority",
    "إدارة": "department",
    "البلدية": "municipality",
    "محافظ": "governor",
    "محافظة": "governorate",
    "المحكمة": "court",
    "النيابة": "prosecution",
    "القضاء": "judiciary",
    "الجيش": "army",
    "الحرس الوطني": "national guard",
    "الدفاع المدني": "civil defense",
    "الإطفاء": "firefighters",
    "الشرطة": "police",
    "الأمن": "security",
    "الداخلية": "interior",
    "الخارجية": "foreign",
    "العدل": "justice",
    "المالية": "finance",
    "الصحة": "health",
    "التربية": "education",
    "التعليم": "education",
    "الكهرباء": "electricity",
    "المياه": "water",
    "النفط": "oil",
    "البترول": "oil",
    "الطيران المدني": "civil aviation",
    "المطار": "airport",
    "الجمارك": "customs",
    "الإسكان": "housing",
    "البيئة": "environment",
    "الغذاء": "food",
    "الصناعة": "industry",
    "التجارة": "trade",
    "الإعلام": "media",
    "الجامعة": "university",
    "الكويت": "kuwait",
    "مجلس التعاون": "gcc",
    "مجلس الأمن": "security council",
    "الأمم المتحدة": "united nations",
    "الجامعة العربية": "arab league",
    # --- districts ---
    "الجهراء": "jahra",
    "الأحمدي": "ahmadi",
    "الفروانية": "farwaniya",
    "حولي": "hawalli",
    "العاصمة": "capital",
    "مبارك الكبير": "mubarak al-kabeer",
    "السالمية": "salmiya",
    "المهبولة": "mahboula",
    "الفحيحيل": "fahaheel",
    "المنقف": "mangaf",
    "خيطان": "khaitan",
    "صباح السالم": "sabah al-salem",
    "الفنطاس": "fintas",
    "الصليبية": "sulaibiya",
    "الرقة": "riqqa",
    "الوفرة": "wafra",
    # --- verbs of news ---
    "يوافق": "approve",
    "وافق": "approve",
    "موافقة": "approve",
    "إلغاء": "cancellation",
    "الغاء": "cancellation",
    "يلغي": "cancel",
    "تعديل": "amendment",
    "يعدل": "amend",
    "إنشاء": "establish",
    "تأسيس": "establish",
    "تطوير": "development",
    "يطور": "develop",
    "يبحث": "discuss",
    "بحث": "discuss",
    "مناقشة": "discuss",
    "يناقش": "discuss",
    "يعلن": "announce",
    "إعلان": "announce",
    "أعلن": "announce",
    "يدين": "condemn",
    "إدانة": "condemn",
    "يستقبل": "receive",
    "استقبال": "receive",
    "يزور": "visit",
    "زيارة": "visit",
    "يوقع": "sign",
    "توقيع": "sign",
    "مذكرة تفاهم": "memorandum",
    "اتفاقية": "agreement",
    "اتفاق": "agreement",
    "يطلق": "launch",
    "إطلاق": "launch",
    "يفتتح": "open",
    "افتتاح": "open",
    "يدشن": "inaugurate",
    "يؤكد": "affirm",
    "تأكيد": "affirm",
    "ينفي": "deny",
    "نفي": "deny",
    "يحذر": "warn",
    "تحذير": "warn",
    "يطالب": "demand",
    "يقر": "pass",
    "إقرار": "pass",
    "يرفض": "reject",
    "رفض": "reject",
    "يواصل": "continue",
    "يعقد": "hold",
    "عقد": "hold",
    "اجتماع": "meeting",
    "مؤتمر": "conference",
    "نشرة": "bulletin",
    "بيان": "statement",
    "تقرير": "report",
    "يحيل": "refer",
    "إحالة": "refer",
    "يقضي": "sentence",
    "قضى": "sentence",
    "يستعرض": "review",
    "يتفقد": "inspect",
    "يشارك": "participate",
    "مشاركة": "participate",
    "يهدد": "threaten",
    "يهاجم": "attack",
    "استهداف": "target",
    "تسجيل": "register",
    "بدء": "start",
    "انطلاق": "start",
    "استكمال": "complete",
    "تسليم": "handover",
    "تأجيل": "postpone",
    "توقف": "suspend",
    "تعليق": "suspend",
    "إيقاف": "suspend",
    "استئناف": "resume",
    "توزيع": "distribution",
    "تكليف": "assign",
    "تعيين": "appointment",
    "استقالة": "resignation",
    "إقالة": "dismissal",
    "ترقية": "promotion",
    "إحصائيات": "statistics",
    "أرقام": "figures",
    "دراسة": "study",
    "تقرير": "report",
    # --- themes ---
    "مدينة": "city",
    "جزيرة": "island",
    "الحرير": "silk",
    "الصبية": "sabiya",
    "بوبيان": "bubiyan",
    "مشروع": "project",
    "مناقصة": "tender",
    "عقود": "contracts",
    "قرار": "decision",
    "مرسوم": "decree",
    "وزراء": "minister",
    "قانون": "law",
    "تشريع": "legislation",
    "الانتخابات": "election",
    "انتخاب": "election",
    "التصويت": "vote",
    "التكويت": "kuwaitization",
    "تكويت": "kuwaitization",
    "الإقامة": "residency",
    "الوافدين": "expat",
    "الوافدة": "expat",
    "الجنسية": "citizenship",
    "الفساد": "corruption",
    "المخدرات": "drugs",
    "تهريب": "smuggling",
    "ضبط": "seize",
    "أحبط": "foil",
    "اعتقال": "arrest",
    "توقيف": "arrest",
    "السجن": "prison",
    "متهم": "suspect",
    "متهما": "suspect",
    "مشتبه": "suspect",
    "مواد": "substance",
    "مخدر": "drug",
    "مخدرة": "drug",
    "مخدرات": "drug",
    "مؤثرات": "psychotropic",
    "عقلية": "psychotropic",
    "سلاح": "weapon",
    "ذخيرة": "ammunition",
    "كحول": "alcohol",
    "خمور": "alcohol",
    "اعتقل": "arrest",
    "أوقف": "arrest",
    "احتجاز": "detention",
    "بلاغ": "complaint",
    "شكوى": "complaint",
    "حبس": "prison",
    "غرامة": "fine",
    "الحكم": "verdict",
    "قضية": "case",
    "جريمة": "crime",
    "قضايا": "cases",
    "متهم": "defendant",
    "الطقس": "weather",
    "الحرارة": "temperature",
    "الأمطار": "rain",
    "الغبار": "dust",
    "الرطوبة": "humidity",
    "حوادث": "accidents",
    "حادث": "accident",
    "حريق": "fire",
    "مرور": "traffic",
    "مخالفة": "violation",
    "ازدحام": "congestion",
    "الرواتب": "salaries",
    "الأجور": "wages",
    "نظام": "system",
    "التضخم": "inflation",
    "الميزانية": "budget",
    "الاستثمار": "investment",
    "تمويل": "financing",
    "قرض": "loan",
    "دعم": "subsidy",
    "أرباح": "profits",
    "إيرادات": "revenues",
    "خسائر": "losses",
    "صادرات": "exports",
    "واردات": "imports",
    "سوق": "market",
    "البورصة": "bourse",
    "مؤشر": "index",
    "شركة": "company",
    "مؤسسة": "institution",
    "بنك": "bank",
    "عقار": "realestate",
    "التنمية": "development",
    "الاستدامة": "sustainability",
    "وظائف": "jobs",
    "توظيف": "employment",
    "تقاعد": "retirement",
    "تدريب": "training",
    "منحة": "scholarship",
    "الطلبة": "students",
    "طالب": "student",
    "المدارس": "schools",
    "مدرسة": "school",
    "امتحان": "exam",
    "اختبار": "test",
    "المناهج": "curriculum",
    "المستشفى": "hospital",
    "مرض": "disease",
    "لقاح": "vaccine",
    "تطعيم": "vaccination",
    "وفيات": "deaths",
    "وفاة": "death",
    "إصابة": "injury",
    "الرياضة": "sport",
    "بطولة": "tournament",
    "مباراة": "match",
    "منتخب": "national team",
    "فريق": "team",
    "لاعب": "player",
    "مدرب": "coach",
    "هدف": "goal",
    "فوز": "win",
    "خسارة": "loss",
    "ميدالية": "medal",
    "ذهبية": "gold",
    "فضية": "silver",
    "برونزية": "bronze",
    "الدوري": "league",
    "الكأس": "cup",
    "كأس الخليج": "gulf cup",
    "كأس آسيا": "asian cup",
    "الألعاب الآسيوية": "asian games",
    "الأولمبياد": "olympics",
    "الأخضر": "green",
    "التعاون": "cooperation",
    "شراكة": "partnership",
    "العلاقات": "relations",
    "الثنائية": "bilateral",
    "النيابية": "parliamentary",
    "المالية": "financial",
    "الاقتصادية": "economic",
    "الاجتماعية": "social",
    "الأمنية": "security",
    "الصحية": "health",
    "امتياز": "concession",

    # -- places, projects and institutions --------------------------------
    "مدينة الحرير": "silk city",
    "الحرير": "silk",
    "الصبية": "sabiya",
    "بوبيان": "bubiyan",
    "حديقة الشهيد": "shaheed park",
    "الفروانية": "farwaniya",
    "الجهراء": "jahra",
    "حولي": "hawalli",
    "الأحمدي": "ahmadi",
    "الزور": "zour",
    "الشعيبة": "shuaiba",
    # -- governance and policy --------------------------------------------
    "تكويت": "kuwaitization",
    "التكويت": "kuwaitization",
    "الإحلال": "replacement",
    "قضائية": "judicial",
    "القضائية": "judicial",
    "قضائي": "judicial",
    "ولي العهد": "crown prince",
    "رئيس الوزراء": "prime minister",
    "مجلس البلدي": "municipal council",
    "المجلس البلدي": "municipal council",
    "البلدية": "municipal",
    "مرسوم": "decree",
    "مشروع قانون": "draft law",
    "قانون": "law",
    "استجواب": "interrogation",
    "إعادة هيكلة": "restructur",
    "الدعم": "subsidy",
    # -- water, power and infrastructure -----------------------------------
    "الربط المائي": "water link",
    "ربط مائي": "water link",
    "المائي": "water",
    "المائية": "water",
    "مائي": "water",
    "تحلية": "desalination",
    "محطة": "station",
    "شبكة": "network",
    "الكهرباء": "electricity",
    # -- economy ------------------------------------------------------------
    "التضخم": "inflation",
    "الفائدة": "interest",
    "المصرفي": "banking",
    "المصرفية": "banking",
    "الأجور": "wage",
    "الرواتب": "salary",
    "التقاعد": "retirement",
    "المعاش": "pension",
    "الأسعار": "price",
    "الدولار": "dollar",
    # -- security and courts -------------------------------------------------
    "النيابة العامة": "prosecution",
    "قضية": "case",
    "قضايا": "case",
    "مستندات": "document",
    "وثائق": "document",
    # -- health and education -------------------------------------------------
    "المستشفى": "hospital",
    "الطوارئ": "emergency",
    "الأمراض": "disease",
    "اللقاح": "vaccine",
    "التطعيم": "vaccination",
    "المناهج": "curriculum",
    "الامتحانات": "exam",
    "الاختبار": "exam",
    "التسجيل": "registration",
    "القبول": "admission",
    # -- sport ------------------------------------------------------------------
    "الآسياد": "asian games",
    "آسياد": "asian games",
    "الأسياد": "asian games",
    "كأس الخليج": "gulf cup",
    "خليجي": "gulf cup",
    "الرماية": "shooting",
    "رماية": "shooting",
    "ذهبيتان": "gold",
    "ذهبيتين": "gold",
    "ذهبية": "gold",
    "برونزيتان": "bronze",
    "برونزيتين": "bronze",
    "برونزية": "bronze",
    "فضية": "silver",
    "الدوري": "league",
    "دوري": "league",
    "المنتخب": "team",
    "منتخب": "team",
    "المباراة": "match",
    "مباراة": "match",
    "البطولة": "tournament",
    "بطولة": "tournament",
    "لاعب": "player",
    "لاعبين": "player",
    "مدرب": "coach",
    # -- the wider world ---------------------------------------------------------
    "بريطانيا": "britain",
    "البريطانية": "britain",
    "أميركا": "us",
    "الأميركية": "us",
    "أميركي": "us",
    "المكسيك": "mexico",
    "اليابان": "japan",
    "الصين": "china",
    "روسيا": "russia",
    "أوكرانيا": "ukraine",
    "فرنسا": "france",
    "ألمانيا": "germany",
    "إعصار": "hurricane",
    "فيضانات": "flood",
    "فيضان": "flood",
    "زلزال": "earthquake",
    "الجفاف": "drought",
    # -- culture ------------------------------------------------------------------
    "مهرجان": "festival",
    "حفل": "ceremony",
    "مطرب": "singer",
    "فيلم": "film",
    "مسلسل": "series",
    "مسرحية": "play",
    "جائزة": "award",
    "معرض": "exhibition",
}

# --- light English stemmer -------------------------------------------------
# Headline writers use approve / approves / approved / approving for the same
# event. Trimming the common inflections makes those match the single form the
# alias table emits. Applied to Latin tokens only and never below 5 characters.
_SUFFIXES = ("ations", "ation", "ements", "ement", "ings", "ing", "ers",
             "er", "ed", "es", "s")


def stem_latin(word: str) -> str:
    if not word.isascii() or len(word) < 5 or not word.isalpha():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    if len(word) >= 5 and word.endswith("e"):
        return word[:-1]
    return word


def _build_alias_table() -> dict[str, str]:
    """Normalise every alias key so it matches the normalised article text."""
    table: dict[str, str] = {}
    for key, value in ALIASES_RAW.items():
        norm = normalize_text(key)
        if norm:
            table[norm] = value
    return table


ALIASES = _build_alias_table()
_ALIAS_ORDER = sorted(ALIASES, key=lambda k: -len(k.split()))
_SINGLE_WORD_ALIASES = {k for k in ALIASES if " " not in k}

# Terms so common they cannot by themselves establish that two stories match.
GENERIC_ANCHORS = {
    "kuwait", "kuwaiti", "kuwaitis", "news", "today", "yesterday", "tomorrow",
    "video", "photo", "photos", "report", "statement", "announce", "official",
    "officials", "source", "sources", "detail", "details", "update", "updates",
    "year", "years", "month", "months", "week", "weeks", "day", "days", "time",
    "times", "people", "person", "government", "minister", "ministry", "state",
    "country", "public", "local", "world", "case", "cases", "number", "first",
    "second", "third", "new", "several", "many", "various", "general", "major",
    "important", "continue", "continues", "hold", "held", "meeting", "discuss",
    "discussion", "review", "receive", "visit", "affirm", "confirm", "stress",
    "call", "calls", "urge", "urges", "welcome", "praise", "honor",
    "start", "end", "close", "sign", "include", "including",
    "release", "issue", "issues",
    "member", "members", "group", "team", "teams", "head", "chief",
    "director", "chairman", "board", "center", "centre", "department", "sector",
    "area", "areas", "region", "regions", "city", "island", "morning", "evening",
    "weekend", "monthly", "annual", "daily", "today", "winter", "summer",
}

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000",
}

_YEAR_RE = re.compile(r"^(19|20)\d{2}$")

_STEM_CACHE: dict[str, str] = {}


def _stem_cached(word: str) -> str:
    cached = _STEM_CACHE.get(word)
    if cached is None:
        cached = stem_latin(word)
        _STEM_CACHE[word] = cached
    return cached


_STOPWORDS_STEMMED = {_stem_cached(w) for w in STOPWORDS} | STOPWORDS
_GENERIC_STEMMED = {_stem_cached(w) for w in GENERIC_ANCHORS} | GENERIC_ANCHORS


def canonical_tokens(text: str) -> set[str]:
    """Normalise, apply the Arabic->English alias map, stem Latin words and
    drop stopwords.

    Generic words are deliberately *kept* here: they still carry a little
    signal when scoring two headlines against each other. They are simply not
    allowed to be the sole evidence for a match — see distinctive_tokens().
    """
    norm = normalize_text(text)
    if not norm:
        return set()
    words = norm.split()

    # Split the Arabic conjunction prefix "و" when what follows is a known
    # alias term, so "وجزيرة" is treated the same as "جزيرة".
    expanded: list[str] = []
    for word in words:
        if (
            word.startswith("و")
            and len(word) > 4
            and word not in _SINGLE_WORD_ALIASES
            and word[1:] in _SINGLE_WORD_ALIASES
        ):
            expanded.append(word[1:])
        else:
            expanded.append(word)
    words = expanded

    # Longest-phrase-first replacement so "مجلس الوزراء" wins over "مجلس".
    out: list[str] = []
    i = 0
    while i < len(words):
        matched = False
        for phrase in _ALIAS_ORDER:
            parts = phrase.split()
            if words[i : i + len(parts)] == parts:
                out.extend(ALIASES[phrase].split())
                i += len(parts)
                matched = True
                break
        if not matched:
            out.append(words[i])
            i += 1

    tokens = {NUMBER_WORDS.get(w, w) for w in out}
    tokens = {_stem_cached(w) for w in tokens}
    return {
        t
        for t in tokens
        if t not in _STOPWORDS_STEMMED
        and len(t) > 1
        # Bare years match every second story in a newspaper; drop them.
        and not _YEAR_RE.match(t)
    }


def distinctive_tokens(text: str) -> set[str]:
    """Tokens with real discriminating power — generic news words removed.

    Two headlines are only ever merged when they agree on at least two of
    these, so a pair can never be joined on boilerplate alone.
    """
    return {t for t in canonical_tokens(text) if t not in _GENERIC_STEMMED}


# The cross-language bridging pass speaks in terms of "anchors".
anchor_tokens = distinctive_tokens


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / float(len(a) + len(b) - inter)


def containment(a: set[str], b: set[str]) -> float:
    """Overlap relative to the smaller set.

    Better than Jaccard when one headline is a short wire line and the
    other is a longer rewrite of the same event.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / float(min(len(a), len(b)))


# --------------------------------------------------------------------------
# HTML handling
# --------------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    """Minimal readability-ish text extractor."""

    SKIP = {"script", "style", "noscript", "iframe", "svg", "form", "nav",
            "footer", "header", "aside", "figure"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
             "tr", "blockquote", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK and self._skip_depth == 0:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.BLOCK and self._skip_depth == 0:
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t\u00a0]+", " ", raw)
        raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
        return raw.strip()


def html_to_text(markup: str) -> str:
    """Strip tags/entities from a fragment or document."""
    if not markup:
        return ""
    if "<" not in markup:
        return _WS_RE.sub(" ", html_mod.unescape(markup)).strip()
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # malformed markup must never kill a run
        pass
    text = parser.text()
    return text or _WS_RE.sub(" ", html_mod.unescape(markup)).strip()


def strip_cdata(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    if value.startswith("<![CDATA["):
        value = value[9:]
    if value.endswith("]]>"):
        value = value[:-3]
    return value.strip()


class _LinkExtractor(HTMLParser):
    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._buf: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_d = dict(attrs)
            self._href = attrs_d.get("href")
            self._buf = []
            self._depth = 1
        elif self._depth:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            text = _WS_RE.sub(" ", "".join(self._buf)).strip()
            self.links.append((self._href, text))
            self._href = None
            self._buf = []
            self._depth = 0
        elif self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if self._href is not None:
            self._buf.append(data)


def extract_links(markup: str, base: str) -> list[tuple[str, str]]:
    """Return (absolute_url, anchor_text) pairs from an HTML document."""
    from urllib.parse import urljoin

    parser = _LinkExtractor(base)
    try:
        parser.feed(markup)
        parser.close()
    except Exception:
        pass
    out: list[tuple[str, str]] = []
    for href, text in parser.links:
        if not href or href.startswith(("javascript:", "mailto:", "#", "tel:")):
            continue
        out.append((urljoin(base, href), text))
    return out


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------


def now_kuwait() -> datetime:
    return datetime.now(KUWAIT_TZ)


def to_kuwait(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KUWAIT_TZ)


def window_start(hours: int, *, reference: datetime | None = None) -> datetime:
    ref = reference or now_kuwait()
    return ref - timedelta(hours=hours)


def humanize_age(dt: datetime | None, *, reference: datetime | None = None) -> str:
    if dt is None:
        return ""
    ref = reference or now_kuwait()
    delta = ref - to_kuwait(dt)
    secs = max(0, int(delta.total_seconds()))
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def fmt_local(dt: datetime | None) -> str:
    if dt is None:
        return "time unknown"
    return to_kuwait(dt).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]).rstrip() + suffix


def dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def slugify(text: str, limit: int = 60) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9\u0600-\u06ff]+", "-", text).strip("-")
    return text[:limit] or "story"
