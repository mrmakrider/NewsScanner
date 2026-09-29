#!/usr/bin/env python3
"""Offline demo — exercises the whole pipeline without any network access.

This is the way to preview the digest's structure before wiring up GitHub
Actions, and the way to verify a change to clustering or rendering.

    python3 scripts/demo.py

It runs the *real* pipeline (clustering, ranking, analysis, rendering, file
output) but feeds it a fixed set of 39 article fixtures drawn from genuine
Kuwaiti headlines, and uses the extractive analyser so no API key is needed.
Nothing is emailed.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from news_scanner import cli  # noqa: E402
from news_scanner.models import Article  # noqa: E402
from news_scanner.sources import SourceHealth  # noqa: E402
from news_scanner.util import now_kuwait  # noqa: E402

# (source_id, outlet name, lang, title, summary)
FIXTURES: list[tuple[str, str, str, str, str]] = [
    # ---------------------------------------------------------- Silk City
    (
        "alrai", "الرأي", "ar",
        "مجلس الوزراء يوافق على إلغاء مرسوم إنشاء جهاز تطوير مدينة الحرير «الصبية» وجزيرة بوبيان",
        "وافق مجلس الوزراء في اجتماعه الأسبوعي على مشروع مرسوم بإلغاء المرسوم الخاص بإنشاء "
        "جهاز تطوير مدينة الحرير وجزيرة بوبيان، على أن تتولى الجهات المختصة استكمال الإجراءات.",
    ),
    (
        "alanba", "الأنباء", "ar",
        "إلغاء تطوير «الحرير» و«بوبيان» وتعديل «الشركات»",
        "أقر مجلس الوزراء إلغاء مرسوم إنشاء جهاز تطوير مدينة الحرير وجزيرة بوبيان، "
        "كما وافق على تعديل قانون الشركات.",
    ),
    (
        "aljarida", "الجريدة", "ar",
        "مجلس الوزراء يوافق على مشروع مرسوم بإلغاء مرسوم إنشاء جهاز تطوير مدينة الحرير وجزيرة بوبيان",
        "وافق مجلس الوزراء أمس على مشروع مرسوم بإلغاء مرسوم إنشاء جهاز تطوير مدينة الحرير "
        "وجزيرة بوبيان، في خطوة تعيد رسم خريطة المشاريع الكبرى.",
    ),
    (
        "kuwaittimes", "Kuwait Times", "en",
        "Cabinet clears abolition of Silk City development body",
        "The Cabinet on Tuesday approved a draft decree abolishing the authority set up to "
        "develop Silk City and Bubiyan Island, saying the file would be handled by other "
        "government bodies.",
    ),
    (
        "timeskuwait", "Times Kuwait", "en",
        "Govt approves cancellation of decree establishing Silk City, BIDA",
        "The Council of Ministers approved a draft law cancelling the decree that established "
        "the Silk City and Bubiyan Island development authority.",
    ),
    # ------------------------------------------------------- Wage system
    (
        "aljarida", "الجريدة", "ar",
        "«المركزي» يطلق نظام الكويت لحماية ودفع الأجور «KWPS»",
        "أعلن بنك الكويت المركزي إطلاق نظام الكويت لحماية ودفع الأجور، الذي يضمن تحويل "
        "رواتب العاملين في القطاع الخاص عبر القنوات المصرفية.",
    ),
    (
        "kuwaittimes", "Kuwait Times", "en",
        "Wage payment system launched",
        "The Central Bank of Kuwait launched the Kuwait Wage Protection System, requiring "
        "private-sector salaries to be paid through approved local banks.",
    ),
    (
        "timeskuwait", "Times Kuwait", "en",
        "Kuwait's New Wage System to Protect Private-Sector Employees' Salaries",
        "A new wage protection system will require companies in the private sector to "
        "transfer salaries electronically, in a bid to cut down on unpaid-wage disputes.",
    ),
    # ------------------------------------------------- University registration
    (
        "alrai", "الرأي", "ar",
        "جامعة الكويت تعلن مواعيد التسجيل لـ«اختبار القدرات» لطلبة الـ12",
        "أعلنت جامعة الكويت عن مواعيد التسجيل في اختبار القدرات لطلبة الصف الثاني عشر "
        "للعام الدراسي المقبل.",
    ),
    (
        "alanba", "الأنباء", "ar",
        "جامعة الكويت تعلن مواعيد التسجيل لـ«الاختبار الوطني الموحد» لطلب التقديم",
        "حددت جامعة الكويت مواعيد التسجيل للاختبار الوطني الموحد لطلبة الثانوية العامة.",
    ),
    # -------------------------------------------------- Swiss state visit
    (
        "alrai", "الرأي", "ar",
        "سمو ولي العهد ورئيس الاتحاد السويسري يترأسان جلسة المباحثات الرسمية بين الكويت وسويسرا",
        "ترأس سمو ولي العهد ورئيس الاتحاد السويسري جلسة المباحثات الرسمية التي عقدت "
        "في برن، وبحث الجانبان تعزيز العلاقات الثنائية.",
    ),
    (
        "kuwaittimes", "Kuwait Times", "en",
        "Crown Prince's visit to Switzerland to boost relations",
        "His Highness the Crown Prince held official talks in Bern with the President of the "
        "Swiss Confederation, covering trade, investment and cooperation.",
    ),
    # ------------------------------------------------------ Judicial Kuwaitization
    (
        "alrai", "الرأي", "ar",
        "السند: انطلاق المرحلة الثانية من تكويت الوظائف القضائية المساندة",
        "أعلن وزير العدل انطلاق المرحلة الثانية من توظيف الكويتيين في الوظائف القضائية "
        "المساندة، ضمن خطة الإحلال التدريجي.",
    ),
    (
        "alanba", "الأنباء", "ar",
        "«العدل»: بدء المرحلة الثانية من تكويت الوظائف القضائية المساندة في النيابة",
        "بدأت وزارة العدل تنفيذ المرحلة الثانية من تكويت الوظائف القضائية المساندة في "
        "النيابة العامة، بحسب ما أعلنته الوزارة.",
    ),
    # ---------------------------------------------------------- Oil price
    (
        "alrai", "الرأي", "ar",
        "سعر برميل النفط الكويتي ينخفض إلى 106.51 دولار",
        "انخفض سعر برميل النفط الكويتي 1.23 دولار ليصل إلى 106.51 دولار للبرميل، "
        "وسط تقلبات أسواق الطاقة العالمية.",
    ),
    (
        "aljarida", "الجريدة", "ar",
        "النفط الكويتي ينخفض إلى 106.51 دولار للبرميل",
        "تراجع سعر برميل النفط الكويتي إلى 106.51 دولار للبرميل مقارنة بالجلسة السابقة.",
    ),
    # ---------------------------------------------------- Al-Shaheed Park
    (
        "alrai", "الرأي", "ar",
        "سمو الأمير يشمل برعايته حفل افتتاح حديقة الشهيد المرحلة الثالثة مساء غد",
        "يشمل حضرة صاحب السمو أمير البلاد برعايته الكريمة حفل افتتاح المرحلة الثالثة "
        "من حديقة الشهيد مساء غد.",
    ),
    (
        "kuwaittimes", "Kuwait Times", "en",
        "Stage set for grand opening of Al-Shaheed Park Phase 3",
        "Al-Shaheed Park's third phase will be inaugurated under the patronage of His "
        "Highness the Amir, adding new gardens and a cultural centre.",
    ),
    # ---------------------------------------------------------- Drug bust
    (
        "alqabas", "القبس", "ar",
        "الداخلية: ضبط 12 متهماً في 8 قضايا بحوزتهم مواد مخدرة ومؤثرات عقلية وسلاح",
        "أعلنت وزارة الداخلية ضبط 12 متهماً في 8 قضايا منفصلة، وضبطت مواد مخدرة "
        "ومؤثرات عقلية وسلاحاً نارياً وذخيرة.",
    ),
    (
        "arabtimes", "Arab Times", "en",
        "Kuwait Arrests 12 Suspects in 8 Cases, Seizes Drugs, Alcohol and Firearm",
        "Anti-narcotics officers arrested 12 suspects in eight separate cases and seized "
        "narcotics, alcohol and a firearm, the Interior Ministry said.",
    ),
    (
        "kuwaittimes", "Kuwait Times", "en",
        "Anti-narcotics officers arrest 12 suspects in eight cases, seize drugs and weapon",
        "The General Department for Drug Control arrested 12 people in eight cases and "
        "referred them to the Public Prosecution.",
    ),
    # ------------------------------------------------- Environment Authority docs
    (
        "alrai", "الرأي", "ar",
        "وزيرة «التنمية والاستدامة» تحيل إلى النيابة العامة واقعة فقد مستندات رسمية من «هيئة البيئة»",
        "أحالت وزيرة الشؤون الاجتماعية والتنمية والاستدامة واقعة فقد مستندات رسمية من "
        "الهيئة العامة للبيئة إلى النيابة العامة.",
    ),
    (
        "arabtimes", "Arab Times", "en",
        "Missing Official Documents at the Environment Authority Referred to Prosecution",
        "The Minister of Social Affairs and Sustainable Development referred the case of "
        "missing official documents at the Environment Public Authority to the prosecution.",
    ),
    # ----------------------------------------------------------- Parliament
    (
        "aljarida", "الجريدة", "ar",
        "مجلس الأمة يبحث تقرير اللجنة المالية بشأن الميزانية العامة للدولة",
        "يناقش مجلس الأمة في جلسته المقبلة تقرير اللجنة المالية والاقتصادية حول "
        "الميزانية العامة للدولة.",
    ),
    (
        "alanba", "الأنباء", "ar",
        "«المالية البرلمانية» تناقش الميزانية العامة وإعادة هيكلة الدعم",
        "ناقشت اللجنة المالية والاقتصادية في مجلس الأمة الميزانية العامة للدولة "
        "وملف إعادة هيكلة الدعم.",
    ),
    # ----------------------------------------------------- Cooperatives
    (
        "alrai", "الرأي", "ar",
        "«الشؤون»: حلّ 15 جمعية تعاونية وعزل 76 عضواً وإحالة 304 أشخاص للنيابة",
        "أعلنت وزارة الشؤون الاجتماعية حل 15 جمعية تعاونية وعزل 76 عضواً وإحالة "
        "304 أشخاص إلى النيابة العامة.",
    ),
    (
        "aljarida", "الجريدة", "ar",
        "رسوب شبه جماعي في اختبارات «إشرافية التعاونيات»",
        "سجلت اختبارات إشرافية التعاونيات نسبة رسوب شبه جماعية، ما أثار جدلاً حول "
        "معايير التعيين في الجمعيات.",
    ),
    # ------------------------------------------------------- Asian Games
    (
        "alrai", "الرأي", "ar",
        "ذهبيتان وبرونزيتان لـ«رماية» الكويت في آسياد اليابان",
        "حققت بعثة الكويت للرماية ذهبيتين وبرونزيتين في دورة الألعاب الآسيوية في اليابان.",
    ),
    (
        "kuwaittimes", "Kuwait Times", "en",
        "Al-Mudhaf wins double gold as Kuwait shooters light up Asian Games",
        "Kuwait's shooters won two gold medals and two bronze at the Asian Games, with "
        "Al-Mudhaf taking an individual double.",
    ),
    (
        "arabtimes", "Arab Times", "en",
        "Kuwait shooters claim two golds at Asian Games",
        "Kuwait's shooting team claimed two gold medals at the Asian Games in Japan, "
        "adding to the country's medal tally.",
    ),
    # --------------------------------------------------- Gulf Cup 28
    (
        "arabtimes", "Arab Times", "en",
        "Iraq to host Gulf Cup 28",
        "The Gulf Cup will be hosted in Iraq, organisers confirmed, with Kuwait expected "
        "to take part in the regional tournament.",
    ),
    (
        "alrai", "الرأي", "ar",
        "تحديد موعد انطلاق «خليجي 28» في العراق",
        "أعلن الاتحاد الخليجي تحديد موعد انطلاق بطولة خليجي 28 في العراق بمشاركة "
        "منتخبات المنطقة.",
    ),
    # ------------------------------------------------------ Water link
    (
        "kuwaittimes", "Kuwait Times", "en",
        "Kuwait, Saudi Arabia advance water interconnection project",
        "Kuwait and Saudi Arabia reviewed progress on the water interconnection project "
        "in a joint technical meeting.",
    ),
    (
        "alrai", "الرأي", "ar",
        "الكويت والسعودية تعززان التنسيق الفني لإنجاز مشروع الربط المائي",
        "بحثت الكويت والسعودية سبل تعزيز التنسيق الفني لإنجاز مشروع الربط المائي "
        "بين البلدين.",
    ),
    # ------------------------------------------------------ Hormuz
    (
        "kuwaittimes", "Kuwait Times", "en",
        "Vessel Struck by Suspected Projectile in Strait of Hormuz",
        "A vessel was struck by a suspected projectile in the Strait of Hormuz, shipping "
        "sources said, raising fresh concerns over Gulf navigation.",
    ),
    # ------------------------------------------------------ Hurricane
    (
        "arabtimes", "Arab Times", "en",
        "Hurricane Polo slams Mexico with flooding and power cuts",
        "Hurricane Polo made landfall in Mexico, bringing flooding and widespread power "
        "cuts to coastal areas.",
    ),
    # ------------------------------------------------------ Taylor Swift
    (
        "arabtimes", "Arab Times", "en",
        "Taylor Swift sets new record at MTV VMAs",
        "Taylor Swift won a record number of MTV Video Music Awards, extending her tally.",
    ),
    # ------------------------------------------------------ Gold price
    (
        "aljarida", "الجريدة", "ar",
        "الذهب قرب أدنى مستوى في 7 أسابيع قبل بيانات أميركية",
        "تراجع الذهب قرب أدنى مستوى في سبعة أسابيع قبل صدور بيانات اقتصادية أميركية.",
    ),
    # ------------------------------------------------------ Rubio
    (
        "alqabas", "القبس", "ar",
        "روبيو: حادثة القاعدة الجوية البريطانية تحمل بصمات «طرف أجنبي»",
        "قال وزير الخارجية الأميركي إن الحادثة في القاعدة الجوية البريطانية تحمل "
        "بصمات طرف أجنبي.",
    ),
    # ------------------------------------------------------ Health
    (
        "alanba", "الأنباء", "ar",
        "«الصحة» تعلن افتتاح أقسام جديدة في مستشفى الفروانية",
        "أعلنت وزارة الصحة افتتاح أقسام طبية جديدة في مستشفى الفروانية ضمن خطة "
        "توسعة الخدمات الصحية.",
    ),
    (
        "kuwaittimes", "Kuwait Times", "en",
        "Kuwait-Swiss ties mark 60 yrs of diplomatic relations",
        "Kuwait and Switzerland marked 60 years of diplomatic relations during the "
        "Crown Prince's visit to Bern.",
    ),
    (
        "arabtimes", "Arab Times", "en",
        "Kuwaitization Axes 62 More Expat Judicial Jobs",
        "Sixty-two expatriate employees in supporting judicial roles will be replaced "
        "by Kuwaiti nationals in the second phase of Kuwaitization.",
    ),
]

SOURCES: list[tuple[str, str, str, int, bool]] = [
    ("alrai", "الرأي", "ar", 1, False),
    ("alanba", "الأنباء", "ar", 1, False),
    ("aljarida", "الجريدة", "ar", 1, False),
    ("alqabas", "القبس", "ar", 1, True),
    ("kuwaittimes", "Kuwait Times", "en", 1, False),
    ("arabtimes", "Arab Times", "en", 2, False),
    ("timeskuwait", "Times Kuwait", "en", 2, False),
]


def build_articles() -> list[Article]:
    now = now_kuwait()
    tier_of = {sid: tier for sid, _, _, tier, _ in SOURCES}
    articles: list[Article] = []
    for index, (sid, name, lang, title, summary) in enumerate(FIXTURES):
        articles.append(
            Article(
                source_id=sid,
                source_name=name,
                lang=lang,
                title=title,
                url=f"https://example.invalid/{sid}/{1000 + index}",
                section="local",
                summary=summary,
                published=now - timedelta(minutes=35 * (index % 14)),
                fetched_at=now,
                tier=tier_of.get(sid, 2),
            )
        )
    return articles


def fake_collect(sources, *, max_workers=12, per_source_limit=120):  # noqa: ARG001
    """Drop-in replacement for sources.collect() that reads the fixtures."""
    articles = build_articles()
    wanted = {s.id for s in sources}
    articles = [a for a in articles if a.source_id in wanted]
    health = [
        SourceHealth(
            source_id=sid,
            name=name,
            tier=tier,
            ok=True,
            article_count=sum(1 for a in articles if a.source_id == sid),
            targets_total=1,
            targets_ok=1,
            optional=optional,
        )
        for sid, name, _lang, tier, optional in SOURCES
        if sid in wanted
    ]
    return articles, health


def main(argv: list[str] | None = None) -> int:
    cli.collect = fake_collect  # type: ignore[assignment]
    args = list(argv or [])
    if not args:
        args = ["run", "--no-llm", "--no-email", "--no-fetch-bodies", "--no-cache", "-v"]
    return cli.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
