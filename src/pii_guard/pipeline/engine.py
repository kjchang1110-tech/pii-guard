"""Core PII anonymization engine using Presidio + CKIP BERT."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import ConflictResolutionStrategy, OperatorConfig

from pii_guard.recognizers.tw_recognizers import TW_ENTITY_TYPES, get_all_tw_recognizers

logger = logging.getLogger(__name__)

# All entity types the engine handles.
# PERSON/ORG/LOCATION come from CKIP BERT NER; the rest from TW_ENTITY_TYPES.
SUPPORTED_ENTITIES: list[str] = [
    "PERSON",
    "ORG",
    "LOCATION",
    *TW_ENTITY_TYPES,
]

# CKIP NER label → Presidio entity type mapping
_CKIP_LABEL_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "FAC": "LOCATION",
}


def _build_analyzer(
    ckip_model: str,
    *,
    english_ner: bool = True,
) -> AnalyzerEngine:
    """
    Build Presidio AnalyzerEngine backed by CKIP BERT (for PERSON/ORG/LOCATION)
    plus Taiwan-specific PatternRecognizers.

    Falls back gracefully if transformers/spaCy models are unavailable.
    Optionally registers an English NER recognizer.
    """
    nlp_engine = _create_nlp_engine(ckip_model)
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["zh"],
    )

    # Remove Presidio built-in recognizers that conflict with our TW variants.
    # The built-in EmailRecognizer uses \b boundaries which produce wrong spans
    # in Chinese text (e.g. "信箱user@x.com" matched as full string instead of
    # just "user@x.com"), and its score=1.0 overrides our TwEmailRecognizer.
    _remove = {"EmailRecognizer", "CreditCardRecognizer"}
    analyzer.registry.recognizers = [
        r for r in analyzer.registry.recognizers
        if r.name not in _remove
    ]

    for recognizer in get_all_tw_recognizers():
        analyzer.registry.add_recognizer(recognizer)

    if english_ner:
        try:
            from pii_guard.recognizers.english_ner_recognizer import EnglishNerRecognizer

            analyzer.registry.add_recognizer(EnglishNerRecognizer())
            logger.info("EnglishNerRecognizer enabled (en_core_web_sm)")
        except Exception as exc:
            logger.warning("EnglishNerRecognizer unavailable (%s)", exc)

    return analyzer


def _create_nlp_engine(ckip_model: str):
    """Create NLP engine: TransformersNlpEngine (CKIP) → SpacyNlpEngine fallback."""
    try:
        from presidio_analyzer.nlp_engine import TransformersNlpEngine

        models = [
            {
                "lang_code": "zh",
                "model_name": {
                    "spacy": "zh_core_web_sm",
                    "transformers": ckip_model,
                },
            }
        ]

        # Presidio ≥ 2.2.34 exposes NerModelConfiguration for label mapping
        try:
            from presidio_analyzer.nlp_engine.transformers_nlp_engine import (
                NerModelConfiguration,
            )

            ner_config = NerModelConfiguration(
                model_to_presidio_entity_mapping=_CKIP_LABEL_MAP,
                aggregation_strategy="simple",
                default_score=0.85,
                low_score_entity_names=["MISC", "NORP", "WORK_OF_ART", "EVENT"],
            )
            engine = TransformersNlpEngine(models=models, ner_model_configuration=ner_config)
            logger.info("Using CKIP TransformersNlpEngine with NerModelConfiguration")
        except (ImportError, TypeError):
            engine = TransformersNlpEngine(models=models)
            logger.info("Using CKIP TransformersNlpEngine (no NerModelConfiguration)")

        return engine

    except (ImportError, OSError, ValueError) as exc:
        # ValueError covers "Can't find factory for 'hf_token_pipe'" when
        # spacy-transformers is not installed.
        logger.warning(
            "TransformersNlpEngine unavailable (%s). "
            "PERSON/ORG/LOCATION detection via CKIP will be disabled. "
            "Taiwan regex recognizers still active.",
            exc,
        )
        from presidio_analyzer.nlp_engine import SpacyNlpEngine

        return SpacyNlpEngine(
            models=[{"lang_code": "zh", "model_name": "zh_core_web_sm"}]
        )


# CKIP BERT max sequence = 512 tokens。TransformersNlpEngine 對超長文本截斷後
# spacy-transformers alignment 整體失效——不是只丟尾段實體、是**全文 NER 實體全滅**
# （實測：300 字命中、600 字含長 ASCII 行全滅；regex recognizer 不受影響）。
# 修法＝分塊偵測 + offset 平移合併。塊長取保守 350 字（中文 ≈ 1 token/字、
# ASCII 子詞另計、留 margin）；優先在換行處切（regex 實體極少跨行、
# 避免地址/電話被硬切兩半）；單行超長才 hard cut。
_CHUNK_MAX_CHARS = 350


def _chunk_spans(text: str, max_chars: int = _CHUNK_MAX_CHARS) -> list[tuple[int, int]]:
    """把 text 切成 [(start, end), ...] 連續覆蓋全文；優先在換行切。"""
    spans: list[tuple[int, int]] = []
    pos = 0
    n = len(text)
    while pos < n:
        if n - pos <= max_chars:
            spans.append((pos, n))
            break
        cut = text.rfind("\n", pos + 1, pos + max_chars)
        if cut <= pos:
            cut = pos + max_chars   # 單行超長、hard cut（罕見、接受）
        spans.append((pos, cut))
        pos = cut
    return spans


def _merge_adjacent_spans(results: list[RecognizerResult]) -> list[RecognizerResult]:
    """Merge adjacent or overlapping spans of the same entity type.

    CKIP sometimes splits a single entity into multiple tokens
    (e.g. "台北市信義區" → [4:9] + [9:10]).  This merges them back.
    """
    if not results:
        return results
    sorted_results = sorted(results, key=lambda r: (r.entity_type, r.start))
    merged: list[RecognizerResult] = []
    for r in sorted_results:
        if (
            merged
            and merged[-1].entity_type == r.entity_type
            and r.start <= merged[-1].end  # adjacent or overlapping
        ):
            prev = merged[-1]
            merged[-1] = RecognizerResult(
                entity_type=prev.entity_type,
                start=prev.start,
                end=max(prev.end, r.end),
                score=max(prev.score, r.score),
            )
        else:
            merged.append(r)
    return merged


def _filter_person_over_date(results: list[RecognizerResult]) -> list[RecognizerResult]:
    """Drop PERSON spans that fully overlap a TW_BIRTH_DATE span.

    CKIP sometimes tags Minguo dates (民國85年12月3日) as PERSON.
    """
    date_spans = {(r.start, r.end) for r in results if r.entity_type == "TW_BIRTH_DATE"}
    if not date_spans:
        return results
    return [
        r for r in results
        if not (
            r.entity_type == "PERSON"
            and any(ds <= r.start and r.end <= de for ds, de in date_spans)
        )
    ]


# Common Taiwan surnames (single-char, ~top 120 by frequency) plus frequent
# compound surnames. Used by _expand_person_surname.
_HAN_RUN_RE = re.compile(r"[一-龥]+")
from pii_guard.recognizers.tw_surnames import (
    TW_SURNAMES_1 as _TW_SURNAMES_1,
    TW_SURNAMES_2 as _TW_SURNAMES_2,
)


def _expand_person_surname(
    results: list[RecognizerResult], text: str
) -> list[RecognizerResult]:
    """Expand PERSON spans leftward until they start with a surname.

    CKIP NER sometimes yields only part of the given name（「陳大文」→ span
    「大文」、or just the last char「文」），leaving「陳」or「陳大」in the output.
    Try absorbing k preceding Han chars (largest k first) so the span becomes a
    plausible Taiwan name: single surname → ≤3 chars（k=1 always allowed so a
    4-char 冠夫姓「陳林小明」still expands as before）；compound surname → ≤4.
    Absorbed chars must be Han and not claimed by another entity span.
    """
    occupied: list[tuple[int, int]] = [
        (r.start, r.end) for r in results
    ]

    def _claimed(pos: int, current: RecognizerResult) -> bool:
        return any(
            s <= pos < e
            for (s, e), r in zip(occupied, results)
            if r is not current
        )

    expanded: list[RecognizerResult] = []
    for r in results:
        if r.entity_type != "PERSON":
            expanded.append(r)
            continue
        start = r.start
        span_len = r.end - r.start
        for k in range(min(3, 4 - span_len), 0, -1):
            s = r.start - k
            if s < 0:
                continue
            seg = text[s:r.start]
            if not _HAN_RUN_RE.fullmatch(seg) or any(_claimed(p, r) for p in range(s, r.start)):
                continue
            if k >= 2 and seg[:2] in _TW_SURNAMES_2 and span_len + k <= 4:
                start = s
                break
            if seg[0] in _TW_SURNAMES_1 and (span_len + k <= 3 or k == 1):
                start = s
                break
        if start != r.start:
            r = RecognizerResult(
                entity_type=r.entity_type, start=start, end=r.end, score=r.score,
            )
        expanded.append(r)
    return expanded


# CKIP 邊界常多吃相鄰虛詞（P-7、2026-09 真件＋重現：「與陳大文於現場」→「陳大|文於」合併成
# 「陳大文於」、同一人兩處映射成兩個 placeholder；「林志和與王小明」→「與王小明」
# 被 merge 跟「林志和」串成一人）。前導＝連接詞、尾字＝介詞／連接詞／助詞；皆非姓氏。
_PERSON_LEAD_PARTICLES = frozenset("與及和跟暨")
_PERSON_TRAIL_PARTICLES = frozenset("於之與及等")
# 剝完只剩 2 字（單姓＋單名）時、名末字真是該字的機率不可忽略（「之」尤常見於名）→
# 只對名末幾乎不用的字放行；其餘靠「全文別處有同名 span」佐證。
_PERSON_TRAIL_SAFE_2 = frozenset("於")


def _surname_led(s: str) -> bool:
    return bool(s) and (s[0] in _TW_SURNAMES_1 or s[:2] in _TW_SURNAMES_2)


def _trim_person_particles(
    results: list[RecognizerResult], text: str, *, lead: bool, trail: bool
) -> list[RecognizerResult]:
    """剝 PERSON span 首尾的虛詞（每端至多一字）。

    - 前導（`lead`）：首字為連接詞、剩餘 ≥2 字且姓氏起頭 → 剝。須在 merge **之前**跑，
      否則「林志|和|與王小明」已被串成一段、剝首字無濟於事。
    - 尾字（`trail`）：末字為虛詞、剩餘姓氏起頭，且（剩餘 ≥3 字〔單姓三字名已滿〕／
      剩餘 2 字而末字屬 `_PERSON_TRAIL_SAFE_2`／剩餘字串與全文另一個 PERSON span 相同）
      → 剝。merge 前後各跑一次：CKIP 可能給「陳大|文於」、單段看「文於」剝完不成姓名形狀。
    """
    others = {text[r.start:r.end] for r in results if r.entity_type == "PERSON"}
    out: list[RecognizerResult] = []
    for r in results:
        if r.entity_type == "PERSON":
            start, end = r.start, r.end
            if lead and end - start >= 3 and text[start] in _PERSON_LEAD_PARTICLES \
                    and _surname_led(text[start + 1:end]):
                start += 1
            if trail and end - start >= 3 and text[end - 1] in _PERSON_TRAIL_PARTICLES:
                rest = text[start:end - 1]
                if _surname_led(rest) and (
                    len(rest) >= 3
                    or text[end - 1] in _PERSON_TRAIL_SAFE_2
                    or rest in others
                ):
                    end -= 1
            if (start, end) != (r.start, r.end):
                r = RecognizerResult(
                    entity_type=r.entity_type, start=start, end=end, score=r.score,
                )
        out.append(r)
    return out


def _boost_tw_name_shape(
    results: list[RecognizerResult], text: str
) -> list[RecognizerResult]:
    """台灣姓名形狀先驗：姓氏起頭的 2-4 字 PERSON 候選 +0.2。

    CKIP 在雜訊文脈（markdown 標記/長 ASCII 行/表格）對真人名的信心會掉到
    threshold 邊緣（實測 0.4999 vs 短文 >0.5）。候選已由 NER 提出、只是信心
    不足——形狀符合台灣姓名（常見姓 + 總長 2-4）就加分、讓邊緣真陽性過門檻。
    """
    boosted: list[RecognizerResult] = []
    for r in results:
        if r.entity_type == "PERSON":
            s = text[r.start:r.end]
            if 2 <= len(s) <= 4 and (s[0] in _TW_SURNAMES_1 or s[:2] in _TW_SURNAMES_2):
                r = RecognizerResult(
                    entity_type=r.entity_type, start=r.start, end=r.end,
                    score=min(1.0, r.score + 0.2),
                )
        boosted.append(r)
    return boosted

class PiiGuardEngine:
    """
    Orchestrates PII detection and reversible anonymization for Traditional Chinese text.

    Usage::

        engine = PiiGuardEngine()
        anonymized, mapping = engine.anonymize("張大明的身分證A123456789")
        # anonymized → "<PERSON_1>的身分證<TW_NATIONAL_ID_1>"
        # mapping    → {"<PERSON_1>": "張大明", "<TW_NATIONAL_ID_1>": "A123456789"}

        original = engine.deanonymize(anonymized, mapping)
        assert original == "張大明的身分證A123456789"
    """

    def __init__(
        self,
        ckip_model: str = "ckiplab/bert-base-chinese-ner",
        score_threshold: float = 0.5,
        english_ner: bool = True,
    ) -> None:
        self.score_threshold = score_threshold
        self._analyzer = _build_analyzer(
            ckip_model,
            english_ner=english_ner,
        )
        self._anonymizer = AnonymizerEngine()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_detect(self, text: str) -> list[RecognizerResult]:
        """Run analyzer + post-processing (merge spans, resolve conflicts).

        長文分塊（見 _chunk_spans docstring）：整篇餵 analyzer 會踩 CKIP 512-token
        截斷 → alignment 全滅（全文 NER 實體歸零）。逐塊 analyze 後把 span offset
        平移回全文座標、再統一跑 merge / filter / surname-expand。
        """
        # 低門檻初篩（讓 threshold 邊緣的真陽性進 pipeline）→ 姓名形狀加分 →
        # 最後才按正式 threshold 過濾（非 PERSON / 非姓名形狀者結果與單段直篩相同）。
        prelim_threshold = min(0.3, self.score_threshold)
        results: list[RecognizerResult] = []
        for c_start, c_end in _chunk_spans(text):
            chunk_results = self._analyzer.analyze(
                text=text[c_start:c_end],
                language="zh",
                entities=SUPPORTED_ENTITIES,
                score_threshold=prelim_threshold,
            )
            for r in chunk_results:
                results.append(RecognizerResult(
                    entity_type=r.entity_type,
                    start=r.start + c_start,
                    end=r.end + c_start,
                    score=r.score,
                ))
        results = _trim_person_particles(results, text, lead=True, trail=True)
        results = _merge_adjacent_spans(results)
        results = _trim_person_particles(results, text, lead=False, trail=True)
        results = _filter_person_over_date(results)
        results = _expand_person_surname(results, text)
        results = _boost_tw_name_shape(results, text)
        results = [r for r in results if r.score >= self.score_threshold]
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anonymize(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Anonymize PII in *text*.

        Returns
        -------
        anonymized_text : str
            Text with PII replaced by numbered placeholders, e.g. ``<PERSON_1>``.
        mapping : dict[str, str]
            ``{placeholder: original_value}`` — needed for :meth:`deanonymize`.
        """
        # Shared mutable state for the operator lambdas (closure)
        entity_mapping: dict[str, str] = {}   # original_value → placeholder
        counters: dict[str, int] = {}          # entity_type → running count

        def make_lambda(entity_type: str):
            def replace_fn(original: str) -> str:
                # Presidio's Custom.validate() always calls lambda("PII") to type-check
                # the return value. Skip this sentinel to avoid polluting entity_mapping.
                if original == "PII":
                    return "<VALIDATION>"
                if original not in entity_mapping:
                    counters[entity_type] = counters.get(entity_type, 0) + 1
                    placeholder = f"<{entity_type}_{counters[entity_type]}>"
                    entity_mapping[original] = placeholder
                return entity_mapping[original]
            return replace_fn

        operators = {
            et: OperatorConfig("custom", {"lambda": make_lambda(et)})
            for et in SUPPORTED_ENTITIES
        }

        results = self._raw_detect(text)

        anonymized_result = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,  # type: ignore[arg-type]
            operators=operators,
            # Default MERGE_SIMILAR_OR_CONTAINED only merges same-type spans and
            # drops fully-contained ones; it leaves partially overlapping spans of
            # DIFFERENT entity types untouched, which double-writes placeholders
            # into the anonymized text and corrupts restore (e.g. "IL" duplicated
            # into "ILIL" when a LOCATION span and an ORG span partially overlap).
            # REMOVE_INTERSECTIONS additionally clips such partial overlaps by score.
            conflict_resolution=ConflictResolutionStrategy.REMOVE_INTERSECTIONS,
        )

        # Reverse: placeholder → original (for deanonymize)
        reverse_mapping: dict[str, str] = {v: k for k, v in entity_mapping.items()}
        return anonymized_result.text, reverse_mapping

    def detect(self, text: str) -> list[RecognizerResult]:
        """Return raw RecognizerResult list without anonymizing."""
        return self._raw_detect(text)

    @staticmethod
    def deanonymize(text: str, mapping: dict[str, str]) -> str:
        """
        Restore anonymized *text* using *mapping*.

        Parameters
        ----------
        text : str
            Text containing placeholders like ``<PERSON_1>``.
        mapping : dict[str, str]
            ``{placeholder: original_value}`` as returned by :meth:`anonymize`.
        """
        # Sort longest first to avoid <PERSON_1> matching inside <PERSON_10>
        for placeholder in sorted(mapping, key=len, reverse=True):
            text = text.replace(placeholder, mapping[placeholder])
        return text

    # ------------------------------------------------------------------
    # Mapping persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_mapping(mapping: dict[str, str], path: Path) -> None:
        """Serialise *mapping* to a JSON file at *path*."""
        path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load_mapping(path: Path) -> dict[str, str]:
        """Load a mapping JSON file previously saved by :meth:`save_mapping`."""
        return json.loads(path.read_text(encoding="utf-8"))
