"""Unit tests for engine._expand_person_surname (no model download needed)."""

from __future__ import annotations

from presidio_analyzer import RecognizerResult

from pii_guard.pipeline.engine import _expand_person_surname


def _person(start: int, end: int, score: float = 0.9) -> RecognizerResult:
    return RecognizerResult(entity_type="PERSON", start=start, end=end, score=score)


def test_expand_single_char_surname():
    text = "驗屋師陳大文到場"
    # CKIP 只抓「大文」(4:6)、姓「陳」(3) 殘留
    results = [_person(4, 6)]
    out = _expand_person_surname(results, text)
    assert (out[0].start, out[0].end) == (3, 6)
    assert text[out[0].start:out[0].end] == "陳大文"


def test_expand_compound_surname():
    text = "會同歐陽志明檢驗"
    results = [_person(4, 6)]   # 只抓「志明」
    out = _expand_person_surname(results, text)
    assert text[out[0].start:out[0].end] == "歐陽志明"


def test_no_expand_when_full_name_already():
    text = "買方：王小明先生"
    results = [_person(3, 6)]   # 已含姓、前一字是「：」
    out = _expand_person_surname(results, text)
    assert (out[0].start, out[0].end) == (3, 6)


def test_no_expand_when_prev_char_claimed_by_other_entity():
    text = "王小林大文"
    p1 = _person(0, 3)          # 「王小林」完整人名
    p2 = _person(3, 5)          # 「大文」缺姓、但前一字「林」屬 p1
    out = _expand_person_surname([p1, p2], text)
    spans = sorted((r.start, r.end) for r in out)
    assert spans == [(0, 3), (3, 5)]   # 「林」被 p1 佔用、p2 不擴


def test_non_person_untouched():
    text = "陳0912345678"
    mobile = RecognizerResult(entity_type="TW_MOBILE", start=1, end=11, score=0.9)
    out = _expand_person_surname([mobile], text)
    assert (out[0].start, out[0].end) == (1, 11)
