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


def test_expand_two_chars_when_ckip_marks_only_last_char():
    # CKIP 只標名末字（驗屋報告真件：「委託人 林宜樺」→ span「樺」、輸出「林宜<PERSON>」）
    text = "委託人 陳大文 "
    results = [_person(6, 7)]
    out = _expand_person_surname(results, text)
    assert text[out[0].start:out[0].end] == "陳大文"


def test_expand_three_chars_compound_surname_from_last_char():
    text = "會同歐陽志明檢驗"
    results = [_person(5, 6)]   # 只抓「明」
    out = _expand_person_surname(results, text)
    assert text[out[0].start:out[0].end] == "歐陽志明"


def test_single_char_span_prefers_three_char_name_over_two():
    # 「大」非姓氏、k=1 不成立；k=2 補到「陳」
    text = "驗屋師陳大文到場"
    results = [_person(5, 6)]   # 只抓「文」
    out = _expand_person_surname(results, text)
    assert text[out[0].start:out[0].end] == "陳大文"


def test_no_expand_across_non_han():
    text = "陳、文"
    results = [_person(2, 3)]
    out = _expand_person_surname(results, text)
    assert (out[0].start, out[0].end) == (2, 3)   # 「、」擋住、不拼「陳、文」


def test_four_char_married_double_surname_still_expands_by_one():
    text = "業主陳林小明女士"
    results = [_person(3, 6)]   # 「林小明」、前一字「陳」為姓（冠夫姓）
    out = _expand_person_surname(results, text)
    assert text[out[0].start:out[0].end] == "陳林小明"


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


# ── _chunk_spans（長文分塊、CKIP 512-token 截斷修）─────────────────────────

from pii_guard.pipeline.engine import _chunk_spans


def test_chunk_spans_cover_text_exactly():
    text = ("第一段內容\n" * 100)   # 600 chars
    spans = _chunk_spans(text, max_chars=350)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 == s2   # 連續無縫
    assert all(e - s <= 350 for s, e in spans)


def test_chunk_spans_prefer_newline_cut():
    text = "甲" * 300 + "\n" + "乙" * 300
    spans = _chunk_spans(text, max_chars=350)
    assert spans[0] == (0, 300)   # 在換行切、不硬切 350


def test_chunk_spans_hard_cut_single_long_line():
    text = "甲" * 800   # 無換行
    spans = _chunk_spans(text, max_chars=350)
    assert spans == [(0, 350), (350, 700), (700, 800)]


def test_chunk_spans_short_text_single_span():
    assert _chunk_spans("短文", max_chars=350) == [(0, 2)]
