"""Unit tests for engine._trim_person_particles (P-7; no model download needed)."""

from __future__ import annotations

from presidio_analyzer import RecognizerResult

from pii_guard.pipeline.engine import _merge_adjacent_spans, _trim_person_particles


def _p(text: str, sub: str, nth: int = 0) -> RecognizerResult:
    start = -1
    for _ in range(nth + 1):
        start = text.index(sub, start + 1)
    return RecognizerResult(entity_type="PERSON", start=start, end=start + len(sub), score=0.9)


def _spans(results, text):
    return [text[r.start:r.end] for r in results]


def test_trailing_particle_after_full_name_stripped():
    # 真件觀察：「陳大|文於」合併成「陳大文於」、同一人兩個 placeholder
    text = "監造人員與陳大文於現場會勘"
    merged = _merge_adjacent_spans([_p(text, "陳大"), _p(text, "文於")])
    assert _spans(_trim_person_particles(merged, text, lead=False, trail=True), text) == ["陳大文"]


def test_trailing_yu_after_two_char_name_stripped():
    text = "王明於昨日到場"
    assert _spans(_trim_person_particles([_p(text, "王明於")], text, lead=False, trail=True),
                  text) == ["王明"]


def test_trailing_zhi_on_two_char_name_kept_without_corroboration():
    # 「之」常見於名末（王羲之型）：剩 2 字時沒有別處同名佐證就不剝
    text = "王羲之書法"
    assert _spans(_trim_person_particles([_p(text, "王羲之")], text, lead=False, trail=True),
                  text) == ["王羲之"]


def test_trailing_zhi_stripped_when_same_name_elsewhere():
    text = "王明之代理人到場，王明確認"
    results = [_p(text, "王明之"), _p(text, "王明", 1)]
    assert _spans(_trim_person_particles(results, text, lead=False, trail=True),
                  text) == ["王明", "王明"]


def test_leading_conjunction_stripped_so_two_people_not_merged():
    # CKIP：「林志|和|與王小明」→ merge 串成一人；前導剝除須在 merge 前
    text = "承辦人林志和與王小明於現場"
    raw = [_p(text, "與王小明"), _p(text, "林志"), _p(text, "和")]
    trimmed = _trim_person_particles(raw, text, lead=True, trail=True)
    merged = _merge_adjacent_spans(trimmed)
    assert sorted(_spans(merged, text)) == ["林志和", "王小明"]


def test_name_not_led_by_surname_untouched():
    text = "與會者之於"
    r = [_p(text, "會者之")]
    assert _spans(_trim_person_particles(r, text, lead=True, trail=True), text) == ["會者之"]


def test_two_char_span_untouched():
    text = "與王先生"
    r = [_p(text, "與王")]
    assert _spans(_trim_person_particles(r, text, lead=True, trail=True), text) == ["與王"]


def test_non_person_entities_untouched():
    text = "台北市於信義區"
    r = [RecognizerResult(entity_type="LOCATION", start=0, end=4, score=0.9)]
    out = _trim_person_particles(r, text, lead=True, trail=True)
    assert (out[0].start, out[0].end) == (0, 4)
