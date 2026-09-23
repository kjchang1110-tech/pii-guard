"""Unit tests for engine._harmonize_person_occurrences (P-8／P-9; no model download needed)."""

from __future__ import annotations

from presidio_analyzer import RecognizerResult

from pii_guard.pipeline.engine import _harmonize_person_occurrences


def _p(text: str, sub: str, nth: int = 0, score: float = 0.9, etype: str = "PERSON"):
    start = -1
    for _ in range(nth + 1):
        start = text.index(sub, start + 1)
    return RecognizerResult(entity_type=etype, start=start, end=start + len(sub), score=score)


def _spans(results, text):
    return [text[r.start:r.end] for r in results if r.entity_type == "PERSON"]


def _h(results, text):
    return _harmonize_person_occurrences(results, text, min_ref_score=0.5)


def test_p8_truncated_second_occurrence_extended():
    text = "承辦人林志和確認。林志和另行補件。"
    results = [_p(text, "林志和"), _p(text, "林志", 1)]
    assert _spans(_h(results, text), text) == ["林志和", "林志和"]


def test_p8_not_extended_when_following_text_differs():
    text = "林志和確認。林志明另行補件。"
    results = [_p(text, "林志和"), _p(text, "林志", 1)]
    assert _spans(_h(results, text), text) == ["林志和", "林志"]


def test_p8_not_extended_into_other_entity():
    text = "林志和確認。林志和平路"
    results = [_p(text, "林志和"), _p(text, "林志", 1),
               _p(text, "和平路", etype="LOCATION")]
    assert _spans(_h(results, text), text) == ["林志和", "林志"]


def test_p8_low_score_fragment_not_used_as_reference():
    text = "林志和確認。林志和另行補件。"
    results = [_p(text, "林志和", score=0.35), _p(text, "林志", 1)]
    assert _spans(_h(results, text), text) == ["林志和", "林志"]


def test_p9_leading_non_surname_char_trimmed():
    text = "業主張美玲之代理人到場，張美玲確認簽名。"
    results = [_p(text, "主張美玲"), _p(text, "張美玲", 1)]
    assert _spans(_h(results, text), text) == ["張美玲", "張美玲"]


def test_p9_married_name_not_trimmed():
    # 冠夫姓「陳林小明」姓氏起頭 → 即使別處有「林小明」也不剪
    text = "陳林小明到場，林小明確認。"
    results = [_p(text, "陳林小明"), _p(text, "林小明", 1)]
    assert _spans(_h(results, text), text) == ["陳林小明", "林小明"]


def test_no_reference_no_change():
    text = "主張美玲到場"
    results = [_p(text, "主張美玲")]
    assert _spans(_h(results, text), text) == ["主張美玲"]
