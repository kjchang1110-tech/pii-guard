"""Taiwan street address recognizer.

Motivation: CKIP LOC reliably tags place *names* (city/district/landmark) but
systematically misses full Taiwan street addresses in structured documents
(e.g. 驗屋報告「物件地址：高雄市鳳山區中山西路 100 號 12 樓之 3」→ zero hits).
A deterministic pattern is both more reliable and gives precise span control.

Span design (deliberate): the match stops at 「號(之N)?」 and NEVER consumes a
trailing floor/unit segment (「12樓之3」「5F」). Use cases like inspection-report
de-identification require stripping the street address while *keeping* the
unit-floor information — cutting at 號 gives exactly that split.
"""

from __future__ import annotations

import re
from typing import ClassVar

from presidio_analyzer import Pattern, PatternRecognizer

# Same flag rationale as tw_phone_recognizer: no IGNORECASE, no \b (breaks in
# Chinese text); lookarounds where boundaries are needed.
_FLAGS = re.DOTALL | re.MULTILINE

# Building blocks（容忍全形/半形空白穿插於數字前後、常見於 PDF 抽出文字）
_SP = r"[ 　]*"
_NUM = rf"[0-9０-９]{{1,5}}{_SP}"
_CITY = r"[一-龥]{1,3}[縣市]"
_DISTRICT = r"[一-龥]{1,3}[鄉鎮市區]"
_VILLAGE = r"[一-龥]{1,4}[村里]"
_NEIGHBOR = rf"{_NUM}鄰"
_ROAD = r"[一-龥0-9０-９]{1,8}(?:路|街|大道)"
_SECTION = rf"(?:[0-9０-９一二三四五六七八九十]{{1,3}}{_SP}段{_SP})?"
_LANE = rf"(?:{_NUM}巷{_SP})?"
_ALLEY = rf"(?:{_NUM}弄{_SP})?"
_NO = rf"{_NUM}(?:之{_SP}[0-9０-９]{{1,3}}{_SP})?號"

# Full form: 縣市 (+區/村里/鄰 optional) + 路街 + (段/巷/弄) + 號 — strong signal
_FULL = (
    rf"{_CITY}{_SP}(?:{_DISTRICT})?{_SP}(?:{_VILLAGE})?{_SP}(?:{_NEIGHBOR})?{_SP}"
    rf"{_ROAD}{_SP}{_SECTION}{_LANE}{_ALLEY}{_NO}"
)
# Partial form: 路街起頭（無縣市）— ambiguous alone, needs context boost
_PARTIAL = rf"{_ROAD}{_SP}{_SECTION}{_LANE}{_ALLEY}{_NO}"


class TwAddressRecognizer(PatternRecognizer):
    """Taiwan street address, matched up to 號 — floor/unit («12樓之3») survives."""

    SUPPORTED_ENTITY: ClassVar[str] = "TW_ADDRESS"
    PATTERNS: ClassVar[list[Pattern]] = [
        Pattern("TW_ADDRESS_FULL", _FULL, 0.9),
        Pattern("TW_ADDRESS_PARTIAL", _PARTIAL, 0.45),
    ]
    CONTEXT: ClassVar[list[str]] = [
        "地址", "住址", "住所", "戶籍", "物件", "位於", "所在地", "案址", "工地",
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity=self.SUPPORTED_ENTITY,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="zh",
            global_regex_flags=_FLAGS,
        )
