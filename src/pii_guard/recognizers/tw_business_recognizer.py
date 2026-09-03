"""Taiwan Unified Business Number (統一編號) recognizer with context + checksum validation."""

from __future__ import annotations

import re
from typing import ClassVar

from presidio_analyzer import LocalRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts


class TwBusinessIdRecognizer(LocalRecognizer):
    """
    Recognizes Taiwan Unified Business Numbers (統一編號, 8 digits).

    Uses two-layer filtering to minimize false positives:
    1. Context keyword within ±50 characters
    2. Official MOF checksum validation (weighted digits, mod 5 per 112/4 新制,
       special rule for 7th digit = 7)
    """

    SUPPORTED_ENTITY: ClassVar[str] = "TW_BUSINESS_ID"
    # Lookaround instead of \b: re.ASCII maps to regex.V1 in Presidio's regex module,
    # breaking boundary detection in Chinese text.
    PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"(?<!\d)\d{8}(?!\d)")
    CONTEXT_KEYWORDS: ClassVar[list[str]] = [
        "統一編號", "統編", "公司", "廠商", "發票", "稅籍", "法人", "營業",
        "企業", "行號", "商號",
    ]
    CONTEXT_WINDOW: ClassVar[int] = 50
    _WEIGHTS: ClassVar[list[int]] = [1, 2, 1, 2, 1, 2, 4, 1]
    _MODULUS: ClassVar[int] = 5   # 112/4 新制；舊制 10（新制為舊制超集）

    def __init__(self) -> None:
        super().__init__(
            supported_entities=[self.SUPPORTED_ENTITY],
            supported_language="zh",
            name="TwBusinessIdRecognizer",
        )

    def load(self) -> None:
        pass

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        if self.SUPPORTED_ENTITY not in entities:
            return []
        results: list[RecognizerResult] = []
        for match in self.PATTERN.finditer(text):
            number = match.group()
            if not self._validate_checksum(number):
                continue
            if not self._has_context(text, match.start(), match.end()):
                continue
            results.append(
                RecognizerResult(
                    entity_type=self.SUPPORTED_ENTITY,
                    start=match.start(),
                    end=match.end(),
                    score=0.85,
                )
            )
        return results

    @classmethod
    def _validate_checksum(cls, number: str) -> bool:
        """
        Taiwan MOF checksum (財政部 112/4 新制): weighted cross-product digit sum
        divisible by **5** (pre-112/4 rule was 10; new numbers issued since then
        only satisfy the mod-5 rule, and every old number still passes).
        Ref: 財政資訊中心「營利事業統一編號檢查碼邏輯修正說明」——
        https://www.fia.gov.tw/singlehtml/3?cntId=c4d9cff38c8642ef8872774ee9987283

        Special rule when the 7th digit (index 6) is '7': its product 7×4=28
        digit-sums to 2+8=10, and the MOF spec lets that term count as either
        0 or 1 (official examples 10458575 → Z2=20 / 19312376 → Z1=30). The
        naive sum below counts it as 10, so the two candidates are
        ``total - 10`` and ``total - 9`` — i.e. ``total`` and ``total + 1`` mod 5.
        (Previous implementation checked ``total - 1``, which rejects the
        official example 19312376.)
        """
        if len(number) != 8 or not number.isdigit():
            return False
        total = sum(
            (int(d) * w) // 10 + (int(d) * w) % 10
            for d, w in zip(number, cls._WEIGHTS)
        )
        if total % cls._MODULUS == 0:
            return True
        if number[6] == "7" and (total + 1) % cls._MODULUS == 0:
            return True
        return False

    def _has_context(self, text: str, start: int, end: int) -> bool:
        window_start = max(0, start - self.CONTEXT_WINDOW)
        window_end = min(len(text), end + self.CONTEXT_WINDOW)
        window = text[window_start:window_end]
        return any(kw in window for kw in self.CONTEXT_KEYWORDS)
