"""Labeled-name recognizer for table/form cells.

CKIP NER reliably tags names in natural sentences but systematically misses
isolated table cells（「|委託人|林宜樺|」— no linguistic context around the
name）. Real-world hit: inspection reports render party info as markdown
tables; the buyer name survived a full CKIP + regex pass twice.

Deterministic rule: a 2-4 char Han string that (a) immediately follows a
party-label cell（委託人/買方/委託單位…、容忍中間空 cell）and (b) starts with a
common Taiwan surname → PERSON with high confidence. Over-scrubbing a rare
non-name value in a party-label cell is acceptable for de-identification
(fail-safe direction).
"""

from __future__ import annotations

import re
from typing import ClassVar

from presidio_analyzer import LocalRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from pii_guard.recognizers.tw_surnames import TW_SURNAMES_1, TW_SURNAMES_2

_LABELS = (
    "委託人", "委託單位", "買方", "承買人", "客戶姓名", "客戶", "業主",
    "甲方", "乙方", "聯絡人", "申請人", "姓名",
)

# 「標籤 [|：:] （可夾空 cell）姓名」— 姓名捕捉群組吃 2-4 個漢字、後面必須接
# cell 邊界（| / ：/ 換行 / 全形空白）避免吃進句子。
_PATTERN = re.compile(
    r"(?:" + "|".join(_LABELS) + r")\s*[|：:]\s*(?:\|\s*)*"
    r"([一-龥]{2,4})(?=\s*(?:[|：:\n）)]|$))",
    re.MULTILINE,
)


class TwLabeledNameRecognizer(LocalRecognizer):
    """PERSON from party-label + name-shaped cell（表格/表單去識別化補洞）。"""

    SUPPORTED_ENTITY: ClassVar[str] = "PERSON"

    def __init__(self) -> None:
        super().__init__(
            supported_entities=[self.SUPPORTED_ENTITY],
            supported_language="zh",
            name="TwLabeledNameRecognizer",
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
        for m in _PATTERN.finditer(text):
            name = m.group(1)
            # 姓氏起頭才算姓名形狀（「先生」「小姐」等稱謂、機關簡稱擋掉）
            if not (name[0] in TW_SURNAMES_1 or name[:2] in TW_SURNAMES_2):
                continue
            results.append(RecognizerResult(
                entity_type=self.SUPPORTED_ENTITY,
                start=m.start(1),
                end=m.end(1),
                score=0.85,
            ))
        return results
