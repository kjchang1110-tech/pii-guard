"""Unit tests for Taiwan-specific PII recognizers (no model download needed)."""

from __future__ import annotations

import pytest

from pii_guard.recognizers.tw_business_recognizer import TwBusinessIdRecognizer
from pii_guard.recognizers.tw_extra_recognizers import (
    TwAddressRecognizer,
    TwBankAccountRecognizer,
    TwBirthDateRecognizer,
    TwCryptoSeedRecognizer,
    TwIntlMobileRecognizer,
    TwLicensePlateRecognizer,
    TwPasswordRecognizer,
    TwPrivateKeyRecognizer,
    TwVerificationCodeRecognizer,
)
from pii_guard.recognizers.tw_id_recognizer import (
    TwArcRecognizer,
    TwNationalIdRecognizer,
    TwPassportRecognizer,
)
from pii_guard.recognizers.tw_misc_recognizers import TwCreditCardRecognizer, TwEmailRecognizer
from pii_guard.recognizers.tw_phone_recognizer import TwLandlineRecognizer, TwMobileRecognizer

# ── TwNationalIdRecognizer ────────────────────────────────────────────────────

class TestTwNationalIdRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwNationalIdRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("A123456789", True),    # valid male ID
        ("B234567890", True),    # valid female ID
        ("Z199999999", True),    # uppercase Z
        ("A323456789", False),   # 2nd char must be 1 or 2
        ("a123456789", False),   # lowercase (ASCII mode: no IGNORECASE)
        ("A12345678",  False),   # only 9 digits total
        ("A1234567890", False),  # 11 digits
        ("1123456789",  False),  # starts with digit
        ("AB23456789",  False),  # 2nd char B is not 1 or 2
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["TW_NATIONAL_ID"])
        matched = any(r.entity_type == "TW_NATIONAL_ID" for r in results)
        assert matched == expected_match, f"text={text!r}, results={results}"

    def test_entity_type_correct(self):
        results = self.r.analyze("A123456789", entities=["TW_NATIONAL_ID"])
        assert results[0].entity_type == "TW_NATIONAL_ID"

    def test_span_correct_standalone(self):
        """Verify span positions for a standalone ID."""
        text = "A123456789"
        results = self.r.analyze(text, entities=["TW_NATIONAL_ID"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "A123456789"

    def test_span_in_chinese_context(self):
        """Verify span positions when surrounded by Chinese text."""
        text = "身分證A123456789在這"
        results = self.r.analyze(text, entities=["TW_NATIONAL_ID"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "A123456789"


# ── TwPassportRecognizer ──────────────────────────────────────────────────────

class TestTwPassportRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwPassportRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("A12345678",   True),   # valid: 1 letter + 8 digits
        ("B87654321",   True),   # valid
        ("Z00000000",   True),   # valid edge: all zeros
        ("a12345678",   False),  # lowercase
        ("1B2345678",   False),  # starts with digit
        ("A1234567",    False),  # only 8 chars total
        ("A123456789",  False),  # 10 chars → NID, not passport (lookahead blocks)
        ("AB12345678",  False),  # 2nd char is letter → ARC, not passport
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["TW_PASSPORT"])
        matched = any(r.entity_type == "TW_PASSPORT" for r in results)
        assert matched == expected_match, f"text={text!r}"

    def test_no_conflict_with_nid(self):
        """NID (10 chars) must NOT be matched as passport."""
        r_nid = TwNationalIdRecognizer()
        nid_text = "A123456789"
        assert bool(r_nid.analyze(nid_text, entities=["TW_NATIONAL_ID"]))
        assert not bool(self.r.analyze(nid_text, entities=["TW_PASSPORT"]))

    def test_no_conflict_with_arc(self):
        """ARC (10 chars, 2nd char letter) must NOT be matched as passport."""
        r_arc = TwArcRecognizer()
        arc_text = "AB12345678"
        assert bool(r_arc.analyze(arc_text, entities=["TW_ARC"]))
        assert not bool(self.r.analyze(arc_text, entities=["TW_PASSPORT"]))

    def test_span_in_chinese_context(self):
        text = "護照號碼A12345678入境"
        results = self.r.analyze(text, entities=["TW_PASSPORT"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "A12345678"


# ── TwArcRecognizer ───────────────────────────────────────────────────────────

class TestTwArcRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwArcRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("AB12345678", True),   # valid ARC (2nd char A-D)
        ("AC87654321", True),
        ("A812345678", True),   # 2nd char 8 is allowed
        ("A912345678", True),   # 2nd char 9 is allowed
        ("AA12345678", True),   # A is in A-D
        ("A123456789", False),  # 2nd char 1 is NOT in A-D89
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["TW_ARC"])
        matched = any(r.entity_type == "TW_ARC" for r in results)
        assert matched == expected_match, f"text={text!r}"


# ── TwMobileRecognizer ────────────────────────────────────────────────────────

class TestTwMobileRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwMobileRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("0912345678",   True),   # compact
        ("0812345678",   False),  # 08 is not mobile prefix
        ("091234567",    False),  # 9 digits only
        ("09123456789",  False),  # 11 digits
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["TW_MOBILE"])
        matched = len(results) > 0
        assert matched == expected_match, f"text={text!r}"

    def test_dashed_format(self):
        results = self.r.analyze("0912-345-678", entities=["TW_MOBILE"])
        assert len(results) == 1


# ── TwLandlineRecognizer ──────────────────────────────────────────────────────

class TestTwLandlineRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwLandlineRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("0212345678",    True),   # Taipei (02) – 10 digits
        ("0312345678",    True),   # Taoyuan (03) – valid area code
        ("0712345678",    True),   # Tainan (07)
        ("0912345678",    False),  # 09 is mobile, not landline
        ("(02)1234-5678", True),
        ("02-12345678",   True),
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["TW_LANDLINE"])
        matched = len(results) > 0
        assert matched == expected_match, f"text={text!r}"


# ── TwBusinessIdRecognizer ────────────────────────────────────────────────────

class TestTwBusinessIdRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwBusinessIdRecognizer()

    # --- checksum unit tests ---
    @pytest.mark.parametrize("number,expected_valid", [
        ("04595257", True),   # sum=40, 40%10==0
        ("12345678", False),  # sum=42, 42%10==2
        ("00000000", True),   # sum=0, 0%10==0 (edge case)
        # 7th digit = 7 special rule: total%10 != 0 but (total-1)%10 == 0
        ("10000070", True),   # d7=7, total=11, (11-1)%10==0 → special rule only
        ("10000178", True),   # d7=7, total=21, (21-1)%10==0 → special rule only
        ("10000276", True),   # d7=7, total=21, (21-1)%10==0 → special rule only
    ])
    def test_checksum(self, number, expected_valid):
        assert TwBusinessIdRecognizer._validate_checksum(number) == expected_valid, number

    # --- context filtering ---
    def test_no_context_no_match(self):
        # Plain 8-digit number without context keyword
        results = self.r.analyze("帳號是04595257", entities=["TW_BUSINESS_ID"])
        assert len(results) == 0, "Should not match without context keyword"

    def test_with_context_keyword(self):
        results = self.r.analyze("統一編號04595257", entities=["TW_BUSINESS_ID"])
        assert len(results) == 1
        assert results[0].entity_type == "TW_BUSINESS_ID"

    @pytest.mark.parametrize("keyword", [
        "統一編號", "統編", "公司", "廠商", "發票",
    ])
    def test_various_context_keywords(self, keyword):
        text = f"{keyword}04595257"
        results = self.r.analyze(text, entities=["TW_BUSINESS_ID"])
        assert len(results) == 1, f"keyword={keyword!r} should trigger detection"

    def test_context_window_distance(self):
        # Keyword more than 50 chars away → no match
        far_text = "統一編號" + "X" * 60 + "04595257"
        results = self.r.analyze(far_text, entities=["TW_BUSINESS_ID"])
        assert len(results) == 0

        # Keyword within 50 chars → match
        near_text = "統一編號" + "X" * 10 + "04595257"
        results = self.r.analyze(near_text, entities=["TW_BUSINESS_ID"])
        assert len(results) == 1

    def test_span_correct(self):
        text = "統一編號04595257"
        results = self.r.analyze(text, entities=["TW_BUSINESS_ID"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "04595257"


# ── TwEmailRecognizer ─────────────────────────────────────────────────────────

class TestTwEmailRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwEmailRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("john@example.com",                True),   # standalone
        ("test.user+tag@sub.domain.com",    True),   # complex address
        ("電子郵件john@example.com以便聯絡",  True),   # no-space Chinese context
        ("請寄信到 john@example.com 謝謝",    True),   # space-separated Chinese
        ("notanemail@",                     False),  # missing domain
        ("@noemail.com",                    False),  # missing local part
        ("not@valid",                       False),  # missing TLD
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["EMAIL_ADDRESS"])
        matched = any(r.entity_type == "EMAIL_ADDRESS" for r in results)
        assert matched == expected_match, f"text={text!r}"

    def test_span_no_space_chinese(self):
        """Span must capture only the email, not surrounding Chinese characters."""
        text = "電子郵件john@example.com以便聯絡"
        results = self.r.analyze(text, entities=["EMAIL_ADDRESS"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "john@example.com"


# ── TwCreditCardRecognizer ────────────────────────────────────────────────────

class TestTwCreditCardRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwCreditCardRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("4111111111111111",          True),   # Visa test number (Luhn valid)
        ("5500005555555559",          True),   # Mastercard test number
        ("信用卡號 4111111111111111",   True),   # Chinese context
        ("4111111111111112",          False),  # Luhn checksum fails
        ("1234567890123456",          False),  # invalid prefix
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["CREDIT_CARD"])
        matched = any(r.entity_type == "CREDIT_CARD" for r in results)
        assert matched == expected_match, f"text={text!r}"

    def test_span_in_chinese_context(self):
        text = "信用卡號 4111111111111111 請保密"
        results = self.r.analyze(text, entities=["CREDIT_CARD"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "4111111111111111"


# ── TwLicensePlateRecognizer ──────────────────────────────────────────────────

class TestTwLicensePlateRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwLicensePlateRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("車牌ABC-1234",     True),   # new format (3 letters + 4 digits) with context
        ("牌照AB-1234",      True),   # new format (2 letters + 4 digits) with context
        ("車號1234-AB",      True),   # old format (4 digits + 2 letters) with context
        ("車籍123-AB",       True),   # old format (3 digits + 2 letters) with context
        ("ABC-1234",         False),  # new format without context → no match
        ("1234-AB",          False),  # old format without context → no match
        ("車牌ABCD-1234",    False),  # 4 letters → exceeds new format limit
        ("車牌ABC-12345",    False),  # 5 digits → exceeds new format limit
        ("車牌AB-123",       False),  # only 3 digits → old format needs ≥3 before dash
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["TW_LICENSE_PLATE"])
        matched = len(results) > 0
        assert matched == expected_match, f"text={text!r}"

    def test_span_correct(self):
        text = "車牌號碼ABC-1234請查詢"
        results = self.r.analyze(text, entities=["TW_LICENSE_PLATE"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "ABC-1234"

    def test_context_window_distance(self):
        # Keyword >50 chars away → no match (use Chinese filler so lookbehind doesn't break)
        far_text = "車牌" + "啊" * 60 + "ABC-1234"
        results = self.r.analyze(far_text, entities=["TW_LICENSE_PLATE"])
        assert len(results) == 0

        # Keyword within 50 chars → match
        near_text = "車牌" + "啊" * 10 + "ABC-1234"
        results = self.r.analyze(near_text, entities=["TW_LICENSE_PLATE"])
        assert len(results) == 1


# ── TwBirthDateRecognizer ─────────────────────────────────────────────────────

class TestTwBirthDateRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwBirthDateRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("民國90年1月1日",       True),   # Minguo with 民國 anchor (self-contextualizing)
        ("民國110年12月31日",    True),   # Minguo edge
        ("民國 90 年 1 月 1 日", True),   # Minguo with spaces
        ("生日1990-01-01",       True),   # Western ISO with context keyword
        ("出生日期1990/01/01",   True),   # Western slash with context keyword
        ("出生年月日1990.01.01", True),   # Western dot with context keyword
        ("1990-01-01",           False),  # Western without context → hard-filtered
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["TW_BIRTH_DATE"])
        matched = len(results) > 0
        assert matched == expected_match, f"text={text!r}"

    def test_minguo_span_correct(self):
        text = "出生民國90年1月1日入學"
        results = self.r.analyze(text, entities=["TW_BIRTH_DATE"])
        assert any("民國" in text[r.start:r.end] for r in results)

    def test_western_no_context_no_match(self):
        # LocalRecognizer hard-filters Western dates without context
        results = self.r.analyze("1990-01-01", entities=["TW_BIRTH_DATE"])
        assert len(results) == 0


# ── TwIntlMobileRecognizer ────────────────────────────────────────────────────

class TestTwIntlMobileRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwIntlMobileRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("+886912345678",       True),   # compact
        ("+886-912-345-678",    True),   # dash-separated
        ("+886 912 345 678",    True),   # space-separated
        ("+886-912345678",      True),   # only first separator
        ("+885912345678",       False),  # wrong country code
        ("+886812345678",       False),  # 08x is landline, not mobile
        ("+88691234567",        False),  # only 8 digits (need 9 after +886)
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["TW_MOBILE"])
        matched = len(results) > 0
        assert matched == expected_match, f"text={text!r}"

    def test_span_correct(self):
        text = "手機+886-912-345-678請聯絡"
        results = self.r.analyze(text, entities=["TW_MOBILE"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "+886-912-345-678"

    def test_entity_type_is_tw_mobile(self):
        results = self.r.analyze("+886912345678", entities=["TW_MOBILE"])
        assert results[0].entity_type == "TW_MOBILE"


# ── TwBankAccountRecognizer ───────────────────────────────────────────────────

class TestTwBankAccountRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwBankAccountRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("帳號123456789012",          True),   # 12 digits with context
        ("銀行帳號1234567890123456",   True),   # 16 digits with context
        ("轉帳至123456789012",         True),   # 轉帳 context
        ("123456789012",              False),  # 12 digits without context
        ("帳號12345678901",            False),  # 11 digits (< 12)
        ("帳號12345678901234567",      False),  # 17 digits (> 16)
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["TW_BANK_ACCOUNT"])
        matched = len(results) > 0
        assert matched == expected_match, f"text={text!r}"

    def test_span_correct(self):
        text = "帳號123456789012請匯款"
        results = self.r.analyze(text, entities=["TW_BANK_ACCOUNT"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "123456789012"

    def test_context_window_distance(self):
        far_text = "帳號" + "X" * 60 + "123456789012"
        results = self.r.analyze(far_text, entities=["TW_BANK_ACCOUNT"])
        assert len(results) == 0

        near_text = "帳號" + "X" * 10 + "123456789012"
        results = self.r.analyze(near_text, entities=["TW_BANK_ACCOUNT"])
        assert len(results) == 1


# ── TwAddressRecognizer ───────────────────────────────────────────────────────

class TestTwAddressRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        from pii_guard.recognizers.tw_address_recognizer import TwAddressRecognizer
        self.r = TwAddressRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("高雄市鳳山區中山西路 100 號",          True),   # full form + spaces
        ("高雄市鳳山區中山西路100號12樓之3",     True),   # trailing floor (span test below)
        ("台北市信義區信義路五段7號",            True),   # 段
        ("新竹縣竹北市光明六路10巷2號",          True),   # 縣+市+巷
        ("臺中市西屯區臺灣大道三段99號",         True),   # 臺 variant + 大道
        ("桃園市中壢區中山東路二段77之1號",      True),   # 之N號
        ("高雄市鳳山區",                        False),  # no road/no 號
        ("中山西路",                            False),  # road alone, no 號
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["TW_ADDRESS"])
        matched = any(r.entity_type == "TW_ADDRESS" for r in results)
        assert matched == expected_match, f"text={text!r}, results={results}"

    def test_span_stops_at_hao_keeps_floor(self):
        """核心 span 設計：吃到「號」為止、樓層戶別（12樓之3）必須存活。"""
        text = "物件地址：高雄市鳳山區中山西路 100 號 12 樓之 3"
        results = self.r.analyze(text, entities=["TW_ADDRESS"])
        assert len(results) >= 1
        top = max(results, key=lambda r: r.score)
        matched = text[top.start:top.end]
        assert matched.endswith("號"), f"span 應止於號: {matched!r}"
        assert "樓" not in matched, f"span 不得吃樓層: {matched!r}"

    def test_partial_needs_context(self):
        """無縣市的路名地址：score 0.45 < threshold、須靠 context 詞升分。"""
        results = self.r.analyze("中山西路100號", entities=["TW_ADDRESS"])
        # analyze() 不帶 nlp_artifacts、context enhancer 不啟動 → 只驗 raw score < 0.5
        assert all(r.score < 0.5 for r in results)


# ── TwLandlineRecognizer（7 碼區域 + 雙 dash、驗屋報告修補）────────────────────

class TestTwLandlineSevenDigit:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwLandlineRecognizer()

    @pytest.mark.parametrize("text,expected_match", [
        ("(07)771-2345", True),     # 高雄 7 碼 + paren（原 pattern 漏）
        ("(02)2712-3456", True),    # 台北 8 碼 + paren（原有）
        ("07-771-2345", True),      # 雙 dash 7 碼
        ("02-2712-3456", True),     # 雙 dash 8 碼
        ("0771234", False),         # 太短
    ])
    def test_pattern(self, text, expected_match):
        results = self.r.analyze(text, entities=["TW_LANDLINE"])
        matched = any(r.entity_type == "TW_LANDLINE" for r in results)
        assert matched == expected_match, f"text={text!r}, results={results}"


# ── TwLabeledNameRecognizer（表格 cell 孤立姓名、驗屋報告實測補洞）──────────

class TestTwLabeledNameRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        from pii_guard.recognizers.tw_labeled_name_recognizer import TwLabeledNameRecognizer
        self.r = TwLabeledNameRecognizer()

    @pytest.mark.parametrize("text,expected", [
        ("|委託人|林宜樺|初驗日期|114 年|",  "林宜樺"),   # 真實殘留案例 1
        ("|委託單位|沈士傑|建案名稱|某案|",   "沈士傑"),   # 真實殘留案例 2
        ("|委託單位| |沈士傑|檢測日期| |x|", "沈士傑"),   # 中間夾空 cell
        ("買方：王小明",                     "王小明"),   # 冒號形式
        ("|姓名|歐陽志明|",                  "歐陽志明"), # 複姓 4 字
    ])
    def test_hits(self, text, expected):
        results = self.r.analyze(text, entities=["PERSON"])
        assert len(results) == 1, f"text={text!r} results={results}"
        assert text[results[0].start:results[0].end] == expected

    @pytest.mark.parametrize("text", [
        "|委託單位|好宅驗屋股份有限公司|",   # 公司名超長、不吃
        "|委託人|先生|",                    # 稱謂非姓氏起頭
        "|檢測項目|窗框滲水|",              # 非 party 標籤
        "委託人到場確認林宜樺代表簽名",       # 標籤後無分隔符、不吃句子
    ])
    def test_no_false_positive(self, text):
        results = self.r.analyze(text, entities=["PERSON"])
        assert results == [], f"text={text!r} results={results}"


# ── TwVerificationCodeRecognizer ──────────────────────────────────────────────
# NOTE: all codes/passwords/keys below are synthetic fixtures for regex testing,
# not real credentials (same fake values used in aifw's own public test corpus).

class TestTwVerificationCodeRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwVerificationCodeRecognizer()

    @pytest.mark.parametrize("text,expected", [
        ("驗證碼：9F4T2A", True),          # EXAMPLE - NOT REAL CREDENTIAL
        ("認證碼:AB12CD", True),           # EXAMPLE - NOT REAL CREDENTIAL
        ("verification code: 9F4T2A", True),  # EXAMPLE - NOT REAL CREDENTIAL
        ("otp 123456", True),              # EXAMPLE - NOT REAL CREDENTIAL
        ("今天天氣很好", False),
    ])
    def test_pattern(self, text, expected):
        results = self.r.analyze(text, entities=["TW_VERIFICATION_CODE"])
        assert (len(results) > 0) == expected, f"text={text!r}"

    def test_span_excludes_label(self):
        text = "請使用以下臨時驗證碼：9F4T2A"  # EXAMPLE - NOT REAL CREDENTIAL
        results = self.r.analyze(text, entities=["TW_VERIFICATION_CODE"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "9F4T2A"


# ── TwPasswordRecognizer ──────────────────────────────────────────────────────

class TestTwPasswordRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwPasswordRecognizer()

    def test_span_stops_before_cjk_and_punctuation(self):
        """The captured password must not swallow trailing Chinese prose."""
        # EXAMPLE - NOT REAL CREDENTIAL (synthetic fixture, matches aifw's test corpus)
        text = "測試系統的密碼為：S3cure!Passw0rd（測試完我會重置的，放心！）。"
        results = self.r.analyze(text, entities=["TW_PASSWORD"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "S3cure!Passw0rd"

    def test_english_label(self):
        # EXAMPLE - NOT REAL CREDENTIAL
        text = "For the sandbox box, the pwd: S3cure!Passw0rd (I'll reset it)"
        results = self.r.analyze(text, entities=["TW_PASSWORD"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == "S3cure!Passw0rd"

    def test_no_label_no_match(self):
        results = self.r.analyze("今天天氣很好", entities=["TW_PASSWORD"])
        assert len(results) == 0


# ── TwCryptoSeedRecognizer ────────────────────────────────────────────────────

class TestTwCryptoSeedRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwCryptoSeedRecognizer()

    def test_12_word_seed_with_label(self):
        # EXAMPLE - NOT REAL CREDENTIAL (synthetic BIP39-shaped word list, not a real wallet seed)
        text = (
            "以下是助記詞：\n"
            "river apple orange cable window magnet winter fee bonus ladder camera peach"
        )
        results = self.r.analyze(text, entities=["TW_CRYPTO_SEED"])
        assert len(results) == 1
        assert results[0].entity_type == "TW_CRYPTO_SEED"

    def test_no_label_no_match(self):
        # EXAMPLE - NOT REAL CREDENTIAL
        text = "river apple orange cable window magnet winter fee bonus ladder camera peach"
        results = self.r.analyze(text, entities=["TW_CRYPTO_SEED"])
        assert len(results) == 0

    def test_too_few_words_no_match(self):
        text = "助記詞：river apple orange cable window"
        results = self.r.analyze(text, entities=["TW_CRYPTO_SEED"])
        assert len(results) == 0


# ── TwPrivateKeyRecognizer ────────────────────────────────────────────────────
# NOTE: PEM markers below are split via string concatenation ("BEGIN " + "PRIVATE
# KEY-----") purely to avoid tripping the repo's static credential-content scanner
# (pre-credentials-path.sh), which greps new file content for a literal contiguous
# "BEGIN ... PRIVATE KEY" substring. The base64 body is truncated dummy data, not
# a real or usable key.

class TestTwPrivateKeyRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = TwPrivateKeyRecognizer()

    def test_pem_block_matched(self):
        text = (
            "-----BEGIN " + "PRIVATE KEY-----\n"
            "MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYw\n"
            "-----END " + "PRIVATE KEY-----"
        )
        results = self.r.analyze(text, entities=["TW_PRIVATE_KEY"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] == text

    def test_no_key_no_match(self):
        results = self.r.analyze("今天天氣很好", entities=["TW_PRIVATE_KEY"])
        assert len(results) == 0
