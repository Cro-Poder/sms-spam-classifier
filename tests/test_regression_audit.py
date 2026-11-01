"""
Regression Test Suite for Post-Deployment Audit (Investigation-Only).
Documents and reproduces the three reported failure modes:
1. Urgency-based electricity disconnection payment scam classified as HAM.
2. Single-word 'WINNER' classified as HAM instead of INCONCLUSIVE.
3. Short English delivery payment scam rejected as UNSUPPORTED_LANGUAGE (Danish 'da').
"""

import pytest
from src.predict_v2 import SpamPredictorV2
from src.language_detector import LanguageDetector


@pytest.fixture(scope="module")
def predictor_v2():
    return SpamPredictorV2()


class TestReportedAuditFailures:
    """
    Direct reproductions and regression benchmarks for the three reported audit cases.
    """

    def test_reproduce_failure_single_word_winner(self, predictor_v2):
        """
        Case 2 Reproduction:
        Single-word 'WINNER' lacks contextual features and should ideally be INCONCLUSIVE,
        but because of the negative SVM intercept (-1.2117) and absence of a token-count
        guardrail, the calibrated probability defaults to ~90.3% HAM.
        """
        raw_text = "WINNER"
        
        # 1. Trace through language detection
        val = LanguageDetector.validate_and_detect(raw_text)
        # Because len(tokens) == 1 (< 3), langdetect is skipped and it passes as VALID
        assert val["status"] == "VALID"
        assert val["detected_language"] == "English"
        
        # 2. Trace through predictor
        res = predictor_v2.predict(raw_text)
        
        # Verify the actual current behavior
        assert res["status"] == "SUCCESS"
        assert res["verdict"] == "HAM", (
            f"Expected current V2 model to exhibit the known defect of predicting HAM for 'WINNER', "
            f"got {res['verdict']}"
        )
        assert res["confidence"] > 0.85
        assert res["probabilities"]["spam"] < 0.15

    def test_reproduce_failure_electricity_disconnection_scam(self, predictor_v2):
        """
        Case 1 Reproduction:
        Electricity disconnection payment scams are misclassified as HAM because:
        1. Historical 2005 training corpus contains zero utility scams.
        2. Words like 'power', 'electricity', 'today', 'pm', 'month', 'thank' carry negative (HAM) weights.
        3. The negative intercept (-1.2117) overwhelms the positive spam weights from 'bill' and 'pay'.
        """
        scam_message = (
            "Dear Customer, your electricity power will be disconnected today at 9:30 PM "
            "from the electricity office. Because your previous month bill was not paid. "
            "Please immediately contact with our electricity officer 9876543210. Thank you."
        )
        
        val = LanguageDetector.validate_and_detect(scam_message)
        assert val["status"] == "VALID"
        
        res = predictor_v2.predict(scam_message)
        # Verify the actual current behavior: misclassified as HAM
        assert res["verdict"] == "HAM", (
            f"Expected current V2 model to exhibit the known defect of predicting HAM for electricity scam, "
            f"got {res['verdict']}"
        )
        assert res["probabilities"]["ham"] > 0.70

    def test_reproduce_failure_delivery_scam_language_rejection(self):
        """
        Case 3 Reproduction:
        Short English delivery payment scams with URLs are falsely rejected as UNSUPPORTED_LANGUAGE because:
        1. Tokens ('package', 'delivery', 'pending', 'post', 'service', 'fee') are missing from _COMMON_ENGLISH.
        2. With common English ratio == 0.0 (< 0.50), the text falls through to langdetect.detect().
        3. langdetect's Naive Bayes n-gram model misclassifies short strings with URLs as Danish ('da').
        """
        scam_text = "Package delivery pending: http://post-service-fee.net"
        
        val = LanguageDetector.validate_and_detect(scam_text)
        
        # Verify reproduction: falsely flagged as UNSUPPORTED_LANGUAGE (Danish 'da')
        assert val["status"] == "UNSUPPORTED_LANGUAGE"
        assert val["detected_language"] == "DA"

    @pytest.mark.xfail(
        reason="Target desired behavior: Single-word 'WINNER' should return INCONCLUSIVE once token-entropy guardrail is added"
    )
    def test_desired_target_winner_inconclusive(self, predictor_v2):
        res = predictor_v2.predict("WINNER")
        assert res["verdict"] == "INCONCLUSIVE"

    @pytest.mark.xfail(
        reason="Target desired behavior: Electricity disconnection scams should be flagged as SPAM once utility templates are added"
    )
    def test_desired_target_electricity_scam_spam(self, predictor_v2):
        scam_message = (
            "Dear Customer, your electricity power will be disconnected today at 9:30 PM "
            "from the electricity office. Because your previous month bill was not paid. "
            "Please immediately contact with our electricity officer 9876543210. Thank you."
        )
        res = predictor_v2.predict(scam_message)
        assert res["verdict"] == "SPAM"

    @pytest.mark.xfail(
        reason="Target desired behavior: English delivery scams must be accepted as VALID English once vocabulary is expanded"
    )
    def test_desired_target_delivery_scam_valid_english(self):
        scam_text = "Package delivery pending: http://post-service-fee.net"
        val = LanguageDetector.validate_and_detect(scam_text)
        assert val["status"] == "VALID"
        assert val["detected_language"] == "English"

