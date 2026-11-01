"""
Unit and Regression Test Suite for Version 2.1.
Verifies all V2.1 improvements, guardrails, and model predictions:
1. Electricity disconnection payment scam -> SPAM.
2. Single-word 'WINNER', 'PRIZE', 'CASH' -> INCONCLUSIVE.
3. English delivery payment scam with URL -> Recognized as English & classified as SPAM.
4. Actionable short text preservation -> Evaluated by model.
5. Non-Latin scripts (Bengali) -> UNSUPPORTED_LANGUAGE.
6. Foreign Latin languages -> UNSUPPORTED_LANGUAGE.
7. Legitimate OTPs and conversational messages -> HAM.
8. Independent coexistence of V1, V2, and V2.1 pipelines.
"""

import pytest
import os
from src.predict import SpamPredictor
from src.predict_v2 import SpamPredictorV2
from src.predict_v2_1 import SpamPredictorV21
from src.language_detector_v2_1 import LanguageDetectorV21


@pytest.fixture(scope="module")
def predictor_v2_1():
    return SpamPredictorV21()


class TestVersion21LanguageDetectionAndGuardrails:
    """
    Tests URL stripping, English vocabulary recognition, and contextual sufficiency.
    """

    def test_single_word_winner_inconclusive(self, predictor_v2_1):
        """Single-word 'WINNER' must return INCONCLUSIVE."""
        res = predictor_v2_1.predict("WINNER")
        assert res["status"] == "INCONCLUSIVE"
        assert res["verdict"] == "INCONCLUSIVE"
        assert res["confidence"] is None

    def test_single_word_prize_and_cash_inconclusive(self, predictor_v2_1):
        """Isolated single words without context must return INCONCLUSIVE."""
        for word in ["PRIZE", "CASH", "FREE", "hello"]:
            res = predictor_v2_1.predict(word)
            assert res["status"] == "INCONCLUSIVE"
            assert res["verdict"] == "INCONCLUSIVE"

    def test_delivery_scam_with_url_accepted_as_english(self, predictor_v2_1):
        """
        English delivery scams with URLs must NOT be rejected as foreign language (e.g. Danish 'da').
        Must be accepted as English and predicted as SPAM.
        """
        scam_text = "Package delivery pending: http://post-service-fee.net"
        val = LanguageDetectorV21.validate_and_detect(scam_text)
        assert val["status"] == "VALID"
        assert val["detected_language"] == "English"
        
        res = predictor_v2_1.predict(scam_text)
        assert res["status"] == "SUCCESS"
        assert res["verdict"] == "SPAM"
        assert res["confidence"] > 0.80

    def test_usps_delivery_scam_detected_as_spam(self, predictor_v2_1):
        """USPS delivery scam with URL must be predicted as SPAM."""
        usps_text = "USPS: Package delivery fee pending: http://usps-pay.net"
        res = predictor_v2_1.predict(usps_text)
        assert res["status"] == "SUCCESS"
        assert res["verdict"] == "SPAM"

    def test_actionable_short_messages_preserved(self, predictor_v2_1):
        """Short messages with actionable signals (phone or shortcode) must NOT be blocked as inconclusive."""
        short_actionable = "Text STOP to 84122"
        val = LanguageDetectorV21.validate_and_detect(short_actionable)
        assert val["status"] == "VALID"
        
        res = predictor_v2_1.predict(short_actionable)
        assert res["status"] == "SUCCESS"
        assert res["verdict"] == "SPAM"

    def test_electricity_disconnection_scam_detected_as_spam(self, predictor_v2_1):
        """
        Electricity disconnection payment scam must be predicted as SPAM in V2.1.
        """
        scam_text = (
            "Dear Customer, your electricity power will be disconnected today at 9:30 PM "
            "from the electricity office. Because your previous month bill was not paid. "
            "Please immediately contact with our electricity officer 9876543210. Thank you."
        )
        res = predictor_v2_1.predict(scam_text)
        assert res["status"] == "SUCCESS"
        assert res["verdict"] == "SPAM"
        assert res["probabilities"]["spam"] > 0.65

    def test_legitimate_otp_and_conversation_ham(self, predictor_v2_1):
        """Modern OTPs and conversational messages must predict HAM."""
        otp_msg = "Your Chase OTP verification code is 482910. Expires in 10 mins."
        res_otp = predictor_v2_1.predict(otp_msg)
        assert res_otp["verdict"] == "HAM"
        assert res_otp["confidence"] > 0.85
        
        chat_msg = "Hey are we still meeting for lunch today at 12:30?"
        res_chat = predictor_v2_1.predict(chat_msg)
        assert res_chat["verdict"] == "HAM"
        assert res_chat["confidence"] > 0.90

    def test_unsupported_language_rejections(self, predictor_v2_1):
        """Non-Latin and foreign Latin languages must be safely rejected."""
        bengali_text = "আপনার বিদ্যুৎ বিল বকেয়া রয়েছে।"
        res_bn = predictor_v2_1.predict(bengali_text)
        assert res_bn["verdict"] == "UNSUPPORTED_LANGUAGE"
        assert res_bn["detected_language"] == "Bengali"
        
        french_text = "Bonjour, votre colis est arrivé au point relais."
        res_fr = predictor_v2_1.predict(french_text)
        assert res_fr["verdict"] == "UNSUPPORTED_LANGUAGE"

    def test_three_way_coexistence(self):
        """Verifies V1, V2, and V2.1 models exist and infer independently."""
        assert os.path.exists("artifacts/final_model.joblib")
        assert os.path.exists("artifacts/final_model_v2.joblib")
        assert os.path.exists("artifacts/final_model_v2_1.joblib")
        
        p1 = SpamPredictor()
        p2 = SpamPredictorV2()
        p21 = SpamPredictorV21()
        
        test_msg = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121"
        res1 = p1.predict(test_msg)
        res2 = p2.predict(test_msg)
        res21 = p21.predict(test_msg)
        
        assert res1["label"] == "SPAM"
        assert res2["verdict"] == "SPAM"
        assert res21["verdict"] == "SPAM"
        assert res21["version"] == "2.1.0"
