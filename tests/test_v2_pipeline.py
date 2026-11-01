"""
Unit and Integration Tests for SMS Spam Classifier Version 2.
Tests deobfuscation, language detection guardrails, modern OTP handling,
and probability calibration.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data_preprocessing_v2 import deobfuscate_text, preprocess_text_v2, char_clean_v2
from src.language_detector import LanguageDetector
from src.predict_v2 import SpamPredictorV2
from src.predict import SpamPredictor as SpamPredictorV1


def test_deobfuscation():
    """Verify adversarial leetspeak and spaced-out letters are restored."""
    assert "claim" in deobfuscate_text("cl@im")
    assert "free" in deobfuscate_text("fr33")
    assert "cash" in deobfuscate_text("c4sh").lower()
    assert "FREE" in deobfuscate_text("F R E E")
    # Pure numbers must remain intact
    assert deobfuscate_text("12345") == "12345"
    assert deobfuscate_text("OTP code 482910") == "OTP code 482910"


def test_language_detector_non_latin():
    """Verify non-Latin scripts are flagged as UNSUPPORTED_LANGUAGE."""
    # Bengali
    res_bn = LanguageDetector.validate_and_detect("আজকে কি তুমি আসবে? দেখা হবে বিকেলে।")
    assert res_bn["status"] == "UNSUPPORTED_LANGUAGE"
    assert "Bengali" in res_bn["detected_language"]
    
    # Chinese
    res_zh = LanguageDetector.validate_and_detect("你好，我们今天晚上一起吃晚餐吗？")
    assert res_zh["status"] == "UNSUPPORTED_LANGUAGE"
    
    # Arabic
    res_ar = LanguageDetector.validate_and_detect("مرحبا كيف حالك اليوم؟")
    assert res_ar["status"] == "UNSUPPORTED_LANGUAGE"


def test_language_detector_latin_foreign():
    """Verify foreign European languages (Spanish/French) are flagged as UNSUPPORTED_LANGUAGE."""
    res_es = LanguageDetector.validate_and_detect("Hola amigo, ¿cómo estás? ¿Nos vemos mañana para almorzar?")
    assert res_es["status"] == "UNSUPPORTED_LANGUAGE"
    
    res_fr = LanguageDetector.validate_and_detect("Bonjour, avez-vous reçu le document pour la réunion?")
    assert res_fr["status"] == "UNSUPPORTED_LANGUAGE"


def test_language_detector_featureless():
    """Verify numbers-only, symbols-only, or empty inputs are flagged appropriately."""
    assert LanguageDetector.validate_and_detect("")["status"] == "EMPTY_INPUT"
    assert LanguageDetector.validate_and_detect("   ")["status"] == "EMPTY_INPUT"
    assert LanguageDetector.validate_and_detect("1234567890")["status"] == "INCONCLUSIVE"
    assert LanguageDetector.validate_and_detect("!@#$%^&*()")["status"] == "INCONCLUSIVE"
    assert LanguageDetector.validate_and_detect("😀🎉🎁💰")["status"] == "INCONCLUSIVE"


def test_modern_otp_message_handling():
    """
    CRITICAL V2 FIX:
    Verify that legitimate modern OTP messages are NOT falsely flagged as SPAM.
    """
    predictor_v2 = SpamPredictorV2()
    otp_sample = "Your Chase OTP verification code is 482910. Do not share this code with anyone. Expires in 10 minutes."
    res = predictor_v2.predict(otp_sample)
    
    assert res["status"] == "SUCCESS"
    assert res["verdict"] == "HAM", f"Modern OTP falsely classified as {res['verdict']}!"
    assert res["probabilities"]["ham"] > 0.60


def test_adversarial_spam_detection():
    """
    CRITICAL V2 FIX:
    Verify that obfuscated leetspeak spam is detected by Version 2.
    """
    predictor_v2 = SpamPredictorV2()
    spam_sample = "W!NNER!! Freeee pr!ze aw@rd wa!ting 4 u. Cl@im now cal 09061701461"
    res = predictor_v2.predict(spam_sample)
    
    assert res["status"] == "SUCCESS"
    assert res["verdict"] == "SPAM", f"Adversarial spam was missed: classified as {res['verdict']}!"


def test_unsupported_language_guardrail():
    """
    CRITICAL V2 FIX:
    Verify non-English text is NOT silently classified as HAM.
    """
    predictor_v2 = SpamPredictorV2()
    res = predictor_v2.predict("Hola amigo, ¿cómo estás? ¿Nos vemos mañana para almorzar?")
    
    assert res["verdict"] == "UNSUPPORTED_LANGUAGE"
    assert res["probabilities"] is None
    assert res["confidence"] is None


def test_inconclusive_guardrail_on_numbers():
    """
    CRITICAL V2 FIX:
    Verify numbers-only input does NOT receive 100% HAM confidence.
    """
    predictor_v2 = SpamPredictorV2()
    res = predictor_v2.predict("9847120491823091823")
    
    assert res["verdict"] == "INCONCLUSIVE"
    assert res["confidence"] is None


def test_v1_and_v2_coexistence():
    """Verify that Version 1 and Version 2 can be instantiated and executed side-by-side."""
    v1 = SpamPredictorV1("artifacts/final_model.joblib")
    v2 = SpamPredictorV2("artifacts/final_model_v2.joblib")
    
    test_text = "Free entry to win £1,000 cash prize! Text WIN to 80085."
    res_v1 = v1.predict(test_text)
    res_v2 = v2.predict(test_text)
    
    assert res_v1["label"] == "SPAM"
    assert res_v2["verdict"] == "SPAM"
    assert res_v1["confidence"] is not None
    assert res_v2["confidence"] is not None
