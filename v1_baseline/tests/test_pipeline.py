"""
Unit and Integration Tests for SMS Spam Classification System.
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    preprocess_text,
    TextPreprocessor,
    load_raw_dataset
)
from src.predict import SpamPredictor


def test_preprocess_text_basic():
    """Verify standard text cleaning steps: lowercase, stopword removal, stemming."""
    raw = "WINNER!! You have been selected to receive a FREE prize reward! Call now."
    cleaned = preprocess_text(raw)
    assert isinstance(cleaned, str)
    # Punctuation removed
    assert "!" not in cleaned
    assert "." not in cleaned
    # Lowercased and stemmed
    assert "winner" in cleaned
    assert "select" in cleaned
    assert "prize" in cleaned
    assert "reward" in cleaned
    # Stopwords removed ('have', 'been', 'a', 'to')
    tokens = cleaned.split()
    assert "have" not in tokens
    assert "been" not in tokens
    assert "a" not in tokens


def test_preprocess_text_edge_cases():
    """Ensure safe behavior with empty, whitespace, None, and numeric inputs."""
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""
    assert preprocess_text(None) == ""
    assert preprocess_text("12345 67890") == "12345 67890"
    assert preprocess_text("!!!???###$$$") == ""


def test_text_preprocessor_transformer():
    """Verify that TextPreprocessor works as a proper Scikit-learn transformer."""
    preprocessor = TextPreprocessor()
    corpus = ["Call FREE now!", "Are you home?"]
    transformed = preprocessor.transform(corpus)
    assert len(transformed) == 2
    assert "call" in transformed[0]
    assert "free" in transformed[0]
    assert "home" in transformed[1]


def test_data_loading_and_deduplication():
    """Verify that dataset loads, handles encodings, and strips duplicates."""
    df = load_raw_dataset("data/spam.csv")
    assert "text" in df.columns
    assert "label" in df.columns
    assert "char_count" in df.columns
    assert "word_count" in df.columns
    assert "sentence_count" in df.columns
    # Ensure no duplicate texts exist
    assert df.duplicated(subset=["text"]).sum() == 0
    # Labels must be binary (0 or 1)
    assert set(df["label"].unique()).issubset({0, 1})


def test_stratification():
    """Ensure train/test splitting maintains exact class proportion."""
    df = load_raw_dataset("data/spam.csv")
    X = df["text"]
    y = df["label"]
    
    total_spam_ratio = y.mean()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    train_spam_ratio = y_train.mean()
    test_spam_ratio = y_test.mean()
    
    # Difference should be negligible (under 0.005)
    assert abs(train_spam_ratio - total_spam_ratio) < 0.005
    assert abs(test_spam_ratio - total_spam_ratio) < 0.005


def test_no_data_leakage():
    """
    Verify that the vectorizer vocabulary and IDF weights are determined
    STRICTLY by training data, with zero knowledge of test data.
    """
    train_corpus = ["urgent claim your reward money", "hello friend see you soon"]
    y_train = [1, 0]
    test_corpus = ["unseen exotic alien vocabulary xyzpdq"]
    
    preprocessor = TextPreprocessor()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    pipeline.fit(train_corpus, y_train)
    vec = pipeline.named_steps['vectorizer']
    
    # Words only in test corpus must NOT be in fitted vocabulary
    assert "exotic" not in vec.vocabulary_
    assert "alien" not in vec.vocabulary_
    assert "xyzpdq" not in vec.vocabulary_
    # Words in train corpus must be in vocabulary
    assert "reward" in vec.vocabulary_
    assert "friend" in vec.vocabulary_


def test_model_serialization_and_prediction():
    """Verify saved model artifact can be reloaded and yields valid predictions."""
    model_path = "artifacts/final_model.joblib"
    assert os.path.exists(model_path), "Model artifact does not exist."
    
    predictor = SpamPredictor(model_path=model_path)
    
    # Obvious spam message
    spam_sample = "URGENT! You have won a 1 week FREE membership prize! Call 09061701461 to claim now!"
    spam_res = predictor.predict(spam_sample)
    assert spam_res["is_valid"] is True
    assert spam_res["label"] == "SPAM"
    assert spam_res["prediction"] == 1
    assert spam_res["confidence"] > 0.50
    
    # Obvious ham message
    ham_sample = "Hi John, could you please send me the notes from today's lecture when you get a chance?"
    ham_res = predictor.predict(ham_sample)
    assert ham_res["is_valid"] is True
    assert ham_res["label"] == "HAM"
    assert ham_res["prediction"] == 0
    assert ham_res["confidence"] > 0.50


def test_invalid_and_empty_inputs():
    """Verify that predictor handles empty, null, and whitespace inputs safely."""
    predictor = SpamPredictor(model_path="artifacts/final_model.joblib")
    
    res_empty = predictor.predict("")
    assert res_empty["is_valid"] is False
    assert res_empty["label"] == "HAM"
    
    res_none = predictor.predict(None)
    assert res_none["is_valid"] is False
    assert res_none["label"] == "HAM"
    
    res_spaces = predictor.predict("      ")
    assert res_spaces["is_valid"] is False
    assert res_spaces["label"] == "HAM"

