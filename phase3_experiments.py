"""
Phase 3 Experiments: Word-level vs Character-level vs Combined Feature Extraction.
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data_preprocessing import load_raw_dataset, preprocess_text, TextPreprocessor

df = load_raw_dataset("data/spam.csv")
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Test cases for adversarial robustness
adversarial_samples = [
    ("W!NNER!! Free prize cash reward claim now!", "SPAM"),
    ("fr33 entry into 1000 cash draw call 09061701461", "SPAM"),
    ("cl@im your £500 walmart giftcard now", "SPAM"),
    ("C4SH prize won! Call now to receive your payout", "SPAM"),
    ("F R E E ringtone text WIN to 80085", "SPAM"),
    ("Urgent! Your account won £1000. Cl@im at 08712460324", "SPAM"),
    ("Hey mom, are we having dinner at 7?", "HAM"),
    ("Please send the class slides for tomorrow", "HAM")
]

# 1. Experiment A: Word-level TF-IDF (Existing V1 baseline)
pipe_word = Pipeline([
    ('preprocessor', TextPreprocessor(stem=True, remove_stopwords=True)),
    ('vectorizer', TfidfVectorizer(max_features=3000)),
    ('classifier', BernoulliNB(alpha=0.2))
])
pipe_word.fit(X_train, y_train)

# 2. Experiment B: Character-level TF-IDF (char_wb, n-grams (3, 5))
# A light preprocessor that lowercases and preserves character sequence
def char_clean(text):
    if not isinstance(text, str):
        return ""
    # Lowercase and normalize whitespace, keep characters
    return re.sub(r'\s+', ' ', text.lower()).strip()

pipe_char = Pipeline([
    ('vectorizer', TfidfVectorizer(
        preprocessor=char_clean,
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=5000,
        sublinear_tf=True
    )),
    ('classifier', BernoulliNB(alpha=0.2))
])
pipe_char.fit(X_train, y_train)

# 3. Experiment C: Combined Word + Character TF-IDF via FeatureUnion
combined_features = FeatureUnion([
    ('word_tfidf', Pipeline([
        ('prep', TextPreprocessor(stem=True, remove_stopwords=True)),
        ('vec', TfidfVectorizer(max_features=3000, ngram_range=(1, 2)))
    ])),
    ('char_tfidf', Pipeline([
        ('vec', TfidfVectorizer(
            preprocessor=char_clean,
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=4000,
            sublinear_tf=True
        ))
    ]))
])

pipe_combined = Pipeline([
    ('features', combined_features),
    ('classifier', BernoulliNB(alpha=0.2))
])
pipe_combined.fit(X_train, y_train)

# Compare on Held-Out Test Set
experiments = {
    "Exp A: Word-level TF-IDF (V1)": pipe_word,
    "Exp B: Char-level TF-IDF (3-5 char_wb)": pipe_char,
    "Exp C: Combined Word+Char TF-IDF": pipe_combined
}

print("=" * 80)
print("PHASE 3: FEATURE EXTRACTION EXPERIMENTAL RESULTS (TEST SET: 1,034 SAMPLES)")
print("=" * 80)
print(f"{'Experiment':<35} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}")
print("-" * 80)

for name, model in experiments.items():
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    print(f"{name:<35} | {acc:.4f}   | {prec:.4f}    | {rec:.4f}   | {f1:.4f}   | {auc:.4f}")

print("\n" + "=" * 80)
print("ADVERSARIAL & OBFUSCATION STRESS TEST COMPARISON")
print("=" * 80)
for sample, expected in adversarial_samples:
    p_w = pipe_word.predict([sample])[0]
    p_c = pipe_char.predict([sample])[0]
    p_comb = pipe_combined.predict([sample])[0]
    
    lbl = lambda p: "SPAM" if p == 1 else "HAM"
    print(f"Sample: '{sample}' (Expected: {expected})")
    print(f"  -> Word-level (V1):    {lbl(p_w):<4} {'[PASS]' if lbl(p_w) == expected else '[FAIL]'}")
    print(f"  -> Char-level:         {lbl(p_c):<4} {'[PASS]' if lbl(p_c) == expected else '[FAIL]'}")
    print(f"  -> Combined Word+Char: {lbl(p_comb):<4} {'[PASS]' if lbl(p_comb) == expected else '[FAIL]'}")
