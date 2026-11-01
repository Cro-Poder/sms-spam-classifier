# SMS Spam Classification System (Versions 1 & 2)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Streamlit](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/library-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end, leak-free, production-grade **SMS Spam Classification System** incorporating both a traditional baseline (Version 1) and an advanced multi-feature calibrated architecture with input validation guardrails (Version 2).

---

## 📌 1. Project Overview & Evolution

### Version 1 (Baseline)
* **Goal:** A mathematically sound, leak-free baseline for classic SMS spam detection.
* **Architecture:** Word-level TF-IDF (3,000 features, 1-gram) $\rightarrow$ Bernoulli Naive Bayes ($\alpha=0.2$).
* **Historical Test Set Performance:** **98.55% Accuracy**, **100% Precision**, **88.55% Recall**, **93.93% F1-score**.
* **Identified Vulnerabilities (Audit Findings):**
  1. *Adversarial Obfuscation Blindspot:* Punctuation stripping split leetspeak words (`W!NNER`, `cl@im`, `fr33`) into isolated letters, evading detection.
  2. *Modern Domain Drift:* False alarms on modern legitimate 2FA / OTP codes (`code`, `verify`, `expires`) due to historical 2005 dataset bias.
  3. *Uncalibrated Overconfidence:* Featureless or foreign-language inputs defaulted to 100% HAM confidence due to Naive Bayes zero-feature likelihood mechanics.

### Version 2 (Advanced Improvement)
* **Goal:** Address all three audit weaknesses while preserving Version 1 as a separate baseline.
* **Architecture:** Multi-Feature Union (Word 1-2 n-grams + Character within-word `char_wb` 3-5 n-grams) $\rightarrow$ Platt-calibrated Linear SVM via `CalibratedClassifierCV(LinearSVC, method='sigmoid', cv=5)`.
* **Guardrails Added:**
  * **Adversarial Deobfuscation:** Automated space-repair (`F R E E` $\rightarrow$ `FREE`) and leetspeak translation (`w!nner` $\rightarrow$ `winner`, `cl@im` $\rightarrow$ `claim`).
  * **Language Detection Guardrail:** Detects non-Latin scripts (Bengali, Chinese, Arabic, Cyrillic) and non-English European languages (Spanish, French, German), returning `UNSUPPORTED_LANGUAGE` rather than forcing a deceptive prediction.
  * **Uncertainty Guardrail:** Inputs without alphabetic features (numbers only, emojis only) or with ambiguous scores near the decision boundary return `INCONCLUSIVE`.
  * **Modern SMS Augmentation:** Training fold augmented with modern 2FA authentication, delivery updates, and contemporary phishing templates, verified against an independent external evaluation set.

---

## 📊 2. Dataset Documentation, Provenance & Licensing

The project strictly distinguishes between **authentic historical benchmark data** and **synthetic modern evaluation templates**:

### 2.1 Historical Dataset (`data/spam.csv` / root `spam.csv`)
* **Source & Citation:** SMS Spam Collection, curated by Tiago A. Almeida and José María Gómez Hidalgo (University of Campinas / University of São Paulo, 2011). Available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
* **Licensing & Redistribution:** Creative Commons Attribution 4.0 International (CC BY 4.0). Redistribution and modification are freely permitted with attribution.
* **Corpus Composition:** 5,572 historical SMS messages collected from the Grumbletext UK forum, NUS SMS Corpus, and Caroline Tag's PhD thesis corpus.
* **Deduplication & Preprocessing:** The dataset contains 403 duplicates and occasional trailing comma fragments (`v3`, `v4`, `v5`). The loader resolves these into **5,169 unique, verified messages** (4,516 Ham, 653 Spam).
* **Data Leakage Safeguard:** Split into an 80% Training set (4,135 samples) and a 20% Held-Out Test set (1,034 samples: 903 Ham, 131 Spam) using stratified sampling before any feature vectorization or vocabulary construction.

### 2.2 Modern Augmented Training Dataset (`data/modern_sms_augmented_train.csv`)
* **Origin & Nature:** **Synthetically authored templates** (120 samples: 60 Ham, 60 Spam).
* **Reasoning for Synthetic Generation:** Modern SMS messages frequently contain Personally Identifiable Information (PII) such as personal phone numbers, bank accounts, or private authentication codes. To eliminate PII privacy risks while adapting the model to modern 2026 SMS patterns (such as Chase/Google 2FA OTP codes, package delivery notifications, crypto phishing, and toll fee scams), realistic synthetic exemplars were authored.
* **Usage:** Merged exclusively into the training fold for Version 2 (Total V2 Train = 4,255 samples). Never included in the historical test fold.

### 2.3 Modern External Held-Out Evaluation Dataset (`data/modern_sms_heldout_eval.csv`)
* **Origin & Nature:** **Synthetically authored templates** (60 samples: 30 Ham, 30 Spam).
* **Isolation Guarantee:** Strictly isolated external benchmark dataset with **zero vocabulary/template overlap** with the training data.
* **Usage:** Used exclusively to objectively stress-test generalization against modern OTP false positives and contemporary phishing evasion. Never seen during training or cross-validation.

---

## 🔬 3. Multi-Model Benchmark & Research Comparison

All models were evaluated on the **Historical Test Set** (1,034 samples) and the **Modern External Evaluation Set** (60 samples):

### 3.1 Historical Test Set (1,034 samples: 903 Ham, 131 Spam)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 Baseline: BernoulliNB (Word)** | 0.9855 | **1.0000** | 0.8855 | 0.9393 | 0.9971 | **0** | 15 |
| **V2 Champion: Calibrated Linear SVM (Combined)** | **0.9874** | 0.9917 | **0.9084** | **0.9482** | **0.9972** | 1 | **12** |
| **BernoulliNB (Char 3-5 n-grams)** | **0.9884** | 0.9837 | **0.9237** | **0.9528** | 0.9940 | 2 | **10** |
| **BernoulliNB (Combined Word+Char)** | 0.9874 | 0.9836 | 0.9160 | 0.9486 | 0.9950 | 2 | 11 |
| **MultinomialNB (Word V2)** | 0.9758 | 0.9569 | 0.8473 | 0.8988 | 0.9883 | 5 | 20 |
| **Logistic Regression (Word V2)** | 0.9710 | 0.9633 | 0.8015 | 0.8750 | 0.9928 | 4 | 26 |

### 3.2 Modern External Held-Out Evaluation Set (60 samples: 30 Ham, 30 Spam)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 Baseline: BernoulliNB (Word)** | 78.33% | 94.74% | 60.00% | 73.47% | 1 | 12 |
| **V2 Champion: Calibrated Linear SVM (Combined)** | **98.33%** | **96.77%** | **100.00%** | **98.36%** | **1** | **0** |
| **BernoulliNB (Combined Word+Char)** | 95.00% | 93.55% | 96.67% | 95.08% | 2 | 1 |
| **MultinomialNB (Word V2)** | 93.33% | 93.33% | 93.33% | 93.33% | 2 | 2 |
| **Logistic Regression (Word V2)** | 91.67% | 100.00% | 83.33% | 90.91% | 0 | 5 |

### 3.3 Key Empirical Research Findings
1. **Modern Generalization Leap:** Version 1 missed **40% of modern phishing/scam texts** (Recall: 60.0%), whereas Version 2 achieved **100% Recall** on modern phishing while cutting False Negatives to **0**.
2. **Historical Recall Improvement:** Version 2 increased historical recall from 88.55% to **90.84%**, intercepting 3 additional historical spam messages while maintaining 99.17% precision (only 1 false positive out of 903 ham messages).
3. **Calibrated Confidence:** Version 2 uses Platt scaling fitted via 5-fold internal CV on the training fold, eliminating extreme 0.0000 / 1.0000 probability overconfidence.

---

## 🏗️ 4. Project Directory Structure

```text
SMS-Spam-Classifier/
│
├── data/
│   ├── spam.csv                          # Original SMS Spam Collection
│   ├── modern_sms_augmented_train.csv    # Dedicated modern training templates (120 samples)
│   └── modern_sms_heldout_eval.csv       # Dedicated modern evaluation templates (60 samples)
│
├── notebooks/
│   └── 01_EDA_and_Experiments.ipynb      # Executed exploratory data analysis notebook
│
├── src/
│   ├── __init__.py                       # Package exports
│   ├── data_preprocessing.py             # Version 1 data loader and preprocessor
│   ├── data_preprocessing_v2.py          # Version 2 deobfuscation and multi-feature cleaners
│   ├── language_detector.py              # Version 2 language detection & input guardrails
│   ├── model_training.py                 # Version 1 baseline training pipeline
│   ├── model_training_v2.py              # Version 2 multi-feature calibrated training pipeline
│   ├── evaluation.py                     # Metric calculation & visualization utilities
│   ├── predict.py                        # Version 1 production predictor
│   └── predict_v2.py                     # Version 2 production predictor with guardrails
│
├── artifacts/
│   ├── final_model.joblib                # Preserved Version 1 model artifact (BernoulliNB)
│   ├── final_model_v2.joblib             # Version 2 champion artifact (Calibrated Linear SVM)
│   ├── model_metadata.json               # Version 1 test metrics
│   ├── model_metadata_v2.json            # Version 2 test metrics (Historical + Modern)
│   ├── model_comparison.csv             # Version 1 benchmark table
│   ├── v2_model_experiments_comparison.csv # Version 2 multi-model benchmark table
│   └── v1_vs_v2_stress_test_comparison.csv # Qualitative 24-case stress test matrix
│
├── figures/
│   ├── champion_confusion_matrix.png     # Version 1 confusion matrix
│   ├── champion_v2_confusion_matrix.png  # Version 2 confusion matrix
│   ├── class_distribution.png            # Dataset class distribution
│   ├── message_length_distributions.png  # Length characteristics
│   ├── model_comparison.png              # Version 1 multi-metric chart
│   ├── roc_curves.png                    # Superimposed ROC curves
│   └── top_words_comparison.png          # Top 20 stems in Ham vs Spam
│
├── tests/
│   ├── test_pipeline.py                  # Version 1 test suite (8 tests)
│   └── test_v2_pipeline.py               # Version 2 test suite (9 tests)
│
├── v1_baseline/                          # Complete, independent copy of Version 1
├── app.py                                # Streamlit web application (Dual-Engine V1 & V2)
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Standard git exclusions
└── README.md                             # Project documentation
```

---

## 🚀 5. Installation & Execution Guide

### 5.1 Environment Setup
```bash
# Clone or navigate to repository
cd "D:\MLP\Spam Detection"

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Training Version 1 and Version 2
```bash
# Train Version 1 baseline (saves to artifacts/final_model.joblib)
python -m src.model_training

# Train Version 2 champion (saves to artifacts/final_model_v2.joblib)
python -m src.model_training_v2
```

### 5.3 Automated Testing
```bash
# Run complete test suite (17 unit and integration tests across V1 and V2)
pytest tests/ -v
```

### 5.4 Command-Line Inference
```bash
# Evaluate with Version 2 Engine (includes language detection & deobfuscation)
python -m src.predict_v2 --message "Your Chase OTP verification code is 482910. Expires in 10 mins."

# Evaluate with Version 1 Baseline
python -m src.predict --message "WINNER! You won 1000 prize. Call 09061701461 to claim now!"
```

### 5.5 Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Use the sidebar switcher to toggle between **Version 1** and **Version 2**.

---

## ⚠️ 6. Remaining Limitations & Future Work

1. **Multilingual Classification:** While Version 2 detects and safely rejects non-English SMS messages as `UNSUPPORTED_LANGUAGE`, it does not classify foreign spam. Future iterations can incorporate multilingual transformer embeddings (e.g., `XLM-RoBERTa`).
2. **Context-Free Numbers:** Numbers without surrounding context (`"9847120491823"`) return `INCONCLUSIVE`. Integrating carrier metadata (e.g. sender short-code reputation lookup) would resolve numeric payloads.
3. **Evolving Phishing URLs:** Integrating a live domain reputation lookup (e.g., Google Safe Browsing API) would enhance detection of freshly registered phishing domains.

---

## 📄 License & Attribution
* **Software Codebase:** Distributed under the [MIT License](LICENSE).
* **Historical SMS Dataset:** Sourced from the SMS Spam Collection by Tiago A. Almeida and José María Gómez Hidalgo, licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
* **Modern Synthetic Datasets:** Authored specifically for this research project and released into the public domain under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
