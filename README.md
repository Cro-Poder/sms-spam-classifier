# SMS Spam Classification System (Versions 1, 2, and 2.1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Streamlit](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/library-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end, leak-free, production-grade **SMS Spam Classification System** featuring a baseline (Version 1), a calibrated multi-feature architecture (Version 2), and an enhanced production release with contextual sufficiency guardrails, URL-sanitized language detection, and utility fraud augmentation (Version 2.1).

---

## 📌 1. Project Overview & Architecture Evolution

### Version 1 (Historical Baseline)
* **Architecture:** Word-level TF-IDF (3,000 features, 1-gram) $\rightarrow$ Bernoulli Naive Bayes ($\alpha=0.2$).
* **Historical Test Set Performance:** **98.55% Accuracy**, **100% Precision**, **88.55% Recall**, **93.93% F1-score**.
* **Identified Vulnerabilities:**
  1. *Adversarial Obfuscation Blindspot:* Punctuation stripping split leetspeak words (`W!NNER`, `cl@im`, `fr33`) into isolated letters, evading detection.
  2. *Modern Domain Drift:* False alarms on modern legitimate 2FA / OTP codes (`code`, `verify`, `expires`) due to historical 2005 dataset bias.
  3. *Uncalibrated Overconfidence:* Featureless or foreign-language inputs defaulted to 100% HAM confidence due to Naive Bayes zero-feature likelihood mechanics.

### Version 2 (Calibrated Multi-Feature & Rollback Baseline)
* **Architecture:** Multi-Feature Union (Word 1-2 n-grams + Character within-word `char_wb` 3-5 n-grams) $\rightarrow$ Platt-calibrated Linear SVM via `CalibratedClassifierCV(LinearSVC, method='sigmoid', cv=5)`.
* **Guardrails Added:**
  * **Adversarial Deobfuscation:** Automated space-repair (`F R E E` $\rightarrow$ `FREE`) and leetspeak translation (`w!nner` $\rightarrow$ `winner`, `cl@im` $\rightarrow$ `claim`).
  * **Language Detection Guardrail:** Detects non-Latin scripts (Bengali, Chinese, Arabic, Cyrillic) and non-English European languages (Spanish, French, German).
  * **Modern SMS Augmentation:** Training fold augmented with modern 2FA authentication, delivery updates, and contemporary phishing templates.
* **Remaining Vulnerabilities Identified in Post-Deployment Audit:**
  1. *Utility Disconnection Scams Missed:* Missed urgent electricity/power disconnection scams due to 2005 corpus bias and absence in V2 augmentation.
  2. *Single-Word Keyword Bias:* Isolated words (`WINNER`, `PRIZE`, `CASH`) predicted HAM with ~90% confidence due to negative SVM intercept ($\beta_0 = -1.2117$).
  3. *URL Language Detection False Positives:* Delivery scams with web links falsely triggered Danish (`da`) in `langdetect`.

### Version 2.1 (Current Production Champion)
* **Architecture:** Calibrated Linear SVM + Extended Multi-Feature Union (8,600 word/char n-grams) + Utility Fraud Augmentation.
* **Key Enhancements:**
  1. **URL-Sanitized Language Detection:** Web URLs and protocol strings are stripped prior to language evaluation, eliminating false foreign language triggers while preserving commercial vocabulary.
  2. **Contextual Sufficiency Guardrail:** Single isolated words without actionable contact details or links return `INCONCLUSIVE`, preventing the class intercept from forcing a false HAM verdict. Actionable short messages (e.g. `Call 09061701461`, `Text STOP to 84122`) are preserved for model classification.
  3. **Utility Fraud Augmentation:** Training fold augmented with 20 synthetic utility-fraud scams and 10 legitimate utility notices, successfully catching electricity, water, gas, and council tax scams with 100% recall.
* **Historical Test Set Performance:** **98.94% Accuracy**, **99.18% Precision**, **92.37% Recall**, **95.65% F1-score**, **99.14% PR-AUC** (False Negatives reduced to 10).
* **Qualitative Stress Test Pass Rate:** **24/24 (100.0%)** vs 17/24 (70.8%) in V2 and 13/24 (54.2%) in V1.

---

## 📊 2. Dataset Documentation, Provenance & Licensing

The project maintains strict data isolation and clearly distinguishes authentic historical data from curated synthetic evaluation templates:

### 2.1 Historical Dataset (`data/spam.csv` / root `spam.csv`)
* **Source & Citation:** SMS Spam Collection, curated by Tiago A. Almeida and José María Gómez Hidalgo (University of Campinas / University of São Paulo, 2011). Available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
* **Licensing & Redistribution:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Permitted for redistribution with attribution.
* **Processing:** 5,169 unique, verified messages (4,516 Ham, 653 Spam) split via stratified sampling into an 80% Training set (4,135 samples) and a 20% Held-Out Test set (1,034 samples: 903 Ham, 131 Spam). Split is identical across V1, V2, and V2.1.

### 2.2 Modern Augmented Training Datasets
* **V2 Training Set (`data/modern_sms_augmented_train.csv`):** 120 synthetically authored modern templates (60 Ham [OTP, banking alerts, delivery] + 60 Spam [delivery phishing, toll scams, account suspension, crypto scams]).
* **V2.1 Training Set (`data/modern_sms_v2_1_augmented_train.csv`):** 150 synthetically authored modern templates (70 Ham + 80 Spam), adding 20 utility disconnection scams (power, water, gas, council tax) and 10 legitimate utility receipts/maintenance notices.
* **Licensing:** Dedicated to the public domain under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

### 2.3 Modern Held-Out Evaluation Datasets
* **Base Modern Eval Set (`data/modern_sms_heldout_eval.csv`):** 60 independent synthetic test templates (30 Ham + 30 Spam).
* **Extended Modern Eval Set (`data/modern_sms_v2_1_heldout_eval.csv`):** 80 independent synthetic test templates (40 Ham + 40 Spam), adding 10 held-out utility scams and 10 held-out utility notices with **zero template overlap** with training data.
* **Strict Isolation Guarantee:** Evaluation data is never seen during training, feature selection, cross-validation, or threshold tuning.

---

## 🔬 3. Multi-Model Benchmark & Research Comparison

All models were evaluated on identical held-out test folds:

### 3.1 Historical Test Set (1,034 samples: 903 Ham, 131 Spam)

| Model Version | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Version 1 (BernoulliNB)** | 0.9855 | **1.0000** | 0.8855 | 0.9393 | 0.9971 | 0.9850 | **0** | 15 |
| **Version 2 (Calibrated SVM)** | 0.9874 | 0.9917 | 0.9084 | 0.9482 | 0.9972 | 0.9905 | 1 | 12 |
| **Version 2.1 (Champion)** | **0.9894** | 0.9918 | **0.9237** | **0.9565** | **0.9973** | **0.9914** | 1 | **10** |

### 3.2 Base Modern Held-Out Evaluation Set (60 samples: 30 Ham, 30 Spam)

| Model Version | Accuracy | Precision | Recall | F1-Score | PR-AUC | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Version 1 (BernoulliNB)** | 78.33% | 94.74% | 60.00% | 73.47% | 0.9497 | 1 | 12 |
| **Version 2 (Calibrated SVM)** | **98.33%** | **96.77%** | **100.00%** | **98.36%** | **1.0000** | **1** | **0** |
| **Version 2.1 (Champion)** | **98.33%** | **96.77%** | **100.00%** | **98.36%** | **1.0000** | **1** | **0** |

### 3.3 Extended Modern Evaluation Set with Utility Scams (80 samples: 40 Ham, 40 Spam)

| Model Version | Accuracy | Precision | Recall | F1-Score | PR-AUC | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Version 1 (BernoulliNB)** | 77.50% | 84.38% | 67.50% | 75.00% | 0.8970 | 5 | 13 |
| **Version 2 (Calibrated SVM)** | 93.75% | 90.70% | 97.50% | 93.98% | 0.9948 | 4 | 1 |
| **Version 2.1 (Champion)** | **96.25%** | **93.02%** | **100.00%** | **96.39%** | **0.9988** | **3** | **0** |

### 3.4 Qualitative 24-Scenario Stress Test Results

| Test Category | Target Expected | Version 1 Baseline | Version 2 Rollback | Version 2.1 Champion |
| :--- | :--- | :---: | :---: | :---: |
| **Single Word (`WINNER`, `PRIZE`, `CASH`)** | `INCONCLUSIVE` | ❌ FAIL (Ham) | ❌ FAIL (Ham) | ✅ **PASS (Inconclusive)** |
| **Utility Disconnection Scams** | `SPAM` | ❌ FAIL (Ham) | ❌ FAIL (Ham) | ✅ **PASS (Spam, 78.8%)** |
| **Delivery Scams with URLs** | `SPAM` | ❌ FAIL (Ham) | ❌ FAIL (Danish) | ✅ **PASS (Spam, 98.0%)** |
| **Adversarial Obfuscations (`W!NNER`, `fr33`)** | `SPAM` | ❌ FAIL (Ham) | ✅ PASS (Spam) | ✅ **PASS (Spam)** |
| **Modern Banking 2FA OTPs** | `HAM` | ❌ FAIL (Spam) | ✅ PASS (Ham) | ✅ **PASS (Ham, 92.0%)** |
| **Foreign Languages (Bengali, French)** | `UNSUPPORTED_LANGUAGE` | ❌ FAIL (Ham) | ✅ PASS (Unsupported)| ✅ **PASS (Unsupported)** |
| **Overall 24-Scenario Pass Rate** | — | **13/24 (54.2%)** | **17/24 (70.8%)** | **24/24 (100.0%)** |

---

## 🏗️ 4. Project Directory Structure

```text
SMS-Spam-Classifier/
│
├── data/
│   ├── spam.csv                              # Historical SMS benchmark (Almeida et al.)
│   ├── modern_sms_augmented_train.csv        # V2 modern training set (120 samples)
│   ├── modern_sms_heldout_eval.csv           # V2 modern evaluation set (60 samples)
│   ├── modern_sms_v2_1_augmented_train.csv   # V2.1 utility augmented train set (150 samples)
│   └── modern_sms_v2_1_heldout_eval.csv      # V2.1 extended evaluation set (80 samples)
│
├── notebooks/
│   └── 01_EDA_and_Experiments.ipynb          # Executed EDA and data exploration notebook
│
├── src/
│   ├── __init__.py                           # Package exports
│   ├── data_preprocessing.py                 # Version 1 preprocessor and data loader
│   ├── data_preprocessing_v2.py              # Version 2 deobfuscation & character cleaners
│   ├── language_detector.py                  # Version 2 language detection module
│   ├── language_detector_v2_1.py              # Version 2.1 contextual sufficiency & URL sanitizer
│   ├── model_training.py                     # Version 1 baseline training pipeline
│   ├── model_training_v2.py                  # Version 2 training pipeline
│   ├── model_training_v2_1.py                # Version 2.1 champion training pipeline
│   ├── evaluation.py                         # Metric evaluation & plotting utilities
│   ├── predict.py                            # Version 1 predictor
│   ├── predict_v2.py                         # Version 2 predictor
│   └── predict_v2_1.py                       # Version 2.1 production predictor
│
├── artifacts/
│   ├── final_model.joblib                    # Version 1 model artifact (198 KB)
│   ├── final_model_v2.joblib                 # Version 2 model artifact (591 KB)
│   ├── final_model_v2_1.joblib               # Version 2.1 champion model artifact (605 KB)
│   ├── model_metadata.json                   # Version 1 metrics
│   ├── model_metadata_v2.json                # Version 2 metrics
│   ├── model_metadata_v2_1.json              # Version 2.1 metrics
│   ├── v1_v2_v2_1_comparison.csv             # 3-way evaluation comparison benchmark
│   ├── v1_v2_v2_1_stress_test.csv            # 24-scenario qualitative comparison
│   └── AUDIT_REPORT_POST_DEPLOYMENT_FAILURE_ANALYSIS.md # Detailed post-deployment audit report
│
├── figures/
│   ├── champion_confusion_matrix.png         # Version 1 confusion matrix
│   ├── champion_v2_confusion_matrix.png      # Version 2 confusion matrix
│   ├── champion_v2_1_confusion_matrix.png    # Version 2.1 confusion matrix
│   ├── class_distribution.png                # Dataset balance
│   ├── message_length_distributions.png      # Payload length characteristics
│   └── top_words_comparison.png              # Top stems in Ham vs Spam
│
├── tests/
│   ├── test_pipeline.py                      # Version 1 unit test suite (8 tests)
│   ├── test_v2_pipeline.py                   # Version 2 unit test suite (9 tests)
│   ├── test_v2_1_pipeline.py                 # Version 2.1 unit test suite (9 tests)
│   └── test_regression_audit.py              # Reproduction & audit regression suite (6 tests)
│
├── v1_baseline/                              # Completely independent copy of Version 1
├── app.py                                    # Streamlit Web Application (Tri-Engine V1, V2, V2.1)
├── requirements.txt                          # Dependencies
├── LICENSE                                   # MIT License
├── .gitignore                                # Exclusions
└── README.md                                 # Documentation
```

---

## 🚀 5. Installation & Execution Guide

### 5.1 Environment Setup
```bash
# Clone and navigate to repository
cd "D:\MLP\Spam Detection"

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Training Any Model Version Independently
```bash
# Train Version 1 Baseline
python -m src.model_training

# Train Version 2 Rollback Baseline
python -m src.model_training_v2

# Train Version 2.1 Champion Release
python -m src.model_training_v2_1
```

### 5.3 Automated Testing
```bash
# Run full regression test suite (32 tests across V1, V2, and V2.1)
pytest tests/ -v
```

### 5.4 Command-Line Inference
```bash
# Evaluate with Version 2.1 Champion
python -m src.predict_v2_1 --message "Dear Customer, your electricity power will be disconnected today at 9:30 PM. Call 0800123456."

# Evaluate with Version 2 Engine
python -m src.predict_v2 --message "Your Chase OTP verification code is 482910."

# Evaluate with Version 1 Baseline
python -m src.predict --message "WINNER! You won 1000 prize."
```

### 5.5 Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501`. Use the sidebar switcher to toggle between **Version 2.1 (Champion)**, **Version 2 (Rollback)**, and **Version 1 (Baseline)**.

---

## ⚠️ 6. Known Limitations & Deployment Guidelines

1. **Multilingual Classification:** Non-English messages are safely intercepted and flagged as `UNSUPPORTED_LANGUAGE`. They are not classified as spam or ham to avoid deceptive predictions. Future iterations can integrate a multilingual transformer embedding layer (`XLM-RoBERTa`).
2. **Context-Free Numbers:** Inputs consisting exclusively of numbers or emojis return `INCONCLUSIVE`. Incorporating carrier metadata (e.g. shortcode registration lookup) would help resolve raw numbers.
3. **Phishing Domain Lifespans:** Brand new, zero-day phishing domains (`.top`, `.cc`) are caught via n-gram and keyword correlations. Integrating live domain reputation lookups (e.g. Google Safe Browsing API) would further boost detection of emerging domains.

---

## 📄 License & Attribution
* **Software Codebase:** Distributed under the [MIT License](LICENSE).
* **Historical SMS Dataset:** Sourced from the SMS Spam Collection by Tiago A. Almeida and José María Gómez Hidalgo, licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
* **Modern Synthetic Datasets:** Authored specifically for this research project and released into the public domain under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
