# SMS Spam Classification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Streamlit](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/library-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end, leak-free, production-ready **SMS Spam Classification System** built with Scikit-learn Pipelines, Natural Language Processing (NLP), and Streamlit.

---

## 📌 1. Project Overview & Problem Statement

Short Message Service (SMS) spam encompasses unsolicited, fraudulent, or malicious text communications, including phishing attacks, fake lottery notifications, and fraudulent financial services. 

Detecting SMS spam presents unique machine learning challenges:
1. **Severe Class Imbalance:** Legitimate messages (Ham) vastly outnumber spam messages (~87.4% vs ~12.6%).
2. **Asymmetric Error Costs:** A **False Positive** (misclassifying a critical legitimate SMS like a bank two-factor code or medical notification as spam) is catastrophic. A **False Negative** (a spam text slipping into an inbox) is merely annoying. Therefore, **Precision on Spam must be prioritized**.
3. **Data Leakage Risk:** Naive implementations often fit vectorizers (TF-IDF) across the whole dataset before train-test splitting, contaminating model evaluation.

This system addresses each challenge with strict ML engineering standards: zero-leakage Scikit-learn Pipelines, stratified cross-validation, hyperparameter tuning, automated tests, and a modern Streamlit user interface.

---

## 📊 2. Dataset Description

The system uses the SMS Spam Collection benchmark dataset located at `data/spam.csv`.

* **Original Format:** CSV format, Latin-1 encoded, 5,572 records.
* **Structural Anomalies Handled:** In raw exports, unescaped commas and quotation marks split text into trailing columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`). Our loader intelligently detects and concatenates these fragments back into the core message.
* **Duplicate Detection:** Identified and removed **403 duplicates**, yielding **5,169 unique messages**.
* **Cleaned Class Distribution:**
  * **HAM (Legitimate):** 4,516 messages (87.37%)
  * **SPAM (Unsolicited):** 653 messages (12.63%)
* **Length Statistics:**
  * **Ham messages:** Mean length 70.4 characters (~17.1 words).
  * **Spam messages:** Mean length 137.8 characters (~27.7 words), tightly clustered near the 160-character cellular SMS limit.

---

## 🏗️ 3. Project Architecture & Directory Structure

```text
SMS-Spam-Classifier/
│
├── data/
│   └── spam.csv                          # Original SMS Spam Collection dataset
│
├── notebooks/
│   └── 01_EDA_and_Experiments.ipynb      # Executed exploratory analysis & experiment log
│
├── src/
│   ├── __init__.py                       # Package exports
│   ├── data_preprocessing.py             # Normalization, tokenization, TextPreprocessor transformer
│   ├── model_training.py                 # Stratified CV, GridSearch, pipeline fitting, serialization
│   ├── evaluation.py                     # Metric calculations and visualization routines
│   └── predict.py                        # Standalone production inference engine (CLI & API)
│
├── artifacts/
│   ├── final_model.joblib                # Fully fitted Scikit-learn Pipeline (Preprocessor + TF-IDF + Classifier)
│   ├── model_metadata.json               # Recorded test metrics, pipeline parameters, and metadata
│   ├── model_comparison.csv             # Test set performance table across 6 models
│   └── cv_summary.csv                    # 5-fold cross-validation performance table
│
├── figures/
│   ├── class_distribution.png            # Class counts and proportions
│   ├── message_length_distributions.png  # Character, word, and sentence distributions
│   ├── top_words_comparison.png          # Top 20 stems in Ham vs Spam
│   ├── wordcloud_ham.png                 # Word cloud for legitimate messages
│   ├── wordcloud_spam.png                # Word cloud for spam messages
│   ├── champion_confusion_matrix.png     # Final model confusion matrix
│   ├── model_comparison.png              # Multi-metric comparative bar chart
│   └── roc_curves.png                    # Superimposed ROC curves
│
├── tests/
│   └── test_pipeline.py                  # Unit and integration test suite (8 tests)
│
├── app.py                                # Modern Streamlit web application
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Standard git exclusions
└── README.md                             # Documentation
```

---

## 🛡️ 4. Preventing Data Leakage

In typical tutorial implementations, `TfidfVectorizer.fit_transform(df['text'])` is executed before calling `train_test_split()`. This creates **severe data leakage**:
* The vectorizer vocabulary incorporates n-grams and vocabulary unique to the test set.
* Inverse Document Frequency (IDF) weights include test document frequencies.

### Our Solution
We construct an end-to-end `Pipeline` where text normalization, TF-IDF vectorization, and the classifier are combined:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from src.data_preprocessing import TextPreprocessor

pipeline = Pipeline([
    ('preprocessor', TextPreprocessor(stem=True, remove_stopwords=True)),
    ('vectorizer', TfidfVectorizer(max_features=3000)),
    ('classifier', BernoulliNB(alpha=0.2))
])

# Vocabulary and IDF weights are computed STRICTLY from training samples
pipeline.fit(X_train, y_train)

# Test samples are transformed using the already-fitted vocabulary
pipeline.predict(X_test)
```

---

## 🔬 5. Model Experimentation & Measured Results

All models were evaluated using **5-Fold Stratified Cross-Validation** on the training set (4,135 samples) and subsequently tested on the held-out test set (1,034 samples: 903 Ham, 131 Spam).

### 5.1 Held-Out Test Set Performance

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score | ROC-AUC | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BernoulliNB (Champion)** | **0.9855** | **1.0000** | **0.8855** | **0.9393** | **0.9971** | **0** | **15** |
| **SVC (Tuned Linear/Sigmoid)** | 0.9797 | 0.9583 | 0.8779 | 0.9163 | 0.9937 | 5 | 16 |
| **Extra Trees Classifier** | 0.9778 | 0.9655 | 0.8550 | 0.9069 | 0.9895 | 4 | 19 |
| **Random Forest** | 0.9749 | 0.9730 | 0.8244 | 0.8926 | 0.9900 | 3 | 23 |
| **MultinomialNB (Tuned)** | 0.9739 | 0.9643 | 0.8244 | 0.8889 | 0.9883 | 4 | 23 |
| **Logistic Regression** | 0.9623 | 1.0000 | 0.7023 | 0.8251 | 0.9912 | 0 | 39 |

### 5.2 Champion Selection Justification
* **Zero False Positives:** `BernoulliNB` achieved **0 False Positives** on the test set (Precision = 1.0000), meaning not a single legitimate SMS was incorrectly flagged as spam.
* **Superior F1-Score & ROC-AUC:** Outperformed Logistic Regression in Recall (88.55% vs 70.23%) and F1 (93.93% vs 82.51%) while maintaining an ROC-AUC of **0.9971**.
* **Domain Fit:** Bernoulli Naive Bayes models the binary presence/absence of tokens rather than frequency. In short SMS texts, whether indicative spam words appear (*'free', 'claim', 'winner', 'urgent'*) is more diagnostic than their repetition count.

---

## 🚀 6. Installation & Quick Start

### 6.1 Prerequisites
Ensure Python 3.10 or higher is installed.

```bash
# Clone or navigate to the repository
cd "D:\MLP\Spam Detection"

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Train and Evaluate the Models
To execute the complete preprocessing, cross-validation, hyperparameter tuning, and pipeline serialization workflow:

```bash
python -m src.model_training
```

### 6.3 Run the Automated Test Suite
To verify data integrity, stratification, leak prevention, and model inference:

```bash
pytest tests/ -v
```

### 6.4 Launch the Streamlit Web Application
To start the interactive web application:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 6.5 Command-Line Predictions
You can also run predictions directly from the command line:

```bash
# Single message evaluation
python -m src.predict --message "WINNER!! Call 09061701461 to claim your £1,000 cash prize!"

# Interactive REPL mode
python -m src.predict
```

---

## 🔮 7. Future Research Potential

The architecture is designed to support further extensions:
1. **Sub-word / Character n-grams:** Capturing adversarial obfuscation (e.g., `w!nner`, `c@sh`, `fr33`).
2. **Transformer Fine-Tuning:** Benchmarking lightweight transformer architectures (e.g., `DistilBERT`, `MobileBERT`) against our traditional ML baseline.
3. **Explainable AI (XAI):** Integrating SHAP (SHapley Additive exPlanations) or LIME to highlight exact word contributions in the Streamlit UI.
4. **Multilingual Adaptation:** Expanding the preprocessing pipeline to support multilingual SMS streams.

---

## 📄 License
This project is licensed under the MIT License.

