# Post-Deployment Failure Analysis & Investigation Audit Report

**Audit Type:** Investigation-Only Deep Dive (Non-Destructive)  
**Date:** November 2, 2026  
**Evaluator:** Senior Machine Learning Engineer & QA Auditor  
**Scope:** Production V2 SMS Spam Classification System  
**Test Suite:** `tests/test_regression_audit.py` (3 Reproduction Passed, 3 Regression XFailed)

---

## Executive Summary

An independent post-deployment investigation was conducted on three reported failure modes in the production SMS Spam Classifier (Version 2). All three failure modes were successfully reproduced in the current codebase without modifying existing production models or application logic:

1. **Failure 1 (Urgency-based electricity disconnection payment scam classified as HAM):**  
   *Observed Output:* `HAM` ($P(\text{Ham}) = 80.72\%$, $P(\text{Spam}) = 19.28\%$).  
   *Root Cause:* Severe historical training corpus bias. The words `power`, `electricity`, `disconnect` appear in the historical 2005 dataset exclusively in legitimate personal messages, assigning them negative (HAM) weights. Crucial conversational markers (`today`, `pm`, `month`, `thank`) contribute strong negative weights that overwhelm positive spam indicators (`bill`, `pay`), compounding with the base SVM intercept ($\beta_0 = -1.2117$).

2. **Failure 2 (Single-word "WINNER" classified as HAM instead of INCONCLUSIVE):**  
   *Observed Output:* `HAM` ($P(\text{Ham}) = 90.34\%$, $P(\text{Spam}) = 9.66\%$).  
   *Root Cause:* The absence of a minimum token-count or minimum Shannon entropy guardrail. An isolated single word yields only 1 word feature (`winner`, weight $+0.2095$) and a small cluster of character n-grams ($+0.3809$), which cannot overcome the $-1.2117$ class-imbalance intercept. The raw decision score is $-0.6213$, which Platt scaling calibrates to $90.34\%$ HAM. The uncertainty band ($0.35 \le P(\text{Spam}) \le 0.65$) is bypassed because the model is statistically "confident" in HAM.

3. **Failure 3 (Short English delivery payment scam rejected as UNSUPPORTED LANGUAGE):**  
   *Observed Output:* `UNSUPPORTED_LANGUAGE` (Detected language: `DA` / Danish).  
   *Root Cause:* A heuristic failure in `src/language_detector.py`. Common logistics keywords (`package`, `delivery`, `pending`, `fee`, `post`, `service`) are missing from `_COMMON_ENGLISH`. Short messages fall below the $50\%$ vocabulary threshold and are passed to `langdetect.detect()`. In the presence of short strings combined with raw URLs (`http://post-service-fee.net`), `langdetect`'s character n-gram model misclassifies English text as Danish (`da`) or Somali (`so`), falsely triggering the language guardrail.

---

## Trace Analysis: Step-by-Step Execution of Failure Cases

### Case 1: Electricity Disconnection Payment Scam
* **Input Text:** `"Dear Customer, your electricity power will be disconnected today at 9:30 PM from the electricity office. Because your previous month bill was not paid. Please immediately contact with our electricity officer 9876543210. Thank you."`

#### Execution Trace:
1. **Language Detection & Input Validation (`LanguageDetector.validate_and_detect`):**
   * Alphabetic characters detected: $> 0$.
   * Non-Latin script search: None.
   * `tokens`: `['dear', 'customer', 'your', 'electricity', 'power', 'will', 'be', 'disconnected', 'today', 'at', 'pm', 'from', 'the', 'electricity', 'office', 'because', 'your', 'previous', 'month', 'bill', 'was', 'not', 'paid', 'please', 'immediately', 'contact', 'with', 'our', 'electricity', 'officer', 'thank', 'you']`
   * `eng_match_count` / `len(tokens)`: $16 / 32 = 50.0\%$.
   * **Result:** `status: VALID`, `detected_language: English`.

2. **Deobfuscation & Preprocessing:**
   * Word pipeline (`V2WordPreprocessor`): lowercasing, punctuation stripping, stopword filtering, Porter stemming.
   * Stems: `['dear', 'custom', 'electr', 'power', 'disconnect', 'today', 'pm', 'electr', 'offic', 'previous', 'month', 'bill', 'paid', 'pleas', 'immedi', 'contact', 'electr', 'offic', 'thank']`.
   * Character pipeline (`char_clean_v2`): normalized spaces, `char_wb` (3 to 5 n-grams).

3. **Feature Extraction & Calibrated Linear SVM Scoring:**
   * Total non-zero features: 273 (Word + Char n-grams).
   * **Base SVM Intercept ($\beta_0$):** $-1.2117$ (heavily biased towards HAM due to 87:13 class prior).
   * **Features pulling towards HAM (Negative weights):**
     * `word: pm`: weight $=-0.3540$, contribution $=-0.0761$
     * `word: month`: weight $=-0.1861$, contribution $=-0.0344$
     * `word: today pm`: weight $=-0.1084$, contribution $=-0.0276$
     * `word: today`: weight $=-0.1394$, contribution $=-0.0204$
     * `word: thank`: weight $=-0.1268$, contribution $=-0.0202$
     * `word: electr`: weight $=-0.0309$, contribution $=-0.0171$
     * `word: power`: weight $=-0.0659$, contribution $=-0.0156$
     * `char:  at `: weight $=-0.3545$, contribution $=-0.0145$
   * **Features pulling towards SPAM (Positive weights):**
     * `word: bill`: weight $=+0.4567$, contribution $=+0.1030$
     * `word: immedi`: weight $=+0.3785$, contribution $=+0.0805$
     * `word: custom`: weight $=+0.2821$, contribution $=+0.0503$
     * `word: pleas`: weight $=+0.1392$, contribution $=+0.0212$
   * **Net Decision Function Score:** $-0.4116$ (Negative, HAM side of the hyperplane).
   * **Platt Sigmoid Calibration:**  
     $$P(\text{Spam}) = \frac{1}{1 + e^{A \cdot (-0.4116) + B}} = 0.1928 \quad (19.28\%)$$  
     $$P(\text{Ham}) = 0.8072 \quad (80.72\%)$$

4. **Decision Thresholds (`SpamPredictorV2`):**
   * Uncertainty bounds: $[0.35, 0.65]$.
   * Because $P(\text{Spam}) = 0.1928 < 0.35$, the prediction is not marked as inconclusive.
   * **Final Output:** `status: SUCCESS`, `verdict: HAM`, `confidence: 80.72%`.

---

### Case 2: Single-Word "WINNER"
* **Input Text:** `"WINNER"`

#### Execution Trace:
1. **Language Detection & Input Validation (`LanguageDetector.validate_and_detect`):**
   * Length $> 0$, non-Latin search is clean.
   * `alpha_chars`: 6 (`W`, `I`, `N`, `N`, `E`, `R`), so line 103 (`len(alpha_chars) == 0`) passes.
   * `tokens`: `['winner']` (length = 1).
   * In `_COMMON_ENGLISH`: `'win'` is present, but `'winner'` is absent. Ratio $= 0.00 < 0.50$.
   * Line 141 condition: `if _LANGDETECT_AVAILABLE and len(tokens) >= 3:`
   * Because `len(tokens) == 1 < 3`, `langdetect` is intentionally skipped.
   * Execution reaches line 161: returns `status: VALID`, `detected_language: English`.

2. **Feature Extraction & Decision Function:**
   * Extracted features: 1 word feature (`winner`) + 12 character n-grams (`win`, `inner`, `ner`, etc.).
   * Base SVM Intercept: $-1.2117$.
   * Total positive evidence from all features: $+0.5904$.
   * Net Decision Score:
     $$\text{score} = -1.2117 + 0.5904 = -0.6213$$

3. **Platt Calibration & Prediction:**
   * Calibrated probabilities: $P(\text{Spam}) = 9.66\%$, $P(\text{Ham}) = 90.34\%$.
   * Uncertainty margin check: $9.66\% < 35.0\%$, so it bypasses `INCONCLUSIVE`.
   * **Final Output:** `status: SUCCESS`, `verdict: HAM`, `confidence: 90.34%`.

---

### Case 3: English Delivery Payment Scam
* **Input Text:** `"Package delivery pending: http://post-service-fee.net"`

#### Execution Trace:
1. **Language Detection & Input Validation (`LanguageDetector.validate_and_detect`):**
   * Tokens extracted by regex `\b[a-zA-Z]+\b`:  
     `['package', 'delivery', 'pending', 'http', 'post', 'service', 'fee', 'net']` (8 tokens).
   * Inspection against `_COMMON_ENGLISH`:
     * None of `package`, `delivery`, `pending`, `http`, `post`, `service`, `fee`, or `net` are present in `_COMMON_ENGLISH` in `src/language_detector.py`.
     * `eng_match_count = 0`, `eng_ratio = 0.00 / 8 = 0.00 < 0.50`.
   * Line 141 condition: `len(tokens) == 8 >= 3` is True.
   * Line 143: `detect("Package delivery pending: http://post-service-fee.net")` is executed.
   * `langdetect` runs a 3-gram Naive Bayes language profile on the raw string:
     * The word `pending` alongside URL components (`post`, `service`, `fee`, `net`) produces highest likelihood in the Danish (`da`) language profile ($P(\text{da}) = 0.99999$).
   * Lines 144–157: `lang == 'da' != 'en'`.
   * **Final Output:** `status: UNSUPPORTED_LANGUAGE`, `verdict: UNSUPPORTED_LANGUAGE`, `detected_language: DA`.
   * The message never reaches the ML pipeline or prediction thresholds.

---

## Detailed Root Cause Analysis

### 1. Why Language Detection Incorrectly Rejects Valid English Messages
`LanguageDetector` implements a two-tier check:
1. Fast-path English dictionary check (`eng_ratio >= 0.50`).
2. Fallback to `langdetect.detect()` if `eng_ratio < 0.50` and `len(tokens) >= 3`.

The fundamental flaw is that the fast-path dictionary `_COMMON_ENGLISH` was seeded with basic stopwords and conversational terms (`ok`, `see`, `u`, `meet`, `home`), but **omits standard commercial, transaction, and delivery terminology** (`package`, `delivery`, `parcel`, `shipping`, `pending`, `fee`, `pay`, `service`, `order`, `post`, `usps`, `dhl`). 

Consequently, short, concise English logistics and banking alerts fail the fast path and get forwarded to `langdetect.detect()`. Furthermore, raw URLs (`http://...`) are not stripped prior to calling `detect()`. URL tokens (such as `.net`, `.cc`, domain sub-stems) distort character n-gram frequencies, causing `langdetect` to output high-confidence false language predictions (e.g., Danish `da`, Somali `so`, Afrikaans `af`).

### 2. Why Ambiguous, Low-Information Messages Return HAM Instead of INCONCLUSIVE
In text classification models trained on class-imbalanced datasets (where HAM represents 87.4% of samples), the model learns a large negative intercept ($\beta_0 = -1.2117$). 

For an isolated word like `"WINNER"`, the total positive weight of the word feature and its constituent character n-grams ($+0.5904$) is numerically insufficient to overcome $-1.2117$. The net score remains negative ($-0.6213$), which Platt scaling converts to a high HAM confidence ($90.34\%$).

The system currently relies exclusively on *probability-proximity uncertainty* ($0.35 \le P(\text{Spam}) \le 0.65$). However, Bayesian/Platt calibration on sparse vectors is structurally incapable of distinguishing between:
* High evidence of innocence (a long, fluent personal message with genuine ham tokens), and
* Lack of evidence (an ultra-short 1-word or 2-word fragment).

An input validation guardrail checking for **contextual sufficiency** (minimum token count, linguistic entropy, or dictionary span) is necessary to intercept isolated keywords before probability calculation.

### 3. Why the Model Misses Urgency-Based Payment Scams
The training corpus exhibits significant historical domain drift:
* In `data/spam.csv` (2005 corpus), utility disconnection scams did not exist. Words like `electricity` (2 samples), `power` (9 samples), and `disconnect` (3 samples) occurred **100% of the time in HAM messages**.
* In `data/modern_sms_augmented_train.csv` (120 samples), augmentation focused on 2FA OTP codes, package deliveries, bank suspensions, and crypto scams, but contained zero electricity or municipal utility payment fraud templates.
* Linear SVM learned strong negative weights for polite conversational tokens (`thank`, `pm`, `today`, `month`), which fraudsters deliberately include to mimic official utility notices.
* When evaluated, the positive weights of `bill` ($+0.4567$) and `pay` ($+0.2829$) are completely offset by the combined negative weights of the polite framing and the base intercept.

---

## Existing Test Suite Review & Added Regression Tests

### Review of Existing Tests:
* `tests/test_pipeline.py` (8 tests): Validates baseline V1 deduplication, lack of data leakage, stratified split, and basic edge cases.
* `tests/test_v2_pipeline.py` (9 tests): Validates deobfuscation, non-Latin script rejection, OTP handling, and dual-pipeline coexistence.
* **Gap Identified:** The previous test suite did not test short messages containing URLs under `LanguageDetector`, did not assert that single words return `INCONCLUSIVE`, and lacked utility/electricity scam test vectors.

### Newly Created Regression Test File:
A non-destructive test file, `tests/test_regression_audit.py`, was added to the repository. It includes:
1. `test_reproduce_failure_single_word_winner`: Asserts the reproduction of the single-word HAM misclassification.
2. `test_reproduce_failure_electricity_disconnection_scam`: Asserts the reproduction of the electricity scam HAM misclassification.
3. `test_reproduce_failure_delivery_scam_language_rejection`: Asserts the reproduction of the Danish `da` false rejection on delivery URLs.
4. Three `@pytest.mark.xfail` target test cases that assert the desired future behavior without failing the existing build pipeline.

Test suite run: **20 passed, 3 xfailed in 7.70s**.

---

## Architectural Recommendations (Without Changing Models or Overwriting V1)

To resolve these three vulnerabilities cleanly in a future release while preserving Version 1 baseline integrity:

### Recommendation 1: Domain-Enriched Pre-Filtering in `LanguageDetector`
1. **URL Sanitization Before Detection:** Strip URLs (`re.sub(r'https?://\S+', '', text)`) and email addresses before passing text to `langdetect.detect()`. URLs are protocol metadata, not natural language text.
2. **Expand Common English Vocabulary:** Expand `_COMMON_ENGLISH` in `src/language_detector.py` to include standard commercial and transactional nouns/verbs:
   ```python
   _COMMERCIAL_ENGLISH = {
       'package', 'delivery', 'parcel', 'shipping', 'shipment', 'pending',
       'fee', 'fees', 'pay', 'payment', 'paid', 'due', 'bill', 'bills',
       'overdue', 'notice', 'service', 'services', 'order', 'orders',
       'post', 'postal', 'usps', 'ups', 'fedex', 'dhl', 'royal', 'mail',
       'electricity', 'power', 'utility', 'officer', 'customer', 'account'
   }
   ```
   If words belong to this expanded dictionary, validate immediately as `English` without delegating to `langdetect`.

### Recommendation 2: Information Entropy / Token Count Guardrail
Introduce a pre-inference contextual guardrail in `LanguageDetector` or `SpamPredictorV2`:
* If an input contains fewer than 2 distinct meaningful tokens (e.g., `"WINNER"`, `"CASH"`), flag it as:
  ```python
  status: INCONCLUSIVE
  reason: "Single-word or low-information input provides insufficient context for reliable classification."
  ```
This prevents single keywords from being swallowed by the $-1.2117$ class intercept.

### Recommendation 3: Augment Modern Training Data with Municipal/Utility Fraud
Add 20 curated synthetic utility disconnection and municipal fee scam templates into `data/modern_sms_augmented_train.csv`:
* Examples: Electricity cut-off notices, water disconnection threats, council tax refund scams, DMV license suspension warnings.
* Retraining V2 on this enriched training fold will re-weight `disconnect`, `overdue`, `utility`, and `power + bill` combinations into strong positive spam signals.

