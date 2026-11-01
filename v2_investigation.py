"""
Phase 2 Investigation: Mathematical and Code-Level Diagnosis of Inconsistencies.
"""

import joblib
import numpy as np
import pandas as pd

model = joblib.load('artifacts/final_model.joblib')
vec = model.named_steps['vectorizer']
clf = model.named_steps['classifier']
prep = model.named_steps['preprocessor']

print("=" * 70)
print("PHASE 2: CORRECTNESS INVESTIGATION REPORT")
print("=" * 70)

print("\n1. WHY EMPTY MESSAGES RECEIVE 50% CONFIDENCE:")
print("Code-level evidence in src/predict.py lines 42-50:")
print("""
if text is None or not isinstance(text, str) or not text.strip():
    return {
        'label': 'HAM',
        'prediction': 0,
        'confidence': 0.50,
        'probabilities': {'ham': 0.50, 'spam': 0.50},
        'is_valid': False,
        'error': 'Input message is empty or invalid.'
    }
""")
print("Finding: The 50% confidence is NOT computed by the model. It is a hardcoded fallback heuristic in predict.py.")

print("\n2. WHY NUMBERS-ONLY OR UNRECOGNIZED MESSAGES RECEIVE 100% HAM CONFIDENCE:")
num_sample = "9847120491823091823"
cleaned_num = prep.transform([num_sample])[0]
print(f"Input: '{num_sample}' -> Cleaned: '{cleaned_num}'")
vec_num = vec.transform([cleaned_num])
print(f"Active vocabulary features (nnz): {vec_num.nnz}")
proba_num = clf.predict_proba(vec_num)[0]
print(f"Model output for 0-feature vector: Ham={proba_num[0]:.8f}, Spam={proba_num[1]:.8f}")

print("\n3. MATHEMATICAL DERIVATION: HOW BERNOULLINB CALCULATES ZERO-FEATURE LIKELIHOOD:")
print("In Bernoulli Naive Bayes, the decision rule incorporates BOTH the presence AND the absence of features:")
print("log P(c | x) = log P(c) + sum_{i=1}^D [ x_i * log P(w_i | c) + (1 - x_i) * log(1 - P(w_i | c)) ]")
print("When an input contains ZERO recognized features (x_i = 0 for all i=1..3000):")
print("log P(c | 0) = log P(c) + sum_{i=1}^D log(1 - P(w_i | c))")

log_prior = clf.class_log_prior_
feature_log_prob = clf.feature_log_prob_ # shape (2, 3000), log P(w_i | c)
log_neg_prob = np.log(1.0 - np.exp(feature_log_prob))
sum_neg_prob = np.sum(log_neg_prob, axis=1)
total_log_lik = log_prior + sum_neg_prob

print(f"Log Class Priors: Ham = {log_prior[0]:.4f}, Spam = {log_prior[1]:.4f} (Spam prior is much lower: ~12.6%)")
print(f"Sum of absence penalty sum_{{i=1}}^D log(1 - P(w_i | c)):")
print(f"  For Ham:  {sum_neg_prob[0]:.4f}")
print(f"  For Spam: {sum_neg_prob[1]:.4f}")
print(f"Total Log-Likelihood for empty vector: Ham = {total_log_lik[0]:.4f}, Spam = {total_log_lik[1]:.4f}")
diff = total_log_lik[0] - total_log_lik[1]
print(f"Log-odds difference: {diff:.4f}")
p_ham_derived = 1.0 / (1.0 + np.exp(-diff))
print(f"Derived P(Ham | x=0) = {p_ham_derived:.10f} (~100.0%)")
print("Diagnosis: When no words match the vocabulary, BernoulliNB penalizes Spam MORE than Ham because spam messages are expected to contain certain diagnostic words. The model is mathematically convinced that the complete absence of any spam words means the message is Ham!")

print("\n4. WHY A SINGLE WORD LIKE 'WINNER' RECEIVES A HAM PREDICTION:")
winner_clean = prep.transform(['WINNER'])[0]
print(f"Input: 'WINNER' -> Cleaned: '{winner_clean}'")
vec_winner = vec.transform([winner_clean])
active_idx = vec_winner.indices[0]
feature_name = vec.get_feature_names_out()[active_idx]
print(f"Active feature: '{feature_name}' at index {active_idx}")
# Calculate log-likelihood with 1 active feature and 2999 inactive features
log_lik_ham = log_prior[0] + feature_log_prob[0, active_idx] + (sum_neg_prob[0] - log_neg_prob[0, active_idx])
log_lik_spam = log_prior[1] + feature_log_prob[1, active_idx] + (sum_neg_prob[1] - log_neg_prob[1, active_idx])
print(f"Log-Likelihood with only 'winner' present:")
print(f"  Ham log-lik:  {log_lik_ham:.4f}")
print(f"  Spam log-lik: {log_lik_spam:.4f}")
diff_winner = log_lik_ham - log_lik_spam
p_spam_winner = 1.0 / (1.0 + np.exp(diff_winner))
print(f"Derived P(Spam | 'WINNER') = {p_spam_winner:.6f} (~3.9%), P(Ham | 'WINNER') = {1.0 - p_spam_winner:.6f} (~96.1%)")
print("Diagnosis: In BernoulliNB, although P('winner' | Spam) is high, the other 2,999 vocabulary words are ABSENT. The joint absence of all other 2,999 words pulls the posterior probability heavily toward HAM!")

print("\n5. ARE MODEL PROBABILITIES RELIABLE CONFIDENCE ESTIMATES?")
print("Naive Bayes is notoriously known for probability overconfidence due to the conditional independence assumption.")
print("Even for ambiguous or featureless inputs, Naive Bayes outputs extreme values (0.9999 or 0.0001) because multiplying 3,000 probability factors drives uncalibrated posteriors to 0 or 1.")
print("Diagnosis: Raw predict_proba from Naive Bayes is an uncalibrated ranking score, NOT a reliable frequentist confidence.")

print("\n6. ARE NON-ENGLISH MESSAGES SILENTLY CLASSIFIED AS HAM?")
non_eng = "Hola amigo como estas nos vemos manana"
vec_non_eng = vec.transform([prep.transform([non_eng])[0]])
print(f"Spanish sample active features: {vec_non_eng.nnz}")
proba_non_eng = clf.predict_proba(vec_non_eng)[0]
print(f"Prediction for Spanish: Ham={proba_non_eng[0]:.6f}, Spam={proba_non_eng[1]:.6f}")
print("Diagnosis: Yes! Because foreign words are out-of-vocabulary, the vector has 0 active features, triggering the exact zero-feature behavior that outputs 100% HAM.")
print("=" * 70)
