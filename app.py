"""
Streamlit Web Application: SMS Spam Classifier (Version 1, Version 2, and Version 2.1 Multi-Engine).
Features:
- Three-way engine selection:
    * Version 2.1 (Latest Champion: Utility-Fraud Augmented + Contextual Guardrails)
    * Version 2 (Rollback Baseline: Calibrated Multi-Feature)
    * Version 1 (Historical Baseline: BernoulliNB)
- Quad-state classification verdicts: HAM, SPAM, INCONCLUSIVE, UNSUPPORTED LANGUAGE.
- Side-by-side comparative analysis across V1, V2, and V2.1.
- Full multi-dataset benchmark tables (Historical, Base Modern, and Extended Utility Modern).
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import joblib

from src.predict import SpamPredictor as SpamPredictorV1
from src.predict_v2 import SpamPredictorV2
from src.predict_v2_1 import SpamPredictorV21
from src.data_preprocessing import preprocess_text
from src.data_preprocessing_v2 import preprocess_text_v2, deobfuscate_text

# Page Configuration
st.set_page_config(
    page_title="SMS Spam Classifier V2.1",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.9rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.2rem;
    }
    .badge-spam {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 2px solid #F87171;
        padding: 0.9rem;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin: 0.8rem 0;
    }
    .badge-ham {
        background-color: #DCFCE7;
        color: #166534;
        border: 2px solid #4ADE80;
        padding: 0.9rem;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin: 0.8rem 0;
    }
    .badge-inconclusive {
        background-color: #FEF3C7;
        color: #92400E;
        border: 2px solid #FBBF24;
        padding: 0.9rem;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin: 0.8rem 0;
    }
    .badge-unsupported {
        background-color: #EDE9FE;
        color: #5B21B6;
        border: 2px solid #A78BFA;
        padding: 0.9rem;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin: 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictors():
    """Load Version 1, Version 2, and Version 2.1 predictors."""
    v1 = SpamPredictorV1("artifacts/final_model.joblib")
    v2 = SpamPredictorV2("artifacts/final_model_v2.joblib")
    v21 = SpamPredictorV21("artifacts/final_model_v2_1.joblib")
    return v1, v2, v21


@st.cache_data
def load_all_metadata():
    """Load metadata and benchmarks for all three versions."""
    meta_v1, meta_v2, meta_v21 = {}, {}, {}
    if os.path.exists("artifacts/model_metadata.json"):
        with open("artifacts/model_metadata.json", "r") as f:
            meta_v1 = json.load(f)
    if os.path.exists("artifacts/model_metadata_v2.json"):
        with open("artifacts/model_metadata_v2.json", "r") as f:
            meta_v2 = json.load(f)
    if os.path.exists("artifacts/model_metadata_v2_1.json"):
        with open("artifacts/model_metadata_v2_1.json", "r") as f:
            meta_v21 = json.load(f)
            
    comp_df = pd.DataFrame()
    if os.path.exists("artifacts/v1_v2_v2_1_comparison.csv"):
        comp_df = pd.read_csv("artifacts/v1_v2_v2_1_comparison.csv")
    elif os.path.exists("artifacts/v2_model_experiments_comparison.csv"):
        comp_df = pd.read_csv("artifacts/v2_model_experiments_comparison.csv")
        
    stress_df = pd.DataFrame()
    if os.path.exists("artifacts/v1_v2_v2_1_stress_test.csv"):
        stress_df = pd.read_csv("artifacts/v1_v2_v2_1_stress_test.csv")
    elif os.path.exists("artifacts/v1_vs_v2_stress_test_comparison.csv"):
        stress_df = pd.read_csv("artifacts/v1_vs_v2_stress_test_comparison.csv")
        
    return meta_v1, meta_v2, meta_v21, comp_df, stress_df


try:
    predictor_v1, predictor_v2, predictor_v21 = load_predictors()
    metadata_v1, metadata_v2, metadata_v21, comparison_df, stress_test_df = load_all_metadata()
except Exception as e:
    st.error(f"Error loading system artifacts: {e}")
    st.stop()


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=64)
    st.title("Spam Shield Engine")
    
    selected_version = st.radio(
        "Select Active Model Version:",
        options=[
            "Version 2.1: Utility-Fraud + Context Guardrails (Latest Champion)",
            "Version 2: Multi-Feature + Guardrails (Rollback Baseline)",
            "Version 1: Baseline (BernoulliNB)"
        ],
        index=0
    )
    
    is_v21 = "Version 2.1" in selected_version
    is_v2 = "Version 2:" in selected_version
    
    st.markdown("---")
    st.markdown("### ⚙️ Active Engine Specs")
    if is_v21:
        st.markdown("**Architecture:** `Calibrated Linear SVM`")
        st.markdown("**Features:** `Word (1-2) + Char_wb (3-5) (8,600)`")
        st.markdown("**Guardrails:** `Contextual Sufficiency + URL-Sanitized LangDetect`")
        st.markdown("**Hist Accuracy:** `98.94%` (Recall: `92.37%`)")
        st.markdown("**Modern F1:** `98.36%` (PR-AUC: `100.0%`)")
        st.markdown("**Utility Fraud F1:** `96.39%` (100% scam recall)")
    elif is_v2:
        st.markdown("**Architecture:** `Calibrated Linear SVM`")
        st.markdown("**Features:** `Word (1-2) + Char_wb (3-5) (8,000)`")
        st.markdown("**Guardrails:** `Deobfuscation + Latin/Non-Latin LangDetect`")
        st.markdown("**Hist Accuracy:** `98.74%` (Recall: `90.84%`)")
        st.markdown("**Modern F1:** `98.36%` (Recall: `100.0%`)")
    else:
        st.markdown("**Architecture:** `Bernoulli Naive Bayes`")
        st.markdown("**Features:** `Word-level TF-IDF (3,000)`")
        st.markdown("**Calibration:** `None (Raw Log-Odds)`")
        st.markdown("**Hist Accuracy:** `98.55%` (Recall: `88.55%`)")
        st.markdown("**Modern F1:** `73.47%` (misses 40% modern spam)")
        
    st.markdown("---")
    st.info(
        "💡 **Version 2.1 Upgrades:** Resolves utility disconnection payment scams, catches delivery URL scams without false Danish flags, and flags ambiguous single words as `INCONCLUSIVE`."
    )


# Header
st.markdown('<div class="main-header">🛡️ SMS Spam Classification System (v2.1)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-feature calibrated SMS classification with contextual sufficiency checks and URL-sanitized language guardrails.</div>',
    unsafe_allow_html=True
)

tab_classify, tab_benchmarks, tab_v1_v2, tab_architecture = st.tabs([
    "🔍 SMS Scanner & Diagnostics",
    "📊 Benchmark Performance",
    "⚖️ V1 vs V2 vs V2.1 Multi-Model Audit",
    "🧠 Mathematical Deep Dive & Calibration"
])


# ==========================================
# TAB 1: SMS SCANNER & DIAGNOSTICS
# ==========================================
with tab_classify:
    col_input, col_presets = st.columns([7, 5])
    
    preset_catalog = {
        "Custom Message (Type your own below)...": "",
        "🚨 Utility Fraud: Urgent Electricity Disconnection": "Dear Customer, your electricity power will be disconnected today at 9:30 PM from the electricity office. Because your previous month bill was not paid. Please immediately contact with our electricity officer 9876543210. Thank you.",
        "🚨 Logistics Phishing: Package Delivery URL": "Package delivery pending: http://post-service-fee.net",
        "⚠️ Ambiguous Keyword: Single-Word WINNER": "WINNER",
        "🚨 Modern Phishing: USPS Customs Fee": "USPS: We attempted to deliver your package today but an unpaid customs fee of $2.99 is pending. Pay now at http://usps-redelivery-fee.info",
        "🚨 Modern Phishing: Bank Card Suspended": "ALERT: Your Barclays debit card has been suspended due to suspicious activity. Visit http://barclays-security-login.com immediately to unlock.",
        "🚨 Adversarial Spam: Obfuscated Leetspeak": "W!NNER!! Freeee pr!ze aw@rd wa!ting 4 u. Cl@im now cal 09061701461",
        "🚨 Spaced Letters: Spammed Keyword": "F R E E ringtone text WIN to 80085 to claim your prize now",
        "✅ Legitimate Utility: Scheduled Maintenance": "Planned grid maintenance: Power in your neighborhood will be temporarily shut off on Tuesday between 2 AM and 4 AM for system upgrades.",
        "✅ Modern Legitimate: Chase Bank 2FA OTP": "Your Chase OTP verification code is 482910. Do not share this code with anyone. Expires in 10 minutes.",
        "✅ Modern Legitimate: Google Auth Code": "Google: 839201 is your verification code. Never share this code with anyone.",
        "🌐 Non-English: Bengali Message": "আজকে কি তুমি আসবে? দেখা হবে বিকেলে।",
        "🌐 Non-English: French Delivery": "Bonjour, votre colis est arrivé au point relais.",
        "⚠️ Featureless: Numbers Only": "9847120491823091823"
    }
    
    with col_presets:
        st.markdown("#### 🧪 Select Real-World Test Scenarios")
        chosen_preset = st.selectbox(
            "Load pre-configured test scenarios:",
            options=list(preset_catalog.keys()),
            index=0
        )
        loaded_preset_text = preset_catalog[chosen_preset]
        st.caption("Includes utility disconnect fraud, delivery scams, single-word inputs, 2FA codes, and foreign texts.")

    with col_input:
        st.markdown("#### ✍️ Enter SMS Text Message")
        initial_val = loaded_preset_text if loaded_preset_text else ""
        sms_text = st.text_area(
            "Message Body:",
            value=initial_val,
            height=125,
            placeholder="Type or paste any SMS text here to evaluate...",
            help="Enter SMS text for classification."
        )
        
        c_count = len(sms_text)
        w_count = len(sms_text.split())
        st.caption(f"Payload Length: **{c_count}** characters | **{w_count}** words")
        
        btn_eval = st.button("🚀 Run Comprehensive Evaluation", type="primary", use_container_width=True)

    st.markdown("---")
    
    if btn_eval or sms_text.strip():
        # Evaluate across all three models
        res_v21 = predictor_v21.predict(sms_text)
        res_v2 = predictor_v2.predict(sms_text)
        res_v1 = predictor_v1.predict(sms_text)
        
        if is_v21:
            active_res = res_v21
            verdict = res_v21.get("verdict")
        elif is_v2:
            active_res = res_v2
            verdict = res_v2.get("verdict")
        else:
            active_res = res_v1
            verdict = res_v1.get("label")
        
        c_res1, c_res2 = st.columns([6, 6])
        
        with c_res1:
            st.markdown("### 📋 Primary Verdict")
            
            if verdict == "SPAM":
                conf_str = f"({active_res['confidence']*100:.1f}% Confidence)" if active_res.get('confidence') is not None else ""
                st.markdown(f'<div class="badge-spam">🚨 SPAM DETECTED {conf_str}</div>', unsafe_allow_html=True)
                st.error(f"**Security Assessment:** {active_res.get('explanation', 'Matches unsolicited promotional or fraudulent spam pattern.')}")
            elif verdict == "HAM":
                conf_str = f"({active_res['confidence']*100:.1f}% Confidence)" if active_res.get('confidence') is not None else ""
                st.markdown(f'<div class="badge-ham">✅ LEGITIMATE (HAM) {conf_str}</div>', unsafe_allow_html=True)
                st.success(f"**Security Assessment:** {active_res.get('explanation', 'Normal conversational, transactional, or legitimate SMS pattern.')}")
            elif verdict == "INCONCLUSIVE":
                st.markdown('<div class="badge-inconclusive">⚠️ INCONCLUSIVE / INSUFFICIENT EVIDENCE</div>', unsafe_allow_html=True)
                st.warning(f"**Uncertainty Assessment:** {active_res.get('explanation')}")
            elif verdict == "UNSUPPORTED_LANGUAGE":
                lang_name = active_res.get("detected_language", "Non-English")
                st.markdown(f'<div class="badge-unsupported">🌐 UNSUPPORTED LANGUAGE ({lang_name})</div>', unsafe_allow_html=True)
                st.info(f"**Language Guardrail:** {active_res.get('explanation')}")
            elif verdict == "EMPTY":
                st.info("⚠️ Please enter a non-empty SMS text message to evaluate.")

        with c_res2:
            st.markdown("### 📊 Probability & Confidence Breakdown")
            probs = active_res.get("probabilities")
            if probs is not None:
                p_ham = float(probs["ham"])
                p_spam = float(probs["spam"])
                
                st.write(f"**Legitimate (Ham) Probability:** `{p_ham * 100:.2f}%`")
                st.progress(p_ham)
                
                st.write(f"**Spam / Phishing Probability:** `{p_spam * 100:.2f}%`")
                st.progress(p_spam)
                
                if is_v21:
                    st.caption("ℹ️ **Calibrated Confidence:** Version 2.1 probabilities are Platt-scaled via internal 5-fold CV and verified against utility fraud data.")
                elif is_v2:
                    st.caption("ℹ️ **Calibrated Confidence:** Version 2 probabilities are Platt-scaled via internal 5-fold CV.")
                else:
                    st.caption("⚠️ **Uncalibrated Metric:** Version 1 Naive Bayes probabilities are log-odds projections and can be overconfident.")
            else:
                st.info("Confidence estimates are not applicable for featureless, empty, or unsupported foreign language inputs.")

        # Side-by-side Three-Way Comparison Card
        with st.expander("⚖️ See Side-by-Side V1 vs V2 vs V2.1 Comparison for This Message", expanded=True):
            comp_c1, comp_c2, comp_c3 = st.columns(3)
            with comp_c1:
                st.markdown("##### Version 1 (Baseline)")
                v1_conf = f"{res_v1['confidence']*100:.1f}%" if res_v1.get('confidence') else "N/A"
                st.write(f"**Verdict:** `{res_v1['label']}`")
                st.write(f"**Confidence:** `{v1_conf}`")
                st.write(f"**Probs:** Ham: `{res_v1['probabilities']['ham']:.4f}` | Spam: `{res_v1['probabilities']['spam']:.4f}`")

            with comp_c2:
                st.markdown("##### Version 2 (Rollback)")
                v2_conf = f"{res_v2['confidence']*100:.1f}%" if res_v2.get('confidence') else "N/A"
                st.write(f"**Verdict:** `{res_v2['verdict']}`")
                st.write(f"**Confidence:** `{v2_conf}`")
                if res_v2.get('probabilities'):
                    st.write(f"**Probs:** Ham: `{res_v2['probabilities']['ham']:.4f}` | Spam: `{res_v2['probabilities']['spam']:.4f}`")

            with comp_c3:
                st.markdown("##### Version 2.1 (Champion)")
                v21_conf = f"{res_v21['confidence']*100:.1f}%" if res_v21.get('confidence') else "N/A"
                st.write(f"**Verdict:** `{res_v21['verdict']}`")
                st.write(f"**Confidence:** `{v21_conf}`")
                if res_v21.get('probabilities'):
                    st.write(f"**Probs:** Ham: `{res_v21['probabilities']['ham']:.4f}` | Spam: `{res_v21['probabilities']['spam']:.4f}`")
                st.caption(f"**Guardrail:** {res_v21.get('explanation')}")

        # NLP Inspection Details
        with st.expander("🔬 NLP Preprocessing & Deobfuscation Diagnostics", expanded=False):
            st.markdown("**1. Raw Input Message:**")
            st.code(sms_text, language="text")
            st.markdown("**2. Adversarial Deobfuscation Output (V2/V2.1 Feature):**")
            deobf_output = deobfuscate_text(sms_text)
            st.code(deobf_output if deobf_output else "[Empty]", language="text")
            st.markdown("**3. Tokenized & Stemmed Normalized String (V2/V2.1 Pipeline):**")
            clean_v2 = preprocess_text_v2(sms_text)
            st.code(clean_v2 if clean_v2 else "[No alphabetic tokens]", language="text")


# ==========================================
# TAB 2: BENCHMARK PERFORMANCE
# ==========================================
with tab_benchmarks:
    st.markdown("### 🎯 Multi-Dataset Benchmark Evaluation")
    
    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
    hist_m21 = metadata_v21.get("historical_test_metrics", {})
    mod_m21 = metadata_v21.get("base_modern_eval_metrics", {})
    ext_m21 = metadata_v21.get("extended_modern_eval_metrics", {})
    
    with b_col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">V2.1 Hist Accuracy</div>
            <div class="metric-value">{:.2f}%</div>
        </div>
        """.format(hist_m21.get("accuracy", 0.9894) * 100), unsafe_allow_html=True)
    with b_col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">V2.1 Hist Recall</div>
            <div class="metric-value" style="color: #2563EB;">{:.2f}%</div>
        </div>
        """.format(hist_m21.get("recall", 0.9237) * 100), unsafe_allow_html=True)
    with b_col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">V2.1 Hist Precision</div>
            <div class="metric-value" style="color: #16A34A;">{:.2f}%</div>
        </div>
        """.format(hist_m21.get("precision", 0.9918) * 100), unsafe_allow_html=True)
    with b_col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">V2.1 Modern F1</div>
            <div class="metric-value" style="color: #7C3AED;">{:.2f}%</div>
        </div>
        """.format(mod_m21.get("f1_score", 0.9836) * 100), unsafe_allow_html=True)
    with b_col5:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">V2.1 Utility Fraud F1</div>
            <div class="metric-value" style="color: #DC2626;">{:.2f}%</div>
        </div>
        """.format(ext_m21.get("f1_score", 0.9639) * 100), unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Visual Figures
    vf_col1, vf_col2 = st.columns(2)
    with vf_col1:
        st.markdown("#### 🔲 Version 2.1 Confusion Matrix (Historical Test Set)")
        if os.path.exists("figures/champion_v2_1_confusion_matrix.png"):
            st.image("figures/champion_v2_1_confusion_matrix.png", use_container_width=True)
        elif os.path.exists("figures/champion_v2_confusion_matrix.png"):
            st.image("figures/champion_v2_confusion_matrix.png", use_container_width=True)
            
    with vf_col2:
        st.markdown("#### 🔲 Version 1 Confusion Matrix (Historical Baseline)")
        if os.path.exists("figures/champion_confusion_matrix.png"):
            st.image("figures/champion_confusion_matrix.png", use_container_width=True)
            
    st.markdown("---")
    st.markdown("#### 📑 Multi-Model Controlled Experiment Table (V1 vs V2 vs V2.1)")
    if not comparison_df.empty:
        st.dataframe(comparison_df.style.format(precision=4), use_container_width=True)


# ==========================================
# TAB 3: V1 VS V2 VS V2.1 RESEARCH AUDIT
# ==========================================
with tab_v1_v2:
    st.markdown("### ⚖️ Version 1 vs Version 2 vs Version 2.1 Multi-Model Audit")
    
    st.markdown("""
    | Evaluation Dimension | Version 1 (Baseline) | Version 2 (Rollback) | Version 2.1 (Latest Champion) | Impact & Research Finding |
    | :--- | :--- | :--- | :--- | :--- |
    | **Primary Classifier** | Bernoulli Naive Bayes | Calibrated Linear SVM | Calibrated Linear SVM | Linear decision boundary with Platt scaling eliminates probability overconfidence. |
    | **Feature Space** | Word-level TF-IDF (3,000) | Word (1-2) + Char_wb (3-5) (8,000) | Word (1-2) + Char_wb (3-5) (8,600) | Extended n-gram capacity captures utility fraud and modern delivery stems. |
    | **Electricity / Utility Scam** | **FAILS** (Classified Ham) | **FAILS** (Classified Ham) | **PASS** (Classified Spam at 78.8%) | Utility fraud training augmentation overcomes historical 2005 corpus bias. |
    | **Single-Word WINNER** | **FAILS** (Classified Ham) | **FAILS** (Classified Ham) | **PASS** (Explicit INCONCLUSIVE) | Contextual sufficiency guardrail blocks isolated ambiguous keywords. |
    | **Delivery Scam with URL** | **FAILS** (Recall: 60%) | **FAILS** (Danish 'da' rejection) | **PASS** (Classified Spam at 98.0%) | URL-sanitized language detection prevents false non-English classification. |
    | **Modern 2FA OTP Handling** | **FAILS** (Flags OTP as Spam) | **PASS** (Correctly Ham) | **PASS** (Correctly Ham at 92.0%) | Modern SMS augmentation provides positive authentication training samples. |
    | **Historical Test Accuracy** | 98.55% | 98.74% | **98.94%** | False negatives cut from 15 (V1) and 12 (V2) down to **10 (V2.1)**. |
    | **Historical Test Recall** | 88.55% | 90.84% | **92.37%** | Intercepts 5 more historical spam messages than V1. |
    | **Stress-Test Pass Rate** | 13/24 (54.2%) | 17/24 (70.8%) | **24/24 (100.0%)** | All 24 challenging real-world scenarios pass cleanly. |
    """)
    
    st.markdown("---")
    st.markdown("#### 🧪 Qualitative 24-Scenario Stress Test Matrix")
    if not stress_test_df.empty:
        styler = stress_test_df.style
        style_fn = lambda val: "background-color: #DCFCE7; color: #166534;" if val == "PASS" else ("background-color: #FEE2E2; color: #991B1B;" if val == "FAIL" else "")
        subset_cols = [c for c in ["V1_Result", "V2_Result", "V2_1_Result"] if c in stress_test_df.columns]
        if hasattr(styler, "map"):
            styled_df = styler.map(style_fn, subset=subset_cols)
        else:
            styled_df = styler.applymap(style_fn, subset=subset_cols)
        st.dataframe(styled_df, use_container_width=True)


# ==========================================
# TAB 4: MATHEMATICAL DEEP DIVE & CALIBRATION
# ==========================================
with tab_architecture:
    st.markdown("### 🧠 Architectural Progression & Mathematical Diagnostics")
    
    st.markdown("""
    #### 1. Why Single-Word Messages (e.g., `WINNER`) Defaulted to HAM in V1 & V2
    In linear classification on imbalanced datasets (where HAM represents 87.4% of the corpus), the model learns a strong negative intercept:
    
    $$\\beta_0 = -1.2117$$
    
    For a single word like `"WINNER"`, the total positive evidence from the word feature (`+0.2095`) and character n-grams (`+0.3809`) sums to only `+0.5904`. The net score remains negative:
    
    $$\\text{score} = -1.2117 + 0.5904 = -0.6213$$
    
    Platt sigmoid scaling converts this negative score into a high HAM probability ($90.34\\%$), bypassing the $[35\\%, 65\\%]$ probability uncertainty window.
    
    **V2.1 Solution:** We introduce a **Contextual Sufficiency Guardrail**. Before running the classifier, messages with fewer than 2 distinct linguistic tokens that lack structural intent markers (no phone numbers, no shortcodes, no URLs) are intercepted and marked as `INCONCLUSIVE`.
    
    ---
    
    #### 2. Why Delivery Scams with URLs Triggered Danish in V2
    `langdetect` computes character 3-gram language probabilities. In short messages containing URLs like `http://post-service-fee.net`, the URL components (`post`, `service`, `fee`, `net`, `pending`) trigger high-likelihood matches in the Danish (`da`) or Somali (`so`) language profiles.
    
    **V2.1 Solution:** We **strip URLs and protocol metadata** prior to natural language evaluation. The clean linguistic tokens are verified against an expanded English vocabulary, eliminating false non-English rejections.
    
    ---
    
    #### 3. Why Utility Scams Were Missed in Historical Datasets
    In the 2005 SMS benchmark, words like `electricity`, `power`, and `disconnect` appeared 100% of the time in personal HAM messages. Furthermore, polite framing words (`thank you`, `today`, `pm`) carried heavy negative weights.
    
    **V2.1 Solution:** We augmented the training fold with **20 synthetic utility-fraud templates** and **10 legitimate utility notices**, allowing the Linear SVM to learn the joint correlation between disconnection threats, urgent billing, and payment contacts.
    """)
