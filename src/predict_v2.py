"""
Version 2 Production Inference Engine for SMS Spam Classification.
Features:
1. Input validation & language detection (rejects non-English & featureless inputs).
2. Deobfuscation of adversarial leetspeak and spaced-out letters.
3. Calibrated probability scoring via Platt-scaled Linear SVM.
4. Uncertainty thresholding (marks borderline predictions as INCONCLUSIVE).
"""

import os
import sys
import argparse
import joblib
from typing import Union, List, Dict, Any

from src.language_detector import LanguageDetector
from src.data_preprocessing_v2 import V2WordPreprocessor, char_clean_v2, deobfuscate_text


class SpamPredictorV2:
    """
    Version 2 Inference Engine with Guardrails and Calibrated Probability Output.
    """
    def __init__(self, model_path: str = "artifacts/final_model_v2.joblib", uncertainty_margin: float = 0.15):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Version 2 model not found at {model_path}. "
                "Please run `python -m src.model_training_v2` first."
            )
        self.model_path = model_path
        self.uncertainty_margin = uncertainty_margin  # [0.5 - margin, 0.5 + margin] is inconclusive
        self.pipeline = joblib.load(model_path)
        
    def predict(self, text: Union[str, Any]) -> Dict[str, Any]:
        """
        Evaluates input SMS text with input validation, language detection,
        adversarial deobfuscation, and calibrated probability estimation.
        """
        # Step 1: Input Validation & Language Detection Guardrail
        validation = LanguageDetector.validate_and_detect(text)
        
        if validation["status"] == "EMPTY_INPUT":
            return {
                "status": "EMPTY",
                "verdict": "EMPTY",
                "confidence": None,
                "probabilities": None,
                "detected_language": None,
                "explanation": validation["reason"],
                "raw_text": "" if text is None else str(text),
                "version": "2.0.0"
            }
            
        if validation["status"] == "UNSUPPORTED_LANGUAGE":
            return {
                "status": "UNSUPPORTED_LANGUAGE",
                "verdict": "UNSUPPORTED_LANGUAGE",
                "confidence": None,
                "probabilities": None,
                "detected_language": validation["detected_language"],
                "explanation": validation["reason"],
                "raw_text": text,
                "version": "2.0.0"
            }
            
        if validation["status"] == "INCONCLUSIVE":
            return {
                "status": "INCONCLUSIVE",
                "verdict": "INCONCLUSIVE",
                "confidence": None,
                "probabilities": None,
                "detected_language": None,
                "explanation": validation["reason"],
                "raw_text": text,
                "version": "2.0.0"
            }
            
        # Step 2: Model Prediction on Valid English Input
        raw_str = text.strip()
        probs = self.pipeline.predict_proba([raw_str])[0]
        prob_ham = float(probs[0])
        prob_spam = float(probs[1])
        
        lower_bound = 0.50 - self.uncertainty_margin
        upper_bound = 0.50 + self.uncertainty_margin
        
        # Step 3: Uncertainty Threshold Check
        if lower_bound <= prob_spam <= upper_bound:
            verdict = "INCONCLUSIVE"
            confidence = max(prob_ham, prob_spam)
            explanation = (
                f"Model uncertainty is high ({prob_spam*100:.1f}% spam score is within the ambiguous "
                f"decision threshold of {lower_bound*100:.0f}%-{upper_bound*100:.0f}%). "
                "Insufficient evidence for a definitive classification."
            )
        elif prob_spam > upper_bound:
            verdict = "SPAM"
            confidence = prob_spam
            explanation = "Calibrated model indicates high probability of unsolicited / fraudulent SMS pattern."
        else:
            verdict = "HAM"
            confidence = prob_ham
            explanation = "Calibrated model indicates high probability of legitimate communication."
            
        return {
            "status": "SUCCESS",
            "verdict": verdict,
            "confidence": float(confidence),
            "probabilities": {"ham": prob_ham, "spam": prob_spam},
            "detected_language": validation.get("detected_language", "English"),
            "explanation": explanation,
            "raw_text": raw_str,
            "version": "2.0.0"
        }
        
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]


def main():
    parser = argparse.ArgumentParser(description="SMS Spam Detection Version 2 CLI")
    parser.add_argument("--message", "-m", type=str, help="SMS text to evaluate")
    parser.add_argument("--model-path", type=str, default="artifacts/final_model_v2.joblib")
    args = parser.parse_args()
    
    predictor = SpamPredictorV2(model_path=args.model_path)
    
    if args.message:
        res = predictor.predict(args.message)
        print("\n--- Version 2 Prediction Result ---")
        print(f"Message:     {args.message}")
        print(f"Verdict:     {res['verdict']}")
        if res['confidence'] is not None:
            print(f"Confidence:  {res['confidence'] * 100:.2f}% (Calibrated)")
            print(f"Probs:       Ham={res['probabilities']['ham']:.4f}, Spam={res['probabilities']['spam']:.4f}")
        print(f"Language:    {res['detected_language']}")
        print(f"Explanation: {res['explanation']}")
    else:
        print("Interactive SMS Spam Classifier V2 (Type 'exit' to quit):")
        print("-" * 60)
        while True:
            try:
                user_msg = input("\nEnter SMS: ").strip()
                if user_msg.lower() in ['exit', 'quit', 'q']:
                    break
                if not user_msg:
                    print("Please enter a non-empty message.")
                    continue
                res = predictor.predict(user_msg)
                print(f"-> Verdict: {res['verdict']} | {res['explanation']}")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
