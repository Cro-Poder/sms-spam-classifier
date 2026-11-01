"""
Prediction module for SMS Spam Classification.
Loads the serialized pipeline and provides single and batch inference with probability scoring.
"""

import os
import sys
import argparse
import joblib
from typing import Union, List, Dict, Any

# Ensure TextPreprocessor is available in namespace when unpickling
from src.data_preprocessing import TextPreprocessor, preprocess_text


class SpamPredictor:
    """
    Production inference engine for SMS Spam Classification.
    Loads and runs predictions with the serialized Pipeline.
    """
    def __init__(self, model_path: str = "artifacts/final_model.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. "
                "Please run `python -m src.model_training` first to train and save the pipeline."
            )
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        
    def predict(self, text: Union[str, Any]) -> Dict[str, Any]:
        """
        Predict whether a single SMS message is SPAM or HAM.
        
        Args:
            text: Raw input message text.
            
        Returns:
            dict containing:
                - label: 'SPAM' or 'HAM'
                - prediction: 1 (Spam) or 0 (Ham)
                - confidence: confidence score of prediction (0.0 - 1.0)
                - probabilities: dict with {'ham': float, 'spam': float}
                - is_valid: whether input was non-empty and valid
        """
        if text is None or not isinstance(text, str) or not text.strip():
            return {
                "label": "HAM",
                "prediction": 0,
                "confidence": 0.50,
                "probabilities": {"ham": 0.50, "spam": 0.50},
                "is_valid": False,
                "error": "Input message is empty or invalid.",
                "raw_text": ""
            }
            
        clean_input = text.strip()
        pred = int(self.pipeline.predict([clean_input])[0])
        
        # Calculate probabilities if supported
        probabilities = {"ham": 0.50, "spam": 0.50}
        confidence = 1.0
        
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([clean_input])[0]
            prob_ham = float(probs[0])
            prob_spam = float(probs[1])
            probabilities = {"ham": prob_ham, "spam": prob_spam}
            confidence = prob_spam if pred == 1 else prob_ham
            
        label = "SPAM" if pred == 1 else "HAM"
        
        return {
            "label": label,
            "prediction": pred,
            "confidence": float(confidence),
            "probabilities": probabilities,
            "is_valid": True,
            "raw_text": clean_input
        }
        
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple text messages.
        """
        return [self.predict(t) for t in texts]


def main():
    parser = argparse.ArgumentParser(description="SMS Spam Detection CLI Predictor")
    parser.add_argument(
        "--message", "-m",
        type=str,
        help="SMS text message to evaluate"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/final_model.joblib",
        help="Path to serialized pipeline artifact"
    )
    args = parser.parse_args()
    
    predictor = SpamPredictor(model_path=args.model_path)
    
    if args.message:
        result = predictor.predict(args.message)
        print("\n--- Prediction Result ---")
        print(f"Message:    {args.message}")
        print(f"Verdict:    {result['label']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print(f"Probabilities: Ham = {result['probabilities']['ham']:.4f}, Spam = {result['probabilities']['spam']:.4f}")
    else:
        print("Interactive SMS Spam Classifier (Type 'exit' to quit):")
        print("-" * 55)
        while True:
            try:
                user_msg = input("\nEnter SMS: ").strip()
                if user_msg.lower() in ['exit', 'quit', 'q']:
                    break
                if not user_msg:
                    print("Please enter a non-empty message.")
                    continue
                result = predictor.predict(user_msg)
                print(f"-> Verdict: {result['label']} (Confidence: {result['confidence'] * 100:.2f}%)")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()

