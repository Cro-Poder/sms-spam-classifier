"""
Phase 9 Controlled Testing: Side-by-Side Stress Test of Version 1 vs Version 2.
Evaluates 24 qualitative stress scenarios covering modern OTP, phishing,
obfuscation, non-English, numbers-only, and edge cases.
"""

import os
import pandas as pd
from src.predict import SpamPredictor as SpamPredictorV1
from src.predict_v2 import SpamPredictorV2

p_v1 = SpamPredictorV1("artifacts/final_model.joblib")
p_v2 = SpamPredictorV2("artifacts/final_model_v2.joblib")

stress_cases = [
    # 1. Legitimate OTP (The primary failure of V1)
    {
        "category": "Legitimate OTP / 2FA",
        "input": "Your Chase OTP verification code is 482910. Do not share this code with anyone. Expires in 10 minutes.",
        "expected": "HAM"
    },
    # 2. Legitimate OTP (Google)
    {
        "category": "Legitimate OTP / 2FA",
        "input": "Google: 839201 is your verification code. Never share this code with anyone.",
        "expected": "HAM"
    },
    # 3. Fraudulent OTP Phishing
    {
        "category": "Fraudulent OTP Phishing",
        "input": "Chase Fraud Alert: Did you send $500? Reply YES or NO. If NO, call 1-888-492-0192 to block your card immediately.",
        "expected": "SPAM"
    },
    # 4. Bank Phishing Scam
    {
        "category": "Bank Phishing Scam",
        "input": "ALERT: Your Barclays debit card has been suspended due to suspicious activity. Visit http://barclays-security-login.com immediately to unlock.",
        "expected": "SPAM"
    },
    # 5. Legitimate Delivery Notification
    {
        "category": "Legitimate Delivery",
        "input": "UPS: Your package #1Z9842109 is out for delivery today between 2:15 PM and 4:15 PM.",
        "expected": "HAM"
    },
    # 6. Fraudulent Delivery Scam
    {
        "category": "Delivery Fee Phishing",
        "input": "USPS: We attempted to deliver your package today but an unpaid customs fee of $2.99 is pending. Pay now at http://usps-redelivery-fee.info",
        "expected": "SPAM"
    },
    # 7. Legitimate Commercial Promotion
    {
        "category": "Legitimate Promotion",
        "input": "Starbucks: Double star day is tomorrow! Earn 2x stars on all handcrafted drinks with the Starbucks app. Reply STOP to cancel.",
        "expected": "HAM"
    },
    # 8. Promotional Retail
    {
        "category": "Legitimate Promotion",
        "input": "Nike: Members get early access to our seasonal sale. Shop online with code MEMBER20. Reply STOP to cancel.",
        "expected": "HAM"
    },
    # 9. Lottery Scam
    {
        "category": "Prize / Lottery Scam",
        "input": "You are the selected winner of the UK National Lottery jackpot of £1,500,000. Send your bank details to claim@lottery-winner.co.uk.",
        "expected": "SPAM"
    },
    # 10. Gift Card Phishing
    {
        "category": "Prize / Lottery Scam",
        "input": "Urgent! Claim your free £500 Amazon gift card before expiry at 12pm. Text GIFT to 82223.",
        "expected": "SPAM"
    },
    # 11. Adversarial Leetspeak
    {
        "category": "Adversarial Obfuscation",
        "input": "W!NNER!! Freeee pr!ze aw@rd wa!ting 4 u. Cl@im now cal 09061701461",
        "expected": "SPAM"
    },
    # 12. Adversarial Leetspeak
    {
        "category": "Adversarial Obfuscation",
        "input": "fr33 entry into 1000 c4sh draw call 09061701461 now",
        "expected": "SPAM"
    },
    # 13. Spaced-out letters
    {
        "category": "Adversarial Spacing",
        "input": "F R E E ringtone text WIN to 80085 to claim your prize now",
        "expected": "SPAM"
    },
    # 14. Misspelled words (ham)
    {
        "category": "Informal / Misspelled Ham",
        "input": "Heyy im gona b late 4 class coz trafick is terible rn see u in 10 mins",
        "expected": "HAM"
    },
    # 15. Short legitimate
    {
        "category": "Short Conversational",
        "input": "Ok see u soon",
        "expected": "HAM"
    },
    # 16. Single word (WINNER)
    {
        "category": "Single Ambiguous Word",
        "input": "WINNER",
        "expected": "SPAM"
    },
    # 17. Numbers only
    {
        "category": "Numbers Only",
        "input": "9847120491823091823091823",
        "expected": "INCONCLUSIVE"
    },
    # 18. Emojis only
    {
        "category": "Emojis Only",
        "input": "😀🎉🎁💰📱💸",
        "expected": "INCONCLUSIVE"
    },
    # 19. Empty Input
    {
        "category": "Empty Input",
        "input": "",
        "expected": "EMPTY"
    },
    # 20. Non-English (Spanish)
    {
        "category": "Non-English (Spanish)",
        "input": "Hola amigo, ¿cómo estás? ¿Nos vemos mañana para almorzar?",
        "expected": "UNSUPPORTED_LANGUAGE"
    },
    # 21. Non-English (French)
    {
        "category": "Non-English (French)",
        "input": "Bonjour, avez-vous reçu le document pour la réunion de demain?",
        "expected": "UNSUPPORTED_LANGUAGE"
    },
    # 22. Non-English (Bengali script)
    {
        "category": "Non-English (Bengali)",
        "input": "আজকে কি তুমি আসবে? দেখা হবে বিকেলে।",
        "expected": "UNSUPPORTED_LANGUAGE"
    },
    # 23. Mixed-Language text
    {
        "category": "Mixed Language",
        "input": "Bonjour my friend, please call me when you arrive at the airport.",
        "expected": "HAM"
    },
    # 24. Standard long conversational ham
    {
        "category": "Standard Long Ham",
        "input": "Hi team, please remember that tomorrow's project review meeting has been shifted from 2 PM to 3:30 PM in Conference Room A. See you all there.",
        "expected": "HAM"
    }
]

records = []
v1_correct = 0
v2_correct = 0

print("=" * 105)
print(f"{'Category':<24} | {'Expected':<12} | {'V1 Pred':<8} | {'V1 Conf':<8} | {'V2 Verdict':<14} | {'V2 Conf':<8} | {'V1 Status':<9} | {'V2 Status':<9}")
print("=" * 105)

for case in stress_cases:
    msg = case["input"]
    exp = case["expected"]
    
    # Version 1
    res1 = p_v1.predict(msg)
    pred1 = res1["label"]
    conf1 = f"{res1['confidence']*100:.1f}%" if res1.get("confidence") is not None else "N/A"
    corr1 = (pred1 == exp)
    if corr1:
        v1_correct += 1
        
    # Version 2
    res2 = p_v2.predict(msg)
    pred2 = res2["verdict"]
    conf2 = f"{res2['confidence']*100:.1f}%" if res2.get("confidence") is not None else "N/A"
    corr2 = (pred2 == exp)
    if corr2:
        v2_correct += 1
        
    s1 = "PASS" if corr1 else "FAIL"
    s2 = "PASS" if corr2 else "FAIL"
    
    print(f"{case['category']:<24} | {exp:<12} | {pred1:<8} | {conf1:<8} | {pred2:<14} | {conf2:<8} | {s1:<9} | {s2:<9}")
    
    records.append({
        "Category": case["category"],
        "Input": msg[:55] + "..." if len(msg) > 55 else msg,
        "Expected": exp,
        "V1_Prediction": pred1,
        "V1_Confidence": conf1,
        "V1_Result": s1,
        "V2_Verdict": pred2,
        "V2_Confidence": conf2,
        "V2_Result": s2,
        "V2_Explanation": res2["explanation"]
    })

print("=" * 105)
print(f"Version 1 Stress Test Score: {v1_correct}/{len(stress_cases)} ({v1_correct/len(stress_cases)*100:.1f}%)")
print(f"Version 2 Stress Test Score: {v2_correct}/{len(stress_cases)} ({v2_correct/len(stress_cases)*100:.1f}%)")
print("=" * 105)

df_comp = pd.DataFrame(records)
os.makedirs("artifacts", exist_ok=True)
df_comp.to_csv("artifacts/v1_vs_v2_stress_test_comparison.csv", index=False)
print("Saved comparison table to artifacts/v1_vs_v2_stress_test_comparison.csv")
