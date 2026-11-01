"""
Version 2.1 Input Validation and Language Detection Guardrail.

Fixes in V2.1:
1. Strips URLs and protocol metadata before language evaluation to eliminate false non-English n-gram triggers.
2. Expands English vocabulary recognition using historical and commercial SMS corpora without requiring heavy downloads.
3. Introduces a Contextual Sufficiency Guardrail:
   - Single-word or information-poor messages (e.g. "WINNER", "PRIZE", "CASH", "FREE") return INCONCLUSIVE.
   - Short messages with actionable signals (e.g. phone numbers, shortcodes, URLs, currency amounts) are preserved for model classification.
4. Preserves robust Unicode detection for non-Latin scripts (Bengali, Cyrillic, Arabic, CJK, etc.).
"""

import re
import string
from typing import Dict, Any, Set, Tuple
from nltk.corpus import stopwords

try:
    import langdetect
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 42
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


# Regex patterns for actionable signals in short messages
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10,11}\b|\b0\d{10}\b"
)
SHORTCODE_REGEX = re.compile(r"\b\d{4,6}\b")
CURRENCY_REGEX = re.compile(r"[\$£€¥]\s*\d+|\b\d+\s*(?:usd|gbp|eur|dollars|pounds|bucks|cents)\b", re.IGNORECASE)

# Non-Latin writing systems
_NON_LATIN_SCRIPTS = {
    "Bengali": re.compile(r'[\u0980-\u09FF]'),
    "Devanagari (Hindi/Sanskrit)": re.compile(r'[\u0900-\u097F]'),
    "Arabic/Persian": re.compile(r'[\u0600-\u06FF]'),
    "Chinese/Japanese/Korean (CJK)": re.compile(r'[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]'),
    "Cyrillic (Russian/Ukrainian)": re.compile(r'[\u0400-\u04FF]'),
    "Greek": re.compile(r'[\u0370-\u03FF]'),
    "Hebrew": re.compile(r'[\u0590-\u05FF]'),
    "Thai": re.compile(r'[\u0E00-\u0E7F]')
}

_EMOJI_PATTERN = re.compile(
    r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]',
    flags=re.UNICODE
)

# Standard English vocabulary, commercial SMS terms, and common shorthand
_BASE_ENGLISH_TERMS = {
    'ok', 'see', 'soon', 'u', 'ur', 'call', 'thanks', 'thx', 'hi', 'hello',
    'yes', 'no', 'come', 'go', 'good', 'love', 'dinner', 'lunch',
    'tonight', 'today', 'tomorrow', 'meet', 'meeting', 'home', 'work',
    'free', 'win', 'winner', 'claim', 'prize', 'cash', 'money', 'txt', 'text',
    'send', 'reply', 'stop', 'code', 'card', 'bank', 'account', 'verify',
    'verification', 'link', 'check', 'late', 'class', 'traffic', 'mins',
    'min', 'now', 'hey', 'heyy', 'guy', 'guys', 'friend', 'dad', 'mom',
    # Logistics and Delivery
    'package', 'delivery', 'parcel', 'shipping', 'shipment', 'pending',
    'post', 'postal', 'courier', 'usps', 'ups', 'fedex', 'dhl', 'royal', 'mail',
    'evri', 'dpd', 'warehouse', 'dispatch', 'dispatched', 'address', 'redelivery',
    'track', 'tracking', 'reschedule', 'customs', 'duty', 'hold', 'transit',
    # Billing, Banking, and Utility
    'fee', 'fees', 'pay', 'payment', 'paid', 'due', 'bill', 'bills',
    'overdue', 'notice', 'service', 'services', 'order', 'orders', 'balance',
    'electricity', 'power', 'utility', 'utilities', 'water', 'gas', 'council',
    'tax', 'cut', 'cutoff', 'disconnect', 'disconnected', 'disconnection',
    'officer', 'customer', 'toll', 'tolls', 'invoice', 'statement', 'refund',
    'chase', 'wells', 'fargo', 'citi', 'apple', 'google', 'amazon', 'netflix',
    'urgent', 'alert', 'warning', 'action', 'required', 'immediately', 'contact'
}

_COMMON_ENGLISH: Set[str] = set(stopwords.words('english')).union(_BASE_ENGLISH_TERMS)


class LanguageDetectorV21:
    """
    Version 2.1 Multi-tier Validator and Contextual Guardrail.
    """

    @staticmethod
    def extract_actionable_signals(text: str) -> Dict[str, Any]:
        """
        Extracts structural indicators that carry explicit communicative intent
        even in short messages (URLs, phone numbers, shortcodes, currencies).
        """
        return {
            "has_url": bool(URL_REGEX.search(text)),
            "has_phone": bool(PHONE_REGEX.search(text)),
            "has_shortcode": bool(SHORTCODE_REGEX.search(text)),
            "has_currency": bool(CURRENCY_REGEX.search(text))
        }

    @staticmethod
    def validate_and_detect(text: Any) -> Dict[str, Any]:
        """
        Validates input text with URL stripping, contextual sufficiency checks,
        and robust multilingual filtering.
        """
        # 1. Null / Non-string / Empty check
        if text is None or not isinstance(text, str):
            return {
                "is_valid_english": False,
                "status": "EMPTY_INPUT",
                "detected_language": None,
                "reason": "Input message is empty or null."
            }
            
        stripped = text.strip()
        if not stripped:
            return {
                "is_valid_english": False,
                "status": "EMPTY_INPUT",
                "detected_language": None,
                "reason": "Input message contains only whitespace."
            }

        # 2. Check for non-Latin writing systems (Bengali, Cyrillic, Arabic, etc.)
        for script_name, script_regex in _NON_LATIN_SCRIPTS.items():
            if script_regex.search(stripped):
                return {
                    "is_valid_english": False,
                    "status": "UNSUPPORTED_LANGUAGE",
                    "detected_language": script_name,
                    "reason": f"Non-Latin script ({script_name}) detected. The classifier is trained exclusively on English text."
                }

        # 3. Check for featureless inputs (numbers only, emojis only, punctuation only)
        alpha_chars = re.findall(r'[a-zA-Z]', stripped)
        digits = re.findall(r'[0-9]', stripped)
        
        if len(alpha_chars) == 0:
            if len(digits) > 0:
                return {
                    "is_valid_english": False,
                    "status": "INCONCLUSIVE",
                    "detected_language": None,
                    "reason": "Input contains numbers only without any linguistic tokens."
                }
            elif _EMOJI_PATTERN.search(stripped):
                return {
                    "is_valid_english": False,
                    "status": "INCONCLUSIVE",
                    "detected_language": None,
                    "reason": "Input contains emojis only without alphabetic tokens."
                }
            else:
                return {
                    "is_valid_english": False,
                    "status": "INCONCLUSIVE",
                    "detected_language": None,
                    "reason": "Input contains symbols/punctuation only with no informative words."
                }

        # 4. Actionable Signals Extraction & URL Stripping
        signals = LanguageDetectorV21.extract_actionable_signals(stripped)
        
        # Strip URLs before natural language analysis so domain tokens do not corrupt n-grams
        text_without_urls = URL_REGEX.sub(" ", stripped)
        tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', text_without_urls)]

        # 5. Contextual Guardrail: Insufficient Evidence for Ultra-Short Messages
        # Single isolated words (WINNER, PRIZE, CASH, FREE) lack context and must return INCONCLUSIVE.
        # However, short messages WITH actionable signals (e.g. "Call 09061701461", "Text STOP to 84122")
        # have sufficient communicative evidence and must NOT be blocked.
        has_actionable_content = (
            signals["has_url"] or 
            signals["has_phone"] or 
            signals["has_shortcode"] or 
            signals["has_currency"]
        )
        
        unique_tokens = set(tokens)
        if len(unique_tokens) < 2 and not has_actionable_content:
            return {
                "is_valid_english": False,
                "status": "INCONCLUSIVE",
                "detected_language": None,
                "reason": (
                    f"Message contains insufficient contextual evidence ({len(unique_tokens)} word). "
                    "Single-word or isolated keyword queries provide ambiguous intent."
                )
            }

        # 6. English Vocabulary Verification
        if tokens:
            eng_match_count = sum(1 for w in tokens if w in _COMMON_ENGLISH)
            eng_ratio = eng_match_count / len(tokens)
            if eng_ratio >= 0.35:
                return {
                    "is_valid_english": True,
                    "status": "VALID",
                    "detected_language": "English",
                    "reason": "Valid English input."
                }

        # 7. Fallback to Langdetect on Clean (URL-stripped) Text
        if _LANGDETECT_AVAILABLE and len(tokens) >= 3:
            clean_sentence = " ".join(tokens)
            try:
                lang_probabilities = detect_langs(clean_sentence)
                top_pred = lang_probabilities[0]
                
                # Check if English is recognized among the probable languages
                en_prob = next((lp.prob for lp in lang_probabilities if lp.lang == 'en'), 0.0)
                
                # Accept as English if top prediction is 'en' or English probability is non-negligible
                if top_pred.lang == 'en' or en_prob >= 0.20:
                    return {
                        "is_valid_english": True,
                        "status": "VALID",
                        "detected_language": "English",
                        "reason": f"Valid English input (langdetect confidence: {en_prob:.2f})."
                    }
                
                # Reject only if a non-English language has high confidence on sufficient token length
                if top_pred.prob >= 0.80 and len(tokens) >= 4:
                    lang_names = {
                        'es': 'Spanish', 'fr': 'French', 'de': 'German',
                        'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch',
                        'ru': 'Russian (transliterated)', 'tl': 'Tagalog', 'id': 'Indonesian',
                        'af': 'Afrikaans', 'so': 'Somali', 'da': 'Danish', 'sv': 'Swedish'
                    }
                    lang_label = lang_names.get(top_pred.lang, top_pred.lang.upper())
                    return {
                        "is_valid_english": False,
                        "status": "UNSUPPORTED_LANGUAGE",
                        "detected_language": lang_label,
                        "reason": f"Non-English language ({lang_label}, {top_pred.prob*100:.1f}%) detected."
                    }
            except Exception:
                pass

        # 8. Default fallback for Latin text
        return {
            "is_valid_english": True,
            "status": "VALID",
            "detected_language": "English",
            "reason": "Valid English input."
        }

