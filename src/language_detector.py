"""
Input Validation and Language Detection Module for SMS Classification.
Protects the model from silently classifying unsupported languages or featureless inputs as HAM,
while handling short colloquial English SMS abbreviations gracefully.
"""

import re
import string
from typing import Dict, Any
from nltk.corpus import stopwords

try:
    import langdetect
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

# Common English vocabulary and SMS shorthand for disambiguating short colloquial texts
_COMMON_ENGLISH = set(stopwords.words('english')).union({
    'ok', 'see', 'soon', 'u', 'ur', 'call', 'thanks', 'thx', 'hi', 'hello',
    'yes', 'no', 'come', 'go', 'good', 'love', 'dinner', 'lunch',
    'tonight', 'today', 'tomorrow', 'meet', 'meeting', 'home', 'work',
    'free', 'win', 'claim', 'prize', 'cash', 'money', 'txt', 'text',
    'send', 'reply', 'stop', 'code', 'card', 'bank', 'account', 'verify',
    'verification', 'link', 'check', 'late', 'class', 'traffic', 'mins',
    'min', 'now', 'hey', 'heyy', 'guy', 'guys', 'friend', 'dad', 'mom'
})

# Regex patterns for non-Latin writing systems
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

# Emoji pattern
_EMOJI_PATTERN = re.compile(
    r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]',
    flags=re.UNICODE
)


class LanguageDetector:
    """
    Multifaceted input validator that verifies:
    1. Input non-emptiness.
    2. Presence of informative alphabetic features (vs numbers/emojis/punctuation only).
    3. Writing system (Latin vs non-Latin).
    4. Language detection for Latin script (English vs Spanish/French/German/etc.).
    """
    
    @staticmethod
    def validate_and_detect(text: Any) -> Dict[str, Any]:
        """
        Validates text and identifies language or feature insufficiency.
        
        Returns:
            dict containing:
                - is_valid_english: bool
                - status: 'VALID', 'EMPTY_INPUT', 'INCONCLUSIVE', or 'UNSUPPORTED_LANGUAGE'
                - detected_language: ISO language code (e.g., 'en', 'es', 'fr') or script name
                - reason: explanatory description for the user
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
            
        # 2. Check for non-Latin alphabets
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
                
        # 4. Check tokens against English vocabulary to prevent short English slang from false triggers
        tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', stripped)]
        if tokens:
            eng_match_count = sum(1 for w in tokens if w in _COMMON_ENGLISH)
            eng_ratio = eng_match_count / len(tokens)
            # If 50% or more tokens are standard English or SMS words, confirm English
            if eng_ratio >= 0.50:
                return {
                    "is_valid_english": True,
                    "status": "VALID",
                    "detected_language": "English",
                    "reason": "Valid English input."
                }
                
        # 5. Language detection on Latin text with insufficient English vocabulary matches
        if _LANGDETECT_AVAILABLE and len(tokens) >= 3:
            try:
                lang = detect(stripped)
                if lang != 'en':
                    lang_names = {
                        'es': 'Spanish', 'fr': 'French', 'de': 'German',
                        'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch',
                        'ru': 'Russian (transliterated)', 'tl': 'Tagalog', 'id': 'Indonesian',
                        'af': 'Afrikaans', 'so': 'Somali'
                    }
                    lang_label = lang_names.get(lang, lang.upper())
                    return {
                        "is_valid_english": False,
                        "status": "UNSUPPORTED_LANGUAGE",
                        "detected_language": lang_label,
                        "reason": f"Non-English language ({lang_label}) detected. English text required."
                    }
            except Exception:
                pass
                
        return {
            "is_valid_english": True,
            "status": "VALID",
            "detected_language": "English",
            "reason": "Valid English input."
        }
