"""
Version 2 Data Preprocessing and Feature Extraction Module.
Introduces:
1. Adversarial deobfuscation (leetspeak normalization & spaced character repair).
2. Dual-mode preprocessing for Word-level and Character-level n-gram extraction.
3. Dataset loading with modern training augmentation support.
"""

import os
import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure NLTK resources
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

_STEMMER = PorterStemmer()
_STOPWORDS = set(stopwords.words('english'))
_LEET_SUB_MAP = {
    '@': 'a',
    '0': 'o',
    '1': 'i',
    '3': 'e',
    '4': 'a',
    '5': 's',
    '$': 's',
    '!': 'i'
}


def deobfuscate_text(text: str) -> str:
    """
    Normalizes adversarial text obfuscations common in SMS spam:
    1. Spaced-out letter padding (e.g., 'F R E E' -> 'FREE', 'C A S H' -> 'CASH')
    2. Embedded leetspeak inside alphabetic tokens (e.g., 'W!NNER' -> 'WiNNER', 'cl@im' -> 'claim')
    Preserves purely numeric strings (e.g., '1000', '482910') without altering them.
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # Repair spaced-out individual letters: look for single letters separated by single space
    repaired = re.sub(r'(?<=\b[a-zA-Z])\s+(?=[a-zA-Z]\b)', '', text)
    
    # Normalize leetspeak characters when embedded with letters
    tokens = repaired.split()
    normalized_tokens = []
    for tok in tokens:
        # If token has both alphabet letters and leetspeak symbols (e.g. 'fr33', 'cl@im')
        if re.search(r'[a-zA-Z]', tok) and re.search(r'[@01345$!]', tok):
            # Replace only the leet characters
            clean_tok = ''.join(_LEET_SUB_MAP.get(c, c) for c in tok)
            normalized_tokens.append(clean_tok)
        else:
            normalized_tokens.append(tok)
            
    return " ".join(normalized_tokens)


def preprocess_text_v2(text: str, stem: bool = True, remove_stopwords: bool = True) -> str:
    """
    Version 2 text cleaning pipeline:
    1. Deobfuscation (spaced-out characters and leetspeak)
    2. Lowercasing
    3. Tokenization with regex fallback
    4. Alphanumeric filtering
    5. Stopwords removal
    6. Porter Stemming
    """
    if text is None or not isinstance(text, str):
        return ""
    
    text = text.strip()
    if not text:
        return ""
        
    deobf = deobfuscate_text(text).lower()
    
    try:
        tokens = nltk.word_tokenize(deobf)
    except Exception:
        tokens = re.findall(r'\b\w+\b', deobf)
        
    tokens = [tok for tok in tokens if tok.isalnum()]
    
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in _STOPWORDS]
        
    if stem:
        tokens = [_STEMMER.stem(tok) for tok in tokens]
        
    return " ".join(tokens)


def char_clean_v2(text: str) -> str:
    """
    Character-level cleaner for char_wb analyzer.
    Applies deobfuscation and lowercasing while preserving character order and whitespace.
    """
    if text is None or not isinstance(text, str):
        return ""
    deobf = deobfuscate_text(text).lower()
    return re.sub(r'\s+', ' ', deobf).strip()


class V2WordPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for Version 2 word-level NLP preprocessing.
    """
    def __init__(self, stem: bool = True, remove_stopwords: bool = True):
        self.stem = stem
        self.remove_stopwords = remove_stopwords
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.apply(lambda t: preprocess_text_v2(t, stem=self.stem, remove_stopwords=self.remove_stopwords))
        return [preprocess_text_v2(t, stem=self.stem, remove_stopwords=self.remove_stopwords) for t in X]


def load_v2_augmented_dataset(
    original_path: str = "data/spam.csv",
    modern_train_path: str = "data/modern_sms_augmented_train.csv"
):
    """
    Loads historical dataset and appends the dedicated modern training dataset.
    Maintains zero contamination with the modern held-out evaluation dataset.
    """
    from src.data_preprocessing import load_raw_dataset
    df_orig = load_raw_dataset(original_path)
    
    if os.path.exists(modern_train_path):
        df_mod = pd.read_csv(modern_train_path)
        combined_text = pd.concat([df_orig['text'], df_mod['text']], ignore_index=True)
        combined_label = pd.concat([df_orig['label'], df_mod['label']], ignore_index=True)
        combined_df = pd.DataFrame({'text': combined_text, 'label': combined_label})
        return combined_df
    return df_orig
