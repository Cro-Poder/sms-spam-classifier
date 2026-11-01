"""
Data preprocessing module for SMS Spam Classification.
Provides robust data loading, cleaning, text normalization, and 
Scikit-learn compatible pipeline transformers with high-performance caching.
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

# Ensure necessary NLTK corpora are downloaded
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Module-level singletons for efficient reuse
_STEMMER = PorterStemmer()
_STOPWORDS = set(stopwords.words('english'))
_CLEAN_CACHE = {}


def preprocess_text(text: str, stem: bool = True, remove_stopwords: bool = True) -> str:
    """
    Standardize and clean raw text for NLP models.
    
    Steps:
    1. Null/type safety: safely converts non-string or None to empty string.
    2. Lowercasing: normalizes capitalization differences.
    3. Tokenization: splits into words using NLTK word_tokenize (with regex fallback).
    4. Alphanumeric filtering: removes punctuation, emojis, and special characters.
    5. Stopwords removal: drops frequent english stopwords (e.g., 'the', 'is').
    6. Stemming: applies Porter Stemmer to reduce words to root forms (e.g., 'running' -> 'run').
    
    Args:
        text (str): Raw input SMS message.
        stem (bool): Whether to apply Porter Stemmer. Default True.
        remove_stopwords (bool): Whether to remove English stopwords. Default True.
        
    Returns:
        str: Normalized, space-separated cleaned tokens.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
        
    text = text.strip()
    if not text:
        return ""
        
    cache_key = (text, stem, remove_stopwords)
    if cache_key in _CLEAN_CACHE:
        return _CLEAN_CACHE[cache_key]
        
    # Lowercase
    lowered = text.lower()
    
    # Tokenize
    try:
        tokens = nltk.word_tokenize(lowered)
    except Exception:
        tokens = re.findall(r'\b\w+\b', lowered)
        
    # Keep alphanumeric tokens
    tokens = [tok for tok in tokens if tok.isalnum()]
    
    # Remove stopwords
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in _STOPWORDS]
        
    # Apply Porter Stemming
    if stem:
        tokens = [_STEMMER.stem(tok) for tok in tokens]
        
    result = " ".join(tokens)
    _CLEAN_CACHE[cache_key] = result
    return result


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for text cleaning.
    Enables seamless integration into an end-to-end Pipeline without data leakage.
    """
    def __init__(self, stem: bool = True, remove_stopwords: bool = True):
        self.stem = stem
        self.remove_stopwords = remove_stopwords
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """
        Transforms an iterable or series of raw text messages.
        """
        if isinstance(X, pd.Series):
            return X.apply(lambda t: preprocess_text(t, stem=self.stem, remove_stopwords=self.remove_stopwords))
        return [preprocess_text(t, stem=self.stem, remove_stopwords=self.remove_stopwords) for t in X]


def load_raw_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads and normalizes the SMS Spam Collection dataset from disk.
    
    Handles:
    - Multiple file encodings (latin-1, utf-8, cp1252)
    - Concatenation of fragmented Unnamed columns resulting from unescaped CSV commas
    - Duplicate detection and removal
    - Label encoding ('ham' -> 0, 'spam' -> 1)
    
    Args:
        filepath (str): Path to CSV dataset.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with columns ['text', 'label', 'char_count', 'word_count', 'sentence_count'].
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        
    df = None
    encodings = ['latin-1', 'cp1252', 'utf-8', 'iso-8859-1']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
            
    if df is None:
        raise ValueError(f"Failed to read {filepath} with supported encodings {encodings}")
        
    # Handle known column variations
    if 'v1' in df.columns and 'v2' in df.columns:
        # Check if Unnamed columns contain trailing text from unescaped commas
        unnamed_cols = [c for c in df.columns if 'Unnamed' in c]
        for c in unnamed_cols:
            has_content = df[c].notna() & (df[c].astype(str).str.strip() != '')
            df.loc[has_content, 'v2'] = df.loc[has_content, 'v2'].astype(str) + ' ' + df.loc[has_content, c].astype(str)
            
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    elif 'label' in df.columns and 'text' in df.columns:
        df = df[['label', 'text']]
    else:
        # Default fallback: take first column as label, second as text
        cols = list(df.columns[:2])
        df = df[cols].rename(columns={cols[0]: 'label', cols[1]: 'text'})
        
    # Ensure text is string and drop NaNs in core fields
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip() != ''].copy()
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    dropped_duplicates = initial_count - len(df)
    
    # Encode label
    if df['label'].dtype == object:
        df['label'] = df['label'].astype(str).str.strip().str.lower().map({'ham': 0, 'spam': 1})
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)
    
    # Compute message length statistics
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda t: len(nltk.word_tokenize(t)))
    df['sentence_count'] = df['text'].apply(lambda t: len(nltk.sent_tokenize(t)))
    
    return df
