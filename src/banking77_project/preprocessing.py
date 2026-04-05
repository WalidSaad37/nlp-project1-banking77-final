from __future__ import annotations

import re
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from banking77_project.config import PreprocessingConfig

CONTRACTIONS = {
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "i'm": "i am",
    "isn't": "is not",
    "it's": "it is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "wasn't": "was not",
    "we're": "we are",
    "weren't": "were not",
    "what's": "what is",
    "won't": "will not",
    "wouldn't": "would not",
    "you're": "you are",
    "you've": "you have",
}

TOKEN_PATTERN = re.compile(r"\b[a-z][a-z']+\b")


def normalize_text(text: str, config: PreprocessingConfig) -> str:
    cleaned = text.strip()
    if config.lowercase:
        cleaned = cleaned.lower()

    if config.expand_contractions:
        for contraction, expanded in CONTRACTIONS.items():
            cleaned = cleaned.replace(contraction, expanded)

    if config.remove_digits:
        cleaned = re.sub(r"\d+", " ", cleaned)

    if config.remove_punctuation:
        cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)

    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str, config: PreprocessingConfig) -> list[str]:
    normalized = normalize_text(text, config)
    tokens = TOKEN_PATTERN.findall(normalized)
    return [
        token
        for token in tokens
        if len(token) >= config.min_token_length
        and (not config.remove_stopwords or token not in ENGLISH_STOP_WORDS)
    ]


def preprocess_for_model(text: str, config: PreprocessingConfig) -> str:
    return " ".join(tokenize(text, config))


def preprocess_many(texts: Iterable[str], config: PreprocessingConfig) -> list[str]:
    return [preprocess_for_model(text, config) for text in texts]
