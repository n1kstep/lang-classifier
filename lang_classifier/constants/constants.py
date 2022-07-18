# pylint: disable=missing-module-docstring

LANGUAGES = [
    "ru",
    "uk",
    "ka",
    "he",
    "en",
    "de",
    "be",
    "kk",
    "az",
    "hy",
]

SEED = 42

PATTERN = r"[\s.,;]+"  # split text, remove punctuation

BASE_MODEL = "xlm-roberta-base"

DROP_COLS = ['Epoch', 'Training Loss', 'Validation Loss']

NEW_ROWS = [
    'az',
    'be',
    'de',
    'en',
    'he',
    'hy',
    'ka',
    'kk',
    'macro_avg',
    'ru',
    'uk',
    'weighted avg',
]

NEW_COLS = [
    'class',
    'f1-score',
    'precision',
    'recall',
    'support',
]
