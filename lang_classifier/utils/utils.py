# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring

from lang_classifier.constants.constants import PATTERN


def smart_truncate(text: str, length: int = 512, suffix="") -> str:
    if len(text) <= length:
        return text
    return " ".join(text[: length + 1].split(" ")[0:-1]) + suffix


def split_text(text: str) -> list:
    tokens = text.split(PATTERN)
    return tokens
