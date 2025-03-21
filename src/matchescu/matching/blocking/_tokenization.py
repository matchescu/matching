import re

from stopwords import clean as remove_stopwords

from matchescu.typing import EntityReference

__TOKEN_RE = re.compile(r"[\d\W_]+")


def _clean(tok: str) -> str:
    if tok is None:
        return ""
    return tok.strip("\t\n\r\a ").lower()


def tokenize_text(text: str, language: str = "en", min_length: int = 3) -> list[str]:
    if text is None:
        return []
    return list(
        {
            t: t
            for t in remove_stopwords(
                list(map(_clean, __TOKEN_RE.split(text))), language
            )
            if t and len(t) >= min_length
        }
    )


def tokenize_reference(
    ref: EntityReference, language: str = "en", min_length: int = 3
) -> list[str]:
    if ref is None:
        return []
    return tokenize_text(" ".join(map(str, ref)), language, min_length)
