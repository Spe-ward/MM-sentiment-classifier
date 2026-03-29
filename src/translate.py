"""Offline JA→EN translation layer using argostranslate."""

import argostranslate.package
import argostranslate.translate
from langdetect import detect, LangDetectException

_JA_EN_INSTALLED = False


def _ensure_ja_en_model():
    """Download and install the JA→EN translation model if not already installed."""
    global _JA_EN_INSTALLED
    if _JA_EN_INSTALLED:
        return

    installed = argostranslate.translate.get_installed_languages()
    lang_codes = {lang.code for lang in installed}

    if "ja" not in lang_codes or "en" not in lang_codes:
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        ja_en = next(
            (p for p in available if p.from_code == "ja" and p.to_code == "en"),
            None,
        )
        if ja_en is None:
            raise RuntimeError("JA→EN translation package not found in Argos index")
        ja_en.install()

    _JA_EN_INSTALLED = True


def detect_language(text: str) -> str:
    """Detect the language of a text string. Returns ISO 639-1 code."""
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def translate_if_needed(text: str) -> tuple[str, str]:
    """Detect language and translate JA→EN if needed.

    Returns:
        (translated_text, detected_language)
    """
    lang = detect_language(text)

    if lang == "ja":
        _ensure_ja_en_model()
        translated = argostranslate.translate.translate(text, "ja", "en")
        return translated, "ja"

    return text, lang
