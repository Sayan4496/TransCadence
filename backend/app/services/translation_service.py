import os
from typing import Any

from huggingface_hub import InferenceClient


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LANGUAGE_NAMES = {
    "hi": "Hindi",
}

_client: InferenceClient | None = None


def get_translation_client() -> InferenceClient:
    global _client

    if _client is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is not configured.")

        _client = InferenceClient(
            model=MODEL_ID,
            token=token,
        )

    return _client


def translate_text(text: str, target_language: str) -> str:
    if target_language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported target language: {target_language}")
    if not text.strip():
        raise ValueError("Cannot translate an empty segment.")

    language_name = LANGUAGE_NAMES[target_language]
    prompt = (
        "You are a professional translator for educational video lectures and voice dubbing.\n\n"
        f"Translate the following English speech into natural, professional {language_name} suitable for spoken video dubbing.\n\n"
        "Requirements:\n"
        "- Preserve the exact meaning.\n"
        "- Do not summarize.\n"
        "- Do not add information.\n"
        "- Do not remove information.\n"
        "- Preserve names and proper nouns.\n"
        "- Preserve numbers and years.\n"
        "- Preserve technical terminology where appropriate.\n"
        "- Preserve abbreviations where appropriate.\n"
        "- Use natural, professional spoken Hindi.\n"
        "- Keep the translation natural for educational video voice dubbing.\n"
        "- Do not provide explanations.\n"
        "- Do not include quotation marks.\n"
        "- Return ONLY the translated text.\n\n"
        f"English:\n{text}"
    )

    response = get_translation_client().chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2048,
    )

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("The translation response was truncated.")

    translated_text = choice.message.content
    if not isinstance(translated_text, str) or not translated_text.strip():
        raise RuntimeError("The translation service returned no translated text.")

    return translated_text.strip()


def translate_segments(
    segments: list[dict[str, Any]],
    target_language: str,
) -> list[dict[str, Any]]:
    if target_language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported target language: {target_language}")

    translated_segments = []
    for segment in segments:
        translated_segments.append(
            {
                **segment,
                "translated_text": translate_text(
                    segment["text"],
                    target_language,
                ),
            }
        )

    return translated_segments