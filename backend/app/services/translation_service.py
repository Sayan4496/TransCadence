"""Server-side English-to-Hindi translation through the Gemini API."""

import os
import json
import time
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from backend.app.config import load_backend_environment


load_backend_environment()


MODEL_ID = "gemini-3.1-flash-lite"
TRANSIENT_RETRY_LIMIT = 3
DEFAULT_TRANSLATION_BATCH_SIZE = 8
MAX_TRANSLATION_BATCH_SIZE = 32
FREE_FLASH_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
)
LANGUAGE_NAMES = {"hi": "Hindi"}
_client: genai.Client | None = None
_resolved_model_id: str | None = None
_resolved_model_setting: str | None = None


class GeminiTranslationError(RuntimeError):
    """Base class for safe, categorized Gemini translation errors."""


class MissingGeminiAPIKeyError(GeminiTranslationError):
    """The server-side Gemini API key is not configured."""


class GeminiAuthenticationError(GeminiTranslationError):
    """Gemini rejected the configured API key."""


class GeminiQuotaError(GeminiTranslationError):
    """Gemini reported a quota or rate-limit failure."""


class GeminiModelUnavailableError(GeminiTranslationError):
    """The configured Gemini model is unavailable to this account."""


class GeminiModelConfigurationError(GeminiTranslationError):
    """The configured model is outside the approved free Flash model set."""


class GeminiNetworkError(GeminiTranslationError):
    """The Gemini API could not be reached."""

    def __init__(
        self,
        message: str,
        api_code: int | None = None,
        api_status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.api_code = api_code
        self.api_status = api_status


class GeminiTimeoutError(GeminiTranslationError):
    """The Gemini API request timed out."""


class GeminiResponseError(GeminiTranslationError):
    """Gemini returned empty, truncated, invalid, or non-Hindi text."""


def get_translation_client() -> genai.Client:
    """Create and reuse a Gemini client using only GEMINI_API_KEY."""
    global _client

    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise MissingGeminiAPIKeyError("GEMINI_API_KEY is not configured.")
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=60_000),
        )

    return _client


def resolve_translation_model(client: genai.Client | None = None) -> str:
    """Choose an account-available Flash model from the free-tier allowlist."""
    global _resolved_model_id, _resolved_model_setting

    requested_model = os.getenv("GEMINI_MODEL", MODEL_ID).strip()
    if requested_model not in FREE_FLASH_MODELS:
        raise GeminiModelConfigurationError(
            "GEMINI_MODEL must name a supported free-tier Flash text model."
        )
    if (
        _resolved_model_id is not None
        and _resolved_model_setting == requested_model
    ):
        return _resolved_model_id

    gemini_client = client or get_translation_client()
    try:
        account_models = gemini_client.models.list(
            config=types.ListModelsConfig(page_size=100)
        )
    except Exception as exc:
        _raise_request_error(exc)

    available_models: set[str] = set()
    for model in account_models:
        model_name = getattr(model, "name", "").rsplit("/", 1)[-1]
        supported_actions = getattr(model, "supported_actions", None) or []
        if (
            model_name in FREE_FLASH_MODELS
            and "generateContent" in supported_actions
        ):
            available_models.add(model_name)

    preference_order = (requested_model,) + tuple(
        candidate for candidate in FREE_FLASH_MODELS if candidate != requested_model
    )
    for model_name in preference_order:
        if model_name in available_models:
            _resolved_model_id = model_name
            _resolved_model_setting = requested_model
            return model_name

    raise GeminiModelUnavailableError(
        "No supported free-tier Gemini Flash text model is available to this API account."
    )


def _translate_prompt(text: str) -> str:
    return (
        "You are a professional English-to-Hindi translator for educational video dubbing.\n\n"
        "Translate the supplied English text into natural, fluent Hindi written in Devanagari.\n\n"
        "Rules:\n"
        "- Translate meaning faithfully.\n"
        "- Do not summarize.\n"
        "- Do not explain.\n"
        "- Do not add information.\n"
        "- Preserve names, numbers, abbreviations, technical terms and proper nouns.\n"
        "- Preserve the original tone, intent and educational context.\n"
        "- Prefer natural spoken Hindi suitable for voice dubbing.\n"
        "- Return ONLY the Hindi translation.\n"
        "- Do not return English.\n"
        "- Do not transliterate Hindi into Latin characters.\n\n"
        f"English text:\n{text}"
    )


def _translation_response_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "id": types.Schema(type=types.Type.INTEGER),
                "translated_text": types.Schema(type=types.Type.STRING),
            },
            required=["id", "translated_text"],
        ),
    )


def _get_batch_size() -> int:
    raw_batch_size = os.getenv(
        "GEMINI_TRANSLATION_BATCH_SIZE",
        str(DEFAULT_TRANSLATION_BATCH_SIZE),
    )
    try:
        batch_size = int(raw_batch_size)
    except ValueError:
        raise ValueError("GEMINI_TRANSLATION_BATCH_SIZE must be an integer.") from None
    if not 1 <= batch_size <= MAX_TRANSLATION_BATCH_SIZE:
        raise ValueError(
            f"GEMINI_TRANSLATION_BATCH_SIZE must be between 1 and {MAX_TRANSLATION_BATCH_SIZE}."
        )
    return batch_size


def _build_batch_prompt(segments: list[dict[str, Any]]) -> str:
    segment_payload = [
        {"id": segment["id"], "text": segment["text"]}
        for segment in segments
    ]
    return (
        "You are a professional English-to-Hindi translator for educational video dubbing.\n\n"
        "Translate every supplied English segment into natural, fluent Hindi written in Devanagari.\n\n"
        "Rules:\n"
        "- Translate meaning faithfully; do not summarize, explain, add, or omit information.\n"
        "- Preserve names, numbers, abbreviations, technical terms, proper nouns, tone, and educational context.\n"
        "- Prefer natural spoken Hindi suitable for voice dubbing.\n"
        "- Return ONLY a JSON array of objects with exactly the input id and translated_text fields.\n"
        "- Return one object for every input id, exactly once; do not reorder IDs.\n"
        "- Every translated_text must be Hindi in Devanagari, with no English explanation.\n"
        "- Do not transliterate Hindi into Latin characters.\n\n"
        f"English segments JSON:\n{json.dumps(segment_payload, ensure_ascii=False)}"
    )


def _raise_api_error(exc: errors.APIError) -> None:
    status_code = getattr(exc, "code", None)
    error_status = str(getattr(exc, "status", "")).upper()
    error_message = str(getattr(exc, "message", "")).lower()
    if (
        status_code in {401, 403}
        or error_status == "UNAUTHENTICATED"
        or "api key not valid" in error_message
    ):
        raise GeminiAuthenticationError("Gemini rejected the configured API key.") from None
    if status_code == 429 or error_status == "RESOURCE_EXHAUSTED":
        raise GeminiQuotaError("Gemini quota or rate limit was reached.") from None
    if status_code == 404 or error_status == "NOT_FOUND":
        raise GeminiModelUnavailableError(
            "The configured Gemini model is unavailable to this API account."
        ) from None
    if status_code in {408, 504} or error_status == "DEADLINE_EXCEEDED":
        raise GeminiTimeoutError("The Gemini API request timed out.") from None
    raise GeminiNetworkError(
        f"The Gemini API request failed with status {status_code}.",
        api_code=status_code,
        api_status=error_status or None,
    ) from None


def _raise_request_error(exc: Exception) -> None:
    if isinstance(exc, errors.APIError):
        _raise_api_error(exc)
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        raise GeminiTimeoutError("The Gemini API request timed out.") from None
    if isinstance(exc, httpx.RequestError):
        raise GeminiNetworkError("The Gemini API could not be reached.") from None
    raise exc


def _translate_batch(
    segments: list[dict[str, Any]],
    target_language: str,
) -> dict[int, str]:
    if target_language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported target language: {target_language}")

    expected_ids: set[int] = set()
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Each translation segment must be an object.")
        segment_id = segment.get("id")
        if isinstance(segment_id, bool) or not isinstance(segment_id, int):
            raise ValueError("Every segment must have an integer id.")
        if segment_id in expected_ids:
            raise ValueError("Segment IDs must be unique.")
        expected_ids.add(segment_id)
        if not isinstance(segment.get("text"), str) or not segment["text"].strip():
            raise ValueError(f"Segment {segment_id} has empty English text.")

    if not segments:
        return {}

    client = get_translation_client()
    selected_model = resolve_translation_model(client)
    for attempt in range(TRANSIENT_RETRY_LIMIT):
        try:
            response = client.models.generate_content(
                model=selected_model,
                contents=_build_batch_prompt(segments),
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=min(8192, 1024 + len(segments) * 512),
                    response_mime_type="application/json",
                    response_schema=_translation_response_schema(),
                ),
            )
            break
        except errors.APIError as exc:
            transient_unavailable = (
                getattr(exc, "code", None) == 503
                and str(getattr(exc, "status", "")).upper() == "UNAVAILABLE"
            )
            if transient_unavailable and attempt < TRANSIENT_RETRY_LIMIT - 1:
                time.sleep(attempt + 1)
                continue
            _raise_api_error(exc)
        except Exception as exc:
            _raise_request_error(exc)

    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise GeminiResponseError("Gemini returned no translation candidates.")
    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason == types.FinishReason.MAX_TOKENS:
        raise GeminiResponseError("The Gemini translation response was truncated.")
    if finish_reason != types.FinishReason.STOP:
        raise GeminiResponseError("Gemini returned an invalid or blocked response.")

    try:
        raw_response = response.text
        translated_items = json.loads(raw_response)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        raise GeminiResponseError("Gemini returned malformed translation JSON.") from None
    if not isinstance(translated_items, list):
        raise GeminiResponseError("Gemini translation response must be a JSON array.")

    translations: dict[int, str] = {}
    for item in translated_items:
        if not isinstance(item, dict):
            raise GeminiResponseError("Gemini returned an invalid translation item.")
        segment_id = item.get("id")
        translated_text = item.get("translated_text")
        if isinstance(segment_id, bool) or not isinstance(segment_id, int):
            raise GeminiResponseError("Gemini returned an invalid segment ID.")
        if segment_id not in expected_ids:
            raise GeminiResponseError("Gemini returned an unexpected segment ID.")
        if segment_id in translations:
            raise GeminiResponseError("Gemini returned a duplicate segment ID.")
        if not isinstance(translated_text, str) or not translated_text.strip():
            raise GeminiResponseError(f"Gemini returned an empty translation for segment {segment_id}.")
        normalized_text = translated_text.strip()
        if not any("\u0900" <= character <= "\u097f" for character in normalized_text):
            raise GeminiResponseError(f"Gemini response for segment {segment_id} was not Hindi in Devanagari.")
        translations[segment_id] = normalized_text

    if translations.keys() != expected_ids:
        missing_ids = expected_ids - translations.keys()
        raise GeminiResponseError(
            f"Gemini omitted translations for {len(missing_ids)} segment(s)."
        )
    return translations


def translate_text(text: str, target_language: str) -> str:
    """Translate one English text using the structured batch interface."""
    if target_language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported target language: {target_language}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot translate empty text.")
    return _translate_batch([{"id": 1, "text": text.strip()}], target_language)[1]


def translate_segments(
    segments: list[dict[str, Any]],
    target_language: str,
) -> list[dict[str, Any]]:
    """Translate segments in configurable structured batches, preserving fields."""
    if target_language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported target language: {target_language}")
    if not isinstance(segments, list):
        raise ValueError("segments must be provided as a list.")

    batch_size = _get_batch_size()
    translated_segments: list[dict[str, Any]] = []
    for start_index in range(0, len(segments), batch_size):
        batch = segments[start_index : start_index + batch_size]
        batch_translations = _translate_batch(batch, target_language)
        translated_segments.extend(
            {**segment, "translated_text": batch_translations[segment["id"]]}
            for segment in batch
        )

    return translated_segments