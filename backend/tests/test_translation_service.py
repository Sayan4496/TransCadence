import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
from google.genai import errors, types
import httpx
import pytest

from backend.app.api import translate as translate_api
from backend.app.config import gemini_api_key_status
from backend.app.main import app
from backend.app.services import translation_service


RUN_REAL_GEMINI = os.getenv("TRANSCADENCE_RUN_REAL_GEMINI") == "1"
REAL_JOB_ID = "73719fc4-3481-42c4-8fd6-51cf246eba1d"


def _gemini_response(
    text: str | None,
    finish_reason: types.FinishReason = types.FinishReason.STOP,
) -> object:
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
    )


def _batch_response(translations: list[dict[str, object]]) -> object:
    return _gemini_response(json.dumps(translations, ensure_ascii=False))


def _api_error(code: int, status: str, message: str = "") -> errors.APIError:
    return errors.APIError(
        code,
        {"error": {"code": code, "status": status, "message": message}},
    )


def _mock_client(monkeypatch: pytest.MonkeyPatch, generate_content) -> None:
    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda **_: [
                SimpleNamespace(
                    name=f"models/{translation_service.MODEL_ID}",
                    supported_actions=["generateContent"],
                )
            ],
            generate_content=generate_content,
        )
    )
    monkeypatch.setattr(
        translation_service,
        "get_translation_client",
        lambda: fake_client,
    )


@pytest.fixture(autouse=True)
def reset_model_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(translation_service, "_resolved_model_id", None)
    monkeypatch.setattr(translation_service, "_resolved_model_setting", None)


def test_hindi_is_the_only_supported_language() -> None:
    assert translation_service.LANGUAGE_NAMES == {"hi": "Hindi"}
    with pytest.raises(ValueError, match="Unsupported target language"):
        translation_service.translate_text("Hello", "bn")
    with pytest.raises(ValueError, match="Unsupported target language"):
        translation_service.translate_segments([], "fr")


def test_service_has_no_hugging_face_or_local_model_dependency() -> None:
    source = Path(translation_service.__file__).read_text(encoding="utf-8").lower()

    for forbidden in ("huggingface_hub", "llama", "mbart", "transformers"):
        assert forbidden not in source


def test_missing_gemini_api_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(translation_service, "_client", None)

    with pytest.raises(translation_service.MissingGeminiAPIKeyError):
        translation_service.get_translation_client()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "missing"), ("local-test-placeholder", "present")],
)
def test_safe_key_status_reports_only_presence(
    value: str | None,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if value is None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GEMINI_API_KEY", value)

    assert gemini_api_key_status() == expected


def test_client_uses_environment_key_and_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_key = "test-key-not-a-credential"
    constructor_calls: list[dict[str, object]] = []
    client = object()

    def fake_client(**kwargs: object) -> object:
        constructor_calls.append(kwargs)
        return client

    monkeypatch.setenv("GEMINI_API_KEY", dummy_key)
    monkeypatch.setattr(translation_service, "_client", None)
    monkeypatch.setattr(translation_service.genai, "Client", fake_client)

    assert translation_service.get_translation_client() is client
    assert translation_service.get_translation_client() is client
    assert constructor_calls == [
        {
            "api_key": dummy_key,
            "http_options": types.HttpOptions(timeout=60_000),
        }
    ]


def test_model_falls_back_only_to_available_free_flash_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda **_: [
                SimpleNamespace(
                    name="models/gemini-3.8-flash",
                    supported_actions=["generateContent"],
                ),
                SimpleNamespace(
                    name="models/gemini-3.1-flash-lite-image",
                    supported_actions=["generateContent"],
                ),
            ]
        )
    )
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    assert translation_service.resolve_translation_model(client) == "gemini-3.8-flash"


def test_unavailable_configured_model_without_free_flash_fallback_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SimpleNamespace(
        models=SimpleNamespace(list=lambda **_: [])
    )
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    with pytest.raises(translation_service.GeminiModelUnavailableError):
        translation_service.resolve_translation_model(client)


def test_successful_mocked_hindi_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def generate_content(**kwargs: object) -> object:
        calls.append(kwargs)
        return _batch_response([{"id": 1, "translated_text": "नमस्ते सभी को।"}])

    _mock_client(monkeypatch, generate_content)
    result = translation_service.translate_text("Hello everyone.", "hi")

    assert result == "नमस्ते सभी को।"
    assert calls[0]["model"] == "gemini-3.1-flash-lite"
    assert calls[0]["config"].temperature == 0.1
    assert calls[0]["contents"].startswith(
        "You are a professional English-to-Hindi translator"
    )
    assert "Return ONLY a JSON array" in calls[0]["contents"]
    assert calls[0]["config"].response_mime_type == "application/json"


def test_transient_unavailable_error_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def generate_content(**_: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _api_error(503, "UNAVAILABLE", "temporary")
        return _batch_response(
            [{"id": 1, "translated_text": "निरंतरता सफलता की कुंजी है।"}]
        )

    _mock_client(monkeypatch, generate_content)
    monkeypatch.setattr(translation_service.time, "sleep", lambda _: None)

    translated = translation_service.translate_text("Consistency brings success.", "hi")

    assert translated == "निरंतरता सफलता की कुंजी है।"
    assert calls == 2


@pytest.mark.parametrize(
    ("code", "status", "message", "error_type"),
    [
        (
            400,
            "INVALID_ARGUMENT",
            "API key not valid",
            translation_service.GeminiAuthenticationError,
        ),
        (401, "UNAUTHENTICATED", "", translation_service.GeminiAuthenticationError),
        (403, "PERMISSION_DENIED", "", translation_service.GeminiAuthenticationError),
        (429, "RESOURCE_EXHAUSTED", "", translation_service.GeminiQuotaError),
        (404, "NOT_FOUND", "", translation_service.GeminiModelUnavailableError),
        (504, "DEADLINE_EXCEEDED", "", translation_service.GeminiTimeoutError),
        (500, "INTERNAL", "", translation_service.GeminiNetworkError),
    ],
)
def test_gemini_api_failures_are_mapped_to_typed_errors(
    code: int,
    status: str,
    message: str,
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_error = _api_error(code, status, message)
    _mock_client(
        monkeypatch,
        lambda **_: (_ for _ in ()).throw(api_error),
    )

    with pytest.raises(error_type):
        translation_service.translate_text("Hello.", "hi")


@pytest.mark.parametrize(
    ("network_error", "error_type"),
    [
        (httpx.ReadTimeout("timeout"), translation_service.GeminiTimeoutError),
        (httpx.ConnectError("offline"), translation_service.GeminiNetworkError),
    ],
)
def test_timeout_and_network_errors_are_typed(
    network_error: Exception,
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_client(
        monkeypatch,
        lambda **_: (_ for _ in ()).throw(network_error),
    )

    with pytest.raises(error_type):
        translation_service.translate_text("Hello.", "hi")


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(
            text=None,
            candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
        ),
        SimpleNamespace(
            text="  ",
            candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
        ),
        SimpleNamespace(text="नमस्ते", candidates=[]),
        SimpleNamespace(text="नमस्ते", candidates=[SimpleNamespace()]),
    ],
)
def test_empty_or_missing_gemini_response_is_rejected(
    response: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_client(monkeypatch, lambda **_: response)

    with pytest.raises(translation_service.GeminiResponseError):
        translation_service.translate_text("Hello.", "hi")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_gemini_response("[]", types.FinishReason.MAX_TOKENS), "truncated"),
        (_gemini_response("[]", types.FinishReason.SAFETY), "invalid or blocked"),
        (
            _batch_response([{"id": 1, "translated_text": "Hello everyone."}]),
            "not Hindi in Devanagari",
        ),
    ],
)
def test_truncated_invalid_or_non_hindi_response_is_rejected(
    response: object,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_client(monkeypatch, lambda **_: response)

    with pytest.raises(translation_service.GeminiResponseError, match=message):
        translation_service.translate_text("Hello.", "hi")

def test_multiple_segments_preserve_source_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_client(
        monkeypatch,
        lambda **_: _batch_response(
            [
                {"id": 1, "translated_text": "नमस्ते।"},
                {"id": 2, "translated_text": "आज आपका स्वागत है।"},
            ]
        ),
    )
    segments = [
        {"id": 1, "start": 1.25, "end": 2.5, "duration": 1.25, "text": "Hello."},
        {"id": 2, "start": 2.5, "end": 4.0, "duration": 1.5, "text": "Welcome."},
    ]

    translated = translation_service.translate_segments(segments, "hi")

    assert len(translated) == 2
    for source, output in zip(segments, translated):
        for field in ("id", "start", "end", "duration", "text"):
            assert output[field] == source[field]
        assert "translated_text" in output
    assert [item["translated_text"] for item in translated] == [
        "नमस्ते।",
        "आज आपका स्वागत है।",
    ]


def test_one_gemini_request_translates_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[list[dict[str, object]]] = []

    def generate_content(**kwargs: object) -> object:
        payload = json.loads(kwargs["contents"].split("English segments JSON:\n", 1)[1])
        requests.append(payload)
        return _batch_response(
            [{"id": segment["id"], "translated_text": f"हिंदी {segment['id']}"} for segment in payload]
        )

    _mock_client(monkeypatch, generate_content)
    segments = [{"id": index, "start": index, "end": index + 1, "text": f"text {index}"} for index in range(1, 4)]

    translated = translation_service.translate_segments(segments, "hi")

    assert len(requests) == 1
    assert len(requests[0]) == 3
    assert len(translated) == 3


def test_large_transcript_uses_multiple_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_TRANSLATION_BATCH_SIZE", "2")
    requests: list[list[dict[str, object]]] = []

    def generate_content(**kwargs: object) -> object:
        payload = json.loads(kwargs["contents"].split("English segments JSON:\n", 1)[1])
        requests.append(payload)
        return _batch_response(
            [{"id": segment["id"], "translated_text": f"हिंदी {segment['id']}"} for segment in payload]
        )

    _mock_client(monkeypatch, generate_content)
    segments = [{"id": index, "start": index, "end": index + 1, "text": f"text {index}"} for index in range(1, 6)]

    translated = translation_service.translate_segments(segments, "hi")

    assert [len(batch) for batch in requests] == [2, 2, 1]
    assert [segment["id"] for segment in translated] == [1, 2, 3, 4, 5]


def test_thirteen_segments_use_two_gemini_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_TRANSLATION_BATCH_SIZE", "8")
    requests: list[list[dict[str, object]]] = []

    def generate_content(**kwargs: object) -> object:
        payload = json.loads(kwargs["contents"].split("English segments JSON:\n", 1)[1])
        requests.append(payload)
        return _batch_response(
            [{"id": segment["id"], "translated_text": f"हिंदी {segment['id']}"} for segment in payload]
        )

    _mock_client(monkeypatch, generate_content)
    segments = [{"id": index, "start": index, "end": index + 1, "text": f"text {index}"} for index in range(1, 14)]

    result = translation_service.translate_segments(segments, "hi")

    assert [len(batch) for batch in requests] == [8, 5]
    assert len(requests) == 2
    assert len(result) == 13


@pytest.mark.parametrize(
    "response_items",
    [
        [{"id": 1, "translated_text": "एक"}],
        [
            {"id": 1, "translated_text": "एक"},
            {"id": 1, "translated_text": "दो"},
        ],
        [
            {"id": 1, "translated_text": "एक"},
            {"id": 2, "translated_text": "दो"},
            {"id": 99, "translated_text": "गलत"},
        ],
        [
            {"id": 1, "translated_text": "एक"},
            {"id": 2, "translated_text": "   "},
        ],
    ],
)
def test_incomplete_duplicate_unexpected_or_empty_batch_response_is_rejected(
    response_items: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_client(monkeypatch, lambda **_: _batch_response(response_items))
    segments = [{"id": 1, "text": "one"}, {"id": 2, "text": "two"}]

    with pytest.raises(translation_service.GeminiResponseError):
        translation_service.translate_segments(segments, "hi")


def test_malformed_batch_json_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_client(monkeypatch, lambda **_: _gemini_response("not JSON"))

    with pytest.raises(translation_service.GeminiResponseError, match="malformed"):
        translation_service.translate_segments([{"id": 1, "text": "Hello"}], "hi")


@pytest.mark.parametrize(
    "segments",
    [
        [{"text": "Missing ID"}],
        [{"id": 1, "text": "one"}, {"id": 1, "text": "duplicate"}],
    ],
)
def test_missing_or_duplicate_input_ids_are_rejected(
    segments: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError):
        translation_service.translate_segments(segments, "hi")


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_detail"),
    [
        (
            translation_service.MissingGeminiAPIKeyError("test"),
            503,
            "not configured",
        ),
        (
            translation_service.GeminiAuthenticationError("test"),
            502,
            "Gemini rejected",
        ),
        (
            translation_service.GeminiQuotaError("test"),
            429,
            "quota or rate limit",
        ),
        (
            translation_service.GeminiModelUnavailableError("test"),
            503,
            "unavailable",
        ),
    ],
)
def test_api_returns_safe_gemini_failures_without_secrets(
    failure: Exception,
    expected_status: int,
    expected_detail: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_sentinel = "NEVER_LOG_THIS_TEST_SENTINEL"
    job_id = str(uuid4())
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    (transcript_dir / f"{job_id}.json").write_text(
        json.dumps({"language": "en", "segments": [{"id": 1, "text": "Hello."}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(translate_api, "TRANSCRIPT_DIR", transcript_dir)
    monkeypatch.setenv("GEMINI_API_KEY", secret_sentinel)

    def raise_failure(*_: object, **__: object) -> list[dict[str, object]]:
        raise type(failure)(secret_sentinel)

    monkeypatch.setattr(translate_api, "translate_segments", raise_failure)
    response = TestClient(app).post(
        f"/api/jobs/{job_id}/translate",
        json={"target_language": "hi"},
    )

    assert response.status_code == expected_status
    assert expected_detail in response.json()["detail"]
    assert secret_sentinel not in response.text
    assert secret_sentinel not in caplog.text


@pytest.mark.skipif(
    not RUN_REAL_GEMINI or gemini_api_key_status() != "present",
    reason="Set GEMINI_API_KEY in backend/.env and TRANSCADENCE_RUN_REAL_GEMINI=1.",
)
def test_real_gemini_translation_and_existing_job() -> None:
    tiny_translation = translation_service.translate_text(
        "Consistency is the key to success.",
        "hi",
    )
    assert any("\u0900" <= character <= "\u097f" for character in tiny_translation)
    assert "Consistency" not in tiny_translation
    print("gemini_tiny_request=status:success")
    print(f"hindi_translation={tiny_translation}")

    source_path = Path("storage/transcripts") / f"{REAL_JOB_ID}.json"
    if not source_path.is_file():
        pytest.skip("The requested English job transcript is not present.")

    response = TestClient(app).post(
        f"/api/jobs/{REAL_JOB_ID}/translate",
        json={"target_language": "hi"},
    )
    assert response.status_code == 200, response.text
    translated_path = Path("storage/transcripts") / f"{REAL_JOB_ID}_hi.json"
    assert translated_path.is_file()
    with translated_path.open("r", encoding="utf-8") as translated_file:
        translated = json.load(translated_file)
    assert translated["target_language"] == "hi"
    assert translated["segments"]
    assert all(segment.get("translated_text") for segment in translated["segments"])
    assert all(
        any("\u0900" <= character <= "\u097f" for character in segment["translated_text"])
        for segment in translated["segments"]
    )
    print("existing_job_translation=status:success")
    print(f"translated_segment_count={len(translated['segments'])}")