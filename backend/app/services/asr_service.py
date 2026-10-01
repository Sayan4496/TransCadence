from pathlib import Path

from faster_whisper import WhisperModel


MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


_model = None


def get_model() -> WhisperModel:
    global _model

    if _model is None:
        _model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )

    return _model


def transcribe_audio(audio_path: Path) -> dict:
    model = get_model()

    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        vad_filter=True,
        word_timestamps=True,
    )

    transcript_segments = []

    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()

        if not text:
            continue

        words = [
            {
                "word": word.word,
                "start": round(word.start, 3),
                "end": round(word.end, 3),
            }
            for word in (segment.words or [])
            if word.word.strip()
        ]

        transcript_segments.append(
            {
                "id": index,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "duration": round(
                    segment.end - segment.start,
                    3,
                ),
                "text": text,
                "words": words,
            }
        )

    return {
        "language": info.language,
        "language_probability": round(
            info.language_probability,
            4,
        ),
        "segments": transcript_segments,
    }