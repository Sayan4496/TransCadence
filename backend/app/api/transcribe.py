from pathlib import Path
import json

from fastapi import APIRouter, HTTPException

from backend.app.services.asr_service import transcribe_audio


router = APIRouter(prefix="/api/jobs", tags=["Transcription"])

AUDIO_DIR = Path("storage/audio")
TRANSCRIPT_DIR = Path("storage/transcripts")

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/{job_id}/transcribe")
def transcribe_job(job_id: str):
    audio_path = AUDIO_DIR / f"{job_id}.wav"

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Extracted audio not found. Run audio extraction first.",
        )

    try:
        result = transcribe_audio(audio_path)

        transcript_path = TRANSCRIPT_DIR / f"{job_id}.json"

        with transcript_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "job_id": job_id,
            "status": "transcribed",
            "transcript_file": str(transcript_path),
            "transcript": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {exc}",
        )