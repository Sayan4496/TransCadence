# TransCadence Backend

## Gemini configuration

Set `GEMINI_API_KEY` in `backend/.env`. Keep the real value local; do not copy it into source, tests, frontend files, logs, or API requests. `backend/.env.example` contains placeholders only. The default model is `gemini-3.1-flash-lite`; `GEMINI_MODEL` can select another documented free-tier Flash text model supported by the account.

The server loads `backend/.env` at startup. It reports only whether the key is present or missing.

## Install and run

From the repository root, install the Google GenAI SDK and dotenv loader into the project environment if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install google-genai python-dotenv
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger.

## Run the backend with Docker

From the repository root, build and start only the backend:

```powershell
docker compose build backend
docker compose up -d backend
docker compose ps
docker compose logs backend --tail 100
```

The backend is available at `http://127.0.0.1:8000`, with Swagger at
`http://127.0.0.1:8000/docs`. Uploads and generated files persist in the root
`storage` directory, and downloaded speech models persist in a named Docker
volume. The container reads Gemini settings from `backend/.env`; that file is
not included in the image.

Keep running the existing React frontend locally, outside Docker. Its Vite
development proxy sends `/api` requests to `http://127.0.0.1:8000`.

## API flow

1. Upload a video with `POST /api/upload` (MP4/WebM, up to 120 seconds). The existing endpoint validates, extracts audio, and writes the English transcript.
2. Optionally translate with `POST /api/jobs/{job_id}/translate` and body `{"target_language":"hi"}`.
3. Run the complete synchronous pipeline with `POST /api/jobs/{job_id}/process`.
4. Check `GET /api/jobs/{job_id}/status`.
5. Download the completed MP4 with `GET /api/jobs/{job_id}/download`.

The completed output is stored under `storage/outputs/{job_id}/hindi_dubbed.mp4`. The server returns a download URL rather than exposing its absolute filesystem path.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```
