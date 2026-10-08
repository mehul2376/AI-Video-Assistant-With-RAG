import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from pydub import AudioSegment

# Optional: faster-whisper is ~4x faster than openai-whisper on CPU.
# Install with:  pip install faster-whisper
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import whisper
except ImportError:
    whisper = None

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25
SARVAM_WORKERS = 4          # parallel requests; lower this if you hit 429 errors
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"

_model = None


# ─── Settings are read when used (so .env is always loaded by then) ─────────────
def _whisper_model_name() -> str:
    return os.getenv("WHISPER_MODEL", "small")


def _sarvam_key() -> str | None:
    return os.getenv("SARVAM_API_KEY")


def _sarvam_model() -> str:
    return os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")


# ─── Whisper (english) ──────────────────────────────────────────────────────────
def load_model():
    global _model
    if _model is None:
        name = _whisper_model_name()
        if WhisperModel is not None:
            try:
                print(f"Loading faster-whisper model: {name} ...")
                _model = WhisperModel(name, device="cpu", compute_type="int8")
            except Exception as e:
                print(f"Warning: faster-whisper failed to load ({e}), falling back to openai-whisper...")
                if whisper is not None:
                    _model = whisper.load_model(name)
                else:
                    raise RuntimeError(f"Failed to load faster-whisper ({e}) and openai-whisper is not installed.") from e
        elif whisper is not None:
            print(f"Loading Whisper model: {name} ...")
            _model = whisper.load_model(name)
        else:
            raise RuntimeError("Install a Whisper backend: pip install faster-whisper or openai-whisper")
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()

    if WhisperModel is not None and isinstance(model, WhisperModel):
        try:
            segments, _ = model.transcribe(
                chunk_path,
                language="en",          # skip language detection = faster
                beam_size=1,            # greedy decoding = faster
                vad_filter=True,        # skip silence
            )
            return " ".join(seg.text.strip() for seg in segments)
        except Exception as e:
            print(f"faster-whisper transcription error: {e}, attempting fallback if available.")

    if hasattr(model, "transcribe"):
        # openai-whisper fallback
        result = model.transcribe(
            chunk_path,
            task="transcribe",
            language="en",
            fp16=False,                 # CPU has no fp16 (avoids warning)
        )
        return result.get("text", "")

    return ""


# ─── Sarvam (hinglish -> english) ───────────────────────────────────────────────
def _send_to_sarvam(piece_path: str, retries: int = 3) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript."""
    key = _sarvam_key()
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    headers = {"api-subscription-key": key}

    for attempt in range(retries):
        with open(piece_path, "rb") as f:
            files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
            data = {"model": _sarvam_model(), "with_diarization": "false"}
            response = requests.post(
                SARVAM_STT_TRANSLATE_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )

        if response.ok:
            return response.json().get("transcript", "")

        # Retry on rate limit / temporary server errors
        if response.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue

        print(f"\n[ERROR] Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return ""


def _sarvam_piece_job(args) -> str:
    chunk_path, i, piece = args
    piece_path = f"{chunk_path}_sv_{i}.wav"
    piece.export(piece_path, format="wav")
    try:
        return _send_to_sarvam(piece_path)
    finally:
        if os.path.exists(piece_path):
            try:
                os.remove(piece_path)
            except Exception:
                pass


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """Split chunk into 25s pieces, send them in PARALLEL, join in order."""
    if not _sarvam_key():
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    jobs = [
        (chunk_path, i, audio[start:start + piece_ms])
        for i, start in enumerate(range(0, len(audio), piece_ms))
    ]
    print(f"  -> Sarvam: sending {len(jobs)} pieces ({SARVAM_WORKERS} at a time)...")

    # executor.map keeps the original order
    with ThreadPoolExecutor(max_workers=SARVAM_WORKERS) as ex:
        texts = list(ex.map(_sarvam_piece_job, jobs))

    return " ".join(t.strip() for t in texts if t.strip())




# ─── Routing ────────────────────────────────────────────────────────────────────
def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    - english  -> Whisper (local model)
    - hinglish -> Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    parts = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        parts.append(transcribe_chunk(chunk, language=language).strip())

    print("Transcription complete.")
    return " ".join(p for p in parts if p)