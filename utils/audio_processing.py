import os
import shutil

import yt_dlp
from pydub import AudioSegment
from yt_dlp.utils import DownloadError

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ─── YouTube download ───────────────────────────────────────────────────────────
def _js_runtimes() -> dict:
    """Use whichever JavaScript runtime is installed (needed by yt-dlp for YouTube)."""
    runtimes = {}
    if shutil.which("deno"):
        runtimes["deno"] = {}
    if shutil.which("node"):
        runtimes["node"] = {}
    return runtimes


def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube and return the path of the .wav file."""
    base_opts = {
        "format": "bestaudio/best",
        # use video id (not title) -> no weird characters in file names
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": False,   # keep warnings visible so problems are not hidden
    }

    runtimes = _js_runtimes()
    if runtimes:
        base_opts["js_runtimes"] = runtimes
    else:
        print(
            "[WARNING] No JavaScript runtime found. Install Deno (winget install DenoLand.Deno) "
            "or Node.js, then restart your terminal."
        )


    # Try the normal way first, then fallbacks if YouTube returns 403
    attempts = [
        {},
        {"extractor_args": {"youtube": {"player_client": ["web_embedded", "web", "tv"]}}},
        {"cookiesfrombrowser": ("edge",)},   # change to "chrome" / "firefox" if needed
    ]

    last_error = None
    for extra in attempts:
        try:
            with yt_dlp.YoutubeDL({**base_opts, **extra}) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded = ydl.prepare_filename(info)
            # After FFmpegExtractAudio the extension is always .wav
            return os.path.splitext(downloaded)[0] + ".wav"
        except DownloadError as e:
            last_error = e
            safe_err = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"Download attempt failed ({', '.join(extra) or 'default'}): {safe_err}")


    raise RuntimeError(
        "YouTube download failed. Update yt-dlp (pip install -U \"yt-dlp[default]\"), "
        "install Deno/Node.js, or use a local video file instead."
    ) from last_error


# ─── Conversion ─────────────────────────────────────────────────────────────────
def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to mono 16 kHz WAV (small + fast to transcribe)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if input_path.endswith("_converted.wav"):
        return input_path

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16 kHz mono
    audio.export(output_path, format="wav")
    return output_path


# ─── Chunking ───────────────────────────────────────────────────────────────────
def chunk_audio(wav_path: str, chunk_length_minutes: int = 10) -> list:
    audio = AudioSegment.from_file(wav_path)
    chunk_ms = chunk_length_minutes * 60 * 1000

    base = os.path.splitext(wav_path)[0]
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{base}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


# ─── Entry point ────────────────────────────────────────────────────────────────
def process_input(source: str) -> list:
    source = source.strip().strip('"').strip("'")   # removes quotes if the path was pasted with them

    if source.startswith(("http://", "https://")):
        print("Detected YouTube URL, downloading audio..")
        wav_path = download_youtube_audio(source)
        # shrink to mono 16 kHz so chunks are small and transcription is faster
        wav_path = convert_to_wav(wav_path)
    else:
        print("Detected local file, converting to wav..")
        wav_path = convert_to_wav(source)

    print("Chunking audio..")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks