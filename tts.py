"""Offline text-to-speech using Piper TTS (neural, local inference)."""

import io
import os
import wave

_voice = None

# Model paths — downloaded once, reused
MODEL_DIR = os.path.join(os.path.dirname(__file__), "piper_models")
MODEL_NAME = "en_US-lessac-medium"
MODEL_FILE = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx")
CONFIG_FILE = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx.json")

# HuggingFace URLs for auto-download
_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
_MODEL_URL = f"{_HF_BASE}/{MODEL_NAME}.onnx"
_CONFIG_URL = f"{_HF_BASE}/{MODEL_NAME}.onnx.json"


def _download_model():
    """Download the Piper voice model if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_FILE) and os.path.exists(CONFIG_FILE):
        return

    import urllib.request
    for url, path in [(_MODEL_URL, MODEL_FILE), (_CONFIG_URL, CONFIG_FILE)]:
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)
            print(f"  Saved to {path}")


def _get_voice():
    """Load the Piper voice model (lazy singleton)."""
    global _voice
    if _voice is None:
        _download_model()
        from piper import PiperVoice
        _voice = PiperVoice.load(MODEL_FILE)
    return _voice


def synthesize_recap(bullets: list[str]) -> bytes:
    """Convert recap bullets to WAV audio bytes.

    Args:
        bullets: List of recap bullet strings (typically 3).

    Returns:
        WAV file content as bytes, playable by st.audio().
    """
    voice = _get_voice()

    # Build narration text with natural pacing
    parts = ["Here are your key takeaways."]
    for i, bullet in enumerate(bullets, 1):
        parts.append(f"Point {i}. {bullet}")
    text = " ".join(parts)

    # Synthesize to in-memory WAV
    buf = io.BytesIO()
    with wave.open(buf, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(voice.config.sample_rate)
        for chunk in voice.synthesize(text):
            wav_file.writeframes(chunk.audio_int16_bytes)

    return buf.getvalue()
