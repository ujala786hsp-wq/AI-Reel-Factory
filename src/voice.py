"""Module 4 — Voice (narration).

Contract:
    what it does : synthesizes narration audio from a script body.
    input        : script_body (str); output dir.
    output       : (audio_path, duration_seconds).
    depends on   : Gemini Developer API TTS → Google Cloud TTS (Chirp 3 HD) → edge-tts → Kokoro
                   (rule 11: every engine has a fallback behind it).

Default engine is **Gemini TTS** on the free `gemini-3.1-flash-tts-preview` model, voice
**Zubenelgenubi** ("Casual"). The chain then falls through to Chirp 3 HD, edge-tts and Kokoro,
resolved by name at call time so a missing key or a withdrawn preview model just advances a link.
Pick the primary via VOICE_ENGINE (gemini|google|edge|kokoro).

**Expressive delivery is per-engine**:
  · `gemini`  — promptable: honours VOICE_STYLE_PROMPT and inline style tags.
  · `google`  — Chirp 3 HD reads `[pause]` tags through its `markup` input field.
  · `edge`/`kokoro` — no tag support; every tag is stripped before synthesis.

**The edge-tts stream now has a 60-second timeout** so a hanging network call can no longer
stall the entire GitHub Actions run.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re

from functools import lru_cache

import requests

from src import config

log = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")

_VOICE = config.get("VOICE", "en-IN-PrabhatNeural")
_RATE = config.get("VOICE_RATE", "+0%")
_TICKS_PER_SECOND = 1e7

_KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
_KOKORO_MODEL = "kokoro-v1.0.int8.onnx"
_KOKORO_VOICES = "voices-v1.0.bin"

_GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

_GEMINI_TTS_RATE = 24000
_GEMINI_MAX_FIELD_BYTES = 4000
_GEMINI_MAX_TOTAL_BYTES = 8000
_GEMINI_TTS_PRIMARY = "gemini-3.1-flash-tts-preview"
_GEMINI_TTS_STABLE = "gemini-2.5-flash-preview-tts"
_TRANSIENT_MARKERS = ("503", "unavailable", "500", "internal", "504", "deadline", "overloaded")

# Timeout for the whole edge-tts stream, in seconds. If Microsoft's endpoint stalls, this
# raises instead of hanging forever.
_EDGE_TIMEOUT = 60.0


def _is_transient(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _TRANSIENT_MARKERS)


_DEFAULT_STYLE_PROMPT = (
    "You are a warm, cinematic storyteller narrating an emotional Hindi romance story. "
    "Speak slowly and clearly, with deliberate pauses before the emotional turns. "
    "Let the words carry the feeling — never rush the ending."
)


def _audio_filename(script_body: str, ext: str = ".mp3") -> str:
    digest = hashlib.sha1(script_body.encode("utf-8")).hexdigest()[:12]
    return f"narration_{digest}{ext}"


def _download(url: str, dest: str) -> None:
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)


def _ensure_kokoro_models() -> tuple[str, str]:
    cache = config.get("KOKORO_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "kokoro"))
    os.makedirs(cache, exist_ok=True)
    paths = []
    for name in (_KOKORO_MODEL, _KOKORO_VOICES):
        dest = os.path.join(cache, name)
        if not (os.path.exists(dest) and os.path.getsize(dest) > 1000):
            log.info("voice: downloading Kokoro asset %s …", name)
            _download(_KOKORO_BASE + name, dest)
        paths.append(dest)
    return paths[0], paths[1]


@lru_cache(maxsize=1)
def _kokoro():
    from kokoro_onnx import Kokoro
    model, voices = _ensure_kokoro_models()
    return Kokoro(model, voices)


_KOKORO_DEFAULT_VOICE = "am_michael"


def _kokoro_voice() -> str:
    return config.get("KOKORO_VOICE", _KOKORO_DEFAULT_VOICE)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _synthesize_kokoro(text: str, out_path: str) -> float:
    import wave
    import numpy as np

    k = _kokoro()
    voice_name = _kokoro_voice()
    speed = float(config.get("KOKORO_SPEED", "1.0"))
    lang = config.get("KOKORO_LANG", "en-us")

    def _create(piece: str):
        samples, sr = k.create(piece, voice=voice_name, speed=speed, lang=lang)
        if samples is None or len(samples) == 0:
            raise RuntimeError("kokoro produced no audio")
        return np.asarray(samples, dtype=np.float32), int(sr)

    sentences = _split_sentences(text) if config.get_bool("ENABLE_DRAMATIC_PACING", True) else [text]
    try:
        if len(sentences) <= 1:
            samples, sr = _create(text)
        else:
            gap = float(config.get("PAUSE_BETWEEN", "0.18"))
            payoff_gap = float(config.get("PAUSE_BEFORE_PAYOFF", "0.5"))
            pieces, sr = [], 0
            for i, sentence in enumerate(sentences):
                chunk, sr = _create(sentence)
                pieces.append(chunk)
                if i < len(sentences) - 1:
                    secs = payoff_gap if i == len(sentences) - 2 else gap
                    pieces.append(np.zeros(int(sr * secs), dtype=np.float32))
            samples = np.concatenate(pieces)
    except Exception as e:  # noqa: BLE001
        log.warning("voice: paced kokoro synth failed (%s); using one-shot.", e)
        samples, sr = _create(text)

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return len(samples) / float(sr)


def _is_chirp_voice(name: str) -> bool:
    return "chirp3-hd" in (name or "").lower()


def _speaking_rate() -> float | None:
    raw = (config.get("GOOGLE_TTS_SPEAKING_RATE", "") or "").strip()
    if not raw:
        return None
    try:
        return max(0.25, min(float(raw), 2.0))
    except (TypeError, ValueError):
        log.warning("voice: GOOGLE_TTS_SPEAKING_RATE=%r is not a number; ignoring", raw)
        return None


def _synthesize_google(text: str, out_dir: str) -> tuple[str, float]:
    import wave

    api_key = (config.get("GOOGLE_TTS_API_KEY", "") or "").strip()
    voice_name = (config.get("GOOGLE_TTS_VOICE", "") or "").strip()
    if not api_key or not voice_name:
        raise RuntimeError("google tts: GOOGLE_TTS_API_KEY / GOOGLE_TTS_VOICE not set")

    lang = config.get("GOOGLE_TTS_LANGUAGE", "en-IN")

    clean = _clean_tts_text(text)
    markup = _pause_markup(text) if config.get_bool("ENABLE_PAUSE_MARKUP", True) else ""
    use_markup = _is_chirp_voice(voice_name) and _has_pause_tag(markup)

    audio_cfg: dict = {"audioEncoding": "LINEAR16"}
    rate = _speaking_rate()
    if rate is not None:
        audio_cfg["speakingRate"] = rate

    def _post(payload_input: dict):
        return requests.post(
            _GOOGLE_TTS_URL,
            params={"key": api_key},
            json={"input": payload_input,
                  "voice": {"languageCode": lang, "name": voice_name},
                  "audioConfig": audio_cfg},
            timeout=60,
        )

    r = _post({"markup": markup} if use_markup else {"text": clean})
    if r.status_code != 200 and use_markup:
        log.warning("voice: Chirp rejected markup (HTTP %d); retrying as plain text.", r.status_code)
        r = _post({"text": clean})
    if r.status_code != 200:
        raise RuntimeError(f"google tts HTTP {r.status_code}: {r.text[:500]}")
    b64 = (r.json() or {}).get("audioContent")
    if not b64:
        raise RuntimeError("google tts: empty audioContent")

    out_path = os.path.join(out_dir, _audio_filename(clean, ".wav"))
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))
    with wave.open(out_path, "rb") as w:
        duration = w.getnframes() / float(w.getframerate())
    return out_path, duration


def _synthesize_gemini(text: str, out_dir: str) -> tuple[str, float]:
    import wave
    from google import genai
    from google.genai import types

    key = (config.get("GEMINI_TTS_API_KEY") or config.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("gemini tts: GEMINI_TTS_API_KEY / GEMINI_API_KEY not set")

    spoken = _style_text(text)
    style = config.get("VOICE_STYLE_PROMPT", _DEFAULT_STYLE_PROMPT)

    t_bytes, p_bytes = len(spoken.encode("utf-8")), len(style.encode("utf-8"))
    if t_bytes > _GEMINI_MAX_FIELD_BYTES or p_bytes > _GEMINI_MAX_FIELD_BYTES \
            or t_bytes + p_bytes > _GEMINI_MAX_TOTAL_BYTES:
        raise RuntimeError(
            f"gemini tts: input too long (script {t_bytes}B, style {p_bytes}B; limits are "
            f"{_GEMINI_MAX_FIELD_BYTES}B each / {_GEMINI_MAX_TOTAL_BYTES}B combined)")
    model = config.get("GEMINI_TTS_MODEL", _GEMINI_TTS_PRIMARY)
    voice_name = config.get("GEMINI_TTS_VOICE", "Zubenelgenubi")

    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )

    attempts = [model] + ([_GEMINI_TTS_STABLE] if model != _GEMINI_TTS_STABLE else [])
    resp = last_err = None
    for attempt in attempts:
        try:
            resp = client.models.generate_content(
                model=attempt, contents=f"{style}\n\n{spoken}", config=cfg)
            if attempt != model:
                log.warning("gemini tts: %s unavailable (%s) — fell back to %s, same voice",
                            model, str(last_err)[:120], attempt)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if not _is_transient(e):
                raise
    if resp is None:
        raise RuntimeError(f"gemini tts: all models failed ({last_err})") from last_err

    try:
        pcm = resp.candidates[0].content.parts[0].inline_data.data
    except (AttributeError, IndexError, TypeError) as e:
        raise RuntimeError(f"gemini tts: unexpected response shape ({e})") from e
    if not pcm:
        raise RuntimeError("gemini tts: empty audio")

    out_path = os.path.join(out_dir, _audio_filename(spoken, ".wav"))
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_GEMINI_TTS_RATE)
        w.writeframes(pcm)
    with wave.open(out_path, "rb") as w:
        duration = w.getnframes() / float(w.getframerate())
    return out_path, duration


def _stream_chunks(text: str, voice: str, rate: str):
    """Yield edge-tts stream chunks. Wrapped in a timeout so a hanging request cannot stall
    the entire GitHub Actions run. Raises RuntimeError if the timeout is exceeded."""
    import edge_tts

    comm = edge_tts.Communicate(text, voice, rate=rate)

    async def _collect():
        chunks = []
        async for chunk in comm.stream():
            chunks.append(chunk)
        return chunks

    try:
        chunks = asyncio.run(asyncio.wait_for(_collect(), timeout=_EDGE_TIMEOUT))
    except asyncio.TimeoutError as e:
        raise RuntimeError(f"edge-tts stream timed out after {_EDGE_TIMEOUT}s") from e

    for chunk in chunks:
        yield chunk


def _synthesize_edge_tts(text: str, out_path: str, voice: str, rate: str) -> float:
    last_end_ticks = 0
    wrote_audio = False
    with open(out_path, "wb") as f:
        for chunk in _stream_chunks(text, voice, rate):
            if chunk["type"] == "audio":
                f.write(chunk["data"])
                wrote_audio = True
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                last_end_ticks = max(last_end_ticks, chunk["offset"] + chunk["duration"])
    if not wrote_audio:
        raise RuntimeError("edge-tts returned no audio (check voice name / connectivity).")
    return last_end_ticks / _TICKS_PER_SECOND


_ENGINE_ORDER = ("google", "edge", "kokoro")
_ENGINE_ALIASES = {"edge-tts": "edge", "chirp": "google", "google-tts": "google",
                   "gemini-tts": "gemini"}


def _engine_gemini(text: str, out_dir: str) -> tuple[str, float]:
    path, dur = _synthesize_gemini(text, out_dir)
    _log_done(path, dur, f"gemini:{config.get('GEMINI_TTS_MODEL', _GEMINI_TTS_PRIMARY)}")
    return path, dur


def _engine_google(text: str, out_dir: str) -> tuple[str, float]:
    path, dur = _synthesize_google(text, out_dir)
    _log_done(path, dur, "google:chirp3hd")
    return path, dur


def _engine_edge(text: str, out_dir: str) -> tuple[str, float]:
    clean = _clean_tts_text(text)
    out_path = os.path.join(out_dir, _audio_filename(clean, ".mp3"))
    dur = _synthesize_edge_tts(clean, out_path, _VOICE, _RATE)
    _log_done(out_path, dur, f"edge-tts:{_VOICE}")
    return out_path, dur


def _engine_kokoro(text: str, out_dir: str) -> tuple[str, float]:
    clean = _clean_tts_text(text)
    out_path = os.path.join(out_dir, _audio_filename(clean, ".wav"))
    dur = _synthesize_kokoro(clean, out_path)
    _log_done(out_path, dur, "kokoro")
    return out_path, dur


_PAUSE_TAGS = ("pause short", "pause", "pause long")

_STYLE_TAGS = (
    "sarcastic", "serious", "curious", "whispers", "tired", "mischievously", "sighs", "laughs",
    "deadpan", "dry", "amused", "flat",
)

_TAG_RE = re.compile(r"\[([^\]]{1,24})\]|<[^>]{1,60}>")


def _tag_limit(key: str, default: int) -> int:
    try:
        return max(0, int(config.get(key, str(default))))
    except (TypeError, ValueError):
        return default


def _filter_tags(text: str, keep: tuple[str, ...], limit: int,
                 pause_as_ellipsis: bool = False) -> str:
    kept = 0

    def _sub(m: re.Match) -> str:
        nonlocal kept
        inner = m.group(1)
        if inner is not None:
            token = inner.strip().lower()
            if token in keep and kept < limit:
                kept += 1
                return f"[{token}]"
            if pause_as_ellipsis and token in _PAUSE_TAGS:
                return " ... "
        return ""

    return re.sub(r"\s+", " ", _TAG_RE.sub(_sub, text)).strip()


def has_style_tag(text: str) -> bool:
    return any(m.group(1) and m.group(1).strip().lower() in _STYLE_TAGS
               for m in _TAG_RE.finditer(text or ""))


def _clean_tts_text(text: str) -> str:
    return _filter_tags(text, (), 0)


def _pause_markup(text: str) -> str:
    return _filter_tags(text, _PAUSE_TAGS, _tag_limit("MAX_PAUSE_TAGS", 3))


def _style_text(text: str) -> str:
    return _filter_tags(text, _STYLE_TAGS, _tag_limit("MAX_STYLE_TAGS", 3),
                        pause_as_ellipsis=True)


def _has_pause_tag(text: str) -> bool:
    return bool(re.search(r"\[(?:pause short|pause long|pause)\]", text))


def synthesize(script_body: str, out_dir: str) -> tuple[str, float]:
    raw = (script_body or "").strip()
    if not _clean_tts_text(raw):
        raise ValueError("voice.synthesize: empty script_body.")
    os.makedirs(out_dir, exist_ok=True)

    primary = str(config.get("VOICE_ENGINE", "gemini")).lower()
    primary = _ENGINE_ALIASES.get(primary, primary)
    order = [primary] + [e for e in _ENGINE_ORDER if e != primary]

    errors: list[str] = []
    for name in order:
        fn = globals().get(f"_engine_{name}")
        if fn is None:
            continue
        try:
            return fn(raw, out_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("voice: engine %s failed (%s); trying next", name, e)
            errors.append(f"{name}: {e}")
    raise RuntimeError("voice.synthesize: all engines failed — " + " | ".join(errors))


def _log_done(out_path: str, duration: float, engine: str) -> None:
    if duration > 60:
        log.warning("voice: narration is %.1fs (>60s) — script likely too long for a Short.", duration)
    log.info("voice: wrote %s (%.1fs, engine=%s)", out_path, duration, engine)
