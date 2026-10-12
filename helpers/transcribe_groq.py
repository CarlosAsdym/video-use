"""Transcribe audio with Groq Whisper and return it in ElevenLabs Scribe's shape.

Selected by transcribe.py when TRANSCRIBE_PROVIDER=groq (env var or .env at the
repo root). The rest of the pipeline (pack_transcripts, timeline_view, render)
reads Scribe JSON, so this module converts Groq's verbose_json into it:

    {"language_code", "text", "words": [
        {"text", "start", "end", "type": "word", "speaker_id": "speaker_0", "logprob"},
        {"text": " ", "start", "end", "type": "spacing"}, ...]}

What Scribe gave that Whisper cannot:
  - diarization: every word gets speaker_0 (single-speaker footage).
  - audio events ("(laughs)"): none are produced.
  - verbatim fillers: Whisper tends to drop "um"/"uh". A filler-laden prompt
    nudges it to keep them (TRANSCRIBE_PROMPT overrides it).

Groq limits (console.groq.com/docs/speech-to-text): 25 MB per upload on the free
tier, 10 s minimum billed per request. Audio is sent as 16 kHz mono FLAC; above
CHUNK_LIMIT_MB it is split into overlapping chunks and stitched back.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import requests


GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-large-v3"
DEFAULT_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
CHUNK_LIMIT_MB = 24.0
CHUNK_OVERLAP_S = 3.0
REPO_ENV = Path(__file__).resolve().parent.parent / ".env"


def env_value(key: str, default: str = "") -> str:
    """Environment first, then the repo-root .env. Never the caller's cwd .env."""
    v = os.environ.get(key, "")
    if v:
        return v
    if REPO_ENV.exists():
        for line in REPO_ENV.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, val = line.split("=", 1)
            if k.strip() == key:
                return val.strip().strip('"').strip("'")
    return default


def load_groq_key() -> str:
    key = env_value("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not found in environment or video-use/.env")
    return key


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _to_flac(src: Path, dest: Path, start: float | None = None, dur: float | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(dest)]
    subprocess.run(cmd, check=True)


def _call_groq(audio: Path, api_key: str, language: str | None) -> dict:
    data: list[tuple[str, str]] = [
        ("model", env_value("GROQ_MODEL", DEFAULT_MODEL)),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
        ("timestamp_granularities[]", "segment"),
        ("temperature", "0"),
    ]
    prompt = env_value("TRANSCRIBE_PROMPT", DEFAULT_PROMPT)
    if prompt:
        data.append(("prompt", prompt))
    if language:
        data.append(("language", language))

    with open(audio, "rb") as f:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (audio.name, f, "audio/flac")},
            data=data,
            timeout=600,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Groq returned {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _hallucinated(seg: dict) -> bool:
    """Whisper's own skip rule: likely silence AND low confidence."""
    return seg.get("no_speech_prob", 0.0) > 0.6 and seg.get("avg_logprob", 0.0) < -1.0


def _norm(token: str) -> str:
    return re.sub(r"[^\w']", "", token.lower())


def _punctuate(words: list[dict], segments: list[dict]) -> None:
    """Word-level output drops punctuation; copy it back from segment text.

    render.py breaks subtitle cues on trailing punctuation, so this matters for
    caption chunking. Tokens are matched in order by normalized form; a word
    that does not match keeps its bare text.
    """
    for seg in segments:
        tokens = seg.get("text", "").split()
        in_seg = [w for w in words if seg["start"] - 0.05 <= w["start"] < seg["end"] + 0.05]
        ti = 0
        for w in in_seg:
            target = _norm(w["text"])
            for j in range(ti, min(ti + 3, len(tokens))):
                if _norm(tokens[j]) == target and target:
                    w["text"] = tokens[j]
                    ti = j + 1
                    break


def _convert(raw: dict, offset: float = 0.0) -> tuple[list[dict], list[dict]]:
    """Return (words, segments) from one Groq response, shifted by offset."""
    segments = [
        {**s, "start": s["start"] + offset, "end": s["end"] + offset}
        for s in raw.get("segments", [])
    ]
    bad = [s for s in segments if _hallucinated(s)]
    seg_logprob = lambda t: next(
        (s.get("avg_logprob") for s in segments if s["start"] <= t < s["end"]), None
    )
    words: list[dict] = []
    for w in raw.get("words", []):
        text = (w.get("word") or "").strip()
        if not text:
            continue
        start, end = w["start"] + offset, w["end"] + offset
        if any(s["start"] <= start < s["end"] for s in bad):
            continue
        words.append({
            "text": text, "start": round(start, 3), "end": round(end, 3),
            "type": "word", "speaker_id": "speaker_0", "logprob": seg_logprob(start),
        })
    _punctuate(words, [s for s in segments if s not in bad])
    return words, segments


def _with_spacing(words: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, w in enumerate(words):
        if i and w["start"] > out[-1]["end"]:
            out.append({"text": " ", "start": out[-1]["end"], "end": w["start"], "type": "spacing"})
        out.append(w)
    return out


def transcribe_wav(wav: Path, api_key: str, language: str | None = None) -> dict:
    """Transcribe a 16 kHz mono wav (from transcribe.extract_audio) via Groq.

    Returns a Scribe-shaped dict, plus "provider", "model" and "groq_segments"
    so the raw segment metadata stays auditable.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        flac = tmp_dir / "full.flac"
        _to_flac(wav, flac)
        size_mb = flac.stat().st_size / (1024 * 1024)

        if size_mb <= CHUNK_LIMIT_MB:
            raw = _call_groq(flac, api_key, language)
            words, segments = _convert(raw)
            lang = raw.get("language")
        else:
            total = _probe_duration(flac)
            chunk_s = max(60.0, total * (CHUNK_LIMIT_MB / size_mb) * 0.9)
            words, segments, lang = [], [], None
            start = 0.0
            while start < total:
                dur = min(chunk_s + CHUNK_OVERLAP_S, total - start)
                part = tmp_dir / f"chunk_{int(start)}.flac"
                _to_flac(flac, part, start, dur)
                raw = _call_groq(part, api_key, language)
                lang = lang or raw.get("language")
                w, s = _convert(raw, offset=start)
                # Keep each word once: a chunk owns the words before the
                # middle of its trailing overlap, the next chunk the rest.
                lo = start + CHUNK_OVERLAP_S / 2 if start > 0 else -1.0
                hi = start + chunk_s + CHUNK_OVERLAP_S / 2
                words += [x for x in w if lo <= x["start"] < hi]
                segments += [x for x in s if lo <= x["start"] < hi]
                start += chunk_s

    words.sort(key=lambda x: x["start"])
    return {
        "provider": "groq",
        "model": env_value("GROQ_MODEL", DEFAULT_MODEL),
        "language_code": lang,
        "text": " ".join(w["text"] for w in words),
        "words": _with_spacing(words),
        "groq_segments": segments,
    }
