"""Module 5 — Visuals (stock B-roll / AI images).

Contract:
    what it does : finds + downloads CC0 vertical B-roll OR generates AI images for a script.
    input        : script_body or keyword list; target duration; output dir.
    output       : list of local clip paths covering the narration length.
    depends on   : Pexels/Pixabay (stock) + Cloudflare Workers AI (ai); requests; src.config.

VISUAL_SOURCE picks the strategy:
    'photos' (default) -> Pexels stock photos + Ken Burns
    'ai'               -> Cloudflare Flux + Ken Burns
    'video'            -> Pexels/Pixabay stock video

The CF_API_TOKEN and CF_ACCOUNT_ID values are stripped of any whitespace/newline characters
before use, because GitHub Secrets are sometimes stored with a trailing newline that breaks
the HTTP Authorization header ("Invalid leading whitespace" error).
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import subprocess

from functools import lru_cache

import requests

from src import config, llm

log = logging.getLogger(__name__)

_PEXELS_VIDEO_SEARCH = "https://api.pexels.com/videos/search"
_PEXELS_PHOTO_SEARCH = "https://api.pexels.com/v1/search"
_PIXABAY_VIDEO_SEARCH = "https://pixabay.com/api/videos/"
_TIMEOUT = 30
_SLICE_SECONDS = 8.0
_PER_KEYWORD = 3

_IMAGE_CLIP_SECONDS = 7.0
_MAX_IMG_CLIPS = 12
_MAX_VIDEO_CLIPS = 12

_STOPWORDS = frozenset(
    "the a an and or but of to in on for with as at by from is are was were be been it its "
    "this that these those they them their there here what which who how why when where will "
    "would can could should may might just not no so than then too very you your we our us i "
    "about into over after before more most some any all has have had do does did up out".split()
)


def _keywords_heuristic(script_body: str, n: int) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", script_body.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w not in _STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq, key=lambda w: (-freq[w], w))
    return ranked[:n]


def _keywords_via_llm(script_body: str, n: int) -> list[str]:
    prompt = (
        f"You are a cinematic storyboard artist picking AI image prompts for a romance "
        f"storytelling Short. Give exactly {n} CONCRETE, VISUAL scene descriptions (2-4 words "
        f"each) that an AI image model can render as anime-style illustrations. Order them to "
        f"follow the story beats so the visuals track what is being said.\n"
        f"KEY RULE: pick specific SCENES with a subject, action, and mood — not abstract concepts. "
        f"Translate emotion into filmable imagery:\n"
        f"  love/longing -> 'young woman staring at rain', 'empty train platform at dusk'\n"
        f"  separation -> 'silhouette walking away in fog', 'closed door, warm light'\n"
        f"  reunion -> 'two hands almost touching', 'tearful smile in crowd'\n"
        f"  letter/memory -> 'old envelope on wooden table', 'faded photograph in hands'\n"
        f"  family -> 'mother and daughter silhouettes', 'empty chair by window'\n"
        f"  nature/season -> 'monsoon rain on window', 'autumn leaves on bench'\n"
        f"  city -> 'streetlamp in evening rain', 'train window passing lights'\n"
        f"  time passing -> 'clock on wall', 'calendar pages fluttering'\n"
        f"Prefer visually striking, emotional subjects that hold attention. AVOID proper nouns, "
        f"logos, and abstract words (love, destiny, hope) — only things a camera can see.\n\n"
        f"NARRATION:\n{script_body}\n\n"
        f'Output ONE valid JSON object and nothing else: '
        f'{{"keywords": ["scene one", "scene two"]}}'
    )
    import json

    raw = llm.generate(prompt, json=True, max_tokens=300, prefer_groq=True)
    start, end = raw.find("{"), raw.rfind("}")
    data = json.loads(raw[start : end + 1], strict=False)
    kws = [str(k).strip() for k in data.get("keywords", []) if str(k).strip()]
    if not kws:
        raise ValueError("llm returned no keywords")
    return kws[:n]


def extract_keywords(script_body: str, n: int = 5) -> list[str]:
    text = (script_body or "").strip()
    if not text:
        return []
    try:
        return _keywords_via_llm(text, n)
    except Exception as e:  # noqa: BLE001
        log.warning("visuals: LLM keyword extraction failed (%s); using heuristic", e)
        return _keywords_heuristic(text, n)


def _pick_portrait_file(video: dict) -> str | None:
    files = [
        f for f in video.get("video_files", [])
        if f.get("file_type") == "video/mp4" and (f.get("height") or 0) > (f.get("width") or 0)
    ]
    if not files:
        return None
    best = min(files, key=lambda f: abs((f.get("width") or 0) - 1080))
    return best.get("link")


def _pexels_search(keyword: str) -> list[dict]:
    resp = requests.get(
        _PEXELS_VIDEO_SEARCH,
        headers={"Authorization": config.require("PEXELS_API_KEY")},
        params={"query": keyword, "orientation": "portrait", "per_page": _PER_KEYWORD,
                "size": "medium"},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for v in resp.json().get("videos", []):
        link = _pick_portrait_file(v)
        if link:
            out.append({"url": link, "duration": float(v.get("duration") or _SLICE_SECONDS)})
    return out


def _pixabay_search(keyword: str) -> list[dict]:
    key = config.get("PIXABAY_API_KEY")
    if not key:
        return []
    resp = requests.get(
        _PIXABAY_VIDEO_SEARCH,
        params={"key": key, "q": keyword, "per_page": _PER_KEYWORD, "safesearch": "true"},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for hit in resp.json().get("hits", []):
        files = hit.get("videos", {})
        chosen = files.get("large") or files.get("medium") or files.get("small")
        if chosen and chosen.get("url"):
            out.append({"url": chosen["url"], "duration": float(hit.get("duration") or _SLICE_SECONDS)})
    return out


def _gather_candidates(keywords: list[str]) -> list[dict]:
    per_kw: list[list[dict]] = []
    for kw in keywords:
        try:
            per_kw.append(_pexels_search(kw))
        except Exception as e:  # noqa: BLE001
            log.warning("visuals: Pexels search failed for %r (%s)", kw, e)
            per_kw.append([])

    interleaved = _interleave(per_kw)
    if interleaved:
        return interleaved

    log.warning("visuals: Pexels returned nothing; trying Pixabay backup")
    for kw in keywords:
        try:
            per_kw_pb = _pixabay_search(kw)
        except Exception as e:  # noqa: BLE001
            log.warning("visuals: Pixabay search failed for %r (%s)", kw, e)
            per_kw_pb = []
        interleaved.extend(per_kw_pb)
    return interleaved


def _interleave(lists: list[list[dict]]) -> list[dict]:
    out: list[dict] = []
    for i in range(max((len(x) for x in lists), default=0)):
        for lst in lists:
            if i < len(lst):
                out.append(lst[i])
    return out


def _clip_filename(url: str) -> str:
    return f"broll_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}.mp4"


def _download(url: str, dest: str) -> None:
    with requests.get(url, stream=True, timeout=_TIMEOUT) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)


def _img_prompt(keyword: str) -> str:
    """Build the AI-image prompt. Style is tunable via IMAGE_STYLE for the channel's look."""
    style = config.get(
        "IMAGE_STYLE",
        "anime style, Makoto Shinkai aesthetic, soft pastel colors, emotional atmosphere, "
        "cinematic composition, golden hour, detailed background, 2D illustration, "
        "no text, no watermark",
    )
    return f"{keyword}, {style}, vertical 9:16 composition"


@lru_cache(maxsize=64)
def _pexels_photo_urls(keyword: str) -> tuple[str, ...]:
    try:
        resp = requests.get(
            _PEXELS_PHOTO_SEARCH,
            headers={"Authorization": config.require("PEXELS_API_KEY")},
            params={"query": keyword, "orientation": "portrait", "per_page": 8, "size": "large"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return tuple(
            p["src"].get("large2x") or p["src"].get("portrait") or p["src"]["original"]
            for p in resp.json().get("photos", []) if p.get("src")
        )
    except Exception as e:  # noqa: BLE001
        log.warning("visuals: Pexels photo search failed for %r (%s)", keyword, e)
        return ()


def _cloudflare_image(prompt: str, dest: str) -> bool:
    """Generate an AI image via Cloudflare Workers AI (Flux). Needs CF_API_TOKEN + CF_ACCOUNT_ID.

    Strips whitespace and newlines from both credentials before use — GitHub Secrets are
    sometimes stored with a trailing newline that produces the "Invalid leading whitespace"
    error on the HTTP Authorization header.
    """
    token = str(config.get("CF_API_TOKEN") or "").strip().replace("\n", "").replace("\r", "")
    acct = str(config.get("CF_ACCOUNT_ID") or "").strip().replace("\n", "").replace("\r", "")
    if not (token and acct):
        return False
    model = config.get("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
    try:
        r = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}",
            headers={"Authorization": f"Bearer {token}"},
            json={"prompt": prompt[:2000]}, timeout=90,
        )
        r.raise_for_status()
        if "application/json" in r.headers.get("content-type", ""):
            b64 = (r.json().get("result") or {}).get("image")
            if not b64:
                return False
            with open(dest, "wb") as f:
                f.write(base64.b64decode(b64))
        else:
            with open(dest, "wb") as f:
                f.write(r.content)
        return os.path.getsize(dest) > 1000
    except Exception as e:  # noqa: BLE001
        log.warning("visuals: Cloudflare image gen failed (%s)", e)
        return False


def _fetch_image(keyword: str, dest: str, seed: int, source: str,
                 variant: str = "") -> bool:
    if source == "ai" and _cloudflare_image(_img_prompt(keyword), dest):
        return True
    urls = _pexels_photo_urls(keyword)
    if not urls:
        return False
    offset = int(hashlib.sha1(str(variant).encode("utf-8")).hexdigest()[:8], 16) if variant else 0
    try:
        _download(urls[(seed + offset) % len(urls)], dest)
        return os.path.getsize(dest) > 1000
    except Exception as e:  # noqa: BLE001
        log.warning("visuals: photo download failed for %r (%s)", keyword, e)
        return False


def _image_to_kenburns_clip(image_path: str, dest: str, seconds: float, index: int = 0) -> None:
    from src.assembly import _ffmpeg

    frames = int(seconds * 30)
    if index % 2 == 0:
        z = "min(zoom+0.0010,1.12)"
    else:
        z = "if(eq(on,0),1.12,max(zoom-0.0009,1.0))"
    vf = (
        "scale=1620:2880:force_original_aspect_ratio=increase,crop=1620:2880,"
        f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s=1080x1920:fps=30,setsar=1"
    )
    proc = subprocess.run(
        [_ffmpeg(), "-y", "-loop", "1", "-i", image_path, "-t", f"{seconds:.2f}",
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", "30", dest],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ken burns failed ({proc.returncode}): {proc.stderr[-400:]}")


def _fetch_image_broll(keywords: list[str], target_seconds: float, out_dir: str, source: str,
                       variant: str = "") -> list[str]:
    from src import assembly

    n = min(_MAX_IMG_CLIPS, assembly.slice_count(target_seconds))
    clips: list[str] = []
    for i in range(n):
        kw = keywords[i % len(keywords)]
        img = os.path.join(out_dir, f"img_{i:02d}.jpg")
        if not _fetch_image(kw, img, i, source, variant=variant):
            continue
        clip = os.path.join(out_dir, f"imgclip_{i:02d}.mp4")
        try:
            _image_to_kenburns_clip(img, clip, _IMAGE_CLIP_SECONDS, index=i)
            clips.append(clip)
        except Exception as e:  # noqa: BLE001
            log.warning("visuals: ken burns failed (%s); skipping", e)
    if not clips:
        raise RuntimeError(f"visuals: produced no image clips from source={source}")
    log.info("visuals: %d %s Ken Burns clips for target %.0fs", len(clips), source, target_seconds)
    return clips


def fetch_broll(keywords: list[str], target_seconds: float, out_dir: str) -> list[str]:
    if not keywords:
        raise ValueError("visuals.fetch_broll: no keywords provided.")
    os.makedirs(out_dir, exist_ok=True)

    source = str(config.get("VISUAL_SOURCE", "photos")).lower()
    if source in ("photos", "ai"):
        try:
            return _fetch_image_broll(keywords, target_seconds, out_dir, source,
                                      variant=os.path.basename(os.path.normpath(out_dir)))
        except Exception as e:  # noqa: BLE001
            log.warning("visuals: %s source failed (%s); falling back to stock video", source, e)

    return _fetch_video_broll(keywords, target_seconds, out_dir)


def _fetch_video_broll(keywords: list[str], target_seconds: float, out_dir: str) -> list[str]:
    candidates = _gather_candidates(keywords)
    if not candidates:
        raise RuntimeError(f"visuals: no B-roll found on Pexels/Pixabay for {keywords}.")

    paths: list[str] = []
    from src import assembly

    needed = min(_MAX_VIDEO_CLIPS, assembly.slice_count(target_seconds))
    for c in candidates:
        if len(paths) >= max(2, needed):
            break
        dest = os.path.join(out_dir, _clip_filename(c["url"]))
        try:
            if not os.path.exists(dest) or os.path.getsize(dest) == 0:
                _download(c["url"], dest)
        except Exception as e:  # noqa: BLE001
            log.warning("visuals: download failed (%s); skipping", e)
            continue
        paths.append(dest)

    if not paths:
        raise RuntimeError("visuals: found candidates but every download failed.")
    log.info("visuals: %d clip(s) for %d cut(s) over %.0fs", len(paths), needed, target_seconds)
    return paths
