"""Module 1 — Romance Story Ideation (free-API, no news feed).

Contract:
    what it does : generates romance/emotional story concepts for 2-3 minute Hindi narrations.
    how to use   : `generate_ideas(n)` for on-demand, `run_fallback_ideation()` for scheduled.
    depends on   : src.llm, src.db, src.config.

This replaces the news-based ideation engine. There are no headlines to fetch and no sources
to verify — a fictional romance story has neither. The LLM is the sole ideation source, and
every idea is human-approved via Telegram before production.

Schema (unchanged so the rest of the pipeline works):
    title        -> the story's headline (English for SEO, emotion intact)
    hook         -> the opening line of the narration (Hindi, sets the emotional scene)
    angle        -> the emotional core / twist the story delivers
    est_score    -> LLM's self-rated strength (0.0-1.0)
    sources      -> always [] for fiction; the caption's citation gate is skipped for this niche
"""
from __future__ import annotations

import json
import logging

from src import config, db, llm

log = logging.getLogger(__name__)

_MAX_IDEAS = 10
_MIN_IDEAS = 2

_ROW_KEYS = ("niche", "title", "hook", "angle", "est_score", "sources")

_PROMPT = """You are the story editor for a Hindi/Urdu romance storytelling channel. \
The channel publishes 2-3 minute cinematic narratives — one-sided love, fate, longing, \
separation, reunion. The audience is South Asian, watches on mobile, and wants to feel \
something real in the first 5 seconds.

Generate {n} DISTINCT romance story concepts. Each must be a complete emotional arc, not a \
premise. Return English titles (for search) but write the hook and angle in English too — \
the scriptwriter will translate to Hindi.

REQUIREMENTS PER IDEA:

1. TITLE (English, <=70 chars, searchable): a curiosity-driven title that names the emotional \
   situation. Examples of the RIGHT style:
     - "She Waited 40 Years For A Letter That Never Came"
     - "He Loved Her From Afar For 12 Years — Then She Asked Him One Question"
     - "The Girl Who Chose The Wrong Brother"
   NOT: "A Story About Love", "True Love", "Heartbreaking Romance"

2. HOOK (English, 1-2 sentences, the first 5 seconds): a SPECIFIC moment, image, or line that \
   drops the viewer into the scene. Use concrete detail — weather, an object, a sound, a \
   gesture. Examples:
     - "The letter was still sealed. She had carried it in her purse for nine years."
     - "He learned her name only on the day she got married."
   NOT: "Love is complicated.", "Sometimes we lose what we love most."

3. ANGLE (English, 2-3 sentences): the emotional core — what the story is really about, and \
   the twist or revelation that makes it worth 2-3 minutes. This is the "why it matters" \
   emotional turn. Examples:
     - "He built a life around a promise she never made. The story is about how we invent \
        reasons for the people we cannot let go of — and what it costs when the invention \
        finally breaks."
   NOT: "It is a sad story about love."

4. EST_SCORE (0.0-1.0): rate this story's emotional pull and shareability. Spread the scores \
   across the range — strongest near 0.9, weakest near 0.4. Do not give everything ~0.8.

GENRE PALETTE (spread across these, one per idea): one-sided love, long-distance separation, \
family disapproval, missed timing, reunion after decades, sacrifice for a sibling or friend, \
love that becomes friendship, forbidden love across class or religion, love across distance \
(letter/phone/internet), love interrupted by war or migration.

TONE: emotionally honest, specific, never melodramatic. NO: violence, communal/religious \
incitement, suicide, explicit content, exploitative tragedy. Characters are ordinary people \
in real situations.

LENGTH TARGET: each story should sustain a 400-500 word Hindi narration (2-3 minutes when \
spoken slowly with pauses). If a concept cannot fill that, replace it.

Return ONLY JSON:
{{"ideas": [{{"title": "...", "hook": "...", "angle": "...", "est_score": 0.0}}]}}
"""


def _parse_ideas(raw: str) -> list[dict]:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"ideation: no JSON object in reply: {raw[:200]!r}")
    data = json.loads(raw[start : end + 1], strict=False)
    ideas = data.get("ideas", [])
    if not isinstance(ideas, list):
        raise ValueError("ideation: 'ideas' is not a list.")
    return ideas


def _to_rows(ideas: list[dict]) -> list[dict]:
    """Project validated ideas to the DB columns. Sources are always [] for fiction."""
    rows = []
    for idea in ideas:
        row = {k: idea.get(k) for k in _ROW_KEYS}
        row["sources"] = []          # fiction has no external sources
        rows.append(row)
    return rows


def _validate(ideas: list[dict]) -> list[dict]:
    """Keep well-formed ideas; coerce fields. No source check — fiction has no sources."""
    niche = config.get("NICHE", "romance-stories")
    clean = []
    seen_titles: set[str] = set()

    for idea in ideas:
        if not isinstance(idea, dict):
            continue
        title = str(idea.get("title", "")).strip()
        hook = str(idea.get("hook", "")).strip()
        angle = str(idea.get("angle", "")).strip()
        if not (title and hook and angle):
            continue
        if title.lower() in seen_titles:
            continue
        try:
            est = float(idea.get("est_score", 0.5))
        except (TypeError, ValueError):
            est = 0.5
        est = min(1.0, max(0.0, est))

        seen_titles.add(title.lower())
        clean.append({
            "niche": niche,
            "title": title,
            "hook": hook,
            "angle": angle,
            "est_score": est,
        })
        if len(clean) >= _MAX_IDEAS:
            break
    return clean


def _produce_ideas(n: int) -> list[dict]:
    """Single LLM call. No headlines, no trends, no grounded search — fiction only."""
    prompt = _PROMPT.format(n=n)
    try:
        raw = llm.generate(prompt, json=True, max_tokens=4096, prefer_groq=True)
    except Exception as e:  # noqa: BLE001
        log.warning("ideation: LLM call failed (%s)", e)
        raise
    ideas = _parse_ideas(raw)
    return _validate(ideas)


def run_fallback_ideation() -> int:
    """Scheduled path: generate a fresh batch if no pending ideas exist. Idempotent."""
    if db.get_pending_ideas():
        log.info("ideation: pending ideas already exist; skipping (idempotent).")
        return 0

    clean = _produce_ideas(3)
    if len(clean) < _MIN_IDEAS:
        raise RuntimeError(
            f"ideation: only {len(clean)} valid ideas (need >= {_MIN_IDEAS}); "
            "not inserting a thin digest."
        )
    inserted = db.insert_ideas(_to_rows(clean))
    log.info("ideation: inserted %d pending story idea(s).", len(inserted))
    return len(inserted)


def generate_ideas(n: int = 3) -> int:
    """On-demand: generate exactly n fresh story ideas and insert as 'pending'."""
    n = max(1, n)
    clean = _produce_ideas(n)
    if not clean:
        raise RuntimeError("ideation: could not generate any valid story idea.")
    inserted = db.insert_ideas(_to_rows(clean[:n]))
    log.info("ideation: generated %d on-demand story idea(s).", len(inserted))
    return len(inserted)


def seed_ideas(n: int = 3) -> int:
    """Seed ~n fresh 'pending' ideas, de-duplicated against existing titles."""
    pool = _produce_ideas(max(n * 2, 4))
    seen = db.existing_idea_titles()
    fresh = [i for i in pool if i["title"].lower() not in seen][:n]
    if not fresh:
        raise RuntimeError("ideation: no fresh story ideas to seed.")
    log.info("ideation: seeding %d story idea(s).", len(fresh))
    return len(db.insert_ideas(_to_rows(fresh)))


def load_routine_ideas() -> list[dict]:
    """Stub — no Claude Routine file for fiction. Returns []."""
    return []
