"""Module 3 — Scriptwriter (Hindi Romance Storytelling).

Contract:
    what it does : turns an approved romance idea (+ its hook and angle) into a 2-3 minute
                   Hindi narration.
    input        : idea dict {id, title, hook, angle, ...}; template name (default 'N').
    output       : {script_id, script_body, caption, hashtags[], title, tags, key_points}
    depends on   : src.llm, src.db, src.config.

WHAT CHANGED FROM THE NEWS/HISTORY VERSION:
    - The story is FICTIONAL. There are no sources to verify, no facts to check.
    - The narration is written in HINDI (Devanagari), not English.
    - Length target is 400-500 words (2-3 minutes when spoken with pauses).
    - The "why it matters" turn is emotional, not factual.
    - Sources are always empty; the caption omits the source line.
    - Delivery tags are [pause], [pause long], [serious], [whispers] — no [sarcastic].

The compliance machinery for AI disclosure and #Shorts is preserved.
"""
from __future__ import annotations

import json
import logging
import re

from src import config, db, llm

log = logging.getLogger(__name__)

DISCLOSURE_LINE = "AI-generated narration; AI-generated visuals."

_SUPPORTED_TEMPLATES = ("N",)

_PROMPT_N = """You are a Hindi/Urdu romance storyteller. Your job is to expand an approved \
story idea into a 2-3 minute narration (400-500 words) that will be read aloud over anime-style \
visuals. The story is FICTION — you are not verifying anything, you are telling a story.

STORY IDEA:
Title: {title}
Hook: {hook}
Emotional core: {angle}

WRITE THE NARRATION IN HINDI (Devanagari script). The title and metadata stay in English, \
but the spoken narration must be in natural, warm Hindi as spoken in India — not formal \
literary Hindi, not Sanskritized. Think of how a grandmother tells a story: simple words, \
concrete images, short sentences.

STRUCTURE (do not label these sections in the output — just write the story):

1. OPENING (first ~10 seconds): Start with the hook exactly as given, or expand it with one \
   sensory detail. Drop the listener into the scene. Use weather, an object, a sound, a gesture.

2. THE MEETING / THE BEGINNING (30-40 seconds): Who are these people? How did they meet? \
   What was the moment that started it all? One specific detail is worth more than three vague ones.

3. THE SEPARATION / THE CONFLICT (60-90 seconds): The heart of the story. What went wrong? \
   A misunderstanding, a family objection, a train missed, a letter never sent, a promise \
   broken by circumstance. This is the longest section — build the ache.

4. THE YEARS PASS (30-40 seconds): Time moves. Show the cost — what each person became \
   while waiting or moving on. Small details of daily life carry the weight.

5. THE TURN / THE REVELATION (30-40 seconds): The emotional core from the idea is delivered \
   here. This is the "why it matters" turn — but emotional, not factual. What did they \
   finally understand? What was the truth all along?

6. THE CLOSE (10-15 seconds): End on an image, not an explanation. Leave the listener with \
   a feeling they can carry. Then a brief call-to-action — "अगर यह कहानी आपको छू गई हो, तो \
   अपनी कहानी कमेंट में बताइए।" or similar.

WRITING RULES:
- Every sentence must be under 25 words. Short sentences carry emotion better than long ones.
- Use concrete details, not abstractions. "उसकी साड़ी का पीला रंग" beats "उसकी खूबसूरती".
- Repeat a single image (the letter, the train, the sari) across sections to create resonance.
- No dialogue tags like "उसने कहा" — let the emotion carry the exchange.
- Do not moralize or explain. Show, then let it land.
- NO: violence, communal/religious incitement, suicide, explicit content, exploitative tragedy.

DELIVERY TAGS (AT LEAST 1, AT MOST 3 in the whole script):
These are stage directions for the voice engine, not narration. Write them in square brackets \
immediately before the line they affect.
- [pause] for a beat before a revelation
- [pause long] for a longer, heavier beat — use at most once
- [serious] for the emotional turn (the "why it matters" moment)
- [whispers] for a quiet, intimate line — use sparingly
Never open the script with a tag. Never write a tag the sentence already says out loud.

ALSO produce, in ENGLISH for the YouTube metadata:
- "title": a YouTube title (<=70 chars) in English, honest to the story, front-loading the \
   most gripping emotional word. This is the SEO title.
- "caption": an English YouTube description, formatted as:
   Line 1: An emotional hook with one emoji (this is what shows in-feed).
   Line 2: A 1-2 sentence teaser + a comment question.
   Do NOT include a sources line — this is fiction.
- "hashtags": 5-8 English hashtags. Include #Shorts. Include romance/storytelling tags like \
   #RomanceStory #HindiKahani #LoveStory #EmotionalStory.
- "tags": 12-15 English YouTube search terms people would type for this kind of story.
- "key_points": 2-3 ULTRA-SHORT English on-screen text cards (<=4 words each).

Return ONLY a JSON object. Write every line break inside a string as the two-character \
escape \\n — a raw newline inside a JSON string is invalid JSON:
{{"title": "English SEO title", "script_body": "हिंदी में पूरी कहानी", "caption": "emoji hook\\n\\ntea ser + comment question", "hashtags": ["#Shorts", "#RomanceStory"], "tags": ["hindi love story", "emotional kahani"], "key_points": ["short card", "another"]}}
"""


def _build_prompt(idea: dict, template: str) -> str:
    if template != "N":
        raise ValueError(
            f"unsupported template {template!r} (MVP supports {_SUPPORTED_TEMPLATES});"
        )
    return _PROMPT_N.format(
        title=idea.get("title", ""),
        hook=idea.get("hook", ""),
        angle=idea.get("angle", ""),
    )


def _parse_llm_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"scriptwriter: no JSON object in LLM reply: {raw[:200]!r}")
    return json.loads(raw[start : end + 1], strict=False)


def _generate_script_json(prompt: str) -> dict:
    """Write the script with ungrounded JSON mode. The story is fiction — no grounded search
    is used, and no source verification happens."""
    return _parse_llm_json(llm.generate(prompt, json=True, max_tokens=4096))


def _visible_words(body: str) -> list[str]:
    """Words the narrator actually says — delivery tags are stage direction, not narration."""
    return re.sub(r"\[[^\]]*\]", " ", body).split()


def _ensure_shorts(hashtags: list[str]) -> list[str]:
    if any(h.lower() == "#shorts" for h in hashtags):
        return hashtags
    return [*hashtags, "#Shorts"]


def _ensure_disclosure(caption: str) -> str:
    if "ai-generated" in caption.lower():
        return caption
    return f"{caption.rstrip()}\n{DISCLOSURE_LINE}" if caption.strip() else DISCLOSURE_LINE


def _truncate_to_words(body: str, max_words: int) -> str:
    """Trim to the last full sentence at or under the cap. Delivery tags are carried through
    and do not count toward the cap."""
    if len(_visible_words(body)) <= max_words:
        return body
    kept, spoken = [], 0
    for piece in re.findall(r"\[[^\]]*\]|[^\s\[]+", body):
        if piece.startswith("[") and piece.endswith("]"):
            kept.append(piece)
            continue
        if spoken >= max_words:
            break
        kept.append(piece)
        spoken += 1
    truncated = " ".join(kept)
    ends = list(re.finditer(r"[.!?।]", truncated))
    return (truncated[: ends[-1].end()] if ends else truncated).strip()


def write_script(idea: dict, template: str = "N") -> dict:
    """Generate {script_body, caption, hashtags[]} for an approved idea and persist it."""
    idea_id = idea.get("id")
    if idea_id is None:
        raise ValueError("scriptwriter: idea has no 'id' (must be a persisted ideas row).")

    data = _generate_script_json(_build_prompt(idea, template))

    body = (data.get("script_body") or "").strip()
    if not body:
        raise ValueError(f"scriptwriter: empty script_body for idea {idea_id}.")

    title = (data.get("title") or "").strip()

    hashtags = data.get("hashtags")
    if not isinstance(hashtags, list):
        hashtags = []
    hashtags = _ensure_shorts([str(h) for h in hashtags])

    caption = _ensure_disclosure(data.get("caption") or "")

    tags = data.get("tags")
    tags = [str(t).lstrip("#").strip() for t in tags if str(t).strip()] if isinstance(tags, list) else []

    kp = data.get("key_points")
    key_points = ([str(p).strip() for p in kp if str(p).strip()][:5]
                  if isinstance(kp, list) else [])

    max_words = int(config.get("SCRIPT_MAX_WORDS", "550"))
    if len(_visible_words(body)) > max_words:
        log.warning("scriptwriter: idea %s script %d words > %d cap; truncating.",
                    idea_id, len(_visible_words(body)), max_words)
        body = _truncate_to_words(body, max_words)
    if len(_visible_words(body)) < 350:
        log.warning("scriptwriter: idea %s script is short (%d words); a 2-min story needs ~400.",
                    idea_id, len(_visible_words(body)))

    script_id = db.insert_script(idea_id, template, body, caption, hashtags, title or None)
    return {"script_id": script_id, "script_body": body, "caption": caption,
            "hashtags": hashtags, "title": title, "tags": tags, "key_points": key_points}
