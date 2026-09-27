"""Module 2 — Approval (Telegram Morning Digest)."""
from __future__ import annotations

import html
import logging
import os
import time
from urllib.parse import urlparse

import requests

from src import config, db

log = logging.getLogger(__name__)

_API_BASE = os.environ.get("TELEGRAM_API_BASE_URL", "https://api.telegram.org").rstrip("/")
_BASE = _API_BASE + "/bot{token}/{method}"
_TIMEOUT = 40

_DECISION_TEXT = {
    "approved": "✅ Approved",
    "rejected": "❌ Rejected",
    "passed": "⏭️ Passed",
    "capped": "⚠️ Daily approval cap reached — not approved",
    "unknown": "Could not process that.",
}


def _api(method: str, **params):
    url = _BASE.format(token=config.require("TELEGRAM_BOT_TOKEN"), method=method)
    resp = requests.post(url, json=params, timeout=_TIMEOUT)
    if resp.status_code >= 400:
        log.error("telegram API error: method=%s status=%s body=%s",
                  method, resp.status_code, resp.text)
        return None
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"telegram {method} failed: {data.get('description', data)}")
    return data["result"]


def _keyboard(idea_id: int) -> dict:
    return {"inline_keyboard": [[
        {"text": "✅ Approve", "callback_data": f"a:{idea_id}"},
        {"text": "❌ Reject", "callback_data": f"r:{idea_id}"},
        {"text": "⏭️ Pass", "callback_data": f"p:{idea_id}"},
    ]]}


_SOURCES_SHOWN = 3
_NO_PREVIEW = {"is_disabled": True}


def _source_label(url: str) -> str:
    host = urlparse(url if "://" in url else "http://" + url).netloc.lower()
    host = host.split("@")[-1].split(":")[0]
    for prefix in ("www.", "m.", "amp."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return "Google News" if host == "news.google.com" else (host or "link")


def _format_sources(sources: list[str]) -> str:
    firsts: dict[str, str] = {}
    for s in sources:
        firsts.setdefault(_source_label(str(s)), str(s))
    if not firsts:
        return "📰 <b>no sources!</b>"
    shown = list(firsts.items())[:_SOURCES_SHOWN]
    links = " · ".join(
        f'<a href="{html.escape(u, quote=True)}">{html.escape(label)}</a>'
        for label, u in shown
    )
    rest = len(sources) - len(shown)
    return f"📰 {links}" + (f" <i>+{rest} more</i>" if rest > 0 else "")


def _format_idea(idea: dict) -> str:
    def esc(x):
        return html.escape(str(x or ""))

    score = idea.get("est_score")
    score_str = f"{float(score):.2f}" if score is not None else "—"
    return (
        f"<b>{esc(idea.get('title'))}</b>\n"
        f"<i>Hook:</i> {esc(idea.get('hook'))}\n"
        f"<i>Why it matters:</i> {esc(idea.get('angle'))}\n"
        f"⭐ {score_str}  {_format_sources(idea.get('sources') or [])}"
    )


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _decided_message(label: str, msg: dict) -> dict:
    original = msg.get("text") or msg.get("caption") or ""
    prefix = f"{label}\n\n" if original else label
    shift = _utf16_len(prefix)
    entities = [{"type": "bold", "offset": 0, "length": _utf16_len(label)}]
    for ent in msg.get("entities") or msg.get("caption_entities") or []:
        if isinstance(ent, dict) and "offset" in ent:
            entities.append({**ent, "offset": int(ent["offset"]) + shift})
    return {"text": prefix + original, "entities": entities,
            "link_preview_options": _NO_PREVIEW}


def send_digest() -> int:
    ideas = db.get_pending_ideas()
    if not ideas:
        log.info("approval: no pending ideas to send.")
        return 0
    chat = config.require("TELEGRAM_CHAT_ID")
    log.info("approval: sending digest to chat_id=%r (base=%s)", chat, _API_BASE)
    sent = 0
    for idea in ideas:
        text = _format_idea(idea)
        # Telegram hard limit: 4096 chars. Truncate defensively.
        if len(text) > 4000:
            text = text[:4000] + "…"
        result = _api("sendMessage", chat_id=chat, text=text,
                      parse_mode="HTML",
                      link_preview_options=_NO_PREVIEW,
                      reply_markup=_keyboard(idea["id"]))
        if result is None:
            log.warning("approval: skipping idea %s (send failed)", idea.get("id"))
        else:
            sent += 1
    log.info("approval: sent %d/%d ideas to the digest.", sent, len(ideas))
    return sent


def _apply_callback(action: str, idea_id: int, cap: int) -> str:
    if action == "a":
        if len(db.get_approved_ideas()) >= cap:
            return "capped"
        db.set_idea_status(idea_id, "approved")
        return "approved"
    if action == "r":
        db.set_idea_status(idea_id, "rejected")
        return "rejected"
    if action == "p":
        db.set_idea_status(idea_id, "passed")
        return "passed"
    return "unknown"


def _handle_update(update: dict, cap: int) -> str | None:
    cq = update.get("callback_query")
    if not cq:
        return None
    msg = cq.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    if str(chat_id) != str(config.require("TELEGRAM_CHAT_ID")):
        log.warning("approval: ignoring callback from unexpected chat %s", chat_id)
        return None

    action, _, sid = (cq.get("data") or "").partition(":")
    try:
        idea_id = int(sid)
    except ValueError:
        idea_id = None
    decision = _apply_callback(action, idea_id, cap) if idea_id is not None else "unknown"

    _api("answerCallbackQuery", callback_query_id=cq["id"], text=_DECISION_TEXT[decision])
    if msg.get("message_id"):
        _api("editMessageText", chat_id=chat_id, message_id=msg["message_id"],
             **_decided_message(_DECISION_TEXT[decision], msg))
    return decision


def process_responses(max_seconds: int = 600, poll_timeout: int = 25, cap: int | None = None) -> int:
    cap = cap if cap is not None else int(config.get("APPROVAL_CAP", "3"))
    deadline = time.monotonic() + max_seconds
    offset = None
    approved = 0
    while time.monotonic() < deadline:
        if not db.get_pending_ideas():
            log.info("approval: all ideas decided.")
            break
        updates = _api("getUpdates", offset=offset, timeout=poll_timeout,
                       allowed_updates=["callback_query"])
        for up in updates:
            offset = up["update_id"] + 1
            if _handle_update(up, cap) == "approved":
                approved += 1
    log.info("approval: %d approved this run.", approved)
    return approved
