"""Telegram command bot — Vercel serverless webhook (stdlib only, zero deps)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler

IST = timezone(timedelta(hours=5, minutes=30))

HELP = (
    "<b>But It Matters — control bot</b>\n"
    "/makeshort [n] — start a batch (n ideas, default 5)\n"
    "/today — Shorts published today (IST)\n"
    "/stats — totals + today + top performer\n"
    "/pending — ideas waiting for approval\n"
    "/latest — last published links\n"
    "/help — this message"
)


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _http(method: str, url: str, headers: dict | None = None, payload: dict | None = None,
          timeout: int = 15) -> tuple[int, str]:
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def tg_send(chat_id, text: str) -> None:
    token = _env("TELEGRAM_BOT_TOKEN")
    if not token:
        return
    try:
        _http("POST", f"https://api.telegram.org/bot{token}/sendMessage",
              {"Content-Type": "application/json"},
              {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": True})
    except Exception:
        pass


def tg_api(method: str, payload: dict) -> tuple[int, str] | None:
    token = _env("TELEGRAM_BOT_TOKEN")
    if not token:
        return None
    try:
        return _http("POST", f"https://api.telegram.org/bot{token}/{method}",
                     {"Content-Type": "application/json"}, payload)
    except Exception:
        return None


def gh_dispatch_make_short(ideas: int, wait_min: int = 30) -> bool:
    repo, pat = _env("GH_REPO"), _env("GH_PAT")
    if not (repo and pat):
        return False
    url = f"https://api.github.com/repos/{repo}/actions/workflows/make-short.yml/dispatches"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "reel-factory-bot",
        "Content-Type": "application/json",
    }
    try:
        status, _ = _http("POST", url, headers,
                          {"ref": "main", "inputs": {"ideas": str(ideas), "wait_min": str(wait_min)}})
        return status == 204
    except Exception:
        return False


def sb_get(path: str) -> list:
    base, key = _env("SUPABASE_URL"), _env("SUPABASE_KEY")
    if not (base and key):
        return []
    headers = {"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"}
    status, body = _http("GET", f"{base}/rest/v1/{path}", headers)
    if status >= 300 or not body:
        return []
    try:
        data = json.loads(body)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def sb_patch(path: str, payload: dict) -> bool:
    base, key = _env("SUPABASE_URL"), _env("SUPABASE_KEY")
    if not (base and key):
        return False
    headers = {"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json",
               "Content-Type": "application/json", "Prefer": "return=minimal"}
    status, _ = _http("PATCH", f"{base}/rest/v1/{path}", headers, payload)
    return status < 300


def _ist_today_start_utc_iso() -> str:
    start_ist = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    return start_ist.astimezone(timezone.utc).isoformat()


def _published_filter() -> str:
    return "platform=eq.youtube&external_id=not.is.null"


_ORDER_NEWEST = "order=published_at.desc.nullslast"


def posts_today() -> list:
    since = urllib.parse.quote(_ist_today_start_utc_iso(), safe="")
    return sb_get(f"posts?select=url,published_at&{_published_filter()}"
                  f"&published_at=gte.{since}&{_ORDER_NEWEST}")


def posts_total() -> int:
    return len(sb_get(f"posts?select=id&{_published_filter()}"))


def latest_posts(n: int = 5) -> list:
    return sb_get(f"posts?select=url,published_at&{_published_filter()}"
                  f"&{_ORDER_NEWEST}&limit={n}")


def top_performer() -> str | None:
    rows = sb_get("analytics?select=views,posts(scripts(title,ideas(title)))&order=views.desc&limit=1")
    for r in rows:
        try:
            script = r["posts"]["scripts"]
            title = (script.get("title") or "").strip() or script["ideas"]["title"]
            return f'"{title}" — {int(r.get("views") or 0):,} views'
        except (TypeError, KeyError):
            return None
    return None


def pending_ideas() -> list:
    rows = sb_get("ideas?select=title&status=eq.pending&order=est_score.desc&limit=10")
    return [r.get("title", "") for r in rows if r.get("title")]


_DECISION_TEXT = {
    "approved": "✅ Approved",
    "rejected": "❌ Rejected",
    "passed": "⏭️ Passed",
    "capped": "⚠️ Daily approval cap reached — not approved",
    "unknown": "Could not process that.",
}


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def decided_message(label: str, msg: dict) -> dict:
    original = msg.get("text") or msg.get("caption") or ""
    prefix = f"{label}\n\n" if original else label
    shift = _utf16_len(prefix)
    entities = [{"type": "bold", "offset": 0, "length": _utf16_len(label)}]
    for ent in msg.get("entities") or msg.get("caption_entities") or []:
        if isinstance(ent, dict) and "offset" in ent:
            entities.append({**ent, "offset": int(ent["offset"]) + shift})
    return {"text": prefix + original, "entities": entities,
            "link_preview_options": {"is_disabled": True}}


_DEFAULT_APPROVAL_CAP = 3


def approval_cap() -> int:
    try:
        return int(_env("APPROVAL_CAP") or _DEFAULT_APPROVAL_CAP)
    except ValueError:
        return _DEFAULT_APPROVAL_CAP


def approved_count() -> int:
    return len(sb_get("ideas?select=id&status=eq.approved"))


def set_idea_status(idea_id: int, status: str) -> bool:
    return sb_patch(f"ideas?id=eq.{idea_id}", {"status": status})


def apply_callback_action(action: str, idea_id: int) -> str:
    if action == "a":
        if approved_count() >= approval_cap():
            return "capped"
        return "approved" if set_idea_status(idea_id, "approved") else "unknown"
    if action == "r":
        return "rejected" if set_idea_status(idea_id, "rejected") else "unknown"
    if action == "p":
        return "passed" if set_idea_status(idea_id, "passed") else "unknown"
    return "unknown"


def handle_callback(cq: dict) -> None:
    msg = cq.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    auth = _env("TELEGRAM_CHAT_ID")
    if auth and str(chat_id) != str(auth):
        return
    action, _, sid = (cq.get("data") or "").partition(":")
    try:
        idea_id = int(sid)
    except ValueError:
        idea_id = 0
    decision = apply_callback_action(action, idea_id) if idea_id else "unknown"
    label = _DECISION_TEXT[decision]
    if cq.get("id"):
        tg_api("answerCallbackQuery", {"callback_query_id": cq["id"], "text": label})
    if chat_id and msg.get("message_id"):
        tg_api("editMessageText", {"chat_id": chat_id, "message_id": msg["message_id"],
                                   **decided_message(label, msg)})


def parse_command(text: str | None) -> tuple[str | None, str]:
    text = (text or "").strip()
    if not text.startswith("/"):
        return None, ""
    parts = text.split()
    cmd = parts[0].lstrip("/").split("@")[0].lower()
    return cmd, " ".join(parts[1:]).strip()


def dispatch(cmd: str, arg: str) -> str:
    if cmd in ("start", "help"):
        return HELP
    if cmd == "makeshort":
        n = max(1, min(8, int(arg))) if arg.isdigit() else 5
        ok = gh_dispatch_make_short(n)
        return (f"🎬 Starting a batch of <b>{n}</b> ideas — the approval digest lands in ~1–2 min."
                if ok else "⚠️ Couldn't start the Action.")
    if cmd == "today":
        rows = posts_today()
        if not rows:
            return "📅 <b>Today (IST):</b> 0 Shorts so far."
        links = "\n".join(f"• {r['url']}" for r in rows if r.get("url"))
        return f"📅 <b>Today (IST): {len(rows)} Short(s)</b>\n{links}"
    if cmd == "stats":
        total, today, top = posts_total(), len(posts_today()), top_performer()
        lines = ["📊 <b>Channel stats</b>",
                 f"• Total published: <b>{total}</b>",
                 f"• Today (IST): <b>{today}</b>"]
        if top:
            lines.append(f"• Top performer: {top}")
        return "\n".join(lines)
    if cmd == "pending":
        titles = pending_ideas()
        if not titles:
            return "✅ No ideas waiting."
        body = "\n".join(f"• {t}" for t in titles)
        return f"⏳ <b>{len(titles)} idea(s):</b>\n{body}"
    if cmd == "latest":
        rows = latest_posts(5)
        if not rows:
            return "No Shorts published yet."
        links = "\n".join(f"• {r['url']}" for r in rows if r.get("url"))
        return f"🎬 <b>Latest Shorts</b>\n{links}"
    return "Unknown command. Send /help."


def handle_update(update: dict) -> None:
    if update.get("callback_query"):
        handle_callback(update["callback_query"])
        return
    msg = update.get("message") or update.get("edited_message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    auth = _env("TELEGRAM_CHAT_ID")
    if auth and str(chat_id) != str(auth):
        return
    cmd, arg = parse_command(msg.get("text"))
    if not cmd:
        return
    reply = dispatch(cmd, arg)
    if reply:
        tg_send(chat_id, reply)


class handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, text: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def do_GET(self):
        self._reply(200, "reel-factory bot ok")

    def do_POST(self):
        secret = _env("WEBHOOK_SECRET")
        if secret and self.headers.get("X-Telegram-Bot-Api-Secret-Token") != secret:
            self._reply(401, "unauthorized")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            handle_update(json.loads(raw or b"{}"))
        except Exception:
            pass
        self._reply(200, "ok")
