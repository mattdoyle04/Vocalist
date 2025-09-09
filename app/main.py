import os, json, base64, secrets, logging, socket as _socket
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, Optional, Set, List
from urllib.parse import urlencode
from urllib.request import Request as URLRequest, urlopen
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, Request, Depends, Body, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

# Prefer IPv4 resolution (belt & braces; REST uses HTTPS anyway)
__orig_getaddrinfo = _socket.getaddrinfo
def __ipv4_first_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    res = __orig_getaddrinfo(host, port, family, type, proto, flags)
    v4 = [r for r in res if r[0] == _socket.AF_INET]
    return v4 or res
_socket.getaddrinfo = __ipv4_first_getaddrinfo

# ---------------- App & config ----------------
load_dotenv()
app = FastAPI()
logger = logging.getLogger("vocalist")
logger.setLevel(logging.INFO)

SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_urlsafe(32)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)
app.mount('/static', StaticFiles(directory=str(Path(__file__).resolve().parent / 'static')), name='static')

TZ = timezone(timedelta(hours=10))  # Australia/Melbourne
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

def get_version() -> str:
    """Return the current VERSION file contents (read per request render)."""
    try:
        return (ROOT_DIR / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        return "0.0.0"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

SUPABASE_URL  = (os.getenv("SUPABASE_URL", "") or "").strip().rstrip("/")
SUPABASE_ANON = (os.getenv("SUPABASE_ANON", "") or os.getenv("SUPABASE_ANON_KEY", "") or "").strip()
SUPABASE_SERVICE_ROLE = (os.getenv("SUPABASE_SERVICE_ROLE", "") or "").strip()
SUPABASE_JWT_ISS = (os.getenv("SUPABASE_JWT_ISS", "") or (SUPABASE_URL + "/auth/v1" if SUPABASE_URL else "")).strip()
SITE_URL = (os.getenv("SITE_URL", "") or "").strip()
ROUND_SECONDS = int(os.getenv("ROUND_SECONDS", "30"))

templates.env.globals.update({
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_ANON": SUPABASE_ANON,
    "SITE_URL": SITE_URL,
    # Expose a callable so templates always show the latest version
    "VERSION": get_version,
})

# ---------------- Minimal Supabase JWT dependency ----------------
def _b64url_json(segment: str) -> Dict[str, Any]:
    padding = '=' * (-len(segment) % 4)
    data = base64.urlsafe_b64decode(segment + padding)
    return json.loads(data.decode("utf-8"))

def require_user(request: Request) -> Dict[str, Any]:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Malformed token")
    try:
        payload = _b64url_json(parts[1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and datetime.now(timezone.utc).timestamp() > float(exp):
        raise HTTPException(status_code=401, detail="Token expired")
    if SUPABASE_JWT_ISS:
        iss = str(payload.get("iss", "")).rstrip("/")
        expected = SUPABASE_JWT_ISS.rstrip("/")
        if iss != expected:
            raise HTTPException(status_code=401, detail="Wrong issuer")
    return payload

# ---------------- Supabase REST client (server-side) ----------------
class SupaREST:
    def __init__(self, base_url: str, service_key: str):
        if not base_url or not service_key:
            raise RuntimeError("Supabase REST not configured")
        self.base = base_url.rstrip("/") + "/rest/v1"
        self.key = service_key

    def _headers(self, extra: Optional[Dict[str,str]]=None) -> Dict[str,str]:
        h = {
            "accept": "application/json",
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Accept-Profile": "public",
            "Content-Profile": "public",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        if extra:
            h.update(extra)
        return h

    def get(self, table_or_path: str, params: Optional[Dict[str, str]]=None) -> Any:
        url = self.base + table_or_path
        if params:
            url += "?" + urlencode(params)
        req = URLRequest(url, headers=self._headers())
        try:
            with urlopen(req, timeout=10) as resp:
                data = resp.read()
                return json.loads(data.decode("utf-8")) if data else None
        except HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            logger.error("Supabase REST GET %s failed: %s %s", table_or_path, e, body)
            raise
        except URLError as e:
            logger.error("Supabase REST GET network error: %s", e)
            raise

    def upsert(self, table: str, obj: Dict[str, Any], on_conflict: Optional[str]=None) -> Any:
        params = {}
        prefer = "resolution=merge-duplicates,return=representation"
        if on_conflict:
            params["on_conflict"] = on_conflict
        url = self.base + "/" + table
        if params:
            url += "?" + urlencode(params)
        req = URLRequest(url, data=json.dumps(obj).encode("utf-8"),
                         headers=self._headers({"Prefer": prefer}), method="POST")
        try:
            with urlopen(req, timeout=10) as resp:
                data = resp.read()
                return json.loads(data.decode("utf-8")) if data else None
        except HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            logger.error("Supabase REST UPSERT %s failed: %s %s", table, e, body)
            raise

    def insert(self, table: str, obj: Dict[str, Any]) -> Any:
        url = self.base + "/" + table
        req = URLRequest(url, data=json.dumps(obj).encode("utf-8"),
                         headers=self._headers(), method="POST")
        try:
            with urlopen(req, timeout=10) as resp:
                data = resp.read()
                return json.loads(data.decode("utf-8")) if data else None
        except HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            logger.error("Supabase REST INSERT %s failed: %s %s", table, e, body)
            raise

SB: Optional[SupaREST] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE:
    try:
        SB = SupaREST(SUPABASE_URL, SUPABASE_SERVICE_ROLE)
    except Exception as e:
        logger.error("Supabase REST init failed: %s", e)

# ---------------- Themes / validation ----------------
DATA_DIR = BASE_DIR / "data" / "themes"
THEMES: List[str] = ["animals"]
_theme_cache: Dict[str, Set[str]] = {}

def _norm_word(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalpha() or ch == " ")

def load_theme_words(theme: str) -> Set[str]:
    key = (theme or "").strip().lower()
    if key in _theme_cache:
        return _theme_cache[key]
    path = DATA_DIR / f"{key}.txt"
    words: Set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                w = _norm_word(line.strip())
                if w: words.add(w)
        logger.info("Loaded theme '%s' (%d words)", key, len(words))
    except Exception as e:
        logger.warning("Theme file missing or unreadable: %s (%s)", path, e)
    _theme_cache[key] = words
    return words

def check_word_server(word: str, letter: str, theme: str) -> Dict[str, Any]:
    w = _norm_word(word)
    if not w or len(w.replace(" ","")) < 3:
        return {"ok": False, "why": "too-short", "theme": theme, "theme_ok": False}
    if not w.startswith(letter.lower()):
        return {"ok": False, "why": "letter-mismatch", "theme": theme, "theme_ok": False}
    bank = load_theme_words(theme)
    return {"ok": True, "theme": theme, "theme_ok": (w in bank if bank else False), "word": w, "theme_mode": "list"}

# ---------------- Letter selection & scoring ----------------
LETTERS = [chr(ord("A") + i) for i in range(26)]
SCRABBLE_COUNTS = {"A":9,"B":2,"C":2,"D":4,"E":12,"F":2,"G":3,"H":2,"I":9,"J":1,"K":1,"L":4,"M":2,"N":6,"O":8,"P":2,"Q":1,"R":6,"S":4,"T":6,"U":4,"V":2,"W":2,"X":1,"Y":2,"Z":1}
SCRABBLE_POINTS = {"A":1,"B":3,"C":3,"D":2,"E":1,"F":4,"G":2,"H":4,"I":1,"J":8,"K":5,"L":1,"M":3,"N":1,"O":1,"P":3,"Q":10,"R":1,"S":1,"T":1,"U":1,"V":4,"W":4,"X":8,"Y":4,"Z":10}

# Custom per-letter scoring (provided rules)
LETTER_POINTS = {
    "A":1, "B":3, "C":2, "D":2, "E":1, "F":3, "G":2, "H":2, "I":1, "J":4, "K":4,
    "L":2, "M":2, "N":1, "O":1, "P":2, "Q":4, "R":1, "S":1, "T":1, "U":2, "V":3,
    "W":4, "X":4, "Y":3, "Z":4,
}

def word_points(word: str) -> int:
    """Sum custom letter values for alphabetic characters in the word."""
    return sum(LETTER_POINTS.get(ch.upper(), 0) for ch in (word or "") if ch.isalpha())

def _hash32(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF

def _rng(seed: int) -> float:
    x = seed or 1
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17)
    x ^= (x << 5) & 0xFFFFFFFF
    return (x & 0xFFFFFFFF) / 2**32

def _weights():
    total = sum(SCRABBLE_COUNTS.values())
    return [SCRABBLE_COUNTS[ch] / total for ch in LETTERS]

def _weighted(seed: int, items: List[str], weights: List[float]) -> str:
    r = _rng(seed); acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc: return it
    return items[-1]

def rarity_bonus(letter: str) -> int:
    return max(0, (SCRABBLE_POINTS.get(letter.upper(), 1) - 1) // 3)

def today_key() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")

def seed_today():
    app.state.round_idx = 0
    app.state.theme_idx = _hash32("T|"+today_key()) % len(THEMES)
    app.state.letter_char = _weighted(_hash32("L|"+today_key()), LETTERS, _weights())

def current_letter() -> str: return getattr(app.state, "letter_char", "A")
def current_theme() -> str:
    idx = getattr(app.state, "theme_idx", 0)
    return THEMES[idx % len(THEMES)]
def advance_round():
    app.state.round_idx = getattr(app.state, "round_idx", 0) + 1
    app.state.theme_idx = (getattr(app.state, "theme_idx", 0) + 1) % len(THEMES)
    seed = _hash32("L|"+today_key()) ^ _hash32(f"R|{app.state.round_idx}")
    app.state.letter_char = _weighted(seed, LETTERS, _weights())

@app.on_event("startup")
def _on_start():
    seed_today()

# ---------------- Supabase REST helpers for data ----------------
def sb_get_player_by_uid(uid: str) -> Optional[Dict[str,Any]]:
    if not SB or not uid: return None
    data = SB.get("/player", {
        "select": "id,name,user_uid",
        "user_uid": f"eq.{uid}",
        "limit": "1"
    })
    return (data or [None])[0]

def sb_upsert_player_from_user(user: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    if not SB: return None
    supa_uid = (user or {}).get("sub") or (user or {}).get("id")
    if not supa_uid: return None
    # Privacy: never store email as display name; generate anonymized label
    try:
        tag = hex(_hash32(str(supa_uid)))[2:6].upper()
    except Exception:
        tag = "0000"
    anon_name = f"Player #{tag}"
    obj = {
        "user_uid": str(supa_uid),
        "name": anon_name,
        "auth_provider": ((user or {}).get("app_metadata") or {}).get("provider","none"),
        "created_at": datetime.now(TZ).replace(tzinfo=None).isoformat(sep=" ")
    }
    rows = SB.upsert("player", obj, on_conflict="user_uid") or []
    return (rows or [None])[0]

def sb_insert_game_run(player_id: int, letter: str, theme: str, duration: int,
                       score: int, words: List[str], inputs: int, off_theme: int):
    if not SB: return
    now = datetime.now(TZ)
    obj = {
        "player_id": int(player_id),
        "play_date": now.date().isoformat(),
        "letter": letter,
        "theme": theme,
        "duration": int(duration),
        "score": int(score),
        "words_json": json.dumps(words),
        "inputs": int(inputs),
        "off_theme_count": int(off_theme),
        "created_at": now.replace(tzinfo=None).isoformat(sep=" "),
    }
    SB.insert("game_run", obj)

def sb_history_for_player(pid: int, limit: int=1000) -> List[Dict[str,Any]]:
    if not SB: return []
    rows = SB.get("/game_run", {
        "select": "play_date,letter,theme,score,duration,words_json,created_at",
        "player_id": f"eq.{pid}",
        "order": "created_at.desc",
        "limit": str(limit),
    }) or []
    out=[]
    for r in rows:
        try: wc = len(json.loads(r.get("words_json") or "[]"))
        except: wc = 0
        out.append({
            "play_date": (r.get("play_date") or "")[:10],
            "letter": str(r.get("letter") or "").upper(),
            "theme": str(r.get("theme") or "").upper(),
            "score": int(r.get("score") or 0),
            "duration": int(r.get("duration") or 0),
            "words": wc
        })
    return out

def _compute_streak(hist: List[Dict[str,Any]]) -> int:
    """Consecutive daily play streak. If played today, streak is up to today;
    otherwise it counts back from the most recent play date."""
    if not hist:
        return 0
    # Build a set of dates you played
    played: Set[date] = set()
    for h in hist:
        try:
            d = datetime.strptime(h["play_date"][:10], "%Y-%m-%d").date()
            played.add(d)
        except Exception:
            continue
    if not played:
        return 0
    today = datetime.now(TZ).date()
    anchor = today if today in played else max(played)
    streak = 0
    d = anchor
    one = timedelta(days=1)
    while d in played:
        streak += 1
        d -= one
    return streak

def sb_stats_for_player(pid: int) -> Dict[str,Any]:
    hist = sb_history_for_player(pid, limit=1000)
    games = len(hist)
    total = sum(h["score"] for h in hist)
    best  = max([h["score"] for h in hist], default=0)
    avg   = (total/games) if games else 0.0
    streak = _compute_streak(hist)
    return {
        "games_played": games,
        "streak": streak,
        "total_score": total,
        "avg_score": avg,
        "best_score": best,
        # kept for backwards-compatibility; not shown in UI now
        "valid_words": total,
        "off_theme": 0,
        "last_played": max([h["play_date"] for h in hist], default="—") if games else "—",
    }

def sb_leaderboard(limit:int=10) -> List[Dict[str,Any]]:
    if not SB: return []
    rows = SB.get("/game_run", {"select":"player_id,score", "limit":"5000"}) or []
    totals: Dict[int,int] = {}
    counts: Dict[int,int] = {}
    for r in rows:
        pid = int(r.get("player_id") or 0)
        totals[pid] = totals.get(pid, 0) + int(r.get("score") or 0)
        counts[pid] = counts.get(pid, 0) + 1
    top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    if not top: return []
    ids = ",".join(str(pid) for pid,_ in top)
    plist = SB.get("/player", {"select":"id,name","id":f"in.({ids})"}) or []
    def anonize(pid: int, name: str) -> str:
        n = (name or "").strip()
        # If looks like an email or empty, replace with anonymized tag
        if ("@" in n) or not n:
            tag = hex(_hash32(str(pid)))[2:6].upper()
            return f"Player #{tag}"
        return n
    name_by_id = {int(p["id"]): anonize(int(p["id"]), p.get("name") or "") for p in plist}
    leaders = [{
        "id": pid,
        "name": name_by_id.get(pid, anonize(pid, "")),
        "score": total,
        "games": counts.get(pid, 0)
    } for pid,total in top]
    return leaders

# ---------------- Pages ----------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def _safe_dialog(request: Request, template: str, ctx: Dict[str,Any]) -> HTMLResponse:
    try:
        return templates.TemplateResponse(template, ctx)
    except Exception as e:
        html = f"""<dialog data-autoshow><div class="modal-card">
<h2>Error</h2><p class="modal-subtle">Couldn’t render <code>{template}</code>.</p>
<pre style="white-space:pre-wrap;font-size:12px">{str(e)}</pre>
<div class="modal-actions"><form method="dialog"><button class="btn btn-primary">Close</button></form></div>
</div></dialog>"""
        return HTMLResponse(html)

@app.get("/my-stats", response_class=HTMLResponse)
def my_stats(request: Request, user: Dict[str,Any] = Depends(require_user)):
    ctx = {"request": request, "stats": {}}
    try:
        player = sb_get_player_by_uid((user or {}).get("sub") or (user or {}).get("id"))
        if player:
            ctx["stats"] = sb_stats_for_player(int(player["id"]))
    except Exception as e:
        logger.exception("my-stats REST error: %s", e)
    return _safe_dialog(request, "_my_stats.html", ctx)

@app.get("/history", response_class=HTMLResponse)
def history(request: Request, user: Dict[str,Any] = Depends(require_user)):
    ctx = {"request": request, "history": []}
    try:
        player = sb_get_player_by_uid((user or {}).get("sub") or (user or {}).get("id"))
        if player:
            ctx["history"] = sb_history_for_player(int(player["id"]), 30)
    except Exception as e:
        logger.exception("history REST error: %s", e)
    return _safe_dialog(request, "_history.html", ctx)

@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard(request: Request, user: Dict[str,Any] = Depends(require_user)):
    ctx = {"request": request, "leaders": [], "me_id": 0}
    try:
        # Identify current player's numeric id for marking "YOU" in UI
        try:
            player = sb_get_player_by_uid((user or {}).get("sub") or (user or {}).get("id"))
            if player: ctx["me_id"] = int(player.get("id") or 0)
        except Exception:
            ctx["me_id"] = 0
        ctx["leaders"] = sb_leaderboard(10)
    except Exception as e:
        logger.exception("leaderboard REST error: %s", e)
    return _safe_dialog(request, "_leaderboard.html", ctx)

# ---------------- API ----------------
@app.get("/health", response_class=JSONResponse)
def health():
    return JSONResponse({"ok": True, "time": datetime.now(TZ).isoformat()})

@app.get("/api/today", response_class=JSONResponse)
def api_today(advance: int = 0):
    if int(advance or 0) == 1:
        advance_round()
    return JSONResponse({
        "letter": current_letter(),
        "theme": current_theme(),
        "roundSeconds": ROUND_SECONDS,
        "letterBonus": rarity_bonus(current_letter()),
    })

@app.post("/api/check-word", response_class=JSONResponse)
def api_check_word(data: Dict[str,Any] = Body(...)):
    return JSONResponse(check_word_server((data or {}).get("word",""), current_letter(), current_theme()))

@app.post("/api/submit-run", response_class=JSONResponse)
def api_submit_run(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    user: Dict[str, Any] = Depends(require_user),
):
    letter = current_letter()
    theme  = current_theme()
    words  = payload.get("words") or []
    inputs = int(payload.get("inputs") or 0)
    duration = int(payload.get("duration") or ROUND_SECONDS)

    valid = 0
    off_theme = 0
    total_points = 0
    for w in words:
        r = check_word_server(w, letter, theme)
        if not r["ok"]:
            continue
        if r["theme_ok"]:
            valid += 1
            total_points += word_points(r.get("word") or w)
        else:
            off_theme += 1

    # New scoring: sum per-letter values across valid (on-theme) words
    base_points = 0
    bonus = 0
    score = int(total_points)

    if SB:
        try:
            supa_uid = (user or {}).get("sub") or (user or {}).get("id")
            player = sb_get_player_by_uid(supa_uid) or sb_upsert_player_from_user(user)
            if player:
                sb_insert_game_run(int(player["id"]), letter, theme, duration, score, words, inputs, off_theme)
            else:
                logger.error("Could not upsert/find player for uid=%s", supa_uid)
        except Exception as e:
            logger.exception("submit-run REST error: %s", e)
    else:
        logger.error("Supabase REST not configured (no SUPABASE_SERVICE_ROLE). Returning score only.")

    return JSONResponse({
        "ok": True,
        "score": score,
        "valid": valid,
        "off_theme": off_theme,
        "letter": letter,
        "theme": theme,
    })

# ---------------- Debug helpers ----------------
@app.get("/debug/auth", response_class=JSONResponse)
def debug_auth():
    return JSONResponse({
        "supabase_url_present": bool(SUPABASE_URL),
        "supabase_anon_present": bool(SUPABASE_ANON),
        "service_role_present": bool(SUPABASE_SERVICE_ROLE),
        "supabase_jwt_iss": SUPABASE_JWT_ISS,
        "site_url": SITE_URL,
        "anon_len": len(SUPABASE_ANON) if SUPABASE_ANON else 0,
    })

@app.get("/debug/db", response_class=JSONResponse)
def debug_db():
    if not SB:
        return JSONResponse({"connected": False, "why": "no SUPABASE_SERVICE_ROLE"}, status_code=500)
    try:
        # simple ping via REST
        SB.get("/player", {"select":"id", "limit":"1"})
        return {"connected": True, "via": "supabase-rest"}
    except Exception as e:
        return JSONResponse({"connected": False, "via":"supabase-rest", "error": str(e)}, status_code=500)

@app.get("/debug/rest-auth", response_class=JSONResponse)
def debug_rest_auth():
    key = SUPABASE_SERVICE_ROLE
    info = {"present": bool(key)}
    try:
        if key:
            header, payload, _sig = key.split(".", 2)
            hdr = _b64url_json(header)
            pl  = _b64url_json(payload)
            info.update({
                "header_alg": hdr.get("alg"),
                "payload_role": pl.get("role"),
                "payload_iss": pl.get("iss"),
                "payload_aud": pl.get("aud"),
            })
    except Exception as e:
        info["error"] = str(e)
    return JSONResponse(info)
