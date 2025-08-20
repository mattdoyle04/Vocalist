from dotenv import load_dotenv
from pathlib import Path

# load the project root .env explicitly (main.py lives in app/)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

import os
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_JWKS_URL = os.getenv("SUPABASE_JWKS_URL") or (SUPABASE_URL.rstrip("/") + "/auth/v1/.well-known/jwks.json" if SUPABASE_URL else "")
SUPABASE_ISS = os.getenv("SUPABASE_ISS") or (SUPABASE_URL.rstrip("/") + "/auth/v1" if SUPABASE_URL else "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///vocalist.db")

import re
import json, time, requests
from collections import deque
from typing import Optional, List, Tuple, Dict, Iterable
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request, HTTPException, Body, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import uuid
from sqlalchemy import func
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlmodel import SQLModel, Field, Session, select, create_engine
from wordfreq import zipf_frequency

import jwt  # PyJWT
from jwt import InvalidTokenError, algorithms

# ======================================================================================
# App setup
# ======================================================================================
app = FastAPI(title="Vocalist")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "dev-secret-change-me"))
templates = Jinja2Templates(directory="app/templates")

THEME_MODE = os.getenv("THEME_MODE", "soft").strip().lower()  # 'soft' or 'strict'
THEMES_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "themes"

# In-memory theme sets: "ANIMALS" -> set([...])
THEME_SETS: dict[str, set[str]] = {}

_PLURAL_PATTERNS = [
    (re.compile(r"ies$"), "y"),        # cities -> city
    (re.compile(r"ves$"), "f"),        # wolves -> wolf (rough)
    (re.compile(r"s$"), ""),           # cats -> cat (very rough)
]

def _norm_word(w: str) -> str:
    t = (w or "").strip().lower()
    if len(t) < 3:
        return t
    for pat, repl in _PLURAL_PATTERNS:
        if pat.search(t):
            return pat.sub(repl, t)
    return t

def load_theme_sets() -> None:
    """Load all *.txt in THEMES_DIR into THEME_SETS (uppercase key)."""
    THEME_SETS.clear()
    if not THEMES_DIR.exists():
        return
    for p in THEMES_DIR.glob("*.txt"):
        key = p.stem.strip().upper()
        words = set()
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    words.add(_norm_word(line))
        except Exception:
            pass
        if words:
            THEME_SETS[key] = words

def theme_match(word: str, theme: str) -> bool:
    """Return True if `word` fits the `theme` based on curated lists."""
    key = (theme or "").strip().upper()
    if not key or key not in THEME_SETS:
        return False
    return _norm_word(word) in THEME_SETS[key]

# ======================================================================================
# Time & daily seed (server-truth, Melbourne)
# ======================================================================================
MEL = ZoneInfo("Australia/Melbourne")
LETTERS = list("ETAOINSHRDLCUMWFGYPBVKJXQZ")
THEMES = ["ANIMALS","FOOD","SPORTS","TRAVEL","MUSIC","NATURE","TECH","COLORS","CITIES","MOVIES"]

def mel_now() -> datetime:
    return datetime.now(tz=MEL)

def mel_today() -> date:
    return mel_now().date()

def _fnv1a32(s: str) -> int:
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def _mulberry32(a: int):
    state = a & 0xFFFFFFFF
    def rnd():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = (t ^ (t >> 15)) * ((t | 1) & 0xFFFFFFFF) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * ((t | 61) & 0xFFFFFFFF) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 2**32
    return rnd

def daily_seed(d: date) -> Tuple[str, str]:
    key = d.isoformat()
    rng = _mulberry32(_fnv1a32(key))
    letter = LETTERS[int(rng() * len(LETTERS))]
    theme  = THEMES[int(rng() * len(THEMES))]
    return letter, theme

# ======================================================================================
# Database
# ======================================================================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///vocalist.db")
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

class Player(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    # use real UUID type so Postgres gets a UUID-typed bind (not VARCHAR)
    user_uid: Optional[uuid.UUID] = Field(
        default=None,
        sa_column=Column(PGUUID(as_uuid=True), unique=True, nullable=True)
    )
    auth_provider: str = Field(default="none")
    created_at: datetime = Field(default_factory=mel_now)

class GameRun(SQLModel, table=True):
    __tablename__ = "game_run"
    id: Optional[int] = Field(default=None, primary_key=True)
    player_id: int = Field(foreign_key="player.id")
    play_date: date = Field(index=True)
    letter: str
    theme: str
    duration: int
    score: int
    words_json: str
    inputs: int
    off_theme_count: int = 0   # <-- add this
    created_at: datetime = Field(default_factory=mel_now)


def _ensure_auth_columns():
    """Tiny migration to add user_uid/auth_provider to Player if missing."""
    with engine.connect() as con:
        if DATABASE_URL.startswith("sqlite"):
            cols = [r[1] for r in con.exec_driver_sql("PRAGMA table_info(player)").fetchall()]
            if "user_uid" not in cols:
                con.exec_driver_sql("ALTER TABLE player ADD COLUMN user_uid TEXT")
                con.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_player_user_uid ON player(user_uid)")
            if "auth_provider" not in cols:
                con.exec_driver_sql("ALTER TABLE player ADD COLUMN auth_provider TEXT DEFAULT 'none'")
        else:
            # ---- Postgres: use IF EXISTS on the table, IF NOT EXISTS on the column ----
            con.exec_driver_sql(
                "ALTER TABLE IF EXISTS public.player "
                "ADD COLUMN IF NOT EXISTS user_uid uuid UNIQUE"
            )
            con.exec_driver_sql(
                "ALTER TABLE IF EXISTS public.player "
                "ADD COLUMN IF NOT EXISTS auth_provider text NOT NULL DEFAULT 'none'"
            )
            con.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS player_user_uid_idx ON public.player(user_uid)"
            )

# ======================================================================================
# Supabase Auth (JWT) — verify Authorization: Bearer <token>
# ======================================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
# Prefer the .well-known path by default
SUPABASE_JWKS_URL = os.getenv("SUPABASE_JWKS_URL") or (SUPABASE_URL.rstrip("/") + "/auth/v1/.well-known/jwks.json" if SUPABASE_URL else "")
SUPABASE_ISS = os.getenv("SUPABASE_ISS") or (SUPABASE_URL.rstrip("/") + "/auth/v1" if SUPABASE_URL else "")

_JWKS_KEYS: Dict[str, object] = {}  # kid -> public key

def _load_jwks():
    if not SUPABASE_JWKS_URL:
        return
    try:
        resp = requests.get(SUPABASE_JWKS_URL, timeout=5)
        data = resp.json()
        _JWKS_KEYS.clear()
        for k in data.get("keys", []):
            kid = k.get("kid")
            if kid:
                _JWKS_KEYS[kid] = algorithms.RSAAlgorithm.from_jwk(json.dumps(k))
    except Exception:
        pass

def _verify_supabase_jwt(token: str) -> Optional[dict]:
    if not token:
        return None
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        key = _JWKS_KEYS.get(kid)
        if not key:
            _load_jwks()
            key = _JWKS_KEYS.get(kid)
            if not key:
                return None
        return jwt.decode(
            token, key=key, algorithms=["RS256"],
            options={"verify_aud": False},  # we don't enforce aud here
            issuer=SUPABASE_ISS if SUPABASE_ISS else None,
        )
    except InvalidTokenError:
        return None

@app.middleware("http")
async def supabase_auth_middleware(request: Request, call_next):
    request.state.user_uid = None
    request.state.user_email = None
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        claims = _verify_supabase_jwt(token)
        if claims:
            request.state.user_uid = claims.get("sub")
            request.state.user_email = claims.get("email")
    return await call_next(request)

def _resolve_player(s: Session, request: Request) -> Player:
    """Prefer Supabase user if present; else fall back to session name."""
    uid = getattr(request.state, "user_uid", None)
    sess_name = (request.session.get("name") or "").strip()

    if uid:
        me = s.exec(select(Player).where(Player.user_uid == uid)).first()
        if me:
            if sess_name and sess_name != me.name:
                taken = s.exec(select(Player).where(Player.name == sess_name, Player.id != me.id)).first()
                if not taken:
                    me.name = sess_name
                    s.add(me); s.commit(); s.refresh(me)
            return me

        # bind existing session-named player if available & unclaimed; else create new
        if sess_name:
            existing_by_name = s.exec(select(Player).where(Player.name == sess_name)).first()
            if existing_by_name and not existing_by_name.user_uid:
                existing_by_name.user_uid = uid
                existing_by_name.auth_provider = "supabase"
                s.add(existing_by_name); s.commit(); s.refresh(existing_by_name)
                return existing_by_name

        new_name = sess_name or f"Player{uid[:8]}"
        me = Player(name=new_name, user_uid=uid, auth_provider="supabase")
        s.add(me); s.commit(); s.refresh(me)
        return me

    # anonymous (session-name) path
    if not sess_name:
        raise HTTPException(401, "Set your name first")
    me = s.exec(select(Player).where(Player.name == sess_name)).first()
    if not me:
        me = Player(name=sess_name, auth_provider="none")
        s.add(me); s.commit(); s.refresh(me)
    return me

# ======================================================================================
# Dictionary & scoring
# ======================================================================================
DICTIONARY_THRESHOLD = float(os.getenv("WORD_ZIPF_MIN", "2.3"))

def _is_dictionary_word(w: str) -> bool:
    t = (w or "").strip().lower()
    if len(t) < 3:
        return False
    for ch in t:
        if not (ch.isalpha() or ch in "'-"):
            return False
    return zipf_frequency(t, "en") >= DICTIONARY_THRESHOLD

def _canonise_words(words: List[str], required_letter: str) -> List[str]:
    if not required_letter:
        return []
    L = required_letter.lower()
    seen = set()
    canon: List[str] = []
    for w in words or []:
        if not isinstance(w, str):
            continue
        t = (w or "").strip().lower()
        if len(t) >= 3 and t.startswith(L) and _is_dictionary_word(t) and t not in seen:
            seen.add(t)
            canon.append(t)
    return canon

SCRABBLE_POINTS = {**{c:1 for c in "eaionrtlsu"}, **{c:2 for c in "dg"}, **{c:3 for c in "bcmp"}, **{c:4 for c in "fhvwy"}, "k":5, "j":8, "x":8, "q":10, "z":10}
def scrabble_score(w: str) -> int:
    return sum(SCRABBLE_POINTS.get(ch, 0) for ch in (w or "").lower())

# ======================================================================================
# Rate limits + Security headers
# ======================================================================================
# path_prefix -> (max_requests, window_seconds)
RATE_LIMITS = {
    "/api/check-word": (40, 10),
    "/api/submit-run": (10, 60),
}
_RATE_BUCKETS: Dict[Tuple[str, str], deque] = {}

def _client_key(request: Request) -> str:
    # Prefer authenticated user (stable across devices)
    uid = getattr(request.state, "user_uid", None)
    if uid:
        return "u:" + uid

    # Next-best: raw session cookie value (no SessionMiddleware needed)
    sess_cookie = request.cookies.get("session")
    if sess_cookie:
        return "s:" + sess_cookie  # in-memory only; fine for rate limiting

    # Fallback: IP (respects proxy header if present)
    xff = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    ip = xff or (request.client.host if request.client else "unknown")
    return "ip:" + ip

@app.middleware("http")
async def security_and_limits(request: Request, call_next):
    # ----- rate limit -----
    path = request.url.path
    limit_cfg = None
    for prefix, cfg in RATE_LIMITS.items():
        if path.startswith(prefix):
            limit_cfg = cfg
            break
    if limit_cfg:
        max_req, window = limit_cfg
        now = time.time()
        key = (path, _client_key(request))
        dq = _RATE_BUCKETS.get(key)
        if dq is None:
            dq = deque()
            _RATE_BUCKETS[key] = dq
        cutoff = now - window
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= max_req:
            return JSONResponse({"detail": "Too Many Requests"}, status_code=429, headers={"Retry-After": str(int(window))})
        dq.append(now)

    # ----- proceed -----
    resp = await call_next(request)

    # ----- security headers -----
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    resp.headers.setdefault("Permissions-Policy", "microphone=(), geolocation=(), camera=()")
    return resp

# ======================================================================================
# Startup
# ======================================================================================
@app.on_event("startup")
def _on_start():
    _load_jwks()
    load_theme_sets()

# ======================================================================================
# Pages & tiny API
# ======================================================================================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    name = request.session.get("name") or ""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "player_name": name,
            "supabase_url": os.getenv("SUPABASE_URL", ""),
            "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY", ""),
        },
    )

@app.get("/api/today")
def api_today():
    t = mel_today()
    letter, theme = daily_seed(t)
    return JSONResponse({"date": t.isoformat(), "letter": letter, "theme": theme})

# ======================================================================================
# API: register, check-word, submit, leaderboard
# ======================================================================================
@app.post("/api/register", response_class=HTMLResponse)
async def register(request: Request):
    form = await request.form()
    name = (form.get("name") or "").strip()
    if not name or len(name) > 40:
        raise HTTPException(400, "Invalid name")
    request.session["name"] = name

    with Session(engine) as s:
        uid = getattr(request.state, "user_uid", None)
        if uid:
            me = s.exec(select(Player).where(Player.user_uid == uid)).first()
            if me:
                someone = s.exec(select(Player).where(Player.name == name, Player.id != me.id)).first()
                if someone:
                    raise HTTPException(409, "Name already taken")
                me.name = name
                s.add(me); s.commit()
            else:
                someone = s.exec(select(Player).where(Player.name == name)).first()
                if someone:
                    raise HTTPException(409, "Name already taken")
                s.add(Player(name=name, user_uid=uid, auth_provider="supabase")); s.commit()
        else:
            existing = s.exec(select(Player).where(Player.name == name)).first()
            if not existing:
                s.add(Player(name=name, auth_provider="none")); s.commit()

    return HTMLResponse(f'<span id="playerBadge" class="badge text-bg-secondary">Player: {name}</span>')

@app.post("/api/check-word")
async def check_word(word: str = Body(..., embed=True)):
    today = mel_today()
    letter, theme = daily_seed(today)
    t = (word or "").strip().lower()

    ok_letter = len(t) >= 3 and t.startswith(letter.lower())
    ok_dict = _is_dictionary_word(t)
    ok = ok_letter and ok_dict

    theme_ok = theme_match(t, theme) if ok else False

    return JSONResponse({
        "ok": ok,
        "letter": letter,
        "theme": theme,
        "theme_ok": theme_ok,
        "theme_mode": THEME_MODE,
    })

@app.post("/api/submit-run")
async def submit_run(request: Request):
    body = await request.json()
    words = body.get("words") or []
    inputs = int(body.get("inputs") or 0)
    duration = int(body.get("duration") or 60)

    today = mel_today()
    letter, theme = daily_seed(today)

    canon = _canonise_words(words, required_letter=letter)
    canon_on_theme = [w for w in canon if theme_match(w, theme)]
    off_theme_count = max(0, len(canon) - len(canon_on_theme))

    score = len(canon_on_theme)

    with Session(engine) as s:
        player = _resolve_player(s, request)
        gr = GameRun(
            player_id=player.id,
            play_date=today,
            letter=letter,
            theme=theme,
            duration=duration,
            score=score,
            words_json=json.dumps(canon_on_theme),  # store only on-theme words
            inputs=inputs,
            off_theme_count=off_theme_count,
        )
        s.add(gr); s.commit()

    return JSONResponse({"ok": True, "score": score, "letter": letter, "theme": theme})

@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard(request: Request):
    """Per-player totals for today's Melbourne date (Best day score)."""
    today = mel_today()
    with Session(engine) as s:
        q = (
            select(
                Player.name.label("name"),
                func.sum(GameRun.score).label("total_score"),
                func.count(GameRun.id).label("runs"),
                func.min(GameRun.created_at).label("first_time"),
            )
            .join(GameRun, GameRun.player_id == Player.id)
            .where(GameRun.play_date == today)
            .group_by(Player.id, Player.name)
            .order_by(func.sum(GameRun.score).desc(), func.min(GameRun.created_at).asc())
            .limit(20)
        )
        rows = s.exec(q).all()

    items = []
    for name, total_score, runs, first_time in rows:
        items.append({
            "name": name,
            "total": int(total_score or 0),
            "runs": int(runs or 0),
            "time": (first_time.astimezone(MEL).strftime("%H:%M") if first_time else "--:--"),
        })

    return templates.TemplateResponse("_leaderboard.html", {"request": request, "items": items, "date": today})

# ======================================================================================
# Player stats & history (server-computed)
# ======================================================================================
def _compute_streaks(dates: List[date]) -> Tuple[int, int]:
    if not dates:
        return 0, 0
    uniq = sorted(set(dates))
    longest = 1
    cur = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i-1] + timedelta(days=1):
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    latest = uniq[-1]
    cur_chain = 1
    d = latest
    s = set(uniq)
    while (d - timedelta(days=1)) in s:
        d = d - timedelta(days=1)
        cur_chain += 1
    return cur_chain, longest

def _player_stats(player_id: int) -> Dict:
    with Session(engine) as s:
        runs: List[GameRun] = s.exec(
            select(GameRun).where(GameRun.player_id == player_id).order_by(GameRun.play_date.asc(), GameRun.created_at.asc())
        ).all()

    total_games = len(runs)
    if total_games == 0:
        return {
            "total_games": 0, "total_valid": 0, "total_invalid": 0,
            "avg_accuracy": 0, "avg_wpm": 0, "avg_word_len": 0,
            "best_day_score": 0, "current_streak": 0, "longest_streak": 0,
            "best_scrabble_word": "", "best_scrabble_points": 0,
        }

    total_valid = sum(r.score for r in runs)
    total_inputs = sum(r.inputs for r in runs)
    total_invalid = max(0, total_inputs - total_valid)
    total_seconds = sum(max(1, r.duration) for r in runs)

    total_letters = 0
    best_word = ""
    best_points = 0
    day_totals: Dict[date, int] = {}
    play_dates: List[date] = []

    for r in runs:
        play_dates.append(r.play_date)
        day_totals[r.play_date] = day_totals.get(r.play_date, 0) + r.score
        try:
            words: Iterable[str] = json.loads(r.words_json or "[]")
        except Exception:
            words = []
        for w in words:
            total_letters += len(w)
            pts = scrabble_score(w)
            if pts > best_points or (pts == best_points and len(w) > len(best_word)):
                best_word, best_points = w.upper(), pts

    avg_accuracy = round(100 * total_valid / (total_valid + total_invalid), 0) if (total_valid + total_invalid) > 0 else 0
    avg_wpm = round(total_valid / (total_seconds / 60.0), 2) if total_seconds > 0 else 0
    avg_word_len = round(total_letters / total_valid, 2) if total_valid > 0 else 0
    best_day_score = max(day_totals.values()) if day_totals else 0
    current_streak, longest_streak = _compute_streaks(play_dates)

    return {
        "total_games": total_games,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "avg_accuracy": int(avg_accuracy),
        "avg_wpm": avg_wpm,
        "avg_word_len": avg_word_len,
        "best_day_score": best_day_score,
        "current_streak": current_streak,
        "longest_streak": longest_streak,
        "best_scrabble_word": best_word,
        "best_scrabble_points": best_points,
    }

@app.get("/my-stats", response_class=HTMLResponse)
def my_stats(request: Request):
    name_sess = request.session.get("name")
    with Session(engine) as s:
        try:
            player = _resolve_player(s, request)
        except HTTPException:
            html = """
            <div class="modal fade show" style="display:block;" tabindex="-1" aria-modal="true" role="dialog">
              <div class="modal-dialog modal-dialog-centered"><div class="modal-content">
                <div class="modal-header"><h5 class="modal-title">Stats</h5>
                  <button type="button" class="btn-close" onclick="this.closest('.modal').remove()"></button>
                </div>
                <div class="modal-body"><div class="alert alert-warning mb-0">Set your name first to view stats.</div></div>
                <div class="modal-footer"><button class="btn btn-primary" onclick="this.closest('.modal').remove()">OK</button></div>
              </div></div>
            </div>
            """
            return HTMLResponse(html)
        stats = _player_stats(player.id)
        name = player.name or name_sess or "Player"
    return templates.TemplateResponse("_my_stats.html", {"request": request, "name": name, "stats": stats})

@app.get("/history", response_class=HTMLResponse)
def history(request: Request, page: int = Query(1, ge=1), partial: int = Query(0, ge=0, le=1)):
    with Session(engine) as s:
        try:
            player = _resolve_player(s, request)
        except HTTPException:
            html = """
            <div class="modal fade show" style="display:block;" tabindex="-1" aria-modal="true" role="dialog">
              <div class="modal-dialog modal-dialog-centered"><div class="modal-content">
                <div class="modal-header"><h5 class="modal-title">History</h5>
                  <button type="button" class="btn-close" onclick="this.closest('.modal').remove()"></button>
                </div>
                <div class="modal-body"><div class="alert alert-warning mb-0">Set your name first to view history.</div></div>
                <div class="modal-footer"><button class="btn btn-primary" onclick="this.closest('.modal').remove()">OK</button></div>
              </div></div>
            </div>
            """
            return HTMLResponse(html)

        PAGE_SIZE = 10
        offset = (page - 1) * PAGE_SIZE
        q = (select(GameRun)
             .where(GameRun.player_id == player.id)
             .order_by(GameRun.created_at.desc())
             .limit(PAGE_SIZE + 1)
             .offset(offset))
        runs = s.exec(q).all()

    has_more = len(runs) > PAGE_SIZE
    if has_more:
        runs = runs[:PAGE_SIZE]

    items = []
    for r in runs:
        try:
            words = json.loads(r.words_json or "[]")
        except Exception:
            words = []
        total_attempts = max(0, int(r.inputs or 0))
        score = int(r.score or 0)
        acc = int(round(100 * score / total_attempts)) if total_attempts > 0 else 0
        wpm = round(score / (max(1, int(r.duration or 60)) / 60.0), 2)
        avg_len = round((sum(len(w) for w in words) / score), 2) if score > 0 else 0
        items.append({
            "date": r.play_date.isoformat(),
            "time": r.created_at.astimezone(MEL).strftime("%H:%M"),
            "letter": r.letter,
            "theme": r.theme,
            "score": score,
            "attempts": total_attempts,
            "acc": acc,
            "wpm": wpm,
            "avg_len": avg_len,
            "words": [str(w).upper() for w in words],
        })

    ctx = {
        "request": request,
        "items": items,
        "page": page,
        "next_page": (page + 1) if has_more else None,
        "partial": bool(partial),
        "player_name": player.name,
    }
    return templates.TemplateResponse("_history.html", ctx)