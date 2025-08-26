# app/main.py
import os
import json
import base64
import secrets
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, Optional, Set, List

from fastapi import FastAPI, Request, Depends, Body, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

# --- DB (optional; used only if DATABASE_URL is set) ---
from sqlalchemy import (
    create_engine, select, Column, Integer, Text, DateTime, Date, ForeignKey
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session as SASession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func, text

try:
    from sqlalchemy.dialects.postgresql import UUID as PGUUID
    HAS_PG = True
except Exception:
    HAS_PG = False

# ------------------------------------------------------
# App & config
# ------------------------------------------------------
load_dotenv()  # load .env

app = FastAPI()

# Session secret for server-side session cookies (not Supabase auth)
SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_urlsafe(32)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# Timezone (Australia/Melbourne)
TZ = timezone(timedelta(hours=10))

# Templates (inject Supabase keys globally so base.html can read them)
templates = Jinja2Templates(directory="app/templates")

ROUND_SECONDS = int(os.getenv("ROUND_SECONDS", "30"))

SUPABASE_URL = (os.getenv("SUPABASE_URL", "") or "").strip()
# Accept either SUPABASE_ANON or SUPABASE_ANON_KEY
SUPABASE_ANON = (os.getenv("SUPABASE_ANON", "") or os.getenv("SUPABASE_ANON_KEY", "") or "").strip()
# Optional issuer; default derived from URL
SUPABASE_JWT_ISS = (os.getenv("SUPABASE_JWT_ISS", "") or (SUPABASE_URL.rstrip("/") + "/auth/v1" if SUPABASE_URL else "")).strip()

templates.env.globals.update({
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_ANON": SUPABASE_ANON,
})

# ------------------------------------------------------
# Static (optional)
# ------------------------------------------------------
# If you later add /static, uncomment:
# from fastapi.staticfiles import StaticFiles
# app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ------------------------------------------------------
# Minimal auth dependency (Supabase JWT)
# ------------------------------------------------------
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
    if isinstance(exp, (int, float)):
        if datetime.now(timezone.utc).timestamp() > float(exp):
            raise HTTPException(status_code=401, detail="Token expired")
    if SUPABASE_JWT_ISS:
        iss = str(payload.get("iss", "")).rstrip("/")
        expected = SUPABASE_JWT_ISS.rstrip("/")
        if iss != expected:
            raise HTTPException(status_code=401, detail="Wrong issuer")
    return payload

# ------------------------------------------------------
# DB setup (optional)
# ------------------------------------------------------
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
# IMPORTANT: default is OFF. Only create tables if you explicitly set AUTO_CREATE_TABLES=1
AUTO_CREATE_TABLES = (os.getenv("AUTO_CREATE_TABLES") or "0").strip() in {"1", "true", "True"}

Base = declarative_base()
SessionLocal = None
engine = None

if DATABASE_URL:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def _uuid_type():
    return PGUUID(as_uuid=False) if HAS_PG else Text

class Player(Base):
    __tablename__ = "player"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name = Column(Text, nullable=False)
    user_uid = Column(_uuid_type(), nullable=True, unique=True)
    auth_provider = Column(Text, nullable=False, server_default="none")
    created_at = Column(DateTime(timezone=False), nullable=False)

class GameRun(Base):
    __tablename__ = "game_run"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    player_id = Column(Integer, ForeignKey("player.id", ondelete="CASCADE"), nullable=False)
    play_date = Column(Date, nullable=False)
    letter = Column(Text, nullable=False)
    theme = Column(Text, nullable=False)
    duration = Column(Integer, nullable=False)
    score = Column(Integer, nullable=False)
    words_json = Column(Text, nullable=False)
    inputs = Column(Integer, nullable=False)
    off_theme_count = Column(Integer, nullable=False, server_default="0")
    created_at = Column(DateTime(timezone=False), nullable=False)

def create_db_and_tables():
    if engine:
        Base.metadata.create_all(bind=engine)

# ------------------------------------------------------
# Themes / validation
# ------------------------------------------------------
DATA_DIR = "app/data/themes"
THEMES: List[str] = ["ANIMALS", "FOOD", "COLORS", "SPORTS", "COUNTRIES", "DOGBREEDS"]
_theme_cache: Dict[str, Set[str]] = {}

def _norm_word(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalpha())

def load_theme_words(theme: str) -> Set[str]:
    if theme in _theme_cache:
        return _theme_cache[theme]
    path = os.path.join(DATA_DIR, f"{theme}.txt")
    words: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = _norm_word(line.strip())
                if w:
                    words.add(w)
    except FileNotFoundError:
        words = set()
    _theme_cache[theme] = words
    return words

def check_word_server(word: str, letter: str, theme: str) -> Dict[str, Any]:
    w = _norm_word(word)
    if not w or len(w) < 3:
        return {"ok": False, "why": "too-short", "theme": theme, "theme_ok": False}
    if not w.startswith(letter.lower()):
        return {"ok": False, "why": "letter-mismatch", "theme": theme, "theme_ok": False}
    theme_words = load_theme_words(theme)
    theme_ok = w in theme_words if theme_words else False
    return {"ok": True, "theme": theme, "theme_ok": theme_ok, "word": w, "theme_mode": "loose"}

# ------------------------------------------------------
# Letter selection & scoring
# ------------------------------------------------------
LETTERS_ORDER = [chr(ord("A") + i) for i in range(26)]
SCRABBLE_COUNTS = {
    "A":9,"B":2,"C":2,"D":4,"E":12,"F":2,"G":3,"H":2,"I":9,"J":1,"K":1,"L":4,
    "M":2,"N":6,"O":8,"P":2,"Q":1,"R":6,"S":4,"T":6,"U":4,"V":2,"W":2,"X":1,"Y":2,"Z":1
}
SCRABBLE_POINTS = {
    "A":1,"B":3,"C":3,"D":2,"E":1,"F":4,"G":2,"H":4,"I":1,"J":8,"K":5,"L":1,
    "M":3,"N":1,"O":1,"P":3,"Q":10,"R":1,"S":1,"T":1,"U":1,"V":4,"W":4,"X":8,"Y":4,"Z":10
}

def _hash32(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF

def _rng_float(seed: int) -> float:
    x = seed or 1
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17)
    x ^= (x << 5) & 0xFFFFFFFF
    return (x & 0xFFFFFFFF) / 2**32

def _scrabble_weights():
    total = sum(SCRABBLE_COUNTS.values())
    return [SCRABBLE_COUNTS[ch] / total for ch in LETTERS_ORDER]

def _weighted_choice(seed: int, items: List[str], weights: List[float]) -> str:
    r = _rng_float(seed)
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]

def rarity_bonus(letter: str) -> int:
    pts = SCRABBLE_POINTS.get(letter.upper(), 1)
    return max(0, (pts - 1) // 3)

def today_key() -> str:
    now = datetime.now(TZ)
    return now.strftime("%Y-%m-%d")

def seed_indices_for_today():
    key = today_key()
    app.state.round_idx = 0
    app.state.theme_idx = _hash32("T|" + key) % len(THEMES)
    weights = _scrabble_weights()
    seed = _hash32("L|" + key)
    app.state.letter_char = _weighted_choice(seed, LETTERS_ORDER, weights)

def current_letter() -> str:
    return getattr(app.state, "letter_char", "A")

def current_theme() -> str:
    idx = getattr(app.state, "theme_idx", 0)
    return THEMES[idx % len(THEMES)]

def advance_round():
    app.state.round_idx = getattr(app.state, "round_idx", 0) + 1
    app.state.theme_idx = (getattr(app.state, "theme_idx", 0) + 1) % len(THEMES)
    key = today_key()
    weights = _scrabble_weights()
    seed = _hash32("L|" + key) ^ _hash32(f"R|{app.state.round_idx}")
    app.state.letter_char = _weighted_choice(seed, LETTERS_ORDER, weights)

def get_player_by_user(db: SessionLocal, user_payload: Dict[str, Any]):
    """Find the Player row for the current Supabase user (by sub/id)."""
    if not db or not user_payload:
        return None
    supa_uid = str(user_payload.get("sub") or user_payload.get("id") or "")
    if not supa_uid:
        return None
    return db.execute(select(Player).where(Player.user_uid == supa_uid)).scalar_one_or_none()

def build_stats_for_player(db: SessionLocal, player_id: int) -> Dict[str, Any]:
    """Aggregate simple lifetime stats for a player."""
    stats = {
        "games_played": 0,
        "total_score": 0,
        "avg_score": 0.0,
        "best_score": 0,
        "valid_words": 0,
        "off_theme": 0,
        "last_played": "—",
    }
    if not db or not player_id:
        return stats

    q = select(
        func.count(GameRun.id),
        func.coalesce(func.sum(GameRun.score), 0),
        func.coalesce(func.max(GameRun.score), 0),
        func.coalesce(func.sum(GameRun.off_theme_count), 0),
        func.coalesce(func.max(GameRun.play_date), None),
    ).where(GameRun.player_id == player_id)
    games, total, best, off_theme, last_played = db.execute(q).one()
    stats["games_played"] = int(games or 0)
    stats["total_score"]  = int(total or 0)
    stats["best_score"]   = int(best or 0)
    stats["off_theme"]    = int(off_theme or 0)
    stats["last_played"]  = (last_played.isoformat() if last_played else "—")
    stats["avg_score"]    = (stats["total_score"] / stats["games_played"]) if stats["games_played"] else 0.0

    # Estimate valid_words from stored score & per-letter bonus:
    # score = valid * (1 + rarity_bonus(letter))  -> valid = score / (1+bonus)
    valid_est = 0
    for (score, letter) in db.execute(
        select(GameRun.score, GameRun.letter).where(GameRun.player_id == player_id)
    ):
        bonus = rarity_bonus(letter or "")
        per  = 1 + bonus
        if per > 0:
            valid_est += int((score or 0) // per)
    stats["valid_words"] = int(valid_est)
    return stats

def build_history_for_player(db: SessionLocal, player_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    """Recent games for a player (most recent first)."""
    rows: List[Dict[str, Any]] = []
    if not db or not player_id:
        return rows
    q = (
        select(
            GameRun.play_date, GameRun.letter, GameRun.theme,
            GameRun.score, GameRun.duration, GameRun.words_json
        )
        .where(GameRun.player_id == player_id)
        .order_by(GameRun.created_at.desc())
        .limit(limit)
    )
    for d, letter, theme, score, dur, words_json in db.execute(q):
        try:
            words_count = len(json.loads(words_json or "[]"))
        except Exception:
            words_count = 0
        rows.append({
            "play_date": d.isoformat() if hasattr(d, "isoformat") else str(d),
            "letter": letter,
            "theme": theme,
            "score": int(score or 0),
            "duration": int(dur or 0),
            "words": words_count,
        })
    return rows

def build_leaderboard(db: SessionLocal, limit: int = 10) -> List[Dict[str, Any]]:
    """Top players by total score."""
    leaders: List[Dict[str, Any]] = []
    if not db:
        return leaders
    q = (
        select(
            Player.name.label("name"),
            func.coalesce(func.sum(GameRun.score), 0).label("total"),
            func.count(GameRun.id).label("games"),
        )
        .join(GameRun, GameRun.player_id == Player.id)
        .group_by(Player.id, Player.name)
        .order_by(text("total DESC"))
        .limit(limit)
    )
    for name, total, games in db.execute(q):
        leaders.append({
            "name": name or "Player",
            "score": int(total or 0),
            "games": int(games or 0),
        })
    return leaders

# ------------------------------------------------------
# Startup
# ------------------------------------------------------
@app.on_event("startup")
def _on_start():
    # Will NOT create tables unless AUTO_CREATE_TABLES=1 is set in the environment
    if AUTO_CREATE_TABLES and engine:
        create_db_and_tables()
    seed_indices_for_today()

# ------------------------------------------------------
# Helpers: safe template rendering for modals (prevents 500s)
# ------------------------------------------------------
def _safe_dialog(request: Request, template_name: str, title_fallback: str) -> HTMLResponse:
    """
    Try to render a template. If it fails (missing file, error, etc.),
    return a minimal <dialog> with the error so you don't get a 500.
    """
    try:
        return templates.TemplateResponse(template_name, {"request": request})
    except Exception as e:
        html = f"""<dialog data-autoshow>
  <div class="modal-card">
    <h2>{title_fallback}</h2>
    <p class="modal-subtle">Couldn’t render template "<code>{template_name}</code>".</p>
    <pre style="white-space:pre-wrap;font-size:12px">{str(e)}</pre>
    <div class="modal-actions">
      <form method="dialog"><button class="btn btn-primary">Close</button></form>
    </div>
  </div>
</dialog>"""
        return HTMLResponse(html)

# ------------------------------------------------------
# Routes: pages
# ------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Globals (SUPABASE_URL/ANON) are injected automatically
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/my-stats", response_class=HTMLResponse)
def my_stats(request: Request, user: Dict[str, Any] = Depends(require_user)):
    # Default empty payload in case DB isn't configured
    ctx = {"request": request, "stats": {}}
    if SessionLocal is None:
        return templates.TemplateResponse("_my_stats.html", ctx)

    with SessionLocal() as db:
        player = get_player_by_user(db, user)
        if not player:
            return templates.TemplateResponse("_my_stats.html", ctx)
        stats = build_stats_for_player(db, player.id)
        ctx["stats"] = stats
        return templates.TemplateResponse("_my_stats.html", ctx)

@app.get("/history", response_class=HTMLResponse)
def history(request: Request, user: Dict[str, Any] = Depends(require_user)):
    ctx = {"request": request, "history": []}
    if SessionLocal is None:
        return templates.TemplateResponse("_history.html", ctx)

    with SessionLocal() as db:
        player = get_player_by_user(db, user)
        if not player:
            return templates.TemplateResponse("_history.html", ctx)
        rows = build_history_for_player(db, player.id, limit=30)
        ctx["history"] = rows
        return templates.TemplateResponse("_history.html", ctx)

@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard(request: Request, user: Dict[str, Any] = Depends(require_user)):
    ctx = {"request": request, "leaders": []}
    if SessionLocal is None:
        return templates.TemplateResponse("_leaderboard.html", ctx)

    with SessionLocal() as db:
        rows = build_leaderboard(db, limit=10)
        ctx["leaders"] = rows
        return templates.TemplateResponse("_leaderboard.html", ctx)

# ------------------------------------------------------
# Routes: api
# ------------------------------------------------------
@app.get("/health", response_class=JSONResponse)
def health():
    return JSONResponse({"ok": True, "time": datetime.now(TZ).isoformat()})

# in api_today()
@app.get("/api/today", response_class=JSONResponse)
def api_today(request: Request, advance: int = 0):
    if int(advance or 0) == 1:
        advance_round()
    return JSONResponse({
        "letter": current_letter(),
        "theme": current_theme(),
        "roundSeconds": ROUND_SECONDS,   # was 60
        "letterBonus": rarity_bonus(current_letter()),
    })


@app.post("/api/check-word", response_class=JSONResponse)
def api_check_word(data: Dict[str, Any] = Body(...)):
    letter = current_letter()
    theme = current_theme()
    word = (data or {}).get("word", "")
    return JSONResponse(check_word_server(word, letter, theme))

@app.post("/api/submit-run", response_class=JSONResponse)
def api_submit_run(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    user: Dict[str, Any] = Depends(require_user),
):
    # current pair
    letter = current_letter()
    theme = current_theme()

    words = payload.get("words") or []
    inputs = int(payload.get("inputs") or 0)
    duration = int(payload.get("duration") or 0)

    valid = 0
    off_theme = 0
    for w in words:
        res = check_word_server(w, letter, theme)
        if not res["ok"]:
            continue
        if res["theme_ok"]:
            valid += 1
        else:
            off_theme += 1

    base_points_per_word = 1
    bonus_per_word = rarity_bonus(letter)
    score = valid * (base_points_per_word + bonus_per_word)

    # Persist if DB configured
    now = datetime.now(TZ)
    play_d = now.date()

    if SessionLocal is not None:
        try:
            with SessionLocal() as db:  # type: SASession
                supa_uid = (user or {}).get("sub") or (user or {}).get("id")
                auth_email = (user or {}).get("email") or "Anonymous"
                provider = ((user or {}).get("app_metadata") or {}).get("provider", "none")

                player = None
                if supa_uid:
                    player = db.execute(select(Player).where(Player.user_uid == supa_uid)).scalar_one_or_none()
                if not player:
                    player = Player(
                        name=auth_email,
                        user_uid=str(supa_uid) if supa_uid else None,
                        auth_provider=provider,
                        created_at=now,
                    )
                    db.add(player)
                    db.flush()

                gr = GameRun(
                    player_id=player.id,
                    play_date=play_d,
                    letter=letter,
                    theme=theme,
                    duration=duration,
                    score=score,
                    words_json=json.dumps(words),
                    inputs=inputs,
                    off_theme_count=off_theme,
                    created_at=now,
                )
                db.add(gr)
                db.commit()
        except SQLAlchemyError:
            # Ignore DB errors during gameplay; still return score
            pass

    return JSONResponse({
        "ok": True,
        "score": score,
        "valid": valid,
        "off_theme": off_theme,
        "letter": letter,
        "theme": theme,
        "bonus_per_word": bonus_per_word,
        "base_points": base_points_per_word,
    })

# ------------------------------------------------------
# Debug: sanity check (never returns the anon key)
# ------------------------------------------------------
@app.get("/debug/auth", response_class=JSONResponse)
def debug_auth():
    from urllib.parse import urlparse
    host = ""
    try:
        host = urlparse(SUPABASE_URL).netloc
    except Exception:
        pass
    return JSONResponse({
        "supabase_url_present": bool(SUPABASE_URL),
        "supabase_anon_present": bool(SUPABASE_ANON),
        "supabase_jwt_iss": SUPABASE_JWT_ISS,
        "supabase_host": host,
        "anon_len": len(SUPABASE_ANON) if SUPABASE_ANON else 0,
    })
