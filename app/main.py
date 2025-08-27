# app/main.py
import os, json, base64, secrets, logging, socket as _socket
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, Optional, Set, List

from fastapi import FastAPI, Request, Depends, Body, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

from sqlalchemy import (
    create_engine, select, Column, Integer, Text, DateTime, Date, ForeignKey,
    func, text as sqla_text
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session as SASession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

try:
    from sqlalchemy.dialects.postgresql import UUID as PGUUID
    HAS_PG = True
except Exception:
    HAS_PG = False

# -------- Force IPv4 DNS resolution (avoid IPv6 AAAA targets) --------
__orig_getaddrinfo = _socket.getaddrinfo
def __ipv4_first_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    res = __orig_getaddrinfo(host, port, family, type, proto, flags)
    v4 = [r for r in res if r[0] == _socket.AF_INET]
    return v4 or res
_socket.getaddrinfo = __ipv4_first_getaddrinfo

# -------- App & config --------
load_dotenv()
app = FastAPI()
logger = logging.getLogger("vocalist")
logger.setLevel(logging.INFO)

SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_urlsafe(32)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

TZ = timezone(timedelta(hours=10))  # Australia/Melbourne

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

SUPABASE_URL  = (os.getenv("SUPABASE_URL", "") or "").strip()
SUPABASE_ANON = (os.getenv("SUPABASE_ANON", "") or os.getenv("SUPABASE_ANON_KEY", "") or "").strip()
SUPABASE_JWT_ISS = (os.getenv("SUPABASE_JWT_ISS", "") or (SUPABASE_URL.rstrip("/") + "/auth/v1" if SUPABASE_URL else "")).strip()
SITE_URL = (os.getenv("SITE_URL", "") or "").strip()

templates.env.globals.update({
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_ANON": SUPABASE_ANON,
    "SITE_URL": SITE_URL,
})

ROUND_SECONDS = int(os.getenv("ROUND_SECONDS", "30"))
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
AUTO_CREATE_TABLES = (os.getenv("AUTO_CREATE_TABLES") or "0").strip() in {"1","true","True"}

# -------- Minimal Supabase JWT dependency --------
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

# -------- DB setup --------
Base = declarative_base()
engine = None
SessionLocal = None

if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        poolclass=NullPool,   # don’t hold connections; works well with Supabase 6543
        future=True,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def _uuid_type():
    return PGUUID(as_uuid=False) if HAS_PG else Text

class Player(Base):
    __tablename__ = "player"
    id          = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name        = Column(Text, nullable=False)
    user_uid    = Column(_uuid_type(), unique=True, nullable=True)
    auth_provider = Column(Text, nullable=False, server_default="none")
    created_at  = Column(DateTime(timezone=False), nullable=False)

class GameRun(Base):
    __tablename__ = "game_run"
    id          = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    player_id   = Column(Integer, ForeignKey("player.id", ondelete="CASCADE"), nullable=False)
    play_date   = Column(Date, nullable=False)
    letter      = Column(Text, nullable=False)
    theme       = Column(Text, nullable=False)
    duration    = Column(Integer, nullable=False)
    score       = Column(Integer, nullable=False)
    words_json  = Column(Text, nullable=False)
    inputs      = Column(Integer, nullable=False)
    off_theme_count = Column(Integer, nullable=False, server_default="0")
    created_at  = Column(DateTime(timezone=False), nullable=False)

def create_db_and_tables():
    if engine:
        Base.metadata.create_all(bind=engine)

# -------- Themes / validation --------
DATA_DIR = BASE_DIR / "data" / "themes"
THEMES: List[str] = ["animals","food","colors","sports","countries","dogbreeds"]
_theme_cache: Dict[str, Set[str]] = {}

def _norm_word(s:str)->str:
    return "".join(ch for ch in s.lower() if ch.isalpha() or ch == " ")

def load_theme_words(theme:str)->Set[str]:
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

def check_word_server(word:str, letter:str, theme:str)->Dict[str,Any]:
    w = _norm_word(word)
    if not w or len(w.replace(" ","")) < 3:
        return {"ok":False,"why":"too-short","theme":theme,"theme_ok":False}
    if not w.startswith(letter.lower()):
        return {"ok":False,"why":"letter-mismatch","theme":theme,"theme_ok":False}
    bank = load_theme_words(theme)
    return {"ok":True,"theme":theme,"theme_ok": (w in bank if bank else False),"word":w,"theme_mode":"list"}

# -------- Letter selection & scoring --------
LETTERS = [chr(ord("A")+i) for i in range(26)]
SCRABBLE_COUNTS = {"A":9,"B":2,"C":2,"D":4,"E":12,"F":2,"G":3,"H":2,"I":9,"J":1,"K":1,"L":4,"M":2,"N":6,"O":8,"P":2,"Q":1,"R":6,"S":4,"T":6,"U":4,"V":2,"W":2,"X":1,"Y":2,"Z":1}
SCRABBLE_POINTS = {"A":1,"B":3,"C":3,"D":2,"E":1,"F":4,"G":2,"H":4,"I":1,"J":8,"K":5,"L":1,"M":3,"N":1,"O":1,"P":3,"Q":10,"R":1,"S":1,"T":1,"U":1,"V":4,"W":4,"X":8,"Y":4,"Z":10}

def _hash32(s:str)->int:
    import zlib
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF

def _rng(seed:int)->float:
    x = seed or 1
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17)
    x ^= (x << 5) & 0xFFFFFFFF
    return (x & 0xFFFFFFFF) / 2**32

def _weights():
    total = sum(SCRABBLE_COUNTS.values())
    return [SCRABBLE_COUNTS[ch]/total for ch in LETTERS]

def _weighted(seed:int, items:List[str], weights:List[float])->str:
    r = _rng(seed); acc=0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc: return it
    return items[-1]

def rarity_bonus(letter:str)->int:
    return max(0, (SCRABBLE_POINTS.get(letter.upper(),1)-1)//3)

def today_key()->str:
    return datetime.now(TZ).strftime("%Y-%m-%d")

def seed_today():
    app.state.round_idx = 0
    app.state.theme_idx = _hash32("T|"+today_key()) % len(THEMES)
    app.state.letter_char = _weighted(_hash32("L|"+today_key()), LETTERS, _weights())

def current_letter()->str: return getattr(app.state,"letter_char","A")
def current_theme()->str:
    idx = getattr(app.state,"theme_idx",0)
    return THEMES[idx % len(THEMES)]
def advance_round():
    app.state.round_idx = getattr(app.state,"round_idx",0)+1
    app.state.theme_idx = (getattr(app.state,"theme_idx",0)+1) % len(THEMES)
    seed = _hash32("L|"+today_key()) ^ _hash32(f"R|{app.state.round_idx}")
    app.state.letter_char = _weighted(seed, LETTERS, _weights())

# -------- Startup --------
@app.on_event("startup")
def _on_start():
    if AUTO_CREATE_TABLES and engine:
        create_db_and_tables()
    seed_today()

# -------- DB helpers --------
def get_player_by_user(db:SASession, user:Dict[str,Any]):
    if not db or not user: return None
    supa_uid = str(user.get("sub") or user.get("id") or "")
    if not supa_uid: return None
    return db.execute(select(Player).where(Player.user_uid == supa_uid)).scalar_one_or_none()

def build_stats_for_player(db:SASession, pid:int)->Dict[str,Any]:
    stats = {"games_played":0,"total_score":0,"avg_score":0.0,"best_score":0,"valid_words":0,"off_theme":0,"last_played":"—"}
    if not db or not pid: return stats
    q = select(
        func.count(GameRun.id),
        func.coalesce(func.sum(GameRun.score),0),
        func.coalesce(func.max(GameRun.score),0),
        func.coalesce(func.sum(GameRun.off_theme_count),0),
        func.coalesce(func.max(GameRun.play_date), None),
    ).where(GameRun.player_id == pid)
    games, total, best, off_theme, last_played = db.execute(q).one()
    stats.update({
        "games_played": int(games or 0),
        "total_score":  int(total or 0),
        "best_score":   int(best or 0),
        "off_theme":    int(off_theme or 0),
        "last_played":  (last_played.isoformat() if last_played else "—")
    })
    stats["avg_score"] = (stats["total_score"]/stats["games_played"]) if stats["games_played"] else 0.0
    # rough estimate from score/bonus
    valid = 0
    for (score, letter) in db.execute(select(GameRun.score, GameRun.letter).where(GameRun.player_id == pid)):
        per = 1 + max(0, (SCRABBLE_POINTS.get((letter or "").upper(),1)-1)//3)
        valid += int((score or 0)//max(1,per))
    stats["valid_words"] = int(valid)
    return stats

def build_history_for_player(db:SASession, pid:int, limit:int=30)->List[Dict[str,Any]]:
    rows=[]
    if not db or not pid: return rows
    q = (select(GameRun.play_date, GameRun.letter, GameRun.theme, GameRun.score, GameRun.duration, GameRun.words_json)
         .where(GameRun.player_id == pid).order_by(GameRun.created_at.desc()).limit(limit))
    for d, letter, theme, score, dur, words_json in db.execute(q):
        try: words_count = len(json.loads(words_json or "[]"))
        except: words_count = 0
        rows.append({
            "play_date": d.isoformat() if hasattr(d,"isoformat") else str(d),
            "letter": (letter or "").upper(),
            "theme": (theme or "").upper(),
            "score": int(score or 0),
            "duration": int(dur or 0),
            "words": words_count,
        })
    return rows

def build_leaderboard(db:SASession, limit:int=10)->List[Dict[str,Any]]:
    leaders=[]
    if not db: return leaders
    q = (select(
            Player.name.label("name"),
            func.coalesce(func.sum(GameRun.score),0).label("total"),
            func.count(GameRun.id).label("games"))
         .join(GameRun, GameRun.player_id == Player.id)
         .group_by(Player.id, Player.name)
         .order_by(sqla_text("total DESC"))
         .limit(limit))
    for name, total, games in db.execute(q):
        leaders.append({"name": name or "Player", "score": int(total or 0), "games": int(games or 0)})
    return leaders

# -------- Pages --------
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
    if SessionLocal is None:
        return _safe_dialog(request, "_my_stats.html", ctx)
    try:
        with SessionLocal() as db:
            player = get_player_by_user(db, user)
            if player:
                ctx["stats"] = build_stats_for_player(db, player.id)
    except SQLAlchemyError as e:
        logger.exception("my-stats DB error: %s", e)
    return _safe_dialog(request, "_my_stats.html", ctx)

@app.get("/history", response_class=HTMLResponse)
def history(request: Request, user: Dict[str,Any] = Depends(require_user)):
    ctx = {"request": request, "history": []}
    if SessionLocal is None:
        return _safe_dialog(request, "_history.html", ctx)
    try:
        with SessionLocal() as db:
            player = get_player_by_user(db, user)
            if player:
                ctx["history"] = build_history_for_player(db, player.id, 30)
    except SQLAlchemyError as e:
        logger.exception("history DB error: %s", e)
    return _safe_dialog(request, "_history.html", ctx)

@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard(request: Request, user: Dict[str,Any] = Depends(require_user)):
    ctx = {"request": request, "leaders": []}
    if SessionLocal is None:
        return _safe_dialog(request, "_leaderboard.html", ctx)
    try:
        with SessionLocal() as db:
            ctx["leaders"] = build_leaderboard(db, 10)
    except SQLAlchemyError as e:
        logger.exception("leaderboard DB error: %s", e)
    return _safe_dialog(request, "_leaderboard.html", ctx)

# -------- API --------
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
    words = payload.get("words") or []
    inputs = int(payload.get("inputs") or 0)
    duration = int(payload.get("duration") or ROUND_SECONDS)

    valid = 0
    off_theme = 0
    for w in words:
        r = check_word_server(w, letter, theme)
        if not r["ok"]: continue
        if r["theme_ok"]: valid += 1
        else: off_theme += 1

    base_points = 1
    bonus = rarity_bonus(letter)
    score = valid * (base_points + bonus)

    if SessionLocal is not None:
        now = datetime.now(TZ)
        try:
            with SessionLocal() as db:
                supa_uid = (user or {}).get("sub") or (user or {}).get("id")
                auth_email = (user or {}).get("email") or "Anonymous"
                provider = ((user or {}).get("app_metadata") or {}).get("provider", "none")

                player = None
                if supa_uid:
                    player = db.execute(select(Player).where(Player.user_uid == supa_uid)).scalar_one_or_none()
                if not player:
                    player = Player(
                        name=auth_email, user_uid=str(supa_uid) if supa_uid else None,
                        auth_provider=provider, created_at=now
                    )
                    db.add(player)
                    db.flush()

                db.add(GameRun(
                    player_id=player.id, play_date=now.date(),
                    letter=letter, theme=theme, duration=duration, score=score,
                    words_json=json.dumps(words), inputs=inputs, off_theme_count=off_theme,
                    created_at=now
                ))
                db.commit()
        except SQLAlchemyError as e:
            logger.exception("submit-run DB error: %s", e)

    return JSONResponse({
        "ok": True, "score": score, "valid": valid, "off_theme": off_theme,
        "letter": letter, "theme": theme, "bonus_per_word": bonus, "base_points": base_points
    })

# -------- Debug helpers --------
@app.get("/debug/auth", response_class=JSONResponse)
def debug_auth():
    return JSONResponse({
        "supabase_url_present": bool(SUPABASE_URL),
        "supabase_anon_present": bool(SUPABASE_ANON),
        "supabase_jwt_iss": SUPABASE_JWT_ISS,
        "site_url": SITE_URL,
        "anon_len": len(SUPABASE_ANON) if SUPABASE_ANON else 0,
    })

@app.get("/debug/db", response_class=JSONResponse)
def debug_db():
    if not engine:
        return {"connected": False, "why": "no DATABASE_URL"}
    try:
        with engine.connect() as conn:
            conn.execute(sqla_text("select 1"))
        return {"connected": True}
    except Exception as e:
        return JSONResponse({"connected": False, "error": str(e)}, status_code=500)
