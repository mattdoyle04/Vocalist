# test_db.py
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

DB_HOST = "db.ijybidlgsfwhrqwivfzr.supabase.co"  # paste yours
DB_USER = "postgres"
DB_PASS = "He23sucK2eByi43V"        # raw, unencoded
DB_NAME = "postgres"

url = URL.create(
    "postgresql+psycopg",
    username=DB_USER,
    password=DB_PASS,   # URL class handles encoding safely
    host=DB_HOST,
    port=5432,
    database=DB_NAME,
    query={"sslmode": "require"},
)
engine = create_engine(url, pool_pre_ping=True)

with engine.connect() as c:
    print(c.execute(text("select version()")).scalar())
    print("OK")

