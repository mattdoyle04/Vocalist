# make_db_url.py
from sqlalchemy.engine import URL
print(URL.create(
    "postgresql+psycopg",
    username="postgres",
    password="He23sucK2eByi43V",   # raw (no URL-encoding needed)
    host="db.ijybidlgsfwhrqwivfzr.supabase.co",
    port=5432,
    database="postgres",
    query={"sslmode":"require"},
))
