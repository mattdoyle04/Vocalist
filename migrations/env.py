import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import make_url
from alembic import context
from sqlmodel import SQLModel

# --- load .env (project root) early, overriding shell exports ---
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)
except Exception:
    pass

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---- read and normalise DATABASE_URL ----
db_url = (os.getenv("DATABASE_URL") or "").strip().strip('"').strip("'")

# Common fixes:
if db_url.startswith("postgres://"):  # old alias
    db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
elif db_url.startswith("postgresql://") and "+psycopg" not in db_url:
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

# Fail early with a helpful message if it’s still bad
try:
    if not db_url:
        raise ValueError("empty")
    make_url(db_url)  # validates format
except Exception:
    raise SystemExit(
        "\n[alembic] Invalid or missing DATABASE_URL.\n"
        "Expected like: postgresql+psycopg://postgres:PASSWORD@db.<ref>.supabase.co:5432/postgres?sslmode=require\n"
        "Tip: generate one with sqlalchemy.engine.URL (see make_db_url.py).\n"
    )

config.set_main_option("sqlalchemy.url", db_url)
target_metadata = SQLModel.metadata

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata,
                      literal_binds=True, dialect_opts={"paramstyle":"named"},
                      compare_type=True, compare_server_default=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}), prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          compare_type=True, compare_server_default=True)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()