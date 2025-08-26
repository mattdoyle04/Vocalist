"""initial schema"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "c03b6c52f6ba"
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # --- player ---
    op.create_table(
        "player",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("user_uid", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("auth_provider", sa.Text(), server_default="none", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
    )
    op.create_index("ix_player_name", "player", ["name"], unique=False)
    op.create_index("player_user_uid_idx", "player", ["user_uid"], unique=True)

    # --- game_run ---
    op.create_table(
        "game_run",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("player_id", sa.Integer(), sa.ForeignKey("player.id", ondelete="CASCADE"), nullable=False),
        sa.Column("play_date", sa.Date(), nullable=False),
        sa.Column("letter", sa.Text(), nullable=False),
        sa.Column("theme", sa.Text(), nullable=False),
        sa.Column("duration", sa.Integer(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("words_json", sa.Text(), nullable=False),
        sa.Column("inputs", sa.Integer(), nullable=False),
        sa.Column("off_theme_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
    )
    op.create_index("ix_game_run_play_date", "game_run", ["play_date"], unique=False)

def downgrade() -> None:
    op.drop_index("ix_game_run_play_date", table_name="game_run")
    op.drop_table("game_run")

    op.drop_index("player_user_uid_idx", table_name="player")
    op.drop_index("ix_player_name", table_name="player")
    op.drop_table("player")
