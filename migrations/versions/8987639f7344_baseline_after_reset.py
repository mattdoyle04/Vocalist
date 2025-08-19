"""baseline after reset

Revision ID: 8987639f7344
Revises: 0fb12212b765
Create Date: 2025-08-18 22:38:14.525623

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8987639f7344'
down_revision: Union[str, Sequence[str], None] = '0fb12212b765'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
