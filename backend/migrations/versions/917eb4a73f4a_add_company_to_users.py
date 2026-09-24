"""add company to users

Revision ID: 917eb4a73f4a
Revises: 7ffa6c72280a
Create Date: 2026-09-24 02:18:24.593283

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "917eb4a73f4a"
down_revision: Union[str, Sequence[str], None] = "7ffa6c72280a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.add_column(
        "users",
        sa.Column(
            "company_id",
            sa.Integer(),
            nullable=False,
        ),
    )

    op.create_index(
        op.f("ix_users_company_id"),
        "users",
        ["company_id"],
        unique=False,
    )

    op.create_foreign_key(
        "fk_users_company_id_companies",
        "users",
        "companies",
        ["company_id"],
        ["id"],
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_constraint(
        "fk_users_company_id_companies",
        "users",
        type_="foreignkey",
    )

    op.drop_index(
        op.f("ix_users_company_id"),
        table_name="users",
    )

    op.drop_column(
        "users",
        "company_id",
    )