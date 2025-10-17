"""add_tool_learning_fields

Revision ID: 16414e12b066
Revises: 15d473f06957
Create Date: 2025-10-17 01:37:36.568092

T033-T034: Add tool learning fields to playbook_bullets for tracking tool usage patterns.
Adds tool_sequence, tool_success_rate, and avg_iterations fields.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


# revision identifiers, used by Alembic.
revision: str = '16414e12b066'
down_revision: Union[str, Sequence[str], None] = '15d473f06957'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - add tool learning fields."""
    # Add tool_sequence to track tool usage patterns
    op.add_column('playbook_bullets',
        sa.Column('tool_sequence', JSON, nullable=True,
                  comment='List[str] - ordered tool names used in this strategy')
    )

    # Add tool_success_rate for strategy effectiveness
    op.add_column('playbook_bullets',
        sa.Column('tool_success_rate', sa.Float(), nullable=True,
                  comment='Success rate for this tool sequence (0.0-1.0)')
    )

    # Add avg_iterations to track convergence efficiency
    op.add_column('playbook_bullets',
        sa.Column('avg_iterations', sa.Float(), nullable=True,
                  comment='Average iterations to completion using this strategy')
    )

    # Add avg_execution_time_ms to track performance
    op.add_column('playbook_bullets',
        sa.Column('avg_execution_time_ms', sa.Float(), nullable=True,
                  comment='Average execution time in milliseconds for tool sequence')
    )


def downgrade() -> None:
    """Downgrade schema - remove tool learning fields."""
    op.drop_column('playbook_bullets', 'avg_execution_time_ms')
    op.drop_column('playbook_bullets', 'avg_iterations')
    op.drop_column('playbook_bullets', 'tool_success_rate')
    op.drop_column('playbook_bullets', 'tool_sequence')
