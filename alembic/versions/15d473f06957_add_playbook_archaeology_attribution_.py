"""add_playbook_archaeology_attribution_metadata

Revision ID: 15d473f06957
Revises: 4c0c6073b96e
Create Date: 2025-10-17 01:08:45.976909

T058: Add attribution metadata to playbook_bullets for traceability.
Adds source_task_id, source_reflection_id, generated_by, and generation_context
to track which task/component created each bullet.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


# revision identifiers, used by Alembic.
revision: str = '15d473f06957'
down_revision: Union[str, Sequence[str], None] = '4c0c6073b96e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - add playbook archaeology fields."""
    # Add source_task_id with index for traceability queries
    op.add_column('playbook_bullets',
        sa.Column('source_task_id', sa.String(64), nullable=True,
                  comment='Task ID that generated this bullet (for traceability)')
    )
    op.create_index('idx_playbook_source_task', 'playbook_bullets', ['source_task_id'])

    # Add source_reflection_id to link back to reflection
    op.add_column('playbook_bullets',
        sa.Column('source_reflection_id', sa.String(36), nullable=True,
                  comment='Reflection ID that created this bullet (foreign key to reflections table)')
    )

    # Add generated_by to track which component created the bullet
    op.add_column('playbook_bullets',
        sa.Column('generated_by', sa.String(32), nullable=True,
                  comment="Component that generated this: 'reflector', 'curator', 'manual', 'import'")
    )

    # Add generation_context for additional metadata
    op.add_column('playbook_bullets',
        sa.Column('generation_context', JSON, nullable=True,
                  comment='Additional context about how this bullet was generated (metadata dict)')
    )


def downgrade() -> None:
    """Downgrade schema - remove playbook archaeology fields."""
    op.drop_index('idx_playbook_source_task', 'playbook_bullets')
    op.drop_column('playbook_bullets', 'generation_context')
    op.drop_column('playbook_bullets', 'generated_by')
    op.drop_column('playbook_bullets', 'source_reflection_id')
    op.drop_column('playbook_bullets', 'source_task_id')
