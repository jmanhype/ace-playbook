"""
Domain validation logic for curator operations.

Extracted from semantic_curator.py to reduce file size and improve modularity.
Implements CHK079-CHK082 domain isolation requirements.
"""

import re
from typing import List

from ace.curator.curator_models import (
    CuratorInput,
    DOMAIN_ISOLATION_PATTERN,
    RESERVED_DOMAINS,
)
from ace.models.playbook import PlaybookBullet


def validate_domain_id(domain_id: str) -> None:
    """
    Validate domain_id against CHK079 namespace pattern.

    Args:
        domain_id: Domain identifier to validate

    Raises:
        ValueError: If domain_id is invalid or reserved
    """
    if not re.match(DOMAIN_ISOLATION_PATTERN, domain_id):
        raise ValueError(
            f"Invalid domain_id '{domain_id}'. Must match pattern: ^[a-z0-9-]+$"
        )
    if domain_id in RESERVED_DOMAINS:
        raise ValueError(f"Reserved domain_id '{domain_id}' cannot be used")


def enforce_domain_isolation(curator_input: CuratorInput) -> None:
    """
    Enforce CHK081-CHK082 domain isolation requirements.

    Args:
        curator_input: CuratorInput with domain_id and current_playbook

    Raises:
        ValueError: If any bullet violates domain isolation
    """
    # Validate domain_id format (CHK079)
    validate_domain_id(curator_input.domain_id)

    # CHK082: Cross-domain guard - verify all existing bullets match domain_id
    for bullet in curator_input.current_playbook:
        if bullet.domain_id != curator_input.domain_id:
            raise ValueError(
                f"Cross-domain access violation: attempted to merge insights from domain "
                f"'{curator_input.domain_id}' into playbook for domain '{bullet.domain_id}'"
            )


def validate_batch_task_insights(task_insights: List[dict]) -> str:
    """
    Validate batch task insights structure and extract domain_id.

    Args:
        task_insights: List of tasks with insights for batch processing

    Returns:
        domain_id: Validated domain_id (all tasks must have same domain)

    Raises:
        ValueError: If task_insights is empty, has mixed domains, or invalid structure
    """
    # T070: Validate input before processing
    if not task_insights:
        raise ValueError("task_insights cannot be empty")

    # Validate all tasks have same domain_id (reject mixed batches)
    try:
        domain_ids = {task["domain_id"] for task in task_insights}
    except KeyError as e:
        raise ValueError(
            f"Invalid task_insights structure: missing 'domain_id' key"
        ) from e

    if len(domain_ids) > 1:
        raise ValueError(
            f"Multiple domain_ids in batch: {domain_ids}. "
            f"Batch operations must operate on a single domain for isolation."
        )

    domain_id = task_insights[0]["domain_id"]

    # Validate domain_id against CHK079 requirements
    validate_domain_id(domain_id)

    # Validate required keys in each task
    for i, task_data in enumerate(task_insights):
        if "task_id" not in task_data:
            raise ValueError(f"Task at index {i} missing required 'task_id' key")
        if "insights" not in task_data:
            raise ValueError(f"Task at index {i} missing required 'insights' key")
        if not isinstance(task_data["insights"], list):
            raise ValueError(f"Task at index {i} 'insights' must be a list")

    return domain_id
