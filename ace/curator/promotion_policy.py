"""
Promotion and quarantine policy logic for playbook bullets.

Extracted from semantic_curator.py to reduce file size and improve modularity.
Implements promotion gates (SHADOW → STAGING → PROD) and quarantine detection.
"""

from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.curator.curator_models import CuratorInput


def should_promote(
    bullet: PlaybookBullet, target_stage: PlaybookStage, curator_input: CuratorInput
) -> bool:
    """
    Check if bullet meets promotion criteria for target stage.

    Promotion Gates:
    - SHADOW: No gates (always True)
    - STAGING: helpful_count ≥ 3, ratio ≥ 3.0
    - PROD: helpful_count ≥ 5, ratio ≥ 5.0

    Args:
        bullet: Bullet to evaluate
        target_stage: Desired stage (STAGING or PROD)
        curator_input: CuratorInput with promotion gate thresholds

    Returns:
        True if bullet meets promotion gates (helpful_count, ratio)
    """
    if target_stage == PlaybookStage.SHADOW:
        return True  # No gates for shadow

    if target_stage == PlaybookStage.STAGING:
        helpful_min = curator_input.promotion_helpful_min  # Default: 3
        ratio_min = curator_input.promotion_ratio_min  # Default: 3.0
    elif target_stage == PlaybookStage.PROD:
        helpful_min = 5  # Hardcoded prod gate
        ratio_min = 5.0
    else:
        return False

    # Check helpful_count threshold
    if bullet.helpful_count < helpful_min:
        return False

    # Check helpful:harmful ratio
    if bullet.harmful_count == 0:
        # No harmful signals = infinite ratio = pass
        return True

    ratio = bullet.helpful_count / bullet.harmful_count
    return ratio >= ratio_min


def should_quarantine(bullet: PlaybookBullet) -> bool:
    """
    Check if bullet should be quarantined (excluded from retrieval).

    Quarantine Criteria:
    - harmful_count ≥ helpful_count
    - helpful_count > 0 (avoid quarantining untested bullets)

    Args:
        bullet: Bullet to evaluate

    Returns:
        True if harmful_count ≥ helpful_count
    """
    return bullet.harmful_count >= bullet.helpful_count and bullet.helpful_count > 0
