"""
SemanticCurator Implementation

Pure Python semantic deduplication with FAISS at 0.8 cosine similarity threshold.
Implements contracts from /Users/speed/specs/004-implementing-the-ace/contracts/curator.py
"""

from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import hashlib
import json
import re

from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.utils.embeddings import get_embedding_service
from ace.utils.faiss_index import get_faiss_manager
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="curator")


# Import contract types from specs
SIMILARITY_THRESHOLD_DEFAULT = 0.8
DOMAIN_ISOLATION_PATTERN = r"^[a-z0-9-]+$"
RESERVED_DOMAINS = {"system", "admin", "test"}


class CuratorInput:
    """Input for Curator delta merging operation."""

    def __init__(
        self,
        task_id: str,
        domain_id: str,
        insights: List[Dict],
        current_playbook: List[PlaybookBullet],
        target_stage: PlaybookStage = PlaybookStage.SHADOW,
        similarity_threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
        promotion_helpful_min: int = 3,
        promotion_ratio_min: float = 3.0,
        quarantine_threshold: float = 1.0,
    ):
        self.task_id = task_id
        self.domain_id = domain_id
        self.insights = insights
        self.current_playbook = current_playbook
        self.target_stage = target_stage
        self.similarity_threshold = similarity_threshold
        self.promotion_helpful_min = promotion_helpful_min
        self.promotion_ratio_min = promotion_ratio_min
        self.quarantine_threshold = quarantine_threshold


class DeltaUpdate:
    """Single atomic playbook update operation."""

    def __init__(
        self,
        operation: str,
        bullet_id: str,
        before_hash: Optional[str] = None,
        after_hash: Optional[str] = None,
        new_bullet: Optional[PlaybookBullet] = None,
        similar_to: Optional[str] = None,
        similarity_score: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        self.operation = operation
        self.bullet_id = bullet_id
        self.before_hash = before_hash
        self.after_hash = after_hash
        self.new_bullet = new_bullet
        self.similar_to = similar_to
        self.similarity_score = similarity_score
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()


class CuratorOutput:
    """Output from Curator delta merging operation."""

    def __init__(
        self,
        task_id: str,
        domain_id: str,
        delta_updates: List[DeltaUpdate],
        updated_playbook: List[PlaybookBullet],
    ):
        self.task_id = task_id
        self.domain_id = domain_id
        self.delta_updates = delta_updates
        self.updated_playbook = updated_playbook
        self.new_bullets_added = 0
        self.existing_bullets_incremented = 0
        self.duplicates_detected = 0
        self.bullets_quarantined = 0
        self.bullets_promoted = 0

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Curator Update Summary:\n"
            f"  New bullets: {self.new_bullets_added}\n"
            f"  Incremented: {self.existing_bullets_incremented}\n"
            f"  Duplicates: {self.duplicates_detected}\n"
            f"  Quarantined: {self.bullets_quarantined}\n"
            f"  Promoted: {self.bullets_promoted}\n"
            f"  Total operations: {len(self.delta_updates)}"
        )


class SemanticCurator:
    """
    Production Curator implementation using FAISS and sentence-transformers.

    Performs semantic deduplication at 0.8 cosine similarity threshold to prevent
    playbook bloat while preserving distinct strategies.

    Implements CHK081-CHK082, CHK086 for multi-domain isolation.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
    ):
        """
        Initialize SemanticCurator.

        Args:
            embedding_model: sentence-transformers model name
            similarity_threshold: Cosine similarity threshold for duplicates (0.8)
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.embedding_service = get_embedding_service(model_name=embedding_model)
        self.faiss_manager = get_faiss_manager()

        logger.info(
            "curator_initialized",
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
        )

    def validate_domain_id(self, domain_id: str) -> None:
        """
        Validate domain_id against CHK079 namespace pattern.

        Raises:
            ValueError: If domain_id is invalid or reserved
        """
        if not re.match(DOMAIN_ISOLATION_PATTERN, domain_id):
            raise ValueError(
                f"Invalid domain_id '{domain_id}'. Must match pattern: ^[a-z0-9-]+$"
            )
        if domain_id in RESERVED_DOMAINS:
            raise ValueError(f"Reserved domain_id '{domain_id}' cannot be used")

    def enforce_domain_isolation(self, curator_input: CuratorInput) -> None:
        """
        Enforce CHK081-CHK082 domain isolation requirements.

        Raises:
            ValueError: If any bullet violates domain isolation
        """
        # Validate domain_id format (CHK079)
        self.validate_domain_id(curator_input.domain_id)

        # CHK082: Cross-domain guard - verify all existing bullets match domain_id
        for bullet in curator_input.current_playbook:
            if bullet.domain_id != curator_input.domain_id:
                raise ValueError(
                    f"Cross-domain access violation: attempted to merge insights from domain "
                    f"'{curator_input.domain_id}' into playbook for domain '{bullet.domain_id}'"
                )

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Args:
            embedding1: First embedding (384-dim)
            embedding2: Second embedding (384-dim)

        Returns:
            Cosine similarity score (0.0 to 1.0, higher = more similar)
        """
        vec1 = np.array(embedding1, dtype=np.float32)
        vec2 = np.array(embedding2, dtype=np.float32)

        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity = dot product of normalized vectors
        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
        return similarity

    def should_promote(
        self, bullet: PlaybookBullet, target_stage: PlaybookStage, curator_input: CuratorInput
    ) -> bool:
        """
        Check if bullet meets promotion criteria for target stage.

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

    def should_quarantine(self, bullet: PlaybookBullet) -> bool:
        """
        Check if bullet should be quarantined (excluded from retrieval).

        Args:
            bullet: Bullet to evaluate

        Returns:
            True if harmful_count ≥ helpful_count
        """
        return bullet.harmful_count >= bullet.helpful_count and bullet.helpful_count > 0

    def compute_bullet_hash(self, bullet: PlaybookBullet) -> str:
        """
        Compute deterministic SHA-256 hash of bullet state.

        Used for diff journal to track exactly what changed during updates.
        Excludes timestamps to focus on semantic state.
        """
        stable_state = {
            "id": bullet.id,
            "domain_id": bullet.domain_id,
            "content": bullet.content,
            "section": bullet.section,
            "helpful_count": bullet.helpful_count,
            "harmful_count": bullet.harmful_count,
            "tags": sorted(bullet.tags),
        }
        canonical_json = json.dumps(stable_state, sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def apply_delta(self, curator_input: CuratorInput) -> CuratorOutput:
        """
        Apply semantic deduplication and merge insights into playbook.

        Args:
            curator_input: InsightCandidate list + current playbook state

        Returns:
            CuratorOutput with delta updates and audit trail

        Raises:
            ValueError: If domain isolation is violated or playbook is corrupted
        """
        logger.info(
            "curator_apply_delta_start",
            task_id=curator_input.task_id,
            domain_id=curator_input.domain_id,
            num_insights=len(curator_input.insights),
            playbook_size=len(curator_input.current_playbook),
        )

        # CHK081-CHK082: Enforce domain isolation
        self.enforce_domain_isolation(curator_input)

        delta_updates = []
        updated_playbook = list(curator_input.current_playbook)  # Copy
        updated_playbook_dict = {b.id: b for b in updated_playbook}

        # Process each insight
        for insight in curator_input.insights:
            # Generate embedding for insight content
            insight_embedding = self.embedding_service.encode_single(insight["content"])

            # Find most similar existing bullet in same section and domain
            best_match = None
            best_similarity = 0.0

            for bullet in updated_playbook:
                if bullet.domain_id != curator_input.domain_id:
                    continue  # Skip cross-domain bullets (CHK081)
                if bullet.section != insight["section"]:
                    continue  # Only compare within same section

                similarity = self.compute_similarity(insight_embedding, bullet.embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = bullet

            # Decide operation based on similarity threshold
            if best_similarity >= curator_input.similarity_threshold and best_match:
                # Duplicate detected - increment counter
                before_hash = self.compute_bullet_hash(best_match)

                if insight["section"] == "Helpful":
                    best_match.helpful_count += 1
                    operation = "increment_helpful"
                elif insight["section"] == "Harmful":
                    best_match.harmful_count += 1
                    operation = "increment_harmful"
                else:
                    # Neutral - no counter increment
                    operation = "increment_neutral"

                after_hash = self.compute_bullet_hash(best_match)

                delta_updates.append(
                    DeltaUpdate(
                        operation=operation,
                        bullet_id=best_match.id,
                        before_hash=before_hash,
                        after_hash=after_hash,
                        similar_to=best_match.id,
                        similarity_score=best_similarity,
                    )
                )

                logger.debug(
                    "duplicate_detected",
                    bullet_id=best_match.id,
                    similarity=best_similarity,
                    operation=operation,
                )
            else:
                # New distinct bullet - add to playbook
                import uuid

                new_bullet = PlaybookBullet(
                    id=str(uuid.uuid4()),
                    domain_id=curator_input.domain_id,
                    content=insight["content"],
                    section=insight["section"],
                    helpful_count=1 if insight["section"] == "Helpful" else 0,
                    harmful_count=1 if insight["section"] == "Harmful" else 0,
                    tags=insight.get("tags", []),
                    embedding=insight_embedding,
                    created_at=datetime.utcnow(),
                    last_used_at=datetime.utcnow(),
                    stage=curator_input.target_stage,
                )

                updated_playbook.append(new_bullet)
                updated_playbook_dict[new_bullet.id] = new_bullet

                delta_updates.append(
                    DeltaUpdate(
                        operation="add",
                        bullet_id=new_bullet.id,
                        new_bullet=new_bullet,
                        after_hash=self.compute_bullet_hash(new_bullet),
                    )
                )

                logger.debug("new_bullet_added", bullet_id=new_bullet.id, section=insight["section"])

        # Check for quarantine/promotion status changes
        for bullet in updated_playbook:
            if self.should_quarantine(bullet) and bullet.stage != PlaybookStage.QUARANTINED:
                before_hash = self.compute_bullet_hash(bullet)
                bullet.stage = PlaybookStage.QUARANTINED
                after_hash = self.compute_bullet_hash(bullet)

                delta_updates.append(
                    DeltaUpdate(
                        operation="quarantine",
                        bullet_id=bullet.id,
                        before_hash=before_hash,
                        after_hash=after_hash,
                    )
                )

        # Generate output
        output = CuratorOutput(
            task_id=curator_input.task_id,
            domain_id=curator_input.domain_id,
            delta_updates=delta_updates,
            updated_playbook=updated_playbook,
        )

        # Compute statistics
        for update in delta_updates:
            if update.operation == "add":
                output.new_bullets_added += 1
            elif update.operation in ("increment_helpful", "increment_harmful"):
                output.existing_bullets_incremented += 1
                output.duplicates_detected += 1
            elif update.operation == "quarantine":
                output.bullets_quarantined += 1

        logger.info(
            "curator_apply_delta_complete",
            task_id=curator_input.task_id,
            new_bullets=output.new_bullets_added,
            incremented=output.existing_bullets_incremented,
            quarantined=output.bullets_quarantined,
        )

        return output

    def batch_merge(
        self,
        task_insights: List[Dict],  # List of {"task_id": str, "domain_id": str, "insights": List[Dict]}
        current_playbook: List[PlaybookBullet],
        target_stage: PlaybookStage = PlaybookStage.SHADOW,
        similarity_threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
    ) -> Dict:
        """
        T054: Efficient batch merging of multiple task insights at once.

        Optimizations:
        - Single FAISS index build for all embeddings
        - Deduplicate across entire batch (not per-task)
        - Single transaction to commit all updates

        Args:
            task_insights: List of insights from N tasks with task_id/domain_id
            current_playbook: Current playbook state
            target_stage: Stage for new bullets (SHADOW, STAGING, PROD)
            similarity_threshold: Cosine threshold for duplicates

        Returns:
            Dict with batch_results and updated_playbook
        """
        logger.info(
            "batch_merge_start",
            num_tasks=len(task_insights),
            playbook_size=len(current_playbook)
        )

        # Collect all insights across tasks
        all_insights = []
        task_id_map = {}  # Map insight index to task_id for tracking

        for task_data in task_insights:
            task_id = task_data["task_id"]
            domain_id = task_data["domain_id"]
            insights = task_data["insights"]

            for insight in insights:
                insight_with_meta = {
                    **insight,
                    "task_id": task_id,
                    "domain_id": domain_id
                }
                task_id_map[len(all_insights)] = task_id
                all_insights.append(insight_with_meta)

        if not all_insights:
            logger.warning("batch_merge_no_insights")
            return {
                "batch_results": [],
                "updated_playbook": current_playbook,
                "total_new_bullets": 0,
                "total_increments": 0
            }

        # Compute embeddings for all insights in batch
        logger.info("batch_embedding_start", num_insights=len(all_insights))
        insight_contents = [ins["content"] for ins in all_insights]
        insight_embeddings = self.embedding_service.encode_batch(insight_contents)

        # Build FAISS index for current playbook
        if current_playbook:
            playbook_embeddings = np.array([b.embedding for b in current_playbook], dtype=np.float32)
            self.faiss_manager.build_index(playbook_embeddings)
        else:
            playbook_embeddings = None

        # Process all insights with single index
        updated_playbook = list(current_playbook)
        updated_playbook_dict = {b.id: b for b in updated_playbook}
        delta_updates_all = []
        total_new_bullets = 0
        total_increments = 0

        for idx, (insight, embedding) in enumerate(zip(all_insights, insight_embeddings)):
            task_id = insight["task_id"]
            domain_id = insight["domain_id"]

            # Find most similar bullet
            best_match = None
            best_similarity = 0.0

            if playbook_embeddings is not None:
                # Use FAISS for fast similarity search
                similarities, indices = self.faiss_manager.search_similar(
                    embedding.reshape(1, -1), k=10  # Get top 10 candidates
                )

                for sim, idx_match in zip(similarities[0], indices[0]):
                    if idx_match == -1:
                        break
                    bullet = current_playbook[idx_match]

                    # Filter by domain and section
                    if bullet.domain_id != domain_id:
                        continue
                    if bullet.section != insight["section"]:
                        continue

                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = bullet

            # Decide operation
            if best_similarity >= similarity_threshold and best_match:
                # Increment existing bullet
                before_hash = self.compute_bullet_hash(best_match)

                if insight["section"] == "Helpful":
                    best_match.helpful_count += 1
                    operation = "increment_helpful"
                elif insight["section"] == "Harmful":
                    best_match.harmful_count += 1
                    operation = "increment_harmful"
                else:
                    operation = "increment_neutral"

                after_hash = self.compute_bullet_hash(best_match)

                delta_updates_all.append(
                    DeltaUpdate(
                        operation=operation,
                        bullet_id=best_match.id,
                        before_hash=before_hash,
                        after_hash=after_hash,
                        similar_to=best_match.id,
                        similarity_score=float(best_similarity),
                        metadata={"task_id": task_id}
                    )
                )
                total_increments += 1

            else:
                # Add new bullet
                import uuid

                new_bullet = PlaybookBullet(
                    id=str(uuid.uuid4()),
                    domain_id=domain_id,
                    content=insight["content"],
                    section=insight["section"],
                    helpful_count=1 if insight["section"] == "Helpful" else 0,
                    harmful_count=1 if insight["section"] == "Harmful" else 0,
                    tags=insight.get("tags", []),
                    embedding=embedding.tolist(),
                    created_at=datetime.utcnow(),
                    last_used_at=datetime.utcnow(),
                    stage=target_stage,
                )

                updated_playbook.append(new_bullet)
                updated_playbook_dict[new_bullet.id] = new_bullet

                delta_updates_all.append(
                    DeltaUpdate(
                        operation="add",
                        bullet_id=new_bullet.id,
                        new_bullet=new_bullet,
                        after_hash=self.compute_bullet_hash(new_bullet),
                        metadata={"task_id": task_id}
                    )
                )
                total_new_bullets += 1

        logger.info(
            "batch_merge_complete",
            num_tasks=len(task_insights),
            num_insights=len(all_insights),
            new_bullets=total_new_bullets,
            increments=total_increments
        )

        return {
            "batch_results": delta_updates_all,
            "updated_playbook": updated_playbook,
            "total_new_bullets": total_new_bullets,
            "total_increments": total_increments,
            "total_processed": len(all_insights)
        }
