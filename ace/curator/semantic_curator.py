"""
SemanticCurator Implementation

Pure Python semantic deduplication with FAISS at 0.8 cosine similarity threshold.
Implements contracts from /Users/speed/specs/004-implementing-the-ace/contracts/curator.py

Refactored in T078 to reduce file size by extracting:
- Data models → curator_models.py
- Domain validation → domain_validator.py
- Utility functions → curator_utils.py
- Promotion policy → promotion_policy.py
"""

from typing import List, Dict
from datetime import datetime
import numpy as np
import uuid

from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.utils.embeddings import get_embedding_service
from ace.utils.faiss_index import get_faiss_manager
from ace.utils.logging_config import get_logger

# Import extracted modules
from ace.curator.curator_models import (
    CuratorInput,
    CuratorOutput,
    DeltaUpdate,
    SIMILARITY_THRESHOLD_DEFAULT,
)
from ace.curator.domain_validator import (
    validate_domain_id,
    enforce_domain_isolation,
    validate_batch_task_insights,
)
from ace.curator.curator_utils import (
    compute_similarity,
    compute_bullet_hash,
)
from ace.curator.promotion_policy import (
    should_promote,
    should_quarantine,
)

logger = get_logger(__name__, component="curator")


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
        enforce_domain_isolation(curator_input)

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

                similarity = compute_similarity(insight_embedding, bullet.embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = bullet

            # Decide operation based on similarity threshold
            if best_similarity >= curator_input.similarity_threshold and best_match:
                # Duplicate detected - increment counter
                before_hash = compute_bullet_hash(best_match)

                if insight["section"] == "Helpful":
                    best_match.helpful_count += 1
                    operation = "increment_helpful"
                elif insight["section"] == "Harmful":
                    best_match.harmful_count += 1
                    operation = "increment_harmful"
                else:
                    # Neutral - no counter increment
                    operation = "increment_neutral"

                after_hash = compute_bullet_hash(best_match)

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
                        after_hash=compute_bullet_hash(new_bullet),
                    )
                )

                logger.debug("new_bullet_added", bullet_id=new_bullet.id, section=insight["section"])

        # Check for quarantine/promotion status changes
        for bullet in updated_playbook:
            if should_quarantine(bullet) and bullet.stage != PlaybookStage.QUARANTINED:
                before_hash = compute_bullet_hash(bullet)
                bullet.stage = PlaybookStage.QUARANTINED
                after_hash = compute_bullet_hash(bullet)

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

        Raises:
            ValueError: If task_insights is empty, has mixed domains, or invalid structure
        """
        # T070: Validate input and extract domain_id
        domain_id = validate_batch_task_insights(task_insights)

        logger.info(
            "batch_merge_start",
            num_tasks=len(task_insights),
            playbook_size=len(current_playbook),
            domain_id=domain_id
        )

        # Collect all insights across tasks
        all_insights = []
        task_id_map = {}  # Map insight index to task_id for tracking

        for task_data in task_insights:
            task_id = task_data["task_id"]
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

        # T072: Use try-finally to ensure FAISS index cleanup (prevent memory leak)
        try:
            # Build FAISS index for current playbook
            if current_playbook:
                # Add current playbook to FAISS index
                playbook_embeddings = np.array([b.embedding for b in current_playbook], dtype=np.float32)
                bullet_ids = [b.id for b in current_playbook]
                self.faiss_manager.add_vectors(domain_id, playbook_embeddings, bullet_ids)

            # Build mapping from bullet_id to bullet for quick lookup
            bullet_id_to_bullet = {b.id: b for b in current_playbook}

            # Process all insights with single index
            updated_playbook = list(current_playbook)
            updated_playbook_dict = {b.id: b for b in updated_playbook}
            delta_updates_all = []
            total_new_bullets = 0
            total_increments = 0

            for idx, (insight, embedding) in enumerate(zip(all_insights, insight_embeddings)):
                task_id = insight["task_id"]
                insight_domain_id = insight["domain_id"]

                # Find most similar bullet
                best_match = None
                best_similarity = 0.0

                if current_playbook:
                    # Use FAISS for fast similarity search
                    # Convert embedding list to numpy array
                    embedding_array = np.array(embedding, dtype=np.float32)
                    search_results = self.faiss_manager.search(
                        insight_domain_id, embedding_array, k=10  # Get top 10 candidates
                    )

                    for bullet_id, similarity in search_results:
                        bullet = bullet_id_to_bullet.get(bullet_id)
                        if not bullet:
                            continue

                        # Filter by domain and section
                        if bullet.domain_id != insight_domain_id:
                            continue
                        if bullet.section != insight["section"]:
                            continue

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = bullet

                # Decide operation
                if best_similarity >= similarity_threshold and best_match:
                    # Increment existing bullet
                    before_hash = compute_bullet_hash(best_match)

                    if insight["section"] == "Helpful":
                        best_match.helpful_count += 1
                        operation = "increment_helpful"
                    elif insight["section"] == "Harmful":
                        best_match.harmful_count += 1
                        operation = "increment_harmful"
                    else:
                        operation = "increment_neutral"

                    after_hash = compute_bullet_hash(best_match)

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
                    # Convert embedding to list if it's a numpy array
                    embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

                    new_bullet = PlaybookBullet(
                        id=str(uuid.uuid4()),
                        domain_id=domain_id,
                        content=insight["content"],
                        section=insight["section"],
                        helpful_count=1 if insight["section"] == "Helpful" else 0,
                        harmful_count=1 if insight["section"] == "Harmful" else 0,
                        tags=insight.get("tags", []),
                        embedding=embedding_list,
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
                            after_hash=compute_bullet_hash(new_bullet),
                            metadata={"task_id": task_id}
                        )
                    )
                    total_new_bullets += 1
        finally:
            # T072: Clean up FAISS index to prevent memory leak
            self.faiss_manager.clear_index(domain_id)
            logger.debug("batch_merge_faiss_cleanup", domain_id=domain_id)

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
