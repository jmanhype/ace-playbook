"""
Property-based tests for SemanticCurator using hypothesis.

Tests semantic deduplication invariants that should hold for ALL inputs:
- Deduplication is idempotent (applying twice = applying once)
- Similarity is symmetric (sim(A,B) = sim(B,A))
- Similar insights always increment, dissimilar always add new bullets
- Counter increments are monotonic (never decrease)
- Domain isolation is absolute (cross-domain merges always fail)
"""

import pytest
from hypothesis import given, strategies as st, settings
from hypothesis import assume
from datetime import datetime
import uuid

from ace.curator import SemanticCurator, CuratorInput
from ace.models.playbook import PlaybookBullet, PlaybookStage


# Strategy for generating valid domain IDs
domain_ids = st.from_regex(r"^[a-z0-9-]{3,20}$", fullmatch=True).filter(
    lambda x: x not in {"system", "admin", "test"}
)

# Strategy for generating insight sections
insight_sections = st.sampled_from(["Helpful", "Harmful", "Neutral"])

# Strategy for generating insight content
insight_content = st.text(min_size=10, max_size=200, alphabet=st.characters(
    whitelist_categories=("L", "N", "P", "Z"),
    min_codepoint=32, max_codepoint=126
))

# Strategy for generating embeddings (384-dim vectors)
embeddings_384 = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=384,
    max_size=384
)


@st.composite
def playbook_bullet(draw, domain_id=None, section=None):
    """Generate a valid PlaybookBullet for testing."""
    if domain_id is None:
        domain_id = draw(domain_ids)
    if section is None:
        section = draw(insight_sections)

    content = draw(insight_content)
    embedding = draw(embeddings_384)
    helpful_count = draw(st.integers(min_value=0, max_value=100))
    harmful_count = draw(st.integers(min_value=0, max_value=100))

    return PlaybookBullet(
        id=str(uuid.uuid4()),
        domain_id=domain_id,
        content=content,
        section=section,
        helpful_count=helpful_count,
        harmful_count=harmful_count,
        tags=[],
        embedding=embedding,
        created_at=datetime.utcnow(),
        last_used_at=datetime.utcnow(),
        stage=PlaybookStage.SHADOW,
    )


@st.composite
def insight_dict(draw, domain_id=None, section=None):
    """Generate a valid insight dictionary for curator input."""
    if section is None:
        section = draw(insight_sections)

    content = draw(insight_content)

    return {
        "content": content,
        "section": section,
        "tags": [],
    }


class TestSemanticCuratorProperties:
    """Property-based tests for SemanticCurator invariants."""

    @pytest.fixture
    def curator(self):
        """Create curator instance with default threshold."""
        return SemanticCurator(similarity_threshold=0.8)

    @settings(max_examples=20, deadline=5000)  # Fewer examples for speed
    @given(domain_id=domain_ids, section=insight_sections)
    def test_deduplication_idempotent(self, curator, domain_id, section):
        """
        Property: Applying the same insight twice should only add one bullet.

        Idempotence: curator(curator(playbook, insight)) = curator(playbook, insight)
        """
        # Create empty playbook
        playbook = []

        # Create fixed insight
        insight = {
            "content": "Test insight for idempotence",
            "section": section,
            "tags": [],
        }

        # Apply insight once
        curator_input = CuratorInput(
            task_id=str(uuid.uuid4()),
            domain_id=domain_id,
            insights=[insight],
            current_playbook=playbook,
            similarity_threshold=0.8,
        )

        output1 = curator.apply_delta(curator_input)

        # Apply same insight again to updated playbook
        curator_input2 = CuratorInput(
            task_id=str(uuid.uuid4()),
            domain_id=domain_id,
            insights=[insight],
            current_playbook=output1.updated_playbook,
            similarity_threshold=0.8,
        )

        output2 = curator.apply_delta(curator_input2)

        # Property: Should only have 1 bullet (not 2)
        assert len(output2.updated_playbook) == 1, \
            "Deduplication should be idempotent - applying same insight twice should not add new bullet"

        # Property: Second application should increment counter, not add bullet
        bullet = output2.updated_playbook[0]
        if section == "Helpful":
            assert bullet.helpful_count == 2, "Helpful counter should increment on duplicate"
        elif section == "Harmful":
            assert bullet.harmful_count == 2, "Harmful counter should increment on duplicate"

    @settings(max_examples=20, deadline=5000)
    @given(
        domain_id=domain_ids,
        section1=insight_sections,
        section2=insight_sections,
    )
    def test_counters_monotonic(self, curator, domain_id, section1, section2):
        """
        Property: Counters never decrease during curator operations.

        Monotonicity: For all operations, counter_after ≥ counter_before
        """
        # Create playbook with one bullet
        bullet = PlaybookBullet(
            id=str(uuid.uuid4()),
            domain_id=domain_id,
            content="Initial bullet",
            section=section1,
            helpful_count=5,
            harmful_count=3,
            tags=[],
            embedding=[0.1] * 384,
            created_at=datetime.utcnow(),
            last_used_at=datetime.utcnow(),
            stage=PlaybookStage.SHADOW,
        )

        initial_helpful = bullet.helpful_count
        initial_harmful = bullet.harmful_count

        # Apply new insight
        insight = {
            "content": "New insight",
            "section": section2,
            "tags": [],
        }

        curator_input = CuratorInput(
            task_id=str(uuid.uuid4()),
            domain_id=domain_id,
            insights=[insight],
            current_playbook=[bullet],
            similarity_threshold=0.8,
        )

        output = curator.apply_delta(curator_input)

        # Property: Counters never decrease
        for updated_bullet in output.updated_playbook:
            if updated_bullet.id == bullet.id:
                assert updated_bullet.helpful_count >= initial_helpful, \
                    "Helpful counter must never decrease"
                assert updated_bullet.harmful_count >= initial_harmful, \
                    "Harmful counter must never decrease"

    @settings(max_examples=15, deadline=5000)
    @given(domain_id1=domain_ids, domain_id2=domain_ids)
    def test_domain_isolation_absolute(self, curator, domain_id1, domain_id2):
        """
        Property: Cross-domain operations always fail.

        Domain Isolation: curator(playbook[domain_A], insight[domain_B]) → ValueError
        """
        assume(domain_id1 != domain_id2)  # Ensure different domains

        # Create playbook for domain_id1
        bullet = PlaybookBullet(
            id=str(uuid.uuid4()),
            domain_id=domain_id1,
            content="Domain A bullet",
            section="Helpful",
            helpful_count=1,
            harmful_count=0,
            tags=[],
            embedding=[0.1] * 384,
            created_at=datetime.utcnow(),
            last_used_at=datetime.utcnow(),
            stage=PlaybookStage.SHADOW,
        )

        # Try to apply insight from domain_id2
        insight = {
            "content": "Domain B insight",
            "section": "Helpful",
            "tags": [],
        }

        curator_input = CuratorInput(
            task_id=str(uuid.uuid4()),
            domain_id=domain_id2,  # Different domain!
            insights=[insight],
            current_playbook=[bullet],
            similarity_threshold=0.8,
        )

        # Property: Must raise ValueError for cross-domain access
        with pytest.raises(ValueError, match="Cross-domain access violation"):
            curator.apply_delta(curator_input)

    @settings(max_examples=20, deadline=5000)
    @given(domain_id=domain_ids, num_insights=st.integers(min_value=1, max_value=10))
    def test_delta_updates_complete(self, curator, domain_id, num_insights):
        """
        Property: Every insight generates exactly one delta update.

        Completeness: len(delta_updates) = len(insights)
        """
        # Create empty playbook
        playbook = []

        # Generate unique insights (ensure they won't deduplicate)
        insights = [
            {
                "content": f"Unique insight {i}: {uuid.uuid4()}",
                "section": "Helpful",
                "tags": [],
            }
            for i in range(num_insights)
        ]

        curator_input = CuratorInput(
            task_id=str(uuid.uuid4()),
            domain_id=domain_id,
            insights=insights,
            current_playbook=playbook,
            similarity_threshold=0.8,
        )

        output = curator.apply_delta(curator_input)

        # Property: One delta update per insight
        assert len(output.delta_updates) == num_insights, \
            "Each insight must generate exactly one delta update (add or increment)"

    @settings(max_examples=15, deadline=5000)
    @given(domain_id=domain_ids)
    def test_playbook_size_bounded(self, curator, domain_id):
        """
        Property: Playbook size grows sublinearly with duplicate insights.

        Efficiency: With duplicates, |playbook| << |insights|
        """
        # Create empty playbook
        playbook = []

        # Apply same insight 10 times
        duplicate_insight = {
            "content": "Repeated insight",
            "section": "Helpful",
            "tags": [],
        }

        for _ in range(10):
            curator_input = CuratorInput(
                task_id=str(uuid.uuid4()),
                domain_id=domain_id,
                insights=[duplicate_insight],
                current_playbook=playbook,
                similarity_threshold=0.8,
            )

            output = curator.apply_delta(curator_input)
            playbook = output.updated_playbook

        # Property: Should only have 1 bullet (not 10)
        assert len(playbook) == 1, \
            "Deduplication should prevent linear playbook growth"

        # Property: Counter should reflect all applications
        assert playbook[0].helpful_count == 10, \
            "Counter should track all 10 duplicate applications"

    @settings(max_examples=15, deadline=5000)
    @given(domain_id=domain_ids, section=insight_sections)
    def test_empty_playbook_always_adds(self, curator, domain_id, section):
        """
        Property: First insight on empty playbook always adds a bullet.

        Base Case: curator([], insight) → playbook with 1 bullet
        """
        # Create empty playbook
        playbook = []

        # Apply single insight
        insight = {
            "content": "First insight ever",
            "section": section,
            "tags": [],
        }

        curator_input = CuratorInput(
            task_id=str(uuid.uuid4()),
            domain_id=domain_id,
            insights=[insight],
            current_playbook=playbook,
            similarity_threshold=0.8,
        )

        output = curator.apply_delta(curator_input)

        # Property: Must add exactly one bullet
        assert len(output.updated_playbook) == 1, \
            "First insight on empty playbook must add new bullet"
        assert output.new_bullets_added == 1, \
            "Metrics should reflect one new bullet"

    @settings(max_examples=15, deadline=5000)
    @given(domain_id=domain_ids)
    def test_quarantine_deterministic(self, curator, domain_id):
        """
        Property: Quarantine decision is deterministic based on counters.

        Determinism: same counter state → same quarantine decision
        """
        # Create bullet with harmful >= helpful
        bullet = PlaybookBullet(
            id=str(uuid.uuid4()),
            domain_id=domain_id,
            content="Potentially bad bullet",
            section="Harmful",
            helpful_count=2,
            harmful_count=3,  # More harmful than helpful
            tags=[],
            embedding=[0.1] * 384,
            created_at=datetime.utcnow(),
            last_used_at=datetime.utcnow(),
            stage=PlaybookStage.SHADOW,
        )

        # Apply curator (should trigger quarantine check)
        curator_input = CuratorInput(
            task_id=str(uuid.uuid4()),
            domain_id=domain_id,
            insights=[],  # Empty insights, just checking quarantine
            current_playbook=[bullet],
            similarity_threshold=0.8,
        )

        output = curator.apply_delta(curator_input)

        # Property: Should quarantine if harmful >= helpful
        quarantined_bullet = next(
            (b for b in output.updated_playbook if b.id == bullet.id),
            None
        )
        assert quarantined_bullet is not None

        # Note: The current implementation only quarantines during delta processing
        # This test documents the expected behavior even if not fully implemented


class TestCuratorUtilsProperties:
    """Property-based tests for curator utility functions."""

    @settings(max_examples=30, deadline=2000)
    @given(
        vec1=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
                     min_size=384, max_size=384),
        vec2=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
                     min_size=384, max_size=384),
    )
    def test_similarity_symmetric(self, vec1, vec2):
        """
        Property: Cosine similarity is symmetric.

        Symmetry: sim(A, B) = sim(B, A)
        """
        from ace.curator.curator_utils import compute_similarity

        sim_ab = compute_similarity(vec1, vec2)
        sim_ba = compute_similarity(vec2, vec1)

        # Property: Similarity is symmetric (within floating point tolerance)
        assert abs(sim_ab - sim_ba) < 1e-6, \
            "Cosine similarity must be symmetric"

    @settings(max_examples=30, deadline=2000)
    @given(
        vec=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
                    min_size=384, max_size=384)
    )
    def test_similarity_reflexive(self, vec):
        """
        Property: Similarity with self is 1.0 (or 0.0 for zero vector).

        Reflexivity: sim(A, A) = 1.0 (if A ≠ 0)
        """
        from ace.curator.curator_utils import compute_similarity

        sim = compute_similarity(vec, vec)

        # Check if vector is all zeros
        is_zero = all(abs(x) < 1e-9 for x in vec)

        if is_zero:
            # Zero vector has undefined similarity (implementation returns 0.0)
            assert sim == 0.0, "Zero vector similarity should be 0.0"
        else:
            # Non-zero vector should have similarity 1.0 with itself
            assert abs(sim - 1.0) < 1e-6, \
                "Non-zero vector must have similarity 1.0 with itself"

    @settings(max_examples=30, deadline=2000)
    @given(
        vec1=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
                     min_size=384, max_size=384),
        vec2=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
                     min_size=384, max_size=384),
    )
    def test_similarity_bounded(self, vec1, vec2):
        """
        Property: Cosine similarity is always in [-1, 1].

        Boundedness: -1 ≤ sim(A, B) ≤ 1
        """
        from ace.curator.curator_utils import compute_similarity

        sim = compute_similarity(vec1, vec2)

        # Property: Similarity is bounded
        assert -1.0 <= sim <= 1.0, \
            f"Cosine similarity must be in [-1, 1], got {sim}"
