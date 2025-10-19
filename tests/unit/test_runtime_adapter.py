"""Unit tests for runtime adaptation layer."""

from ace.runtime.adaptation import RuntimeAdapter


class DummyCoordinator:
    def __init__(self):
        self.submissions = []

    def submit(self, domain_id, event):
        self.submissions.append((domain_id, event))
        return None


def test_runtime_adapter_ingest_and_cache():
    coordinator = DummyCoordinator()
    adapter = RuntimeAdapter("demo", coordinator, ttl_seconds=1_000)

    adapter.ingest(
        task_id="task-1",
        insight={
            "content": "Use [bullet-xyz] for arithmetic",
            "section": "Helpful",
            "confidence": 0.9,
        },
    )

    assert len(coordinator.submissions) == 1
    domain_id, event = coordinator.submissions[0]
    assert domain_id == "demo"
    assert event.task_id == "task-1"

    hot_entries = adapter.get_hot_entries()
    assert len(hot_entries) == 1
    assert hot_entries[0]["content"].startswith("Use")
