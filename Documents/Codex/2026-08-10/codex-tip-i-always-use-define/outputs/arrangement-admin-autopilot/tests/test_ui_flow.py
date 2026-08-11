from pathlib import Path

from streamlit.testing.v1 import AppTest

from core import ARTIFACT_KEYS

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def make_app(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "test-password")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return AppTest.from_file(APP_PATH, default_timeout=10)


def seed_review(at, unknown=False):
    at.session_state["authenticated"] = True
    drafts = {key: f"Draft {key}" for key in ARTIFACT_KEYS}
    if unknown:
        drafts[ARTIFACT_KEYS[0]] = "Service date: UNKNOWN"
    at.session_state["drafts"] = drafts
    for key, value in drafts.items():
        at.session_state[f"edit_{key}"] = value
    at.session_state["approver_name"] = ""
    at.session_state["unknown_acknowledged"] = False


def test_password_gate_unlocks_intake_without_network(monkeypatch):
    at = make_app(monkeypatch)
    at.run()
    assert not list(at.exception)
    assert [item.label for item in at.text_input] == ["Pilot password"]
    assert [item.label for item in at.button] == ["Unlock"]

    at.text_input[0].input("test-password")
    at.button[0].click().run()
    assert not list(at.exception)
    assert at.session_state["authenticated"] is True
    assert [item.label for item in at.button] == ["Generate Drafts"]


def test_approval_exports_and_edit_invalidation(monkeypatch):
    at = make_app(monkeypatch)
    seed_review(at)
    at.run()
    assert not list(at.exception)
    assert len(at.tabs) == 5
    assert at.button[1].label == "Approve & Export"
    assert at.button[1].disabled is True
    assert not at.get("download_button")

    at.text_input[0].input("Jane Smith").run()
    assert at.button[1].disabled is False
    at.button[1].click().run()
    assert not list(at.exception)
    assert len(at.get("download_button")) == 6  # one PDF plus five text files

    at.text_area[1].input("A corrected internal summary").run()
    assert not list(at.exception)
    assert not at.get("download_button")
    assert "approved_pack" not in at.session_state


def test_unknown_requires_explicit_acknowledgement(monkeypatch):
    at = make_app(monkeypatch)
    seed_review(at, unknown=True)
    at.run()
    at.text_input[0].input("Jane Smith").run()
    assert len(at.checkbox) == 1
    assert at.button[1].disabled is True

    at.checkbox[0].check().run()
    assert at.button[1].disabled is False
