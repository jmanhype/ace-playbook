import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core import (
    ARTIFACT_KEYS,
    OUTPUT_SCHEMA,
    ApprovedPack,
    ValidationError,
    build_pdf_bytes,
    build_text_exports,
    can_approve,
    contains_unknown,
    create_approved_pack,
    generate_drafts,
    highlight_unknown_html,
    load_system_prompt,
    normalise_approver_name,
    parse_claude_response,
    prepare_notes,
    verify_password,
)


def valid_artifacts(suffix=""):
    return {key: f"Reviewed content {key}{suffix}" for key in ARTIFACT_KEYS}


def response_for(payload, stop_reason="end_turn", block_type="text"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type=block_type, text=json.dumps(payload))],
    )


def test_schema_has_exactly_five_required_string_fields():
    assert tuple(OUTPUT_SCHEMA["properties"]) == ARTIFACT_KEYS
    assert set(OUTPUT_SCHEMA["required"]) == set(ARTIFACT_KEYS)
    assert len(OUTPUT_SCHEMA["properties"]) == 5
    assert OUTPUT_SCHEMA["additionalProperties"] is False
    assert all(
        item["type"] == "string" for item in OUTPUT_SCHEMA["properties"].values()
    )


def test_system_prompt_contains_core_safety_rules():
    prompt = load_system_prompt()
    assert "exactly five" in prompt
    assert "Never invent" in prompt
    assert "`UNKNOWN`" in prompt
    assert "untrusted data" in prompt
    assert "Nothing in your response is approved or sent automatically" in prompt


def test_prepare_notes_combines_typed_and_utf8_upload():
    result = prepare_notes("Typed fact", "notes.MD", "Uploaded fact".encode())
    assert "TYPED NOTES" in result
    assert "Typed fact" in result
    assert "UPLOADED NOTES (notes.MD)" in result


@pytest.mark.parametrize(
    "name,data,message",
    [
        (None, None, "Enter notes"),
        ("notes.pdf", b"text", ".txt or .md"),
        ("notes.txt", b"\xff", "UTF-8"),
    ],
)
def test_prepare_notes_rejects_invalid_input(name, data, message):
    with pytest.raises(ValidationError, match=message):
        prepare_notes("", name, data)


def test_parse_claude_response_accepts_only_exact_complete_payload():
    expected = valid_artifacts()
    assert parse_claude_response(response_for(expected)) == expected


@pytest.mark.parametrize("reason", ["max_tokens", "refusal", None])
def test_parse_claude_response_rejects_non_completion(reason):
    with pytest.raises(ValidationError, match="complete"):
        parse_claude_response(response_for(valid_artifacts(), stop_reason=reason))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop(ARTIFACT_KEYS[0]),
        lambda value: value.update({"sixth_output": "not allowed"}),
        lambda value: value.update({ARTIFACT_KEYS[0]: ""}),
        lambda value: value.update({ARTIFACT_KEYS[0]: ["wrong type"]}),
    ],
)
def test_parse_claude_response_fails_closed_on_bad_fields(mutation):
    payload = valid_artifacts()
    mutation(payload)
    with pytest.raises(ValidationError):
        parse_claude_response(response_for(payload))


def test_parse_claude_response_rejects_non_text_and_malformed_json():
    with pytest.raises(ValidationError, match="non-text"):
        parse_claude_response(response_for(valid_artifacts(), block_type="tool_use"))
    malformed = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="not json")],
    )
    with pytest.raises(ValidationError, match="malformed"):
        parse_claude_response(malformed)


def test_generate_drafts_uses_structured_output_and_no_live_api():
    class FakeMessages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return response_for(valid_artifacts())

    fake_messages = FakeMessages()
    client = SimpleNamespace(messages=fake_messages)
    assert generate_drafts(client, "case notes", model="test-model") == valid_artifacts()
    kwargs = fake_messages.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    schema = kwargs["output_config"]["format"]["schema"]
    assert set(schema["properties"]) == set(ARTIFACT_KEYS)
    assert schema["additionalProperties"] is False
    assert "untrusted case data" in kwargs["messages"][0]["content"]


def test_unknown_detection_and_highlight_escape_user_content():
    assert contains_unknown("Date: UNKNOWN")
    assert not contains_unknown("unknown")
    rendered = highlight_unknown_html('<script>alert("x")</script> UNKNOWN')
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert '<mark class="unknown-token"' in rendered


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("Jane Smith", "Jane Smith"),
        ("  Dr  Jane O'Neil-Smith ", "Dr Jane O'Neil-Smith"),
        ("Jane", None),
        ("UNKNOWN", None),
        ("Jane <Smith>", None),
    ],
)
def test_approver_name_validation(candidate, expected):
    assert normalise_approver_name(candidate) == expected


def test_password_gate_uses_correct_comparison_behaviour():
    assert verify_password("pilot-secret", "pilot-secret")
    assert not verify_password("wrong", "pilot-secret")
    assert not verify_password("anything", "")


def test_approval_gate_requires_name_and_unknown_acknowledgement():
    clean = valid_artifacts()
    assert can_approve(clean, "Jane Smith", False)
    assert not can_approve(clean, "Jane", False)
    unresolved = valid_artifacts()
    unresolved[ARTIFACT_KEYS[0]] = "Date: UNKNOWN"
    assert not can_approve(unresolved, "Jane Smith", False)
    assert can_approve(unresolved, "Jane Smith", True)


def test_approved_snapshot_is_deeply_immutable_and_timestamped():
    source = valid_artifacts()
    when = datetime(2026, 8, 10, 12, 30, tzinfo=timezone.utc)
    pack = create_approved_pack(source, "Jane Smith", False, now=when)
    source[ARTIFACT_KEYS[0]] = "changed later"
    assert pack.artifacts[ARTIFACT_KEYS[0]] != "changed later"
    assert pack.approved_at_utc == "2026-08-10T12:30:00Z"
    with pytest.raises(TypeError):
        pack.artifacts[ARTIFACT_KEYS[0]] = "cannot change"


def test_export_helpers_require_approved_pack_and_return_six_downloads():
    with pytest.raises(ValidationError):
        build_pdf_bytes(valid_artifacts())
    with pytest.raises(ValidationError):
        build_text_exports(valid_artifacts())

    pack = create_approved_pack(valid_artifacts(" – café"), "Jane Smith", False)
    pdf = build_pdf_bytes(pack)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1_000
    text_files = build_text_exports(pack)
    assert len(text_files) == 5
    assert all(content.count(b"Approved by: Jane Smith") == 1 for content in text_files.values())
    assert all("café".encode() in content for content in text_files.values())


def test_approved_pack_cannot_be_constructed_with_bad_fields():
    bad = valid_artifacts()
    bad["extra"] = "sixth"
    with pytest.raises(ValidationError):
        ApprovedPack(bad, "Jane Smith", "2026-08-10T12:30:00Z")
