"""Integration tests for TaskSpec validation (IT-005).

Tests the full TaskSpec validation workflow including:
- Strictness tiers (learning, permissive, strict, locked)
- Input/output schema validation
- Policy violation detection and messaging
- Deviation tracking
"""

from __future__ import annotations

import pytest

from blackice.primitives.types import StrictnessLevel
from blackice.schemas.taskspec import (
    SchemaDefinition,
    TaskSpec,
    TaskSpecVersion,
    ValidationRule,
)


class TestTaskSpecValidation:
    """Test TaskSpec validation of inputs."""

    def test_valid_input_passes_validation(self) -> None:
        """Valid input matching schema should pass."""
        spec = TaskSpec(
            id="test-spec",
            name="Test Spec",
            input_schema=SchemaDefinition(
                properties={
                    "vision": {"type": "string"},
                    "language": {"type": "string"},
                },
                required=["vision"],
            ),
        )

        is_valid, errors = spec.validate_input({"vision": "Build a CLI tool", "language": "python"})
        assert is_valid is True
        assert errors == []

    def test_missing_required_field_fails(self) -> None:
        """Missing required field should fail validation."""
        spec = TaskSpec(
            id="test-spec",
            name="Test Spec",
            input_schema=SchemaDefinition(
                properties={"vision": {"type": "string"}},
                required=["vision"],
            ),
        )

        is_valid, errors = spec.validate_input({})
        assert is_valid is False
        assert "Missing required field: vision" in errors

    def test_wrong_type_fails_validation(self) -> None:
        """Wrong field type should fail validation."""
        spec = TaskSpec(
            id="test-spec",
            name="Test Spec",
            input_schema=SchemaDefinition(
                properties={"count": {"type": "integer"}},
                required=["count"],
            ),
        )

        is_valid, errors = spec.validate_input({"count": "not a number"})
        assert is_valid is False
        assert any("must be an integer" in e for e in errors)

    def test_additional_properties_rejected(self) -> None:
        """Unknown fields should be rejected when additional_properties is False."""
        spec = TaskSpec(
            id="test-spec",
            name="Test Spec",
            input_schema=SchemaDefinition(
                properties={"vision": {"type": "string"}},
                additional_properties=False,
            ),
        )

        is_valid, errors = spec.validate_input({"vision": "test", "unknown": "field"})
        assert is_valid is False
        assert "Unknown field: unknown" in errors


class TestStrictnessTiers:
    """Test different strictness levels."""

    def test_learning_mode_allows_deviations(self) -> None:
        """Learning mode should allow deviations with warnings."""
        spec = TaskSpec(
            id="test-spec",
            name="Test Spec",
            strictness=StrictnessLevel.LEARNING,
            forbidden_patterns=["sudo"],
        )

        is_allowed, message = spec.check_deviation("sudo rm -rf /")
        # Learning mode is most permissive - allows with warning
        assert is_allowed is True
        assert "Warning" in message or "forbidden" in message.lower()

    def test_permissive_mode_warns_on_deviations(self) -> None:
        """Permissive mode should warn but allow deviations."""
        spec = TaskSpec(
            id="test-spec",
            name="Test Spec",
            strictness=StrictnessLevel.PERMISSIVE,
            forbidden_patterns=["dangerous"],
        )

        is_allowed, message = spec.check_deviation("dangerous command")
        assert is_allowed is True
        assert "Warning" in message

    def test_strict_mode_blocks_with_override_option(self) -> None:
        """Strict mode should block but indicate override is possible."""
        spec = TaskSpec(
            id="test-spec",
            name="Test Spec",
            strictness=StrictnessLevel.STRICT,
            forbidden_patterns=["forbidden"],
        )

        is_allowed, message = spec.check_deviation("forbidden action")
        assert is_allowed is False
        assert "override possible" in message.lower()

    def test_locked_mode_blocks_completely(self) -> None:
        """Locked mode should block with no override."""
        spec = TaskSpec(
            id="test-spec",
            name="Test Spec",
            strictness=StrictnessLevel.LOCKED,
            forbidden_patterns=["blocked"],
        )

        is_allowed, message = spec.check_deviation("blocked operation")
        assert is_allowed is False
        assert "override" not in message.lower()


class TestDeviationTracking:
    """Test deviation detection and tracking."""

    def test_allowed_action_passes(self) -> None:
        """Actions not matching forbidden patterns should pass."""
        spec = TaskSpec(
            id="test-spec",
            name="Test Spec",
            forbidden_patterns=["danger", "risk"],
        )

        is_allowed, message = spec.check_deviation("safe action")
        assert is_allowed is True
        assert "allowed" in message.lower()

    def test_forbidden_pattern_case_insensitive(self) -> None:
        """Forbidden pattern matching should be case-insensitive."""
        spec = TaskSpec(
            id="test-spec",
            name="Test Spec",
            strictness=StrictnessLevel.STRICT,
            forbidden_patterns=["SECRET"],
        )

        is_allowed, _ = spec.check_deviation("contains secret data")
        assert is_allowed is False

    def test_multiple_forbidden_patterns(self) -> None:
        """Multiple forbidden patterns should all be checked."""
        spec = TaskSpec(
            id="test-spec",
            name="Test Spec",
            strictness=StrictnessLevel.STRICT,
            forbidden_patterns=["pattern1", "pattern2", "pattern3"],
        )

        is_allowed1, _ = spec.check_deviation("contains pattern1")
        is_allowed2, _ = spec.check_deviation("contains pattern2")
        is_allowed3, _ = spec.check_deviation("contains pattern3")
        is_allowed_safe, _ = spec.check_deviation("safe action")

        assert is_allowed1 is False
        assert is_allowed2 is False
        assert is_allowed3 is False
        assert is_allowed_safe is True


class TestTaskSpecHashIntegrity:
    """Test TaskSpec hash computation for integrity."""

    def test_same_spec_produces_same_hash(self) -> None:
        """Identical specs should produce identical hashes."""
        spec1 = TaskSpec(id="test-spec", name="Test", strictness=StrictnessLevel.STRICT)
        spec2 = TaskSpec(id="test-spec", name="Test", strictness=StrictnessLevel.STRICT)

        hash1 = spec1.compute_hash()
        hash2 = spec2.compute_hash()

        assert hash1.value == hash2.value

    def test_different_specs_produce_different_hashes(self) -> None:
        """Different specs should produce different hashes."""
        spec1 = TaskSpec(id="spec-a", name="Spec A")
        spec2 = TaskSpec(id="spec-b", name="Spec B")

        hash1 = spec1.compute_hash()
        hash2 = spec2.compute_hash()

        assert hash1.value != hash2.value

    def test_hash_changes_with_version(self) -> None:
        """Version changes should affect the hash."""
        spec1 = TaskSpec(
            id="test-spec",
            name="Test",
            version=TaskSpecVersion(major=1, minor=0, patch=0),
        )
        spec2 = TaskSpec(
            id="test-spec",
            name="Test",
            version=TaskSpecVersion(major=2, minor=0, patch=0),
        )

        assert spec1.compute_hash().value != spec2.compute_hash().value


class TestTaskSpecRegistry:
    """Test TaskSpec registry for organization management."""

    def test_register_and_retrieve(self) -> None:
        """Should be able to register and retrieve specs."""
        from blackice.schemas.taskspec import TaskSpecRegistry

        registry = TaskSpecRegistry(organization_id="org-123")
        spec = TaskSpec(
            id="my-spec",
            name="My Spec",
            version=TaskSpecVersion(major=1, minor=0, patch=0),
        )

        registry.register(spec)
        retrieved = registry.get("my-spec", "1.0.0")

        assert retrieved is not None
        assert retrieved.id == "my-spec"

    def test_get_latest_version(self) -> None:
        """Should return latest version when version not specified."""
        from blackice.schemas.taskspec import TaskSpecRegistry

        registry = TaskSpecRegistry(organization_id="org-123")

        spec_v1 = TaskSpec(
            id="my-spec",
            name="My Spec v1",
            version=TaskSpecVersion(major=1, minor=0, patch=0),
        )
        spec_v2 = TaskSpec(
            id="my-spec",
            name="My Spec v2",
            version=TaskSpecVersion(major=2, minor=0, patch=0),
        )

        registry.register(spec_v1)
        registry.register(spec_v2)

        latest = registry.get("my-spec")

        assert latest is not None
        assert latest.version.major == 2

    def test_list_versions(self) -> None:
        """Should list all versions of a spec."""
        from blackice.schemas.taskspec import TaskSpecRegistry

        registry = TaskSpecRegistry(organization_id="org-123")

        for minor in range(3):
            spec = TaskSpec(
                id="versioned-spec",
                name=f"Versioned Spec v1.{minor}",
                version=TaskSpecVersion(major=1, minor=minor, patch=0),
            )
            registry.register(spec)

        versions = registry.list_versions("versioned-spec")

        assert len(versions) == 3


class TestValidationRules:
    """Test custom validation rules."""

    def test_validation_rule_creation(self) -> None:
        """Should be able to create validation rules."""
        rule = ValidationRule(
            name="max-length",
            description="Maximum string length",
            rule_type="range",
            max_value=100,
            error_message="Value too long",
        )

        assert rule.name == "max-length"
        assert rule.max_value == 100

    def test_spec_with_validation_rules(self) -> None:
        """Spec should accept validation rules."""
        spec = TaskSpec(
            id="validated-spec",
            name="Validated Spec",
            validation_rules=[
                ValidationRule(
                    name="vision-length",
                    rule_type="range",
                    max_value=5000,
                    error_message="Vision too long",
                ),
            ],
        )

        assert len(spec.validation_rules) == 1
        assert spec.validation_rules[0].name == "vision-length"


class TestComplianceFeatures:
    """Test Enterprise compliance features."""

    def test_spec_with_compliance_frameworks(self) -> None:
        """Spec should support compliance framework tags."""
        spec = TaskSpec(
            id="compliant-spec",
            name="Compliant Spec",
            organization_id="org-enterprise",
            compliance_frameworks=["SOC2", "HIPAA", "GDPR"],
        )

        assert "SOC2" in spec.compliance_frameworks
        assert "HIPAA" in spec.compliance_frameworks
        assert spec.organization_id == "org-enterprise"

    def test_spec_with_required_capabilities(self) -> None:
        """Spec should define required agent capabilities."""
        spec = TaskSpec(
            id="capable-spec",
            name="Capable Spec",
            required_capabilities=["code_review", "security_audit"],
        )

        assert "code_review" in spec.required_capabilities
        assert "security_audit" in spec.required_capabilities

    def test_spec_with_allowed_tools(self) -> None:
        """Spec should define allowed tools."""
        spec = TaskSpec(
            id="tooled-spec",
            name="Tooled Spec",
            allowed_tools=["pytest", "ruff", "black"],
            forbidden_patterns=["curl", "wget"],
        )

        assert "pytest" in spec.allowed_tools
        assert "curl" in spec.forbidden_patterns
