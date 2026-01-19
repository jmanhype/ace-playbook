"""TaskSpec schema for BLACKICE 3.0 Enterprise.

TaskSpecs define reusable, versioned specifications for reproducible
software generation with strict validation and compliance controls.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from blackice.primitives.types import (
    Hash,
    StrictnessLevel,
    Timestamp,
)


class ValidationRule(BaseModel):
    """A validation rule for TaskSpec inputs or outputs."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    rule_type: str = Field(..., description="Type of rule: regex, schema, range, custom")
    pattern: str | None = Field(default=None, description="Regex pattern or JSON schema")
    min_value: float | None = Field(default=None)
    max_value: float | None = Field(default=None)
    required: bool = Field(default=True)
    error_message: str = Field(default="Validation failed")


class SchemaDefinition(BaseModel):
    """JSON Schema definition for input or output validation."""

    type: str = Field(default="object")
    properties: dict[str, dict[str, Any]] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additional_properties: bool = Field(default=False)


class TaskSpecVersion(BaseModel):
    """Version information for a TaskSpec."""

    major: int = Field(ge=0)
    minor: int = Field(ge=0)
    patch: int = Field(ge=0)
    prerelease: str | None = Field(default=None)

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        return version

    @classmethod
    def parse(cls, version_str: str) -> TaskSpecVersion:
        """Parse a version string into TaskSpecVersion."""
        prerelease = None
        if "-" in version_str:
            version_str, prerelease = version_str.split("-", 1)

        parts = version_str.split(".")
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
            prerelease=prerelease,
        )


class TaskSpec(BaseModel):
    """A reusable specification for reproducible software generation.

    TaskSpecs (Enterprise feature) define strict, versioned specifications
    that enable:
    - Reproducible builds from the same spec
    - Compliance with organizational policies
    - Audit trails for generated software
    - Version control of generation parameters

    Attributes:
        id: Unique identifier for this TaskSpec
        name: Human-readable name
        version: Semantic version of the spec
        strictness: How strictly to enforce the spec
        description: Detailed description of what this spec produces
        input_schema: JSON Schema for valid inputs
        output_schema: JSON Schema for expected outputs
        validation_rules: Additional validation rules
    """

    id: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9-_]*$")
    name: str = Field(..., min_length=1, max_length=200)
    version: TaskSpecVersion = Field(default_factory=lambda: TaskSpecVersion(major=1, minor=0, patch=0))
    strictness: StrictnessLevel = Field(default=StrictnessLevel.STRICT)

    # Description
    description: str = Field(default="", max_length=5000)
    tags: list[str] = Field(default_factory=list)

    # Schemas
    input_schema: SchemaDefinition = Field(default_factory=SchemaDefinition)
    output_schema: SchemaDefinition = Field(default_factory=SchemaDefinition)

    # Validation
    validation_rules: list[ValidationRule] = Field(default_factory=list)

    # Generation parameters
    required_capabilities: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)

    # Compliance
    organization_id: str | None = Field(default=None)
    compliance_frameworks: list[str] = Field(default_factory=list)

    # Metadata
    created_at: Timestamp = Field(default_factory=Timestamp.now)
    updated_at: Timestamp = Field(default_factory=Timestamp.now)
    created_by: str | None = Field(default=None)
    approved_by: str | None = Field(default=None)

    # Hash for integrity
    content_hash: Hash | None = Field(default=None)

    class Config:
        """Pydantic configuration."""

        frozen = False

    def compute_hash(self) -> Hash:
        """Compute hash of the TaskSpec content for integrity verification."""
        import hashlib

        # Create deterministic representation
        content = f"{self.id}:{self.name}:{self.version}:{self.strictness.value}"
        content += f":{self.input_schema.model_dump_json()}:{self.output_schema.model_dump_json()}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()
        return Hash(value=hash_value)

    def validate_input(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate input data against the input schema.

        Returns (is_valid, list_of_errors).
        """
        errors: list[str] = []

        # Check required fields
        for field in self.input_schema.required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check property types
        for field, value in data.items():
            if field in self.input_schema.properties:
                prop = self.input_schema.properties[field]
                expected_type = prop.get("type")
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Field '{field}' must be a string")
                elif expected_type == "integer" and not isinstance(value, int):
                    errors.append(f"Field '{field}' must be an integer")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Field '{field}' must be a number")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Field '{field}' must be a boolean")
                elif expected_type == "array" and not isinstance(value, list):
                    errors.append(f"Field '{field}' must be an array")
                elif expected_type == "object" and not isinstance(value, dict):
                    errors.append(f"Field '{field}' must be an object")

        # Check additional properties
        if not self.input_schema.additional_properties:
            for field in data:
                if field not in self.input_schema.properties:
                    errors.append(f"Unknown field: {field}")

        return len(errors) == 0, errors

    def check_deviation(self, action: str) -> tuple[bool, str]:
        """Check if an action deviates from the spec.

        Returns (is_allowed, message) based on strictness level:
        - LEARNING: Always allow, warn about forbidden patterns
        - PERMISSIVE: Allow, warn about forbidden patterns
        - STRICT: Block forbidden patterns, but override is possible
        - LOCKED: Block forbidden patterns completely, no override
        """
        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            if pattern.lower() in action.lower():
                message = f"Action contains forbidden pattern: {pattern}"
                if self.strictness == StrictnessLevel.LOCKED:
                    return False, message
                elif self.strictness == StrictnessLevel.STRICT:
                    return False, f"{message} (override possible)"
                elif self.strictness == StrictnessLevel.LEARNING:
                    return True, f"Warning: {message} (learning mode - all actions allowed)"
                else:  # PERMISSIVE
                    return True, f"Warning: {message}"

        return True, "Action allowed"


class TaskSpecRegistry(BaseModel):
    """Registry of TaskSpecs for an organization."""

    organization_id: str
    specs: dict[str, TaskSpec] = Field(default_factory=dict)

    def register(self, spec: TaskSpec) -> None:
        """Register a TaskSpec."""
        key = f"{spec.id}@{spec.version}"
        self.specs[key] = spec

    def get(self, spec_id: str, version: str | None = None) -> TaskSpec | None:
        """Get a TaskSpec by ID and optional version."""
        if version:
            return self.specs.get(f"{spec_id}@{version}")

        # Find latest version
        matching = [s for key, s in self.specs.items() if key.startswith(f"{spec_id}@")]
        if not matching:
            return None
        return max(matching, key=lambda s: (s.version.major, s.version.minor, s.version.patch))

    def list_versions(self, spec_id: str) -> list[str]:
        """List all versions of a TaskSpec."""
        versions = []
        for key in self.specs:
            if key.startswith(f"{spec_id}@"):
                versions.append(key.split("@")[1])
        return sorted(versions)
