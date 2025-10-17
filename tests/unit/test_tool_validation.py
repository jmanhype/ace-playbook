"""
Unit tests for tool validation in ReActGenerator

Tests T011: Verify valid/invalid tool signatures are correctly identified
"""

import pytest
from typing import List, Dict
from ace.generator.react_generator import validate_tool, ToolValidationError


@pytest.mark.unit
@pytest.mark.tool_validation
class TestToolValidation:
    """Test tool signature validation for ReAct agents."""

    def test_valid_tool_with_annotations(self):
        """Valid tool with type annotations should pass validation."""

        def search_database(query: str, limit: int = 10) -> List[str]:
            """Search database for records."""
            return []

        errors = validate_tool(search_database)
        assert len(errors) == 0, f"Valid tool should have no errors, got: {errors}"

    def test_valid_tool_with_docstring(self):
        """Tool with docstring should pass validation."""

        def rank_results(results: List[str], criteria: str) -> List[str]:
            """Rank results by criteria."""
            return results

        errors = validate_tool(rank_results)
        assert len(errors) == 0

    def test_invalid_tool_missing_annotations(self):
        """Tool without type annotations should fail validation."""

        def bad_tool(query, limit=10):
            """Missing type annotations."""
            return []

        errors = validate_tool(bad_tool)
        assert len(errors) > 0
        assert any("missing type annotation" in err.lower() for err in errors)

    def test_invalid_tool_no_parameters(self):
        """Tool with no parameters should fail validation."""

        def no_params_tool() -> str:
            """Tool with no parameters."""
            return "result"

        errors = validate_tool(no_params_tool)
        assert len(errors) > 0
        assert any("at least one parameter" in err.lower() for err in errors)

    def test_tool_missing_docstring_warns(self):
        """Tool without docstring should generate warning."""

        def no_docstring_tool(query: str) -> str:
            return query

        errors = validate_tool(no_docstring_tool)
        assert any("docstring" in err.lower() for err in errors)

    def test_non_callable_fails_validation(self):
        """Non-callable object should fail validation."""
        not_a_function = "this is a string"

        errors = validate_tool(not_a_function)
        assert len(errors) > 0
        assert any("callable" in err.lower() for err in errors)

    def test_complex_tool_signatures(self):
        """Tool with complex types should validate correctly."""

        def complex_tool(
            data: List[Dict[str, str]], filters: Dict[str, int], threshold: float = 0.5
        ) -> Dict[str, List[str]]:
            """Process complex data structures."""
            return {}

        errors = validate_tool(complex_tool)
        assert len(errors) == 0


@pytest.mark.unit
@pytest.mark.tool_validation
class TestReActGeneratorToolRegistration:
    """Test tool registration in ReActGenerator."""

    def test_register_valid_tool(self):
        """Should successfully register a valid tool."""
        from ace.generator.react_generator import ReActGenerator

        def search(query: str) -> List[str]:
            """Search function."""
            return []

        agent = ReActGenerator(tools=[search])
        assert "search" in agent.tools
        assert len(agent.tools) == 1

    def test_register_invalid_tool_raises_error(self):
        """Should raise ToolValidationError for invalid tool."""
        from ace.generator.react_generator import ReActGenerator

        def bad_tool(query):  # Missing type annotation
            return []

        with pytest.raises(ToolValidationError):
            ReActGenerator(tools=[bad_tool])

    def test_register_duplicate_tool_raises_error(self):
        """Should raise DuplicateToolError when registering duplicate."""
        from ace.generator.react_generator import ReActGenerator, DuplicateToolError

        def search(query: str) -> List[str]:
            """Search function."""
            return []

        agent = ReActGenerator(tools=[search])

        with pytest.raises(DuplicateToolError):
            agent.register_tool(search)

    def test_validate_tools_returns_errors(self):
        """validate_tools() should return list of all tool errors."""
        from ace.generator.react_generator import ReActGenerator

        def valid_tool(query: str) -> str:
            """Valid tool."""
            return ""

        agent = ReActGenerator(tools=[valid_tool])
        errors = agent.validate_tools()
        assert isinstance(errors, list)
        # Valid tool should have no errors (or only docstring warnings)
        assert len([e for e in errors if "missing type annotation" in e.lower()]) == 0
