"""Tests for BLACKICE API endpoints."""

import pytest
from fastapi.testclient import TestClient

from blackice.api import app, create_app
from blackice.api.schemas import Edition, ProviderType, RunStatus


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_api_info(self, client: TestClient) -> None:
        """Root endpoint returns API metadata."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "BLACKICE API"
        assert data["version"] == "3.0.0"
        assert data["docs"] == "/docs"


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_liveness_probe(self, client: TestClient) -> None:
        """Liveness probe returns alive status."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_probe(self, client: TestClient) -> None:
        """Readiness probe returns ready/not_ready status."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] in ["ready", "not_ready"]

    def test_health_check(self, client: TestClient) -> None:
        """Health check returns system status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert data["version"] == "3.0.0"
        assert "providers" in data


class TestProvidersEndpoints:
    """Tests for provider listing endpoints."""

    def test_list_providers(self, client: TestClient) -> None:
        """List providers returns available LLM providers."""
        response = client.get("/api/v1/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "default" in data
        assert data["default"] == "ollama"


class TestRunsEndpoints:
    """Tests for run management endpoints."""

    def test_list_runs_empty(self, client: TestClient) -> None:
        """List runs returns empty list initially."""
        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_create_run(self, client: TestClient) -> None:
        """Create run starts a new BLACKICE run."""
        response = client.post(
            "/api/v1/runs",
            json={
                "vision": "Create a hello world CLI app",
                "edition": "core",
                "provider": "ollama",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "run_id" in data
        assert data["run_id"].startswith("run-")
        assert data["status"] == "created"
        assert "workspace" in data

    def test_get_run(self, client: TestClient) -> None:
        """Get run returns run details."""
        # Create a run first
        create_response = client.post(
            "/api/v1/runs",
            json={"vision": "Test vision"},
        )
        run_id = create_response.json()["run_id"]

        # Get the run
        response = client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == run_id
        assert data["vision"] == "Test vision"
        assert "status" in data
        assert "workspace" in data

    def test_get_run_not_found(self, client: TestClient) -> None:
        """Get nonexistent run returns 404."""
        response = client.get("/api/v1/runs/run-nonexistent")
        assert response.status_code == 404

    def test_delete_run(self, client: TestClient) -> None:
        """Delete run removes the run."""
        # Create a run first
        create_response = client.post(
            "/api/v1/runs",
            json={"vision": "Test to delete"},
        )
        run_id = create_response.json()["run_id"]

        # Delete the run
        response = client.delete(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200

        # Verify it's gone
        get_response = client.get(f"/api/v1/runs/{run_id}")
        assert get_response.status_code == 404


class TestOpenAPISchema:
    """Tests for OpenAPI schema generation."""

    def test_openapi_schema_generated(self, client: TestClient) -> None:
        """OpenAPI schema is generated correctly."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "BLACKICE API"
        assert data["info"]["version"] == "3.0.0"
        assert "/api/v1/runs" in data["paths"]
        assert "/api/v1/health" in data["paths"]

    def test_docs_endpoint_available(self, client: TestClient) -> None:
        """Swagger docs endpoint is available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint_available(self, client: TestClient) -> None:
        """ReDoc endpoint is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
