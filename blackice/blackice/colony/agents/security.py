"""Security Agent for BLACKICE 3.0 colony.

The Security Agent is responsible for:
- Reviewing code for security vulnerabilities
- Ensuring secure coding practices are followed
- Validating input handling and authentication
- Checking for OWASP top 10 vulnerabilities

Uses the SECURITY role from the agent schema.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from blackice.colony.agents.base import BaseColonyAgent
from blackice.primitives.types import AgentId, AgentRole, TaskId
from blackice.schemas.agent import Agent, AgentCapabilities, AGENT_PROMPTS

if TYPE_CHECKING:
    from blackice.adapters.models.base import ModelProvider
    from blackice.adapters.memory.base import MemoryProvider


logger = structlog.get_logger(__name__)


SECURITY_SYSTEM_PROMPT = """You are a security expert and application security agent. Your role is to:
- Review code for security vulnerabilities and attack vectors
- Ensure secure coding practices are followed throughout
- Validate authentication, authorization, and input handling
- Check for OWASP Top 10 and common vulnerability patterns
- Recommend security improvements and mitigations

Security areas to consider:
1. Injection attacks (SQL, command, LDAP, XPath, etc.)
2. Broken authentication and session management
3. Cross-Site Scripting (XSS)
4. Insecure direct object references
5. Security misconfiguration
6. Sensitive data exposure
7. Missing function-level access control
8. Cross-Site Request Forgery (CSRF)
9. Using components with known vulnerabilities
10. Unvalidated redirects and forwards

When reporting issues:
- Classify severity (Critical, High, Medium, Low)
- Reference CWE/OWASP categories when applicable
- Provide specific remediation steps
- Include code examples of secure alternatives

Always err on the side of security over convenience."""


class SecurityAgent(BaseColonyAgent):
    """Security agent for vulnerability detection and secure coding.

    The Security agent performs security-focused reviews and provides
    recommendations for securing the codebase.
    """

    def __init__(
        self,
        agent: Agent,
        model_provider: ModelProvider | None = None,
        memory_provider: MemoryProvider | None = None,
    ) -> None:
        """Initialize the Security agent."""
        super().__init__(agent, model_provider, memory_provider)
        if agent.system_prompt == AGENT_PROMPTS.get(AgentRole.SECURITY):
            agent.system_prompt = SECURITY_SYSTEM_PROMPT

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a security task.

        Args:
            task: Task definition with 'type' and task-specific fields

        Returns:
            Task result with security findings and recommendations
        """
        task_type = task.get("type", "security_review")
        task_id = TaskId(task.get("id", "unknown"))

        await self.start_execution(task_id)

        try:
            if task_type == "security_review":
                result = await self._security_review(task)
            elif task_type == "threat_model":
                result = await self._threat_modeling(task)
            elif task_type == "auth_review":
                result = await self._authentication_review(task)
            elif task_type == "input_validation":
                result = await self._input_validation_review(task)
            elif task_type == "secrets_scan":
                result = await self._secrets_scan(task)
            else:
                result = await self._generic_security(task)

            await self.complete_execution(success=True, output=json.dumps(result))
            return result

        except Exception as e:
            await self.complete_execution(success=False, error=str(e))
            raise

    async def _security_review(self, task: dict[str, Any]) -> dict[str, Any]:
        """Comprehensive security review of code.

        Args:
            task: Task with 'code', optional 'file_path', 'context'

        Returns:
            Security findings and recommendations
        """
        code = task.get("code", "")
        file_path = task.get("file_path", "")
        context = task.get("context", "")
        language = task.get("language", "python")

        context_section = f"\n\nContext:\n{context}" if context else ""

        prompt = f"""Perform a comprehensive security review of this code:

File: {file_path or 'Unknown'}
Language: {language}

```
{code}
```{context_section}

Analyze for:
1. Injection vulnerabilities (SQL, command, XSS, etc.)
2. Authentication and authorization issues
3. Cryptographic weaknesses
4. Sensitive data handling
5. Input validation gaps
6. Error handling that leaks information
7. Race conditions and TOCTOU issues
8. Insecure defaults
9. Missing security headers/controls
10. Dependency vulnerabilities (if identifiable)

Format as JSON with:
- findings: list with 'severity', 'category', 'cwe', 'location', 'description', 'remediation', 'code_example'
- risk_score: 0-10 overall risk
- critical_count: number of critical issues
- high_count: number of high issues
- recommendations: prioritized security improvements
- approval: 'approved' | 'changes_required' | 'rejected'"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
            else:
                result = {"raw_review": response}
        except json.JSONDecodeError:
            result = {"raw_review": response}

        return result

    async def _threat_modeling(self, task: dict[str, Any]) -> dict[str, Any]:
        """Create or review a threat model.

        Args:
            task: Task with 'architecture' or 'system_description'

        Returns:
            Threat model with identified threats and mitigations
        """
        architecture = task.get("architecture", task.get("system_description", ""))
        data_flows = task.get("data_flows", "")
        trust_boundaries = task.get("trust_boundaries", "")

        data_section = f"\n\nData Flows:\n{data_flows}" if data_flows else ""
        trust_section = f"\n\nTrust Boundaries:\n{trust_boundaries}" if trust_boundaries else ""

        prompt = f"""Create a threat model for the following system:

System Architecture:
{architecture}{data_section}{trust_section}

Using STRIDE methodology, identify:
1. Spoofing threats
2. Tampering threats
3. Repudiation threats
4. Information Disclosure threats
5. Denial of Service threats
6. Elevation of Privilege threats

For each threat:
- Severity (Critical/High/Medium/Low)
- Attack vector
- Affected components
- Mitigation strategy
- Detection approach

Format as JSON with:
- threats: list with 'category', 'severity', 'description', 'attack_vector', 'affected_components', 'mitigation', 'detection'
- attack_surface: summary of attack surface
- high_risk_areas: components needing most protection
- security_controls: recommended controls by layer"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_threat_model": response}
        except json.JSONDecodeError:
            return {"raw_threat_model": response}

    async def _authentication_review(self, task: dict[str, Any]) -> dict[str, Any]:
        """Review authentication and authorization implementation.

        Args:
            task: Task with 'code' related to auth

        Returns:
            Auth-specific security findings
        """
        code = task.get("code", "")
        auth_type = task.get("auth_type", "unknown")

        prompt = f"""Review authentication/authorization implementation:

Auth Type: {auth_type}

```
{code}
```

Check for:
1. Secure password handling (hashing, salting)
2. Session management security
3. Token security (JWT issues, expiration)
4. Brute force protection
5. Account enumeration prevention
6. Privilege escalation vectors
7. Secure credential storage
8. Multi-factor authentication support
9. Logout and session invalidation
10. Password reset security

Format as JSON with:
- findings: list with 'severity', 'category', 'description', 'remediation'
- auth_strength: assessment of overall auth strength
- missing_controls: security controls that should be added
- compliance_notes: relevant compliance considerations (OWASP, etc.)"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_auth_review": response}
        except json.JSONDecodeError:
            return {"raw_auth_review": response}

    async def _input_validation_review(self, task: dict[str, Any]) -> dict[str, Any]:
        """Review input validation and sanitization.

        Args:
            task: Task with 'code' handling user input

        Returns:
            Input validation findings
        """
        code = task.get("code", "")
        input_sources = task.get("input_sources", [])

        sources_str = "\n".join(f"- {s}" for s in input_sources) if input_sources else "Not specified"

        prompt = f"""Review input validation and sanitization:

```
{code}
```

Input Sources:
{sources_str}

Check for:
1. Input validation at all entry points
2. Proper encoding/escaping for output context
3. Length and format validation
4. Type checking
5. Whitelist vs blacklist approaches
6. Canonicalization of paths
7. Handling of special characters
8. Validation of file uploads
9. API input validation
10. Client-side validation (should not be only validation)

Format as JSON with:
- findings: list with 'input_point', 'severity', 'issue', 'remediation'
- missing_validation: inputs not properly validated
- encoding_gaps: output encoding issues
- recommendations: improvements to validation strategy"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_input_review": response}
        except json.JSONDecodeError:
            return {"raw_input_review": response}

    async def _secrets_scan(self, task: dict[str, Any]) -> dict[str, Any]:
        """Scan code for exposed secrets.

        Args:
            task: Task with 'code' to scan

        Returns:
            Found secrets and recommendations
        """
        code = task.get("code", "")

        prompt = f"""Scan the following code for exposed secrets:

```
{code}
```

Look for:
1. Hardcoded passwords
2. API keys and tokens
3. Private keys and certificates
4. Database connection strings
5. AWS/GCP/Azure credentials
6. OAuth secrets
7. Encryption keys
8. JWT secrets
9. Internal URLs and endpoints
10. Any other sensitive data

Format as JSON with:
- secrets_found: list with 'type', 'location', 'severity', 'excerpt' (redacted)
- risk_assessment: overall risk from exposed secrets
- remediation: steps to secure each secret type
- prevention: how to prevent future exposure"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_secrets_scan": response}
        except json.JSONDecodeError:
            return {"raw_secrets_scan": response}

    async def _generic_security(self, task: dict[str, Any]) -> dict[str, Any]:
        """Handle generic security requests."""
        description = task.get("description", str(task))

        prompt = f"""Security task:

{description}

Provide a security-focused analysis with findings and recommendations.
Format as JSON with 'findings', 'risk_score', and 'recommendations'."""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_security": response}
        except json.JSONDecodeError:
            return {"raw_security": response}


def create_security(
    agent_id: AgentId | None = None,
    model_provider: ModelProvider | None = None,
    memory_provider: MemoryProvider | None = None,
) -> SecurityAgent:
    """Factory function to create a Security agent."""
    from blackice.schemas.agent import create_agent

    agent = create_agent(
        role=AgentRole.SECURITY,
        agent_id=agent_id,
        system_prompt=SECURITY_SYSTEM_PROMPT,
        capabilities=AgentCapabilities(
            can_write_code=False,  # Security doesn't write code
            can_execute_commands=False,
            can_read_files=True,
            can_write_files=False,
            domains=["security", "appsec", "vulnerability"],
        ),
    )

    return SecurityAgent(agent, model_provider, memory_provider)
