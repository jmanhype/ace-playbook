#!/usr/bin/env python3
"""
Letta MAS Maintenance Script v2 - Design Patterns Edition

Applied Patterns:
1. Strategy Pattern - Interchangeable healing strategies
2. Observer Pattern - Event-based notifications
3. Chain of Responsibility - Cascading health actions
4. Circuit Breaker - Prevent repeated failures
5. Factory Pattern - Create checkers/healers
6. Template Method - Standardized maintenance workflow
7. State Pattern - Agent health states

Run via cron every 5 minutes:
*/5 * * * * /usr/bin/python3 /home/straughter/letta_maintenance_v2.py >> /tmp/letta_maintenance.log 2>&1
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Protocol, Optional
import json
import requests


# =============================================================================
# CONFIGURATION (could be externalized to YAML/JSON)
# =============================================================================
@dataclass(frozen=True)
class Config:
    """Immutable configuration - Single source of truth."""
    letta_url: str = "http://localhost:8283"
    router_url: str = "http://localhost:3000"
    comfyui_url: str = "http://localhost:8188"

    # Thresholds (Strategy parameters)
    summarize_threshold: int = 300
    reset_threshold: int = 500
    critical_threshold: int = 600
    max_summarize_failures: int = 2

    # Circuit breaker settings
    circuit_breaker_threshold: int = 3
    circuit_breaker_timeout: int = 300  # seconds

    # Video monitoring
    video_stale_minutes: int = 60
    video_stuck_minutes: int = 15  # Stale + pending = stuck production
    video_dir: Path = field(default_factory=lambda: Path("/home/straughter/ComfyUI/output/video"))

    # Director for unsticking
    director_id: str = "agent-22069f59-7a79-4890-bf4f-1f2a69696267"

    # State persistence
    state_file: Path = field(default_factory=lambda: Path("/tmp/letta_maintenance_state.json"))


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================
class HealthState(Enum):
    """State Pattern - Agent health states."""
    HEALTHY = auto()
    WARNING = auto()
    CRITICAL = auto()
    FAILED = auto()


class HealingAction(Enum):
    """Actions that can be taken to heal an agent."""
    NONE = auto()
    SUMMARIZE = auto()
    RESET = auto()
    RESTART = auto()


@dataclass
class AgentStatus:
    """Value object for agent status."""
    name: str
    agent_id: str
    message_count: int
    state: HealthState
    last_healed: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "agent_id": self.agent_id,
            "message_count": self.message_count,
            "state": self.state.name,
            "last_healed": self.last_healed.isoformat() if self.last_healed else None
        }


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    healthy: bool
    message: str = ""
    metadata: dict = field(default_factory=dict)


# =============================================================================
# OBSERVER PATTERN - Event System
# =============================================================================
class Event:
    """Base event class."""
    def __init__(self, source: str, data: dict = None):
        self.timestamp = datetime.now()
        self.source = source
        self.data = data or {}


class HealthCheckEvent(Event):
    """Emitted when health check completes."""
    pass


class HealingEvent(Event):
    """Emitted when healing action taken."""
    pass


class AlertEvent(Event):
    """Emitted when alert condition detected."""
    pass


class Observer(Protocol):
    """Observer protocol for type hints."""
    def on_event(self, event: Event) -> None: ...


class EventBus:
    """
    Observer Pattern - Central event dispatcher.

    Benefits:
    - Decouples event producers from consumers
    - Easy to add new observers (logging, alerting, metrics)
    - Enables async processing if needed
    """
    def __init__(self):
        self._observers: list[Observer] = []

    def subscribe(self, observer: Observer) -> None:
        self._observers.append(observer)

    def unsubscribe(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def emit(self, event: Event) -> None:
        for observer in self._observers:
            try:
                observer.on_event(event)
            except Exception as e:
                print(f"[ERROR] Observer failed: {e}")


class LoggingObserver:
    """Logs all events to console."""
    def on_event(self, event: Event) -> None:
        timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        event_type = type(event).__name__

        if isinstance(event, AlertEvent):
            level = "WARN"
        elif isinstance(event, HealingEvent):
            level = "INFO" if event.data.get("success") else "ERROR"
        else:
            level = "INFO"

        print(f"[{timestamp}] [{level}] [{event_type}] {event.source}: {event.data.get('message', '')}")


class MetricsObserver:
    """Collects metrics for monitoring."""
    def __init__(self):
        self.metrics = {
            "health_checks": 0,
            "healing_actions": 0,
            "failures": 0,
            "alerts": 0
        }

    def on_event(self, event: Event) -> None:
        if isinstance(event, HealthCheckEvent):
            self.metrics["health_checks"] += 1
        elif isinstance(event, HealingEvent):
            self.metrics["healing_actions"] += 1
            if not event.data.get("success"):
                self.metrics["failures"] += 1
        elif isinstance(event, AlertEvent):
            self.metrics["alerts"] += 1


# =============================================================================
# STRATEGY PATTERN - Healing Strategies
# =============================================================================
class HealingStrategy(ABC):
    """
    Strategy Pattern - Abstract healing strategy.

    Benefits:
    - Encapsulates healing algorithms
    - Easy to add new strategies
    - Strategies can be swapped at runtime
    """
    @abstractmethod
    def can_heal(self, status: AgentStatus, context: dict) -> bool:
        """Check if this strategy applies."""
        pass

    @abstractmethod
    def heal(self, status: AgentStatus, config: Config) -> bool:
        """Execute healing action."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def action(self) -> HealingAction:
        pass


class SummarizeStrategy(HealingStrategy):
    """Strategy: Summarize messages to reduce context."""

    @property
    def name(self) -> str:
        return "Summarize"

    @property
    def action(self) -> HealingAction:
        return HealingAction.SUMMARIZE

    def can_heal(self, status: AgentStatus, context: dict) -> bool:
        failures = context.get("summarization_failures", {}).get(status.name, 0)
        return (
            status.state == HealthState.WARNING and
            failures < context.get("max_failures", 2)
        )

    def heal(self, status: AgentStatus, config: Config) -> bool:
        try:
            resp = requests.post(
                f"{config.letta_url}/v1/agents/{status.agent_id}/summarize",
                timeout=120
            )
            return resp.status_code == 200
        except:
            return False


class ResetStrategy(HealingStrategy):
    """Strategy: Reset messages (nuclear option)."""

    @property
    def name(self) -> str:
        return "Reset"

    @property
    def action(self) -> HealingAction:
        return HealingAction.RESET

    def can_heal(self, status: AgentStatus, context: dict) -> bool:
        failures = context.get("summarization_failures", {}).get(status.name, 0)
        return (
            status.state == HealthState.CRITICAL or
            (status.state == HealthState.WARNING and failures >= context.get("max_failures", 2))
        )

    def heal(self, status: AgentStatus, config: Config) -> bool:
        try:
            resp = requests.patch(
                f"{config.letta_url}/v1/agents/{status.agent_id}/reset-messages",
                headers={"Content-Type": "application/json"},
                json={},
                timeout=30
            )
            return resp.status_code == 200
        except:
            return False


# =============================================================================
# CHAIN OF RESPONSIBILITY - Cascading Health Actions
# =============================================================================
class HealthHandler(ABC):
    """
    Chain of Responsibility Pattern - Health check handler.

    Benefits:
    - Decouples sender from receivers
    - Allows dynamic chain modification
    - Each handler can decide to process or pass
    """
    def __init__(self, next_handler: 'HealthHandler' = None):
        self._next = next_handler

    def set_next(self, handler: 'HealthHandler') -> 'HealthHandler':
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, status: AgentStatus, context: dict) -> Optional[HealingAction]:
        pass

    def _pass_to_next(self, status: AgentStatus, context: dict) -> Optional[HealingAction]:
        if self._next:
            return self._next.handle(status, context)
        return HealingAction.NONE


class CriticalHandler(HealthHandler):
    """Handle critical state - immediate reset."""
    def handle(self, status: AgentStatus, context: dict) -> Optional[HealingAction]:
        if status.state == HealthState.CRITICAL:
            return HealingAction.RESET
        return self._pass_to_next(status, context)


class WarningHandler(HealthHandler):
    """Handle warning state - try summarize first."""
    def handle(self, status: AgentStatus, context: dict) -> Optional[HealingAction]:
        if status.state == HealthState.WARNING:
            failures = context.get("summarization_failures", {}).get(status.name, 0)
            if failures < context.get("max_failures", 2):
                return HealingAction.SUMMARIZE
            else:
                return HealingAction.RESET
        return self._pass_to_next(status, context)


class HealthyHandler(HealthHandler):
    """Handle healthy state - no action needed."""
    def handle(self, status: AgentStatus, context: dict) -> Optional[HealingAction]:
        if status.state == HealthState.HEALTHY:
            return HealingAction.NONE
        return self._pass_to_next(status, context)


# =============================================================================
# CIRCUIT BREAKER PATTERN - Prevent Repeated Failures
# =============================================================================
class CircuitBreaker:
    """
    Circuit Breaker Pattern - Prevents cascading failures.

    States:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Too many failures, requests blocked
    - HALF_OPEN: Testing if system recovered

    Benefits:
    - Prevents overwhelming failing services
    - Allows graceful degradation
    - Auto-recovery after timeout
    """

    class State(Enum):
        CLOSED = auto()
        OPEN = auto()
        HALF_OPEN = auto()

    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = self.State.CLOSED

    def can_execute(self) -> bool:
        """Check if operation can proceed."""
        if self.state == self.State.CLOSED:
            return True

        if self.state == self.State.OPEN:
            if self._timeout_elapsed():
                self.state = self.State.HALF_OPEN
                return True
            return False

        # HALF_OPEN - allow one test request
        return True

    def record_success(self) -> None:
        """Record successful operation."""
        self.failures = 0
        self.state = self.State.CLOSED

    def record_failure(self) -> None:
        """Record failed operation."""
        self.failures += 1
        self.last_failure_time = datetime.now()

        if self.failures >= self.failure_threshold:
            self.state = self.State.OPEN

    def _timeout_elapsed(self) -> bool:
        if not self.last_failure_time:
            return True
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    @property
    def is_open(self) -> bool:
        return self.state == self.State.OPEN


# =============================================================================
# FACTORY PATTERN - Create Health Checkers
# =============================================================================
class HealthCheckerFactory:
    """
    Factory Pattern - Creates appropriate health checkers.

    Benefits:
    - Centralizes object creation
    - Easy to add new checker types
    - Decouples creation from usage
    """

    @staticmethod
    def create_letta_checker(config: Config) -> Callable[[], HealthCheckResult]:
        def check() -> HealthCheckResult:
            try:
                resp = requests.get(f"{config.letta_url}/v1/agents/", timeout=10)
                return HealthCheckResult(
                    name="Letta",
                    healthy=resp.status_code == 200,
                    message="OK" if resp.status_code == 200 else f"Status {resp.status_code}"
                )
            except Exception as e:
                return HealthCheckResult(name="Letta", healthy=False, message=str(e))
        return check

    @staticmethod
    def create_router_checker(config: Config) -> Callable[[], HealthCheckResult]:
        def check() -> HealthCheckResult:
            try:
                resp = requests.get(f"{config.router_url}/health", timeout=5)
                healthy = resp.status_code in [200, 404]
                return HealthCheckResult(
                    name="Router",
                    healthy=healthy,
                    message="OK" if healthy else f"Status {resp.status_code}"
                )
            except Exception as e:
                return HealthCheckResult(name="Router", healthy=False, message=str(e))
        return check

    @staticmethod
    def create_comfyui_checker(config: Config) -> Callable[[], HealthCheckResult]:
        def check() -> HealthCheckResult:
            try:
                resp = requests.get(f"{config.comfyui_url}/queue", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    return HealthCheckResult(
                        name="ComfyUI",
                        healthy=True,
                        message="OK",
                        metadata={
                            "running": len(data.get("queue_running", [])),
                            "pending": len(data.get("queue_pending", []))
                        }
                    )
                return HealthCheckResult(name="ComfyUI", healthy=False, message=f"Status {resp.status_code}")
            except Exception as e:
                return HealthCheckResult(name="ComfyUI", healthy=False, message=str(e))
        return check


# =============================================================================
# TEMPLATE METHOD - Maintenance Workflow
# =============================================================================
class MaintenanceRunner:
    """
    Template Method Pattern - Defines maintenance workflow skeleton.

    Benefits:
    - Enforces consistent workflow
    - Allows customization via hooks
    - Separates invariant from variant behavior
    """

    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.state = self._load_state()

        # Build chain of responsibility
        self.health_chain = HealthyHandler()
        warning_handler = WarningHandler()
        critical_handler = CriticalHandler()
        self.health_chain.set_next(warning_handler).set_next(critical_handler)

        # Strategies
        self.strategies: dict[HealingAction, HealingStrategy] = {
            HealingAction.SUMMARIZE: SummarizeStrategy(),
            HealingAction.RESET: ResetStrategy(),
        }

        # Circuit breakers per agent
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Health checkers (Factory)
        self.health_checkers = [
            HealthCheckerFactory.create_letta_checker(config),
            HealthCheckerFactory.create_router_checker(config),
            HealthCheckerFactory.create_comfyui_checker(config),
        ]

        # Agent registry
        self.agents = {
            "director": "agent-22069f59-7a79-4890-bf4f-1f2a69696267",
            "writer": "agent-e565b3e8-4a59-440a-89ab-6c279d61cfb0",
            "cameraman": "agent-f939736a-46fc-4115-a584-0a8cf896212a",
        }

    def run(self) -> None:
        """Template method - main workflow."""
        self._log_header()

        # Step 1: Health checks
        health_results = self._run_health_checks()
        if not self._can_proceed(health_results):
            self._save_state()
            return

        # Step 2: Agent maintenance
        for name, agent_id in self.agents.items():
            self._maintain_agent(name, agent_id)

        # Step 3: Video health
        self._check_video_health()

        # Step 4: Summary
        self._log_summary()
        self._save_state()

    def _run_health_checks(self) -> list[HealthCheckResult]:
        """Run all health checks."""
        results = []
        for checker in self.health_checkers:
            result = checker()
            results.append(result)
            self.event_bus.emit(HealthCheckEvent(
                source=result.name,
                data={"healthy": result.healthy, "message": result.message}
            ))

        status_str = ", ".join(f"{r.name}={'OK' if r.healthy else 'FAIL'}" for r in results)
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Health: {status_str}")
        return results

    def _can_proceed(self, results: list[HealthCheckResult]) -> bool:
        """Check if we can proceed with maintenance."""
        letta_healthy = any(r.name == "Letta" and r.healthy for r in results)
        if not letta_healthy:
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [ERROR] Letta not healthy - cannot proceed")
        return letta_healthy

    def _maintain_agent(self, name: str, agent_id: str) -> None:
        """Maintain a single agent."""
        # Get circuit breaker
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_threshold,
                recovery_timeout=self.config.circuit_breaker_timeout
            )
        breaker = self.circuit_breakers[name]

        # Check circuit breaker
        if not breaker.can_execute():
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [WARN] Circuit breaker OPEN for {name} - skipping")
            return

        # Get agent status
        status = self._get_agent_status(name, agent_id)
        if not status:
            breaker.record_failure()
            return

        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] {name}: {status.message_count} msgs ({status.state.name})")

        # Determine action via chain of responsibility
        context = {
            "summarization_failures": self.state.get("summarization_failures", {}),
            "max_failures": self.config.max_summarize_failures
        }
        action = self.health_chain.handle(status, context)

        # Execute action via strategy
        if action and action != HealingAction.NONE:
            strategy = self.strategies.get(action)
            if strategy:
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Executing {strategy.name} for {name}")
                success = strategy.heal(status, self.config)

                self.event_bus.emit(HealingEvent(
                    source=name,
                    data={"action": action.name, "success": success, "message": f"{strategy.name} {'succeeded' if success else 'failed'}"}
                ))

                if success:
                    breaker.record_success()
                    self._update_state_on_success(name, action)
                else:
                    breaker.record_failure()
                    self._update_state_on_failure(name, action)

    def _get_agent_status(self, name: str, agent_id: str) -> Optional[AgentStatus]:
        """Get current agent status."""
        try:
            resp = requests.get(f"{self.config.letta_url}/v1/agents/{agent_id}", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                msg_count = len(data.get("message_ids", []))

                # Determine state
                if msg_count >= self.config.critical_threshold:
                    state = HealthState.CRITICAL
                elif msg_count >= self.config.summarize_threshold:
                    state = HealthState.WARNING
                else:
                    state = HealthState.HEALTHY

                return AgentStatus(
                    name=name,
                    agent_id=agent_id,
                    message_count=msg_count,
                    state=state
                )
        except Exception as e:
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [ERROR] Failed to get status for {name}: {e}")
        return None

    def _check_video_health(self) -> None:
        """Check video generation health."""
        try:
            videos = list(self.config.video_dir.glob("*.mp4"))
            video_count = len(videos)

            latest_time = None
            if videos:
                latest = max(videos, key=lambda p: p.stat().st_mtime)
                latest_time = datetime.fromtimestamp(latest.stat().st_mtime)

            # Get queue status
            queue = {"running": 0, "pending": 0}
            try:
                resp = requests.get(f"{self.config.comfyui_url}/queue", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    queue["running"] = len(data.get("queue_running", []))
                    queue["pending"] = len(data.get("queue_pending", []))
            except:
                pass

            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Videos: {video_count}, Queue: {queue['running']} running / {queue['pending']} pending")

            if latest_time:
                age_minutes = (datetime.now() - latest_time).total_seconds() / 60
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Latest video: {age_minutes:.1f} minutes ago")

                if age_minutes > self.config.video_stale_minutes and queue["running"] == 0 and queue["pending"] == 0:
                    self.event_bus.emit(AlertEvent(
                        source="VideoMonitor",
                        data={"message": f"VIDEO STALE - No new video in {age_minutes:.1f} min"}
                    ))

                # NEW: Detect stuck production (stale videos but queue should be active)
                if age_minutes > self.config.video_stuck_minutes and queue["running"] == 0 and queue["pending"] == 0:
                    self._check_and_unstick_production(age_minutes)

            self.state["last_video_count"] = video_count
            self.state["last_video_time"] = latest_time.isoformat() if latest_time else None

        except Exception as e:
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [ERROR] Video health check failed: {e}")

    def _check_and_unstick_production(self, stale_minutes: float) -> None:
        """
        Detect and fix stuck production pipeline.

        If videos are stale but Director has pending work, the pipeline is stuck
        (likely on a failed video). Send skip command to continue production.
        """
        try:
            # Check Director's production_queue for pending videos
            resp = requests.get(
                f"{self.config.letta_url}/v1/agents/{self.config.director_id}",
                timeout=30
            )
            if resp.status_code != 200:
                return

            data = resp.json()
            blocks = {b['label']: b for b in data.get('memory', {}).get('blocks', [])}
            queue_block = blocks.get('production_queue', {}).get('value', '')

            # Parse queue status
            has_pending = 'PENDING_VIDEOS:' in queue_block and 'remaining' in queue_block.lower()
            has_in_progress = 'IN_PROGRESS:' in queue_block
            has_failure = 'FAILURE' in queue_block or 'FAILED' in queue_block

            if has_pending and has_failure:
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [WARN] STUCK PRODUCTION DETECTED")
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [WARN]   Videos stale for {stale_minutes:.1f} min")
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [WARN]   Director has pending videos but pipeline stuck")
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Sending skip command to Director...")

                # Send skip command
                skip_resp = requests.post(
                    f"{self.config.letta_url}/v1/agents/{self.config.director_id}/messages",
                    headers={"Content-Type": "application/json"},
                    json={"messages": [{
                        "role": "user",
                        "content": "The current video has failed permanently. Skip it, clear IN_PROGRESS, and continue with the next video in PENDING_VIDEOS. Resume autonomous production immediately."
                    }]},
                    timeout=120
                )

                success = skip_resp.status_code == 200
                self.event_bus.emit(HealingEvent(
                    source="ProductionUnstick",
                    data={
                        "action": "SKIP_FAILED_VIDEO",
                        "success": success,
                        "message": f"Skip command {'sent' if success else 'failed'}"
                    }
                ))

                if success:
                    self.state.setdefault("unstick_count", 0)
                    self.state["unstick_count"] += 1
                    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SUCCESS] Production unstuck")

        except Exception as e:
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [ERROR] Failed to unstick production: {e}")

    def _update_state_on_success(self, name: str, action: HealingAction) -> None:
        """Update state after successful healing."""
        if action == HealingAction.RESET:
            self.state.setdefault("reset_count", {})[name] = self.state.get("reset_count", {}).get(name, 0) + 1
            self.state.setdefault("summarization_failures", {})[name] = 0
        elif action == HealingAction.SUMMARIZE:
            self.state.setdefault("summarization_failures", {})[name] = 0

    def _update_state_on_failure(self, name: str, action: HealingAction) -> None:
        """Update state after failed healing."""
        if action == HealingAction.SUMMARIZE:
            failures = self.state.get("summarization_failures", {}).get(name, 0) + 1
            self.state.setdefault("summarization_failures", {})[name] = failures

    def _load_state(self) -> dict:
        """Load persistent state."""
        if self.config.state_file.exists():
            try:
                return json.loads(self.config.state_file.read_text())
            except:
                pass
        return {
            "last_video_count": 0,
            "last_video_time": None,
            "summarization_failures": {},
            "reset_count": {},
            "circuit_breakers": {}
        }

    def _save_state(self) -> None:
        """Save persistent state."""
        self.state["last_run"] = datetime.now().isoformat()
        self.config.state_file.write_text(json.dumps(self.state, indent=2))

    def _log_header(self) -> None:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] {'=' * 50}")
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] LETTA MAINTENANCE v2 (Design Patterns)")
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] {'=' * 50}")

    def _log_summary(self) -> None:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] {'-' * 40}")
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] SUMMARY")
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO]   Resets: {self.state.get('reset_count', {})}")
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO]   Failures: {self.state.get('summarization_failures', {})}")
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO]   Unsticks: {self.state.get('unstick_count', 0)}")
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO]   Videos: {self.state.get('last_video_count', 'unknown')}")
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] {'=' * 50}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Configuration
    config = Config()

    # Event bus with observers
    event_bus = EventBus()
    event_bus.subscribe(LoggingObserver())
    event_bus.subscribe(MetricsObserver())

    # Run maintenance
    runner = MaintenanceRunner(config, event_bus)
    runner.run()


if __name__ == "__main__":
    main()
