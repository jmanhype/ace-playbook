#!/usr/bin/env python3
"""
Letta MAS Maintenance v3 - Simple & Effective

Does exactly what's needed:
1. Check agent message counts
2. Summarize at 300+, reset at 500+
3. Detect stuck production, auto-skip
4. Log everything

Run via cron every 5 minutes:
*/5 * * * * /usr/bin/python3 /home/straughter/letta_maintenance_v3.py >> /tmp/letta_maintenance.log 2>&1
"""

import json
import requests
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    letta_url: str = "http://localhost:8283"
    comfyui_url: str = "http://localhost:8188"

    # Thresholds
    warn_threshold: int = 300
    critical_threshold: int = 500
    emergency_threshold: int = 600
    max_summarize_failures: int = 2

    # Video monitoring
    video_stuck_minutes: int = 15
    video_dir: Path = field(default_factory=lambda: Path("/home/straughter/ComfyUI/output/video"))

    # State
    state_file: Path = field(default_factory=lambda: Path("/tmp/letta_maintenance_state.json"))


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def log(level: str, msg: str):
    """Simple structured logging."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{level}] {msg}")


def load_state(config: Config) -> dict:
    """Load persistent state."""
    if config.state_file.exists():
        try:
            return json.loads(config.state_file.read_text())
        except Exception:
            pass
    return {"summarize_failures": {}, "reset_count": {}, "unstick_count": 0}


def save_state(config: Config, state: dict):
    """Save persistent state."""
    state["last_run"] = datetime.now().isoformat()
    config.state_file.write_text(json.dumps(state, indent=2))


def get_agents(config: Config) -> dict[str, str]:
    """Fetch agents from API - no hardcoded IDs."""
    try:
        resp = requests.get(f"{config.letta_url}/v1/agents/", timeout=10)
        if resp.ok:
            # Filter to our MAS agents by name pattern
            agents = {}
            for a in resp.json():
                name = a.get("name", "").lower()
                if name in ("director", "writer", "cameraman"):
                    agents[name] = a["id"]
            return agents
    except requests.RequestException as e:
        log("ERROR", f"Failed to fetch agents: {e}")
    return {}


def get_agent_message_count(agent_id: str, config: Config) -> Optional[int]:
    """Get message count for an agent."""
    try:
        resp = requests.get(f"{config.letta_url}/v1/agents/{agent_id}", timeout=30)
        if resp.ok:
            return len(resp.json().get("message_ids", []))
    except requests.RequestException as e:
        log("ERROR", f"Failed to get agent {agent_id}: {e}")
    return None


def summarize_agent(agent_id: str, config: Config) -> bool:
    """Trigger summarization for an agent."""
    try:
        resp = requests.post(
            f"{config.letta_url}/v1/agents/{agent_id}/summarize",
            timeout=120
        )
        return resp.ok
    except requests.RequestException as e:
        log("ERROR", f"Summarize failed: {e}")
        return False


def reset_agent(agent_id: str, config: Config) -> bool:
    """Reset an agent's messages (nuclear option)."""
    try:
        resp = requests.patch(
            f"{config.letta_url}/v1/agents/{agent_id}/reset-messages",
            headers={"Content-Type": "application/json"},
            json={},
            timeout=30
        )
        return resp.ok
    except requests.RequestException as e:
        log("ERROR", f"Reset failed: {e}")
        return False


def check_services(config: Config) -> dict[str, bool]:
    """Quick health check of dependent services."""
    health = {}

    # Letta
    try:
        resp = requests.get(f"{config.letta_url}/v1/agents/", timeout=5)
        health["letta"] = resp.ok
    except Exception:
        health["letta"] = False

    # ComfyUI
    try:
        resp = requests.get(f"{config.comfyui_url}/queue", timeout=5)
        health["comfyui"] = resp.ok
    except Exception:
        health["comfyui"] = False

    return health


def get_video_status(config: Config) -> tuple[int, Optional[float], dict]:
    """Get video count, age of latest, and queue status."""
    videos = list(config.video_dir.glob("*.mp4"))
    count = len(videos)

    age_minutes = None
    if videos:
        latest = max(videos, key=lambda p: p.stat().st_mtime)
        age_minutes = (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).total_seconds() / 60

    queue = {"running": 0, "pending": 0}
    try:
        resp = requests.get(f"{config.comfyui_url}/queue", timeout=5)
        if resp.ok:
            data = resp.json()
            queue["running"] = len(data.get("queue_running", []))
            queue["pending"] = len(data.get("queue_pending", []))
    except Exception:
        pass

    return count, age_minutes, queue


def check_stuck_production(config: Config, agents: dict) -> bool:
    """Detect and fix stuck production pipeline."""
    director_id = agents.get("director")
    if not director_id:
        return False

    try:
        resp = requests.get(f"{config.letta_url}/v1/agents/{director_id}", timeout=30)
        if not resp.ok:
            return False

        blocks = {b["label"]: b for b in resp.json().get("memory", {}).get("blocks", [])}
        queue_block = blocks.get("production_queue", {}).get("value", "")

        has_pending = "PENDING_VIDEOS:" in queue_block and "remaining" in queue_block.lower()
        has_failure = "FAILURE" in queue_block or "FAILED" in queue_block

        if has_pending and has_failure:
            log("WARN", "STUCK PRODUCTION - sending skip command")
            skip_resp = requests.post(
                f"{config.letta_url}/v1/agents/{director_id}/messages",
                headers={"Content-Type": "application/json"},
                json={"messages": [{
                    "role": "user",
                    "content": "The current video failed. Skip it and continue with the next video in PENDING_VIDEOS."
                }]},
                timeout=120
            )
            return skip_resp.ok
    except requests.RequestException as e:
        log("ERROR", f"Stuck check failed: {e}")

    return False


# =============================================================================
# MAIN
# =============================================================================
def main():
    config = Config()
    state = load_state(config)

    log("INFO", "=" * 50)
    log("INFO", "LETTA MAINTENANCE v3")
    log("INFO", "=" * 50)

    # Health check
    health = check_services(config)
    log("INFO", f"Services: Letta={'OK' if health['letta'] else 'FAIL'}, ComfyUI={'OK' if health['comfyui'] else 'FAIL'}")

    if not health["letta"]:
        log("ERROR", "Letta unavailable - aborting")
        save_state(config, state)
        return

    # Get agents dynamically
    agents = get_agents(config)
    if not agents:
        log("WARN", "No MAS agents found")
        save_state(config, state)
        return

    # Check each agent
    for name, agent_id in agents.items():
        count = get_agent_message_count(agent_id, config)
        if count is None:
            log("ERROR", f"{name}: failed to get status")
            continue

        # Determine action
        action = None
        if count >= config.emergency_threshold:
            action = "reset"
        elif count >= config.critical_threshold:
            action = "reset"
        elif count >= config.warn_threshold:
            failures = state["summarize_failures"].get(name, 0)
            action = "reset" if failures >= config.max_summarize_failures else "summarize"

        # Execute
        if action:
            log("WARN", f"{name}: {count} msgs -> {action}")

            if action == "summarize":
                success = summarize_agent(agent_id, config)
                if success:
                    state["summarize_failures"][name] = 0
                    log("SUCCESS", f"{name} summarized")
                else:
                    state["summarize_failures"][name] = state["summarize_failures"].get(name, 0) + 1
                    log("ERROR", f"{name} summarize failed ({state['summarize_failures'][name]} failures)")

            elif action == "reset":
                success = reset_agent(agent_id, config)
                if success:
                    state["reset_count"][name] = state["reset_count"].get(name, 0) + 1
                    state["summarize_failures"][name] = 0
                    log("SUCCESS", f"{name} reset")
                else:
                    log("ERROR", f"{name} reset failed")
        else:
            log("INFO", f"{name}: {count} msgs (healthy)")

    # Video health
    video_count, age_minutes, queue = get_video_status(config)
    log("INFO", f"Videos: {video_count}, Queue: {queue['running']} running / {queue['pending']} pending")

    if age_minutes:
        log("INFO", f"Latest video: {age_minutes:.1f} min ago")

        # Stuck detection
        if age_minutes > config.video_stuck_minutes and queue["running"] == 0 and queue["pending"] == 0:
            if check_stuck_production(config, agents):
                state["unstick_count"] = state.get("unstick_count", 0) + 1
                log("SUCCESS", "Production unstuck")

    # Summary
    log("INFO", "-" * 40)
    log("INFO", f"Resets: {state.get('reset_count', {})}")
    log("INFO", f"Unsticks: {state.get('unstick_count', 0)}")
    log("INFO", f"Videos: {video_count}")
    log("INFO", "=" * 50)

    save_state(config, state)


if __name__ == "__main__":
    main()
