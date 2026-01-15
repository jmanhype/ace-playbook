#!/usr/bin/env python3
"""
Letta MAS Maintenance Script - Bulletproofing Layer

Monitors and auto-heals the Letta MAS video production system:
1. Message count monitoring with auto-summarization
2. Auto-reset if summarization fails
3. Video generation health checks
4. Router health checks
5. Comprehensive logging

Run via cron every 5 minutes:
*/5 * * * * /home/straughter/letta_maintenance.py >> /tmp/letta_maintenance.log 2>&1
"""

import requests
import subprocess
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# === CONFIGURATION ===
LETTA_URL = "http://localhost:8283"
ROUTER_URL = "http://localhost:3000"
COMFYUI_URL = "http://localhost:8188"

# Agent IDs
AGENTS = {
    "director": "agent-22069f59-7a79-4890-bf4f-1f2a69696267",
    "writer": "agent-e565b3e8-4a59-440a-89ab-6c279d61cfb0",
    "cameraman": "agent-f939736a-46fc-4115-a584-0a8cf896212a",
}

# Thresholds
MESSAGE_SUMMARIZE_THRESHOLD = 300  # Summarize when messages exceed this
MESSAGE_RESET_THRESHOLD = 500      # Force reset if summarization fails above this
MESSAGE_CRITICAL_THRESHOLD = 600   # Emergency reset, no questions asked
VIDEO_STALE_MINUTES = 60           # Alert if no new video in this time

# Paths
VIDEO_DIR = Path("/home/straughter/ComfyUI/output/video")
STATE_FILE = Path("/tmp/letta_maintenance_state.json")
LOG_FILE = Path("/tmp/letta_maintenance.log")

# === LOGGING ===
def log(level: str, message: str):
    """Log with timestamp and level."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level.upper()}] {message}")

def log_info(msg): log("info", msg)
def log_warn(msg): log("warn", msg)
def log_error(msg): log("error", msg)
def log_success(msg): log("success", msg)

# === STATE MANAGEMENT ===
def load_state() -> dict:
    """Load persistent state from file."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except:
            pass
    return {
        "last_video_count": 0,
        "last_video_time": None,
        "summarization_failures": {},
        "reset_count": {},
        "last_run": None
    }

def save_state(state: dict):
    """Save state to file."""
    state["last_run"] = datetime.now().isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))

# === HEALTH CHECKS ===
def check_letta_health() -> bool:
    """Check if Letta server is responding."""
    try:
        resp = requests.get(f"{LETTA_URL}/v1/agents/", timeout=10)
        return resp.status_code == 200
    except Exception as e:
        log_error(f"Letta health check failed: {e}")
        return False

def check_router_health() -> bool:
    """Check if router is responding."""
    try:
        resp = requests.get(f"{ROUTER_URL}/health", timeout=5)
        return resp.status_code in [200, 404]  # 404 means running but no health endpoint
    except Exception as e:
        log_error(f"Router health check failed: {e}")
        return False

def check_comfyui_health() -> bool:
    """Check if ComfyUI is responding."""
    try:
        resp = requests.get(f"{COMFYUI_URL}/queue", timeout=5)
        return resp.status_code == 200
    except Exception as e:
        log_error(f"ComfyUI health check failed: {e}")
        return False

# === AGENT MONITORING ===
def get_agent_message_count(agent_id: str) -> int:
    """Get message count for an agent."""
    try:
        resp = requests.get(f"{LETTA_URL}/v1/agents/{agent_id}", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return len(data.get("message_ids", []))
    except Exception as e:
        log_error(f"Failed to get message count for {agent_id}: {e}")
    return -1

def summarize_agent(agent_id: str) -> bool:
    """Trigger summarization for an agent."""
    try:
        log_info(f"Triggering summarization for {agent_id}")
        resp = requests.post(
            f"{LETTA_URL}/v1/agents/{agent_id}/summarize",
            timeout=120
        )
        if resp.status_code == 200:
            log_success(f"Summarization successful for {agent_id}")
            return True
        else:
            log_error(f"Summarization failed for {agent_id}: {resp.status_code} - {resp.text[:200]}")
            return False
    except Exception as e:
        log_error(f"Summarization exception for {agent_id}: {e}")
        return False

def reset_agent_messages(agent_id: str) -> bool:
    """Reset agent messages (emergency measure)."""
    try:
        log_warn(f"RESETTING messages for {agent_id}")
        resp = requests.patch(
            f"{LETTA_URL}/v1/agents/{agent_id}/reset-messages",
            headers={"Content-Type": "application/json"},
            json={},  # Empty body required by API
            timeout=30
        )
        if resp.status_code == 200:
            log_success(f"Message reset successful for {agent_id}")
            return True
        else:
            log_error(f"Message reset failed for {agent_id}: {resp.status_code}")
            return False
    except Exception as e:
        log_error(f"Message reset exception for {agent_id}: {e}")
        return False

# === VIDEO MONITORING ===
def get_video_count() -> int:
    """Get current video count."""
    try:
        videos = list(VIDEO_DIR.glob("*.mp4"))
        return len(videos)
    except Exception as e:
        log_error(f"Failed to count videos: {e}")
        return -1

def get_latest_video_time() -> datetime | None:
    """Get modification time of latest video."""
    try:
        videos = sorted(VIDEO_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if videos:
            return datetime.fromtimestamp(videos[0].stat().st_mtime)
    except Exception as e:
        log_error(f"Failed to get latest video time: {e}")
    return None

def check_comfyui_queue() -> dict:
    """Check ComfyUI queue status."""
    try:
        resp = requests.get(f"{COMFYUI_URL}/queue", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "running": len(data.get("queue_running", [])),
                "pending": len(data.get("queue_pending", []))
            }
    except:
        pass
    return {"running": 0, "pending": 0}

# === MAIN MAINTENANCE LOGIC ===
def maintain_agent(name: str, agent_id: str, state: dict) -> dict:
    """Run maintenance for a single agent."""
    log_info(f"Checking {name} ({agent_id})")

    msg_count = get_agent_message_count(agent_id)
    if msg_count < 0:
        log_error(f"Could not get message count for {name}")
        return state

    log_info(f"{name} has {msg_count} messages")

    # Critical threshold - immediate reset
    if msg_count >= MESSAGE_CRITICAL_THRESHOLD:
        log_warn(f"{name} at CRITICAL level ({msg_count} >= {MESSAGE_CRITICAL_THRESHOLD})")
        if reset_agent_messages(agent_id):
            state.setdefault("reset_count", {})[name] = state.get("reset_count", {}).get(name, 0) + 1
            state.setdefault("summarization_failures", {})[name] = 0
        return state

    # Reset threshold - summarization has failed, force reset
    if msg_count >= MESSAGE_RESET_THRESHOLD:
        failures = state.get("summarization_failures", {}).get(name, 0)
        if failures >= 2:
            log_warn(f"{name} at reset threshold with {failures} summarization failures - forcing reset")
            if reset_agent_messages(agent_id):
                state.setdefault("reset_count", {})[name] = state.get("reset_count", {}).get(name, 0) + 1
                state.setdefault("summarization_failures", {})[name] = 0
            return state

    # Summarize threshold - try to summarize
    if msg_count >= MESSAGE_SUMMARIZE_THRESHOLD:
        log_info(f"{name} above summarize threshold ({msg_count} >= {MESSAGE_SUMMARIZE_THRESHOLD})")
        if summarize_agent(agent_id):
            state.setdefault("summarization_failures", {})[name] = 0
        else:
            failures = state.get("summarization_failures", {}).get(name, 0) + 1
            state.setdefault("summarization_failures", {})[name] = failures
            log_warn(f"{name} summarization failure count: {failures}")

    return state

def check_video_health(state: dict) -> dict:
    """Check video generation health."""
    video_count = get_video_count()
    latest_time = get_latest_video_time()
    queue = check_comfyui_queue()

    log_info(f"Videos: {video_count}, Queue: {queue['running']} running / {queue['pending']} pending")

    if latest_time:
        age_minutes = (datetime.now() - latest_time).total_seconds() / 60
        log_info(f"Latest video: {age_minutes:.1f} minutes ago")

        # Check for stale production
        if age_minutes > VIDEO_STALE_MINUTES and queue["running"] == 0 and queue["pending"] == 0:
            log_warn(f"VIDEO GENERATION STALE - No new video in {age_minutes:.1f} minutes and queue empty!")
            # Could trigger alert here (email, slack, etc.)

    state["last_video_count"] = video_count
    state["last_video_time"] = latest_time.isoformat() if latest_time else None

    return state

def run_maintenance():
    """Main maintenance routine."""
    log_info("=" * 60)
    log_info("LETTA MAINTENANCE RUN STARTING")
    log_info("=" * 60)

    state = load_state()

    # Health checks
    letta_ok = check_letta_health()
    router_ok = check_router_health()
    comfyui_ok = check_comfyui_health()

    log_info(f"Health: Letta={'OK' if letta_ok else 'FAIL'}, Router={'OK' if router_ok else 'FAIL'}, ComfyUI={'OK' if comfyui_ok else 'FAIL'}")

    if not letta_ok:
        log_error("Letta server not responding - cannot proceed")
        save_state(state)
        return

    # Maintain each agent
    for name, agent_id in AGENTS.items():
        state = maintain_agent(name, agent_id, state)

    # Check video health
    if comfyui_ok:
        state = check_video_health(state)

    # Summary
    log_info("-" * 40)
    log_info("MAINTENANCE SUMMARY")
    log_info(f"  Resets performed: {state.get('reset_count', {})}")
    log_info(f"  Summarization failures: {state.get('summarization_failures', {})}")
    log_info(f"  Video count: {state.get('last_video_count', 'unknown')}")
    log_info("=" * 60)

    save_state(state)

if __name__ == "__main__":
    run_maintenance()
