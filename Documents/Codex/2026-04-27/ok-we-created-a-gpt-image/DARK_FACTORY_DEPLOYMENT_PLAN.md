# Dark Factory Deployment Plan

Status: APPROVED  
Date: 2026-05-13  
Author: BatmanOsama + Kiro

---

## Architecture Overview

```
ZIMABOARD (services + persistence)          3090 (compute + execution)
─────────────────────────────────           ──────────────────────────
LXC 100: Coolify                            Paperclip AI (management)
LXC 102: WireGuard                          Paseo (coding orchestration)
LXC 106: MCP Gateway                       GitLab (source control)
LXC 108: Plex                               Hermes (agent runtime)
LXC 110: Media Automation (Arr stack)       Claude/Codex/OpenCode CLIs
LXC 111: Invoice Ninja                      Skills Library
LXC NEW: InsForge (agent backend)           ACE-Step, ComfyUI, Demucs, Whisper
                                            Proactive Watcher (code/agent events)
Proactive Watcher (media events)
```

---

## Deployment Order

### Step 1: GitLab (3090)
- Self-hosted GitLab CE on the 3090
- Local repos for all agent-generated code
- CI runners for testing agent output
- `glab` CLI for agent interaction
- No external telemetry — code stays local
- Storage: `/mnt/bulk/gitlab/`

### Step 2: Paseo + Litter (3090)
- Install Paseo from github.com/getpaseo/paseo
- Headless daemon mode
- Connect to existing Claude/Codex/OpenCode CLIs already in `~/.npm-global/bin/`
- Configure Paperclip adapter to route coding issues through Paseo
- Key features: /paseo-handoff, /paseo-committee, /paseo-loop
- Cross-device access (phone/tablet via QR)
- Install Litter (github.com/dnakov/litter / kittylitter.app)
- Native iOS + Android client for Codex, Claude Code, OpenCode
- Mobile interface to control agents from phone
- Paseo = headless orchestration daemon, Litter = mobile UI to interact with agents directly

### Step 3: InsForge (Zimaboard, new LXC)
- Deploy via Docker in a new LXC container
- Postgres backend for agent state persistence
- Auth, storage, serverless functions
- MCP server exposed to agents on 3090
- Each Paperclip company gets its own InsForge project
- Agents can self-provision databases, deploy endpoints, persist state

### Step 4: Skills Library (3090)
- Location: `/mnt/bulk/home/straughter/skills/`
- Formalize all proven procedures as skill tuples (M, R, C)
- `skills_index.json` for retrieval
- Shared across all 8 Paperclip companies
- Update all agent AGENTS.md to reference the skill library

### Step 5: Proactive Watchers
- 3090 watcher: monitors git repos, agent outputs, audio factory, sgflix_runs
- Zimaboard watcher: monitors Radarr/Sonarr webhooks, download completions
- Both create Paperclip issues via API when events match trigger conditions

### Step 6: Wire It All Together
- Paperclip issue → Paseo for code tasks (via new adapter type)
- Paperclip issue → Hermes for non-code tasks (existing z-ai adapter)
- Agents write code → GitLab (local)
- Agents need backend → InsForge (MCP)
- Agents need skills → Skills Library (local filesystem)
- Events happen → Watchers create issues → agents execute

---

## Connection Flow

```
YOU (Mac/Phone)
  │
  ├── Paseo (remote control coding agents from anywhere)
  │     │
  │     ├── Claude Code (coding worker)
  │     ├── Codex (coding worker)
  │     └── OpenCode (coding worker)
  │           │
  │           └── writes code to → GitLab (local, 3090)
  │
  ├── Paperclip (management plane, 3090)
  │     │
  │     ├── Issues → triggers Paseo for code tasks
  │     ├── Issues → triggers Hermes for non-code tasks
  │     └── Heartbeats → monitors all agents
  │
  ├── InsForge (agent backend, Zimaboard)
  │     │
  │     ├── Agents self-provision databases
  │     ├── Agents deploy functions
  │     └── MCP access to everything
  │
  └── Zimaboard + 3090 (infrastructure)
        │
        ├── Media stack (Arr suite)
        ├── GPU compute (generation, inference)
        ├── Plex (serving)
        └── MCP Gateway (API bridge)
```

---

## Role Separation

| Layer | Tool | Purpose |
|-------|------|---------|
| Management | Paperclip AI | What needs doing, who does it, approvals, budgets |
| Code Execution | Paseo | How coding agents collaborate (handoff, committee, loop) |
| Mobile Control | Litter | Native iOS/Android client for agent interaction from phone |
| Source Control | GitLab | Where code lives, CI/CD, no external telemetry |
| Agent Backend | InsForge | Agents' self-service infra (DB, auth, storage, functions) |
| Skills | Skills Library | Reusable procedures, composable workflows |
| Proactive | Watchers | Event-driven issue creation, eventually developer model |
| Compute | 3090 GPU | Inference, generation, rendering |
| Services | Zimaboard | Persistent services (media, backend, networking) |

---

## What's Already Working

- [x] Paperclip AI (8 companies, agents, issues, heartbeats)
- [x] Hermes agent runtime (z-ai adapter)
- [x] Media Automation (Radarr/Sonarr/SABnzbd/qBit)
- [x] Media Pipeline API (audio sourcing + GPU bridge)
- [x] MCP Gateway (exposes Arr stack to agents)
- [x] Audio Factory (ACE-Step, Demucs, Whisper, proxy critic)
- [x] ComfyUI + WAN 2.1 (video generation)
- [x] Tailscale mesh (all machines connected)
- [x] Claude/Codex CLIs (installed + authenticated on 3090)

## What Needs Deploying

- [ ] GitLab CE (3090)
- [ ] Paseo (3090)
- [ ] InsForge (Zimaboard, new LXC)
- [ ] Skills Library structure (3090)
- [ ] Proactive Watchers (both machines)
- [ ] Paperclip → Paseo adapter wiring
- [ ] Cross-company skill sharing config

---

## Notes

- Paseo does NOT replace Paperclip. Paperclip = management. Paseo = coding execution.
- InsForge does NOT replace the media pipeline. InsForge = agent backend. Media pipeline = audio/video sourcing.
- GitLab does NOT replace GitHub. GitLab = local/private agent code. GitHub = public/collaborative repos.
- The Skills Library concept comes from "Quantizing AI into Executable Skills as Math Operators" (Chinese University of Hong Kong, May 8 2026).
- The Proactive Layer concept comes from "Gigantic Coding Needs Proactivity" (Google Labs, May 7 2026).
