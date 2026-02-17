# VAOS Voice Bridge

Real-time voice AI bridge implementing the Talker-Reasoner cognitive architecture
from DeepMind's "Agents Thinking Fast and Slow" paper, combined with the CoALA
framework for structured agent memory.

Two cognitive systems work in concert:

- **System 1 (Talker)** -- PersonaPlex/Moshi 7B on an NVIDIA GPU. Always-on,
  full-duplex voice conversation at ~200ms latency. Fast, intuitive, but limited
  to what a 7B parameter model knows.
- **System 2 (Reasoner)** -- Letta AI agent backed by a large LLM (Claude,
  Ollama, etc.) with persistent memory, web search, and tool use. Slow and
  deliberate, activated only when System 1 fails or the user requests it.

The bridge connects a browser client to both systems via an event bus, manages
memory compression, and handles the handoff between fast intuition and slow
reasoning.

```
                         +------------------+
                         |    Browser       |
                         | Opus mic stream  |
                         | AudioWorklet out |
                         | SpeechRecognition|
                         +--------+---------+
                                  |
                            WSS (TLS)
                                  |
                    +-------------+-------------+
                    |      Voice Bridge         |
                    |      (Bun + Hono)         |
                    |                           |
                    |  EventBus (X-Talk style)  |
                    |  talker.* | trigger.*     |
                    |  reasoner.* | memory.*    |
                    +---+-------------------+---+
                        |                   |
              WSS (Moshi binary)      REST API
                        |                   |
              +---------+-------+   +-------+--------+
              |  PersonaPlex    |   |  Letta Agent   |
              |  Moshi 7B       |   |  (System 2)    |
              |  (System 1)     |   |  + Claude/LLM  |
              |  NVIDIA GPU     |   |  + web_search  |
              |  ~19GB VRAM     |   |  + memory tools|
              +-----------------+   +----------------+
```


## Prerequisites

| Component      | Requirement                                         |
|----------------|-----------------------------------------------------|
| Runtime        | [Bun](https://bun.sh) v1.0+                         |
| GPU            | NVIDIA GPU with ~20GB VRAM (RTX 3090, A5000, etc.)  |
| PersonaPlex    | NVIDIA Moshi server (`pip install moshi`)            |
| Letta          | Letta server on port 8283 (Docker or pip)            |
| Ollama         | Optional -- local LLM for fast System 2 path        |
| Supabase       | Optional -- shared state for VAOS control plane      |
| OpenSSL        | For generating self-signed TLS certs                 |
| Browser        | Chrome/Edge (SpeechRecognition requires secure context) |


## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> vaos-voice-bridge
cd vaos-voice-bridge
bun install

# 2. Configure environment
cp .env.example .env
# Edit .env with your PersonaPlex host, Letta URL, API keys, etc.

# 3. Generate self-signed TLS certs (required for Chrome SpeechRecognition)
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/key.pem -out certs/cert.pem \
  -days 365 -subj '/CN=localhost'

# 4. Start PersonaPlex on the GPU server (see PersonaPlex Setup below)

# 5. Start the bridge
bun run start          # production
bun run dev            # watch mode with auto-reload

# 6. Open in Chrome
#    https://localhost:9001
#    Accept the self-signed certificate warning.
#    Click the mic button to begin.
```


## Configuration

All configuration is validated at startup via Zod. Required variables will cause
a clear error message if missing.

| Variable                    | Required | Default                              | Description                                      |
|-----------------------------|----------|--------------------------------------|--------------------------------------------------|
| `SUPABASE_URL`              | Yes      | --                                   | Supabase project URL                             |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes      | --                                   | Supabase service role key                        |
| `ANTHROPIC_AUTH_TOKEN`      | Yes      | --                                   | LLM API key (Anthropic, Z.AI proxy, etc.)        |
| `ANTHROPIC_BASE_URL`       | No       | `https://api.z.ai/api/anthropic`     | LLM API base URL                                 |
| `PORT`                      | No       | `9001`                               | Bridge server port                               |
| `LOG_LEVEL`                 | No       | `info`                               | `debug`, `info`, `warn`, or `error`              |
| `PERSONAPLEX_HOST`          | No       | `10.0.0.3`                           | PersonaPlex server IP/hostname                   |
| `PERSONAPLEX_PORT`          | No       | `8998`                               | PersonaPlex port                                 |
| `PERSONAPLEX_WS_PATH`      | No       | `/ws`                                | PersonaPlex WebSocket path                       |
| `LETTA_BASE_URL`            | No       | `http://10.0.0.3:8283`              | Letta server URL                                 |
| `LETTA_AGENT_NAME`          | No       | `voice-reasoner`                     | Letta agent name to resolve                      |
| `REASONER_TIMEOUT_MS`       | No       | `30000`                              | System 2 response timeout (ms)                   |
| `BELIEF_UPDATE_TIMEOUT_MS`  | No       | `15000`                              | Belief state update timeout (ms)                 |
| `OLLAMA_URL`                | No       | `http://192.168.1.143:11434`         | Ollama URL for fast LLM path                     |
| `OLLAMA_MODEL`              | No       | `qwen2.5:7b`                         | Ollama model for fast path                       |
| `VOICE_PROMPT_PATH`         | No       | `/opt/moshi/voices/NATF0.pt`        | Voice timbre file on PersonaPlex server           |
| `TTS_SAMPLE_RATE`           | No       | `24000`                              | Audio sample rate (Hz)                           |
| `NODE_TLS_REJECT_UNAUTHORIZED` | No    | --                                   | Set to `0` for self-signed cert connections       |


## Architecture

### System 1: Talker (PersonaPlex / Moshi 7B)

The Talker is always on. It maintains a persistent WebSocket to PersonaPlex
using the Moshi binary protocol. Audio flows full-duplex -- the user speaks and
PersonaPlex responds simultaneously, with ~200ms end-to-end latency.

PersonaPlex accepts a `text_prompt` query parameter on WebSocket connect. This
is the bridge's primary control surface: the Reasoner updates the belief state,
the Memory module compresses it into ~150 words, and the Talker reconnects with
the new prompt. This is how System 2 steers System 1.

The Talker is intentionally "dumb" -- it responds with whatever the 7B model
can generate. When it fails (deflects, hallucinates, or the user asks for
something requiring tools), the Trigger fires and System 2 takes over.

### System 2: Reasoner (Letta + LLM)

The Reasoner is activated only when needed. It has:

- **Persistent memory** (Letta core memory blocks): belief_state,
  conversation_context, action_queue, fact_check
- **Archival memory** (Letta vector store): long-term facts, searchable
- **Tool access**: web_search, core_memory_replace, execute_mission
- **LLM backend**: Claude, Ollama, or any Anthropic-compatible API

When System 2 activates:

1. PersonaPlex audio/text is muted to the browser (suppresses hallucination)
2. Browser audio buffer is reset (cuts off mid-sentence hallucination)
3. Thinking indicator shown in the UI
4. Reasoner processes the request, updates memory blocks
5. Memory compresses updated blocks into a new text_prompt
6. PersonaPlex reconnects with the answer embedded in the prompt
7. PersonaPlex delivers the answer in its own voice

This is a single reconnect -- one audio disruption, not two.

### Trigger (Semantic Gate)

The Trigger decides when to activate System 2. It evaluates:

- **User requests**: explicit action keywords, imperative phrases
- **Frustration**: "I already told you", "are you listening"
- **Deflection**: PersonaPlex admitting it cannot do something
- **Echo detection**: PersonaPlex hallucinating search results

Social turns (greetings, "thanks", "okay") never trigger System 2.

### Event Bus

All components communicate through a priority-sorted event bus (X-Talk pattern):

| Event                   | Producer  | Consumers              |
|-------------------------|-----------|------------------------|
| `talker.text`           | Talker    | Bridge, Memory         |
| `talker.turn`           | Talker    | Trigger, Bridge        |
| `talker.audio`          | Talker    | Bridge                 |
| `talker.state`          | Talker    | Bridge                 |
| `trigger.activate`      | Trigger   | Reasoner               |
| `reasoner.thinking`     | Reasoner  | Bridge                 |
| `reasoner.interjection` | Reasoner  | Bridge                 |
| `reasoner.belief`       | Reasoner  | Memory                 |
| `memory.compressed`     | Memory    | Bridge, Talker         |
| `user.text`             | Bridge    | Trigger, Reasoner      |
| `error.occurred`        | Any       | Bridge                 |

Higher priority handlers execute first. Convention: ASR=100, Trigger=90,
Memory=80, Output=10, Error=5.

### Memory (CoALA Framework)

Four memory types following the CoALA taxonomy:

| Type       | Implementation                       | Purpose                              |
|------------|--------------------------------------|--------------------------------------|
| Working    | 5 Letta core memory blocks           | Current session state                |
| Episodic   | Letta recall memory                  | Conversation history (auto-managed)  |
| Semantic   | Letta archival memory (vector store) | Long-term facts, searchable          |
| Procedural | Agent tool definitions               | web_search, memory tools, missions   |

The Memory module compresses working memory into PersonaPlex's ~200-token
text_prompt budget. Priority order: persona > conversation context > user model
> fact corrections > System 2 answers > action queue.


## PersonaPlex Setup

PersonaPlex runs NVIDIA's Moshi 7B model as a WebSocket server. It requires an
NVIDIA GPU with ~19GB VRAM.

```bash
# On the GPU server (e.g., Proxmox LXC with GPU passthrough)
pip install moshi

# Start the Moshi server with TLS (required for WSS from the bridge)
moshi-server \
  --host 0.0.0.0 \
  --port 8998 \
  --cert /path/to/cert.pem \
  --key /path/to/key.pem
```

PersonaPlex only allows one WebSocket session at a time (single-session lock).
The bridge handles this automatically -- if a second browser tab connects, it
adopts into the existing session rather than creating a duplicate connection
that would deadlock.

### Moshi Binary Protocol

All PersonaPlex messages use a binary frame format with a 1-byte type prefix:

| Byte | Type      | Direction       | Description                        |
|------|-----------|------------------|------------------------------------|
| 0x00 | Handshake | Server to client | Session established, ready for audio |
| 0x01 | Audio     | Bidirectional   | Opus-encoded audio chunks          |
| 0x02 | Text      | Server to client | UTF-8 text tokens (model output)   |
| 0x03 | Control   | Reserved         | Future use                         |
| 0x04 | Metadata  | Reserved         | Future use                         |

The client sends audio frames prefixed with 0x01. The server sends raw
Opus packets (not Ogg containers). Sending Ogg pages crashes PersonaPlex
with "ValueError: sending on a closed channel".

Turn detection: 350ms of silence after the last text token marks a turn
boundary.


## Letta Setup

Letta provides the persistent memory and tool-calling backend for System 2.

```bash
# Docker (recommended)
docker run -d \
  --name letta-server \
  -p 8283:8283 \
  -v letta-data:/root/.letta \
  letta/letta:latest

# Or pip install
pip install letta
letta server --port 8283
```

The bridge auto-resolves the agent by name (`LETTA_AGENT_NAME`). If the agent
does not exist, it will be created on first use.


## Echo Suppression

Full-duplex voice creates an echo problem: Chrome's SpeechRecognition picks up
PersonaPlex's audio output from the speakers and transcribes it as "user speech",
creating false triggers.

The bridge uses three layers of echo suppression:

1. **Browser echo gate** -- SpeechRecognition results are suppressed for 8
   seconds after PersonaPlex produces speech (text tokens or large audio frames
   >200 bytes). This is the first line of defense, running client-side.

2. **Streaming text match** -- The server accumulates PersonaPlex's streaming
   text tokens. When a speech transcript arrives, it is compared against the
   streaming buffer using word overlap (40% threshold, 2+ matching words).
   Matches are suppressed.

3. **Turn dedup** -- The server maintains a window of completed PersonaPlex
   turns (last 30 seconds). Incoming speech is compared against this history
   using the same word-overlap algorithm. This catches delayed transcriptions
   that arrive after the streaming buffer has been trimmed.

A length heuristic is also applied: transcripts over 120 characters during
active PersonaPlex conversation are almost certainly echo (mixed user + speaker
audio in one transcript) and are suppressed outright.

Typed text input (`text:` prefix) is never echo-suppressed. Only speech input
(`speech:` prefix) passes through the echo filters.


## Text Drought Auto-Reconnect

PersonaPlex occasionally enters a state where it sends audio frames but no text
tokens -- the model produces silence or noise but no meaningful output. The
Talker detects this after 45 seconds of text silence and automatically
reconnects with the current text_prompt, recovering the session without user
intervention.


## API Endpoints

| Method | Path                              | Description                          |
|--------|-----------------------------------|--------------------------------------|
| GET    | `/`                               | Embedded browser client              |
| GET    | `/api/v1/health`                  | Server health and session count      |
| GET    | `/api/v1/sessions`                | List active voice sessions           |
| GET    | `/api/v1/sessions/:id/memory`     | Get memory blocks for a session      |
| POST   | `/api/v1/sessions/:id/inject`     | Inject context into a session        |
| POST   | `/api/v1/speak`                   | TTS playback to active session       |


## Browser Client

The client is embedded directly in `server.ts` (no separate build step). It
provides:

- **Opus mic streaming** at 24kHz via opus-recorder (loaded from CDN)
- **Opus audio playback** via an AudioWorklet (Moshi processor) with jitter
  buffering
- **Chrome SpeechRecognition** for parallel text transcription of user speech
- **Memory panel** showing live CoALA block state (phase, goals, topic, etc.)
- **Text input** for typed messages (bypasses echo suppression)

Messages from the browser use a prefix system:

| Prefix    | Source             | Echo suppressed? |
|-----------|--------------------|------------------|
| `speech:` | SpeechRecognition  | Yes (3 layers)   |
| `text:`   | Typed input        | No               |
| (binary)  | Opus mic audio     | N/A (audio path) |


## Project Structure

```
src/
  server.ts          Main bridge server + embedded client HTML
  talker.ts          PersonaPlex WebSocket client (System 1)
  reasoner.ts        Letta System 2 integration
  memory.ts          CoALA memory manager + compression
  trigger.ts         Semantic gate (System 1 -> System 2)
  events.ts          Typed event bus (X-Talk pattern)
  belief.ts          Belief state schema and prompt generation
  coordinator.ts     Component coordination
  tts.ts             Text-to-speech utilities
  lib/
    config.ts        Zod-validated environment configuration
    db.ts            Supabase client
    logger.ts        Pino structured logging
  executors/
    voice.ts         VAOS control plane executor integration
certs/
  cert.pem           Self-signed TLS certificate
  key.pem            TLS private key
```


## License

MIT
