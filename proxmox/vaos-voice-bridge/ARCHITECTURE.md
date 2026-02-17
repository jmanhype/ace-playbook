# Voice Bridge: A Dual-Process Architecture for Tool-Augmented Voice AI

**Abstract.** Current real-time voice models (Moshi 7B, GPT-4o Realtime) trade off latency for capability: they speak fluently but cannot use tools, search the web, or maintain persistent memory. Adding these capabilities via "side-tool" mechanisms (Unmute #77) introduces three problems: (1) information flows only to the speech model, never back; (2) the speech model is unaware which tools exist; (3) filler audio creates a double disruption when the tool returns. We present Voice Bridge, an open-source implementation of the Talker-Reasoner dual-process architecture that solves all three problems by treating WebSocket reconnect as a context injection channel. We describe the architecture, the belief state compression mechanism, a three-layer echo suppression system for full-duplex voice, and an experimental integration of Voxtral Mini 3B as a parallel audio listener for speech-to-tool-call classification.

---

## 1. Introduction

The release of Kyutai's Moshi (Defossez et al., 2024) and OpenAI's GPT-4o Realtime API made sub-200ms full-duplex voice conversation possible. For the first time, users could interrupt a model mid-sentence and receive overlapping speech -- a fundamental shift from the turn-based ASR-LLM-TTS pipeline.

But these models are constrained by what fits in a single forward pass. A 7B-parameter speech model cannot reliably search the web, execute code, or remember facts across sessions. The community response has been "side-tool" architectures: a separate process listens to the conversation, detects when a tool is needed, executes it, and injects the result back into the speech stream.

The Unmute project (Kyutai, 2024) proposed this approach in their issue tracker (#77), and the discussion surfaced three fundamental limitations:

1. **Write-only information flow.** Tools can receive context from the speech model's transcript, but results cannot flow back into the model's next response. The speech model continues generating from its original context, unaware of the tool's output.

2. **Tool-unaware speech model.** The speech model has no representation of available tools. It cannot negotiate with the user about which tools to invoke, explain what it's doing, or ask clarifying questions before execution.

3. **Double disruption.** When a tool is invoked, the speech model must produce filler audio ("Let me look that up...") while waiting. When the result arrives, the filler must be interrupted and the result injected -- creating two audio discontinuities that break conversational flow.

Voice Bridge addresses all three limitations by implementing the Talker-Reasoner dual-process pattern (Google DeepMind, 2024) with a belief state compression mechanism that turns the speech model's WebSocket reconnect into a unidirectional context injection channel.

## 2. Related Work

### 2.1 Dual-Process Theory and the Talker-Reasoner Pattern

Kahneman's dual-process theory (2011) distinguishes between System 1 (fast, intuitive, automatic) and System 2 (slow, deliberate, analytical). Google DeepMind's "Agents Thinking Fast and Slow" paper (2024) applied this framework to conversational AI, proposing a **Talker** (System 1) that handles real-time conversation and a **Reasoner** (System 2) that performs slow, tool-augmented deliberation. The Reasoner updates a shared belief state that the Talker consults on its next turn.

The original paper describes the architecture abstractly. Voice Bridge is, to our knowledge, the first open-source implementation that operates on a real-time speech model rather than a text-based chatbot.

### 2.2 CoALA: Cognitive Architectures for Language Agents

The CoALA framework (Sumers et al., 2023) proposes a standardized taxonomy of agent memory: **working memory** (current session state), **episodic memory** (conversation history), **semantic memory** (long-term facts, vector-indexed), and **procedural memory** (tool definitions and action schemas).

Voice Bridge implements all four CoALA memory types using Letta AI as the backing store:

| CoALA Type  | Implementation                        |
|-------------|---------------------------------------|
| Working     | 5 Letta core memory blocks            |
| Episodic    | Letta recall memory (auto-managed)    |
| Semantic    | Letta archival memory (vector store)  |
| Procedural  | Agent tool definitions                |

The key challenge is **compression**: the speech model accepts at most ~200 tokens of context via its `text_prompt` parameter, so the entire CoALA working memory must be compressed to fit.

### 2.3 Moshi / PersonaPlex and the "Side-Tool" Problem

Moshi (Defossez et al., 2024) is a 7B-parameter speech-text foundation model that generates audio and text tokens simultaneously from a single transformer. It supports a `text_prompt` parameter that steers the model's persona and context at session start.

NVIDIA's PersonaPlex extends Moshi with voice timbre control and exposes it as a WebSocket server. The WebSocket accepts Opus audio and returns interleaved Opus audio + UTF-8 text tokens in a binary frame format.

The "side-tool" approach from Unmute #77 proposes adding tool use by having an external process listen to the conversation transcript and inject tool results. This is architecturally simple but creates the three limitations described above.

### 2.4 Other Approaches

**GPT-4o Realtime API** provides built-in function calling but is proprietary, cloud-only, and does not support self-hosted speech models. **LiveKit Agents** and **Pipecat** provide pipeline frameworks but use the traditional ASR-LLM-TTS chain rather than a native speech model, sacrificing the low latency and natural interruption handling of models like Moshi.

## 3. Architecture

### 3.1 Overview

Voice Bridge connects three subsystems through a session-scoped event bus:

```
Browser (Opus mic + AudioWorklet + SpeechRecognition)
    |
    WSS (TLS)
    |
Voice Bridge (Bun + Hono + EventBus)
    |                          |                    |
    WSS (Moshi binary)        REST (Letta)         REST (Voxtral)
    |                          |                    |
PersonaPlex / Moshi 7B     Letta Agent          Voxtral Mini 3B
(System 1 / Talker)        (System 2 / Reasoner) (Parallel Listener)
~19GB VRAM, ~200ms         Claude/Ollama + tools  ~4GB VRAM, audio->intent
```

The event bus follows the X-Talk priority-sorted pattern: handlers are registered with numeric priorities, and higher-priority handlers execute first within each event type. This provides deterministic ordering without explicit dependency graphs.

### 3.2 System 1: Talker (PersonaPlex)

The Talker maintains a persistent WebSocket connection to PersonaPlex. Audio flows full-duplex: the user's Opus-encoded microphone stream is forwarded to PersonaPlex, and PersonaPlex's Opus audio responses are forwarded to the browser's AudioWorklet for playback. Text tokens are emitted as `talker.text` events for the Trigger and Memory components.

PersonaPlex only allows one WebSocket session at a time. The bridge handles multi-tab scenarios by adopting subsequent connections into the existing session.

The critical design insight is that PersonaPlex's `text_prompt` parameter -- intended for persona configuration -- can be repurposed as a **context injection channel**. By disconnecting and reconnecting with an updated `text_prompt`, the bridge can steer PersonaPlex's next response without modifying the Moshi protocol.

### 3.3 System 2: Reasoner (Letta + LLM)

The Reasoner is activated only when the Trigger fires. It uses the Letta AI framework, which provides:

- **Persistent memory blocks**: The Reasoner's state survives across sessions. Memory blocks include `belief_state` (user model), `conversation_context` (topic and summary), `fact_check` (corrections to prevent hallucination loops), and `action_queue` (in-flight tasks).
- **Tool access**: `web_search`, `core_memory_replace`, `execute_mission`, and custom tools.
- **Archival memory**: A vector-indexed long-term store that the Reasoner can search and insert into.
- **LLM backends**: Claude (via Anthropic API), Ollama (local), or any Anthropic-compatible proxy.

When System 2 activates:

1. PersonaPlex audio is muted to the browser (suppresses in-flight hallucination)
2. The browser's AudioWorklet buffer is reset (cuts off mid-sentence audio)
3. A thinking indicator is shown in the UI
4. The Reasoner processes the request and updates Letta memory blocks
5. The Memory module compresses all blocks into ~150 words
6. PersonaPlex reconnects with the compressed prompt as `text_prompt`
7. PersonaPlex delivers the answer in its own voice

This produces **one audio disruption** (the reconnect), not two (filler start + filler interrupt).

### 3.4 Trigger (Semantic Gate)

The Trigger evaluates every PersonaPlex turn and every user transcript to decide when to activate System 2. It uses pattern matching against four signal categories:

- **User action requests**: imperative verbs ("search", "look up", "build", "deploy")
- **User frustration**: phrases indicating the Talker has failed ("I already told you", "are you listening", "that's wrong")
- **Talker deflection**: PersonaPlex admitting inability ("I don't have access to", "I'm not able to search")
- **Echo artifacts**: PersonaPlex hallucinating search results or tool use

Social turns (greetings, thanks, acknowledgments) are explicitly excluded to prevent false triggers during normal conversation.

### 3.5 Belief State Compression

The compression algorithm maps the full CoALA working memory into PersonaPlex's ~200-token budget. Blocks are allocated by priority:

| Priority | Block                | Budget   | Purpose                                |
|----------|----------------------|----------|----------------------------------------|
| 1        | Persona              | ~30 words | Identity -- non-negotiable             |
| 2        | Conversation context | ~50 words | Current topic and summary              |
| 3        | User model           | ~30 words | Who the user is, their project/goals   |
| 4        | Fact corrections     | ~25 words | Prevent hallucination loops            |
| 5        | System 2 answer      | ~150 chars | The answer to deliver                  |
| 6        | Action queue         | ~15 words | Active background tasks                |

The System 2 answer is framed as: `[New information available to share with the user: "..."] Naturally incorporate this into your next response.` This phrasing causes PersonaPlex to deliver the information as if it were its own knowledge, maintaining conversational coherence.

Fact corrections use the format `NOT "wrong thing" -> "right thing"` to prevent PersonaPlex from re-hallucinating corrected information.

## 4. Echo Suppression in Full-Duplex Voice

Full-duplex voice introduces an echo problem not present in turn-based systems: the speech model's audio output plays through the user's speakers, gets picked up by the microphone, and is transcribed by Chrome's SpeechRecognition as "user speech." This creates false System 2 triggers -- for example, PersonaPlex saying "I could search for that" triggers the web search tool on a hallucinated statement.

Voice Bridge uses three layers of suppression:

### Layer 1: Browser-Side Timing Gate

SpeechRecognition results are suppressed for 8 seconds after PersonaPlex produces any text token or audio frame larger than 200 bytes. This is the primary defense and catches >90% of echo. The 8-second window accounts for the full chain: Opus decode (~50ms) + AudioWorklet buffering (~80ms) + speaker-to-mic propagation + SpeechRecognition processing (500-1500ms) + safety margin.

### Layer 2: Streaming Text Content Match

The server accumulates PersonaPlex's streaming text output into a rolling buffer. When a speech transcript arrives, its words (length > 2 characters) are compared against the buffer. If >= 40% of the transcript's words appear in the streaming buffer AND at least 2 words match, the transcript is classified as echo.

### Layer 3: Completed Turn Dedup

The server maintains a window of completed PersonaPlex turns (last 30 seconds). The same word-overlap algorithm is applied against this history, catching delayed transcriptions that arrive after the streaming buffer has been trimmed.

A length heuristic provides a fourth check: transcripts over 120 characters during active PersonaPlex speech are almost certainly mixed user-speaker audio and are suppressed outright.

Typed text (`text:` prefix) bypasses all echo filters. Only speech transcriptions (`speech:` prefix) are filtered.

## 5. Voxtral Integration: Audio-Native Intent Detection

The current Trigger system operates on text -- it requires SpeechRecognition to transcribe user speech before it can classify intent. This introduces latency and loses information present in the audio signal (tone, emphasis, hesitation).

Voxtral Mini 3B (Mistral AI, 2025) is a 3B-parameter speech understanding model that supports function calling directly from audio input. Voice Bridge integrates it as a **parallel audio listener**: the same Opus stream sent to PersonaPlex is simultaneously buffered and sent to Voxtral for intent classification.

```
Browser mic audio
    |
    +---> PersonaPlex (conversation)
    |
    +---> Voxtral Mini 3B (intent classification)
              |
              tool_call detected? ---> trigger.activate event
              plain text? ---> optional transcript
```

Four intent tools are defined:

| Tool           | Triggers on                                    |
|----------------|------------------------------------------------|
| `search_web`   | "search for X", "look up Y", "what is Z"      |
| `remember`     | "remember that", "what did I say about"        |
| `execute_task` | "build", "deploy", "create", "send"            |
| `get_status`   | "what's the status of", "how is X going"       |

Voxtral operates in two API modes:

- **Chat mode** (vLLM, Mistral API): Audio is sent as `input_audio` alongside tool definitions in a single `chat/completions` request. Voxtral returns either a tool call or a plain text response.
- **Transcribe mode** (Together AI fallback): Audio is first transcribed via `/audio/transcriptions`, then the text is classified via a second `chat/completions` request with tools.

The listener flushes every 4 seconds, with a 5-second cooldown between triggers to prevent rapid-fire activation. Maximum 2 concurrent API requests.

**VRAM budget**: PersonaPlex ~19GB + Voxtral quantized ~4GB = 23GB on a 24GB RTX 3090.

## 6. Solving the Three Limitations

Returning to the three limitations from Unmute #77:

**1. Write-only information flow.** Solved. System 2 results flow back to System 1 via belief state compression into `text_prompt`. PersonaPlex reconnects and delivers the answer in its own voice. The information flows both ways: System 1 -> event bus -> System 2, and System 2 -> memory compression -> `text_prompt` -> System 1.

**2. Tool-unaware speech model.** Solved differently than expected. Rather than making the speech model aware of tools (which would require retraining), we externalize tool awareness entirely. The speech model's job is to talk; when it fails, the Trigger detects the failure pattern and routes to System 2. The speech model does not need to know about tools because it never invokes them -- it only delivers their results.

**3. Double disruption.** Solved. PersonaPlex continues talking normally while System 2 works asynchronously. When the answer is ready, a single reconnect with the new `text_prompt` replaces PersonaPlex's context. One audio disruption, not two. The user experiences: question -> brief thinking indicator -> answer delivered in the same voice.

## 7. Implementation Notes

**Runtime**: Bun (TypeScript), chosen for native WebSocket support and fast startup.

**Event bus**: Session-scoped, priority-sorted (X-Talk pattern). All inter-component communication goes through the bus; no direct function calls between components. This makes the system testable and extensible.

**TLS**: Required because Chrome's SpeechRecognition API requires a secure context. The bridge auto-detects `certs/key.pem` and `certs/cert.pem` at startup.

**Text drought recovery**: PersonaPlex occasionally enters a degenerate state where it produces audio but no text tokens. The Talker auto-reconnects after 45 seconds of text silence.

**Session adoption**: PersonaPlex enforces a single-session lock. When a second browser tab connects, the bridge detects the existing session and transparently adopts the new WebSocket into it.

## 8. Limitations and Future Work

- **Voxtral function calling from audio is undocumented**: While Mistral advertises the capability, no hosted API provider publishes working examples of audio + tool calling in a single request. The transcribe-then-classify fallback adds ~1-2s latency.

- **200-token compression budget**: PersonaPlex's `text_prompt` is limited. Complex multi-step tool results must be aggressively summarized, which loses nuance.

- **Single-session lock**: PersonaPlex allows only one WebSocket client. Multi-user deployments require one GPU per concurrent conversation.

- **No streaming System 2**: The Reasoner produces a complete response before injecting it. Streaming partial results into `text_prompt` would require multiple reconnects, each causing an audio disruption.

- **Echo suppression is heuristic**: The three-layer system works well in practice but can suppress genuine user speech that happens to overlap with PersonaPlex's vocabulary.

Future work includes: (1) training a lightweight trigger model on the audio signal directly, replacing pattern matching; (2) exploring PersonaPlex v2's native tool calling when available; (3) multi-agent orchestration where multiple Reasoner agents specialize in different tool domains.

## References

- Defossez, A., Copet, J., Synnaeve, G., & Adi, Y. (2024). Moshi: a speech-text foundation model for real-time dialogue. *Kyutai*.
- Google DeepMind. (2024). Agents Thinking Fast and Slow. *arXiv:2410.08328*.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2023). Cognitive Architectures for Language Agents. *arXiv:2309.02427*.
- Kyutai. (2024). Unmute: Function calling discussion. *GitHub issue #77*.
- Mistral AI. (2025). Voxtral: Speech understanding with function calling. *mistral.ai/news/voxtral*.
- Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.

## License

MIT. See [README.md](README.md) for setup instructions.
