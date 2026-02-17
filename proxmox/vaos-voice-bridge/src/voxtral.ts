/**
 * VoxtralListener — Parallel audio listener for intent detection.
 *
 * Listens to the same user audio stream as PersonaPlex, buffers chunks,
 * and periodically sends them to Voxtral Mini 3B (via Together AI) for
 * speech-to-tool-call classification.
 *
 * When Voxtral detects tool-worthy intent in the audio, it fires
 * trigger.activate on the event bus — replacing the text-based
 * semantic trigger with a single model that goes audio → tool call.
 *
 * PersonaPlex still handles the conversation. Voxtral is the ear
 * for System 2.
 *
 * Flow:
 *   Browser audio → VoxtralListener.feedAudio()
 *                 → buffer accumulates ~3-5s of audio
 *                 → flush to Together AI (Voxtral Mini 3B)
 *                 → tool_call detected? → bus.emit(trigger.activate)
 *                 → plain text? → optional transcript (future: replace SpeechRecognition)
 */

import { createLogger } from './lib/logger.js';
import { type EventBus, E } from './events.js';

const logger = createLogger('voxtral');

// ─── Tool Definitions ────────────────────────────────────────────
// These are intent categories, not actual tool implementations.
// Voxtral classifies user speech into these intents; the Reasoner
// handles actual execution via Letta.

const VOXTRAL_TOOLS = [
  {
    type: 'function' as const,
    function: {
      name: 'search_web',
      description: 'User wants to search the web, look something up, or get current information about a topic.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The search query extracted from the user\'s speech',
          },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'remember',
      description: 'User wants to store information in memory, recall something previously discussed, or update their preferences.',
      parameters: {
        type: 'object',
        properties: {
          action: {
            type: 'string',
            enum: ['store', 'recall', 'update'],
            description: 'Whether to store new info, recall existing info, or update preferences',
          },
          content: {
            type: 'string',
            description: 'What to remember or recall',
          },
        },
        required: ['action', 'content'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'execute_task',
      description: 'User wants to perform an action: create something, build, deploy, run code, send a message, generate content.',
      parameters: {
        type: 'object',
        properties: {
          task: {
            type: 'string',
            description: 'Description of the task to execute',
          },
          urgency: {
            type: 'string',
            enum: ['immediate', 'normal', 'background'],
            description: 'How urgently the task should be handled',
          },
        },
        required: ['task'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'get_status',
      description: 'User wants to check the status of something: a process, system, task, or get a report.',
      parameters: {
        type: 'object',
        properties: {
          subject: {
            type: 'string',
            description: 'What to check the status of',
          },
        },
        required: ['subject'],
      },
    },
  },
];

// ─── Configuration ───────────────────────────────────────────────

export interface VoxtralConfig {
  /** API key (Together AI, Mistral, or self-hosted vLLM). */
  apiKey: string;
  /** API base URL (default: Together AI). */
  baseUrl?: string;
  /** Model ID for transcription. */
  model?: string;
  /**
   * Model ID for tool-calling classification step (transcribe mode only).
   * Voxtral on Together AI serverless doesn't return proper tool_calls
   * (backend not configured with --tool-call-parser), so we use a separate
   * model that supports native tool calling.
   */
  classifierModel?: string;
  /** How many seconds of audio to accumulate before flushing. */
  bufferSeconds?: number;
  /** Minimum audio bytes before attempting a flush. */
  minBufferBytes?: number;
  /** Maximum concurrent API requests. */
  maxConcurrent?: number;
  /** Whether to emit plain transcriptions as user.text events. */
  emitTranscriptions?: boolean;
  /**
   * API mode:
   * - 'chat': Send audio via chat/completions with input_audio + tools (vLLM, Mistral native)
   * - 'transcribe': Voxtral transcribes audio, emits user.text for existing trigger.ts to classify
   */
  mode?: 'chat' | 'transcribe';
}

const DEFAULTS = {
  baseUrl: 'https://api.together.xyz/v1',
  model: 'mistralai/Voxtral-Mini-3B-2507',
  classifierModel: 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
  bufferSeconds: 4,
  minBufferBytes: 4800, // ~0.1s of Opus at 48kbps
  maxConcurrent: 2,
  emitTranscriptions: false,
  mode: 'chat' as const,
};

// ─── VoxtralListener ─────────────────────────────────────────────

export class VoxtralListener {
  private bus: EventBus;
  private sessionId: string;
  private config: Required<VoxtralConfig>;

  /** Raw Opus frame buffer — accumulated between flushes. */
  private audioBuffer: Uint8Array[] = [];
  private audioBufferBytes = 0;

  /** Flush timer. */
  private flushTimer: ReturnType<typeof setInterval> | null = null;

  /** Track in-flight requests to limit concurrency. */
  private inflight = 0;

  /** Suppress duplicate triggers within a cooldown window. */
  private lastTriggerTime = 0;
  private readonly TRIGGER_COOLDOWN_MS = 5_000;

  /** System prompt — very strict to minimize false positives. */
  private readonly systemPrompt = `You are a strict voice intent classifier. Your job: decide if the user EXPLICITLY asked for an action, or if they are just having a conversation.

ONLY call a tool when the user uses EXPLICIT action language like:
- "Search for...", "Look up...", "Google..." → search_web
- "Remember that...", "Don't forget..." → remember (ONLY when user explicitly asks you to remember something specific)
- "Build...", "Deploy...", "Create...", "Send..." → execute_task
- "What's the status of...", "How is X going" → get_status

NEVER call a tool for:
- Agreements ("yeah", "that makes sense", "okay")
- Opinions ("I think...", "probably...")
- Questions that are part of normal conversation
- Greetings, small talk, acknowledgments
- Anything that isn't a DIRECT command or request

If in doubt, do NOT call a tool. Respond with "no action" instead. False positives are worse than missed detections.`;

  constructor(bus: EventBus, sessionId: string, config: VoxtralConfig) {
    this.bus = bus;
    this.sessionId = sessionId;
    this.config = {
      apiKey: config.apiKey,
      baseUrl: config.baseUrl ?? DEFAULTS.baseUrl,
      model: config.model ?? DEFAULTS.model,
      classifierModel: config.classifierModel ?? DEFAULTS.classifierModel,
      bufferSeconds: config.bufferSeconds ?? DEFAULTS.bufferSeconds,
      minBufferBytes: config.minBufferBytes ?? DEFAULTS.minBufferBytes,
      maxConcurrent: config.maxConcurrent ?? DEFAULTS.maxConcurrent,
      emitTranscriptions: config.emitTranscriptions ?? DEFAULTS.emitTranscriptions,
      mode: config.mode ?? DEFAULTS.mode,
    };

    logger.info({
      model: this.config.model,
      classifierModel: this.config.classifierModel,
      mode: this.config.mode,
      bufferSeconds: this.config.bufferSeconds,
    }, 'VoxtralListener initialized');
  }

  // ─── Audio Pipeline ──────────────────────────────────────────

  /**
   * Feed raw Opus audio frames from the browser.
   * Called on every binary WebSocket message alongside PersonaPlex.
   */
  feedAudio(data: ArrayBuffer): void {
    const chunk = new Uint8Array(data);
    this.audioBuffer.push(chunk);
    this.audioBufferBytes += chunk.length;
  }

  /** Start the periodic flush timer. */
  start(): void {
    if (this.flushTimer) return;
    this.flushTimer = setInterval(() => {
      this.maybeFlush();
    }, this.config.bufferSeconds * 1000);
    logger.info({ intervalMs: this.config.bufferSeconds * 1000 }, 'VoxtralListener started');
  }

  /** Stop listening and clear buffers. */
  stop(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
    this.audioBuffer = [];
    this.audioBufferBytes = 0;
    logger.info('VoxtralListener stopped');
  }

  // ─── Flush Logic ─────────────────────────────────────────────

  private maybeFlush(): void {
    // Skip if buffer too small (silence or mic not active)
    if (this.audioBufferBytes < this.config.minBufferBytes) {
      return;
    }

    // Skip if too many in-flight requests
    if (this.inflight >= this.config.maxConcurrent) {
      logger.debug({ inflight: this.inflight }, 'Skipping flush — max concurrent reached');
      return;
    }

    // Take the buffer and reset
    const chunks = this.audioBuffer;
    const totalBytes = this.audioBufferBytes;
    this.audioBuffer = [];
    this.audioBufferBytes = 0;

    // Concatenate chunks into a single buffer
    const combined = new Uint8Array(totalBytes);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    // Fire and forget — don't block the audio pipeline
    this.sendToVoxtral(combined).catch(err => {
      logger.error({ err: err instanceof Error ? err.message : String(err) }, 'Voxtral API error');
    });
  }

  // ─── API Call ────────────────────────────────────────────────

  private async sendToVoxtral(audioData: Uint8Array): Promise<void> {
    this.inflight++;
    const startTime = Date.now();

    try {
      if (this.config.mode === 'transcribe') {
        return await this.transcribeAndClassify(audioData, startTime);
      }

      // Encode audio as base64 for the API
      const audioBase64 = Buffer.from(audioData).toString('base64');

      // Build the multimodal message with audio + tool definitions.
      // Format: vLLM/Mistral-compatible `input_audio` (base64 wav/opus).
      // Together AI may use `audio` key instead — we try `input_audio` first
      // (works with self-hosted vLLM and Mistral API).
      const response = await fetch(`${this.config.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.config.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.config.model,
          messages: [
            {
              role: 'system',
              content: this.systemPrompt,
            },
            {
              role: 'user',
              content: [
                {
                  type: 'input_audio',
                  input_audio: {
                    data: audioBase64,
                    format: 'opus',
                  },
                },
                {
                  type: 'text',
                  text: 'Listen to the audio and determine if the user is requesting an action.',
                },
              ],
            },
          ],
          tools: VOXTRAL_TOOLS,
          tool_choice: 'auto',
          temperature: 0.1,
          max_tokens: 256,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        logger.error({
          status: response.status,
          error: errorText.slice(0, 500),
          latencyMs: Date.now() - startTime,
        }, 'Voxtral API error response');
        return;
      }

      const result = await response.json() as VoxtralResponse;
      const latencyMs = Date.now() - startTime;
      const choice = result.choices?.[0];

      if (!choice) {
        logger.warn({ latencyMs }, 'Voxtral returned no choices');
        return;
      }

      // Check for tool calls
      if (choice.message?.tool_calls && choice.message.tool_calls.length > 0) {
        await this.handleToolCalls(choice.message.tool_calls, latencyMs);
        return;
      }

      // Plain text response — Voxtral decided no tool was needed
      const text = choice.message?.content;
      if (text) {
        logger.debug({ text: text.slice(0, 200), latencyMs }, 'Voxtral transcript (no tool call)');

        // Optionally emit as a user.text event (more reliable than SpeechRecognition)
        if (this.config.emitTranscriptions && text.length > 5) {
          this.bus.emit(E.userText(this.sessionId, text));
        }
      }
    } finally {
      this.inflight--;
    }
  }

  // ─── Transcribe + Classify Fallback ───────────────────────────
  // For providers that don't support audio in chat/completions (e.g. Together AI),
  // we transcribe first, then classify the text with tool calling.

  private async transcribeAndClassify(audioData: Uint8Array, startTime: number): Promise<void> {
    // Transcribe audio via Voxtral, then emit as user.text.
    // The existing trigger.ts pattern matching handles intent classification.
    // This is more reliable than LLM-based classification which has high false positive rates.
    const formData = new FormData();
    formData.append('file', new Blob([audioData], { type: 'audio/opus' }), 'audio.opus');
    formData.append('model', this.config.model);
    formData.append('language', 'en');

    const transcribeRes = await fetch(`${this.config.baseUrl}/audio/transcriptions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.config.apiKey}`,
      },
      body: formData,
    });

    if (!transcribeRes.ok) {
      const err = await transcribeRes.text();
      logger.error({ status: transcribeRes.status, error: err.slice(0, 300) }, 'Voxtral transcription failed');
      return;
    }

    const transcription = await transcribeRes.json() as { text?: string };
    const text = transcription.text?.trim();
    if (!text || text.length < 3) return;

    const latencyMs = Date.now() - startTime;

    // Filter out hallucinated transcriptions from silence/noise.
    // Voxtral hallucinates coherent text from non-speech audio.
    if (this.isLikelyHallucination(text)) {
      logger.debug({ text: text.slice(0, 100), latencyMs }, 'Voxtral: filtered hallucination');
      return;
    }

    logger.info({ text: text.slice(0, 200), latencyMs }, 'Voxtral transcription');

    // Emit as user.text — the existing Trigger subscribes to this and handles
    // intent classification via pattern matching (deflections, action keywords, etc.)
    this.bus.emit(E.userText(this.sessionId, text));
  }

  /**
   * Detect hallucinated transcriptions from silence/noise.
   * Voxtral generates coherent but fabricated text when given non-speech audio.
   */
  private isLikelyHallucination(text: string): boolean {
    const lower = text.toLowerCase();
    // Common hallucination patterns from silence
    const HALLUCINATION_PATTERNS = [
      "i'm sorry",
      "i didn't catch",
      "could you repeat",
      "i can't hear",
      "thank you for watching",
      "please subscribe",
      "the end",
      "music playing",
      "[music]",
    ];
    return HALLUCINATION_PATTERNS.some(p => lower.includes(p));
  }

  // ─── Tool Call Handling ──────────────────────────────────────

  private async handleToolCalls(
    toolCalls: VoxtralToolCall[],
    latencyMs: number,
  ): Promise<void> {
    const now = Date.now();

    // Cooldown — don't fire triggers in rapid succession
    if (now - this.lastTriggerTime < this.TRIGGER_COOLDOWN_MS) {
      logger.debug({ cooldownRemaining: this.TRIGGER_COOLDOWN_MS - (now - this.lastTriggerTime) },
        'Voxtral tool call suppressed (cooldown)');
      return;
    }

    for (const tc of toolCalls) {
      const fnName = tc.function?.name;
      const fnArgs = tc.function?.arguments;

      if (!fnName) continue;

      let parsedArgs: Record<string, any> = {};
      try {
        parsedArgs = typeof fnArgs === 'string' ? JSON.parse(fnArgs) : (fnArgs ?? {});
      } catch {
        logger.warn({ fnName, fnArgs }, 'Failed to parse Voxtral tool call args');
      }

      // Build context string from the tool call
      const context = this.buildContextFromToolCall(fnName, parsedArgs);

      logger.info({
        tool: fnName,
        args: parsedArgs,
        context: context.slice(0, 200),
        latencyMs,
      }, 'Voxtral detected intent — firing trigger');

      // Fire the semantic trigger
      this.bus.emit(E.triggerActivate(
        this.sessionId,
        'semantic',   // reason
        0.85,         // high confidence — Voxtral is purpose-trained for this
        context,
      ));

      this.lastTriggerTime = now;

      // Only fire for the first tool call per flush
      break;
    }
  }

  /**
   * Convert a tool call into a natural language context string
   * for the Reasoner to act on.
   */
  private buildContextFromToolCall(
    fnName: string,
    args: Record<string, any>,
  ): string {
    switch (fnName) {
      case 'search_web':
        return `User wants to search the web: "${args.query ?? 'unknown'}"`;
      case 'remember':
        return `User wants to ${args.action ?? 'recall'} memory: "${args.content ?? 'unknown'}"`;
      case 'execute_task':
        return `User wants to execute a task: "${args.task ?? 'unknown'}" (urgency: ${args.urgency ?? 'normal'})`;
      case 'get_status':
        return `User wants status check on: "${args.subject ?? 'unknown'}"`;
      default:
        return `User intent detected via Voxtral: ${fnName}(${JSON.stringify(args)})`;
    }
  }
}

// ─── API Response Types ──────────────────────────────────────────

interface VoxtralToolCall {
  id?: string;
  type?: 'function';
  function?: {
    name: string;
    arguments: string | Record<string, any>;
  };
}

interface VoxtralMessage {
  role: string;
  content?: string;
  tool_calls?: VoxtralToolCall[];
}

interface VoxtralChoice {
  index: number;
  message?: VoxtralMessage;
  finish_reason?: string;
}

interface VoxtralResponse {
  id?: string;
  choices?: VoxtralChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}
