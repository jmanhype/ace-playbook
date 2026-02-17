/**
 * VoxtralListener — Parallel audio listener for intent detection.
 *
 * Listens to the same user audio stream as PersonaPlex, buffers chunks,
 * and periodically sends them to Voxtral (via Mistral API) for
 * speech-to-tool-call classification.
 *
 * When Voxtral detects tool-worthy intent in the audio, it fires
 * trigger.activate on the event bus — replacing Chrome SpeechRecognition,
 * text-based pattern matching, and separate intent extraction with a
 * single model family that goes audio → tool call.
 *
 * Two-step pipeline (both Mistral API):
 *   1. Voxtral Mini (3B) transcribes audio via /audio/transcriptions
 *   2. Voxtral Small (24B) classifies transcript with tool definitions
 *      → returns proper tool_calls or "no action"
 *
 * This replaces three separate components:
 *   - Chrome SpeechRecognition (ASR)
 *   - trigger.ts pattern matching (semantic gate)
 *   - manual intent extraction (structured query)
 *
 * PersonaPlex still handles the conversation. Voxtral is the ear
 * for System 2.
 *
 * Flow:
 *   Browser audio → VoxtralListener.feedAudio()
 *                 → buffer accumulates ~4s of audio
 *                 → flush: Voxtral Mini transcribes → Voxtral Small classifies
 *                 → tool_call detected? → bus.emit(trigger.activate)
 *                 → no action? → discard (PersonaPlex handles conversation)
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
  /** API key for the Voxtral provider (Mistral API or Together AI). */
  apiKey: string;
  /** API base URL. Default: Mistral API. */
  baseUrl?: string;
  /** Model ID for transcription (Voxtral Mini 3B). */
  model?: string;
  /**
   * Model ID for tool-calling classification (Voxtral Small 24B on Mistral API).
   * Voxtral Mini doesn't support function calling; Voxtral Small does.
   * Only used in 'chat' mode.
   */
  classifierModel?: string;
  /** How many seconds of audio to accumulate before flushing. */
  bufferSeconds?: number;
  /** Minimum audio bytes before attempting a flush. */
  minBufferBytes?: number;
  /** Maximum concurrent API requests. */
  maxConcurrent?: number;
  /** Whether to also emit transcriptions as user.text events. */
  emitTranscriptions?: boolean;
  /**
   * API mode:
   * - 'chat': Two-step Mistral pipeline — Voxtral Mini transcribes, Voxtral Small classifies with tools
   * - 'transcribe': Voxtral Mini transcribes only, emits user.text for existing trigger.ts to classify
   */
  mode?: 'chat' | 'transcribe';
}

const DEFAULTS = {
  baseUrl: 'https://api.mistral.ai/v1',
  model: 'voxtral-mini-2507',
  classifierModel: 'voxtral-small-2507',
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

  /** System prompt for the Voxtral Small classifier — strict to minimize false positives. */
  private readonly classifierPrompt = `You are a strict intent classifier for a voice assistant. ONLY call a tool when the user EXPLICITLY requests an action. For normal conversation, agreements, small talk, or anything ambiguous, respond with "no action needed". False positives are worse than missed detections.`;

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
        return await this.transcribeOnly(audioData, startTime);
      }

      // ─── Chat Mode: Two-Step Mistral Pipeline ──────────────────
      // Step 1: Voxtral Mini transcribes raw audio (accepts Opus via FormData)
      // Step 2: Voxtral Small classifies transcript with tool definitions
      //
      // This replaces: Chrome SpeechRecognition + trigger.ts + intent extraction

      // Step 1: Transcribe
      const transcript = await this.transcribe(audioData);
      if (!transcript) return;

      const transcribeMs = Date.now() - startTime;
      logger.debug({ text: transcript.slice(0, 200), latencyMs: transcribeMs }, 'Voxtral transcribed');

      // Optionally emit the transcription as user.text
      if (this.config.emitTranscriptions) {
        this.bus.emit(E.userText(this.sessionId, transcript));
      }

      // Step 2: Classify with tools via Voxtral Small
      const classifyStart = Date.now();
      const response = await fetch(`${this.config.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.config.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.config.classifierModel,
          messages: [
            { role: 'system', content: this.classifierPrompt },
            { role: 'user', content: transcript },
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
        }, 'Voxtral classifier error');
        return;
      }

      const result = await response.json() as VoxtralResponse;
      const totalMs = Date.now() - startTime;
      const classifyMs = Date.now() - classifyStart;
      const choice = result.choices?.[0];

      if (!choice) {
        logger.warn({ totalMs }, 'Voxtral classifier returned no choices');
        return;
      }

      // Check for tool calls
      if (choice.message?.tool_calls && choice.message.tool_calls.length > 0) {
        logger.info({
          transcribeMs,
          classifyMs,
          totalMs,
          transcript: transcript.slice(0, 100),
        }, 'Voxtral pipeline: tool call detected');
        await this.handleToolCalls(choice.message.tool_calls, totalMs);
        return;
      }

      // No tool call — Voxtral Small decided it's normal conversation
      logger.debug({
        text: choice.message?.content?.slice(0, 100),
        totalMs,
      }, 'Voxtral pipeline: no action');
    } finally {
      this.inflight--;
    }
  }

  // ─── Transcription ──────────────────────────────────────────────
  // Shared audio → text via Voxtral Mini's /audio/transcriptions endpoint.
  // Accepts raw Opus frames via FormData. Returns null if silence/hallucination.

  private async transcribe(audioData: Uint8Array): Promise<string | null> {
    const formData = new FormData();
    formData.append('file', new Blob([audioData as BlobPart], { type: 'audio/opus' }), 'audio.opus');
    formData.append('model', this.config.model);
    formData.append('language', 'en');

    const res = await fetch(`${this.config.baseUrl}/audio/transcriptions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.config.apiKey}`,
      },
      body: formData,
    });

    if (!res.ok) {
      const err = await res.text();
      logger.error({ status: res.status, error: err.slice(0, 300) }, 'Voxtral transcription failed');
      return null;
    }

    const transcription = await res.json() as { text?: string };
    const text = transcription.text?.trim();
    if (!text || text.length < 3) return null;

    // Filter hallucinations from silence/noise
    if (this.isLikelyHallucination(text)) {
      logger.debug({ text: text.slice(0, 100) }, 'Voxtral: filtered hallucination');
      return null;
    }

    return text;
  }

  // ─── Transcribe-Only Mode ──────────────────────────────────────
  // For setups using existing trigger.ts for classification.
  // Voxtral Mini transcribes, emits user.text for pattern matching.

  private async transcribeOnly(audioData: Uint8Array, startTime: number): Promise<void> {
    const text = await this.transcribe(audioData);
    if (!text) return;

    const latencyMs = Date.now() - startTime;
    logger.info({ text: text.slice(0, 200), latencyMs }, 'Voxtral transcription');

    // Emit as user.text — trigger.ts handles intent classification
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
