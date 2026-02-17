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

  /**
   * Cached Ogg BOS + comment header pages.
   * opus-recorder streams Ogg pages: the first 2 pages contain OpusHead and
   * OpusTags headers. Subsequent pages are audio data. Mistral's
   * /audio/transcriptions endpoint requires a valid Ogg/Opus file, so we
   * prepend these headers to every flush to make each chunk decodable.
   */
  private oggHeaderPages: Uint8Array | null = null;
  private oggHeaderPagesCollected = 0;
  private oggHeaderChunks: Uint8Array[] = [];

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
   *
   * opus-recorder with streamPages:true sends Ogg pages. The first 2 pages
   * are BOS (OpusHead) and comment (OpusTags) headers. We cache these and
   * prepend them to every flush so Mistral can decode each chunk.
   */
  feedAudio(data: ArrayBuffer): void {
    const chunk = new Uint8Array(data);

    // Capture Ogg header pages (first 2 pages: BOS + OpusTags)
    if (!this.oggHeaderPages) {
      this.oggHeaderChunks.push(chunk);
      // Count Ogg page boundaries in this chunk
      for (let i = 0; i <= chunk.length - 4; i++) {
        if (chunk[i] === 0x4F && chunk[i+1] === 0x67 &&
            chunk[i+2] === 0x67 && chunk[i+3] === 0x53) {
          this.oggHeaderPagesCollected++;
        }
      }
      // Once we have 3+ OggS markers, the first 2 pages are headers
      // and the 3rd starts audio data. Cache everything before the 3rd marker.
      if (this.oggHeaderPagesCollected >= 3) {
        // Concatenate all collected chunks
        const totalLen = this.oggHeaderChunks.reduce((s, c) => s + c.length, 0);
        const all = new Uint8Array(totalLen);
        let offset = 0;
        for (const c of this.oggHeaderChunks) {
          all.set(c, offset);
          offset += c.length;
        }
        // Find the 3rd OggS marker — everything before it is headers
        let markerCount = 0;
        let headerEnd = all.length;
        for (let i = 0; i <= all.length - 4; i++) {
          if (all[i] === 0x4F && all[i+1] === 0x67 &&
              all[i+2] === 0x67 && all[i+3] === 0x53) {
            markerCount++;
            if (markerCount === 3) {
              headerEnd = i;
              break;
            }
          }
        }
        this.oggHeaderPages = all.slice(0, headerEnd);
        // Feed remaining audio data into the buffer
        if (headerEnd < all.length) {
          const audioData = all.slice(headerEnd);
          this.audioBuffer.push(audioData);
          this.audioBufferBytes += audioData.length;
        }
        this.oggHeaderChunks = []; // Free memory
        logger.info({ headerBytes: this.oggHeaderPages.length }, 'Cached Ogg header pages for Voxtral');
        return;
      }
      return; // Still collecting headers
    }

    // Only buffer chunks that look like Ogg pages (start with "OggS")
    // opus-recorder with streamPages:true should always send complete Ogg pages
    if (chunk.length >= 4 && chunk[0] === 0x4F && chunk[1] === 0x67 &&
        chunk[2] === 0x67 && chunk[3] === 0x53) {
      this.audioBuffer.push(chunk);
      this.audioBufferBytes += chunk.length;
    }
    // else: skip non-Ogg data (possible if opus-recorder sends partial/non-page data)
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
    this.oggHeaderPages = null;
    this.oggHeaderPagesCollected = 0;
    this.oggHeaderChunks = [];
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

    // Concatenate chunks into a single buffer, prepending Ogg headers
    const headerLen = this.oggHeaderPages?.length ?? 0;
    const combined = new Uint8Array(headerLen + totalBytes);
    let offset = 0;
    if (this.oggHeaderPages) {
      combined.set(this.oggHeaderPages, 0);
      offset = headerLen;
    }
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    // Validate: combined should start with OggS (our prepended headers)
    const hasOggHeader = combined.length >= 4 &&
      combined[0] === 0x4F && combined[1] === 0x67 &&
      combined[2] === 0x67 && combined[3] === 0x53;
    if (!hasOggHeader) {
      logger.warn({ totalBytes: combined.length, first4: Array.from(combined.slice(0, 4)).map(b => b.toString(16)) },
        'Voxtral flush: audio missing OggS header — skipping');
      return;
    }

    // Rewrite Ogg page sequence numbers to be contiguous (0, 1, 2, ...).
    // Our buffer has header pages (seq 0, 1) followed by data pages from
    // the middle of the stream (seq N, N+1, ...). Mistral's decoder rejects
    // the gap, so we renumber all pages sequentially.
    this.rewriteOggPageSequences(combined);

    logger.debug({ totalBytes: combined.length, headerLen: headerLen, dataBytes: totalBytes },
      'Voxtral flush: sending audio to Mistral');

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
   * Rewrite Ogg page sequence numbers and CRC in a buffer to be contiguous.
   *
   * Ogg page layout (27-byte header):
   *   0-3:   "OggS" magic
   *   4:     version (0)
   *   5:     header type (BOS=0x02, continuation=0x01, EOS=0x04)
   *   6-13:  granule position (8 bytes, little-endian)
   *   14-17: serial number (4 bytes, little-endian)
   *   18-21: page sequence number (4 bytes, little-endian)
   *   22-25: CRC checksum (4 bytes, little-endian)
   *   26:    number of segments
   *   27+:   segment table (N bytes), then payload
   *
   * After rewriting sequence numbers, we must recalculate the CRC.
   */
  private rewriteOggPageSequences(buf: Uint8Array): void {
    let pos = 0;
    let pageSeq = 0;

    while (pos + 27 <= buf.length) {
      // Find OggS magic
      if (buf[pos] !== 0x4F || buf[pos+1] !== 0x67 ||
          buf[pos+2] !== 0x67 || buf[pos+3] !== 0x53) {
        break; // Not at a page boundary — done
      }

      // Rewrite page sequence number (bytes 18-21, little-endian)
      buf[pos + 18] = pageSeq & 0xFF;
      buf[pos + 19] = (pageSeq >> 8) & 0xFF;
      buf[pos + 20] = (pageSeq >> 16) & 0xFF;
      buf[pos + 21] = (pageSeq >> 24) & 0xFF;

      // Calculate page size to advance: 27 (header) + numSegments + sum(segments)
      const numSegments = buf[pos + 26];
      if (pos + 27 + numSegments > buf.length) break;

      let payloadSize = 0;
      for (let i = 0; i < numSegments; i++) {
        payloadSize += buf[pos + 27 + i];
      }

      const pageSize = 27 + numSegments + payloadSize;

      // Recalculate CRC (zero it first, then compute)
      buf[pos + 22] = 0;
      buf[pos + 23] = 0;
      buf[pos + 24] = 0;
      buf[pos + 25] = 0;
      const crc = this.oggCrc32(buf, pos, pageSize);
      buf[pos + 22] = crc & 0xFF;
      buf[pos + 23] = (crc >> 8) & 0xFF;
      buf[pos + 24] = (crc >> 16) & 0xFF;
      buf[pos + 25] = (crc >> 24) & 0xFF;

      pos += pageSize;
      pageSeq++;
    }
  }

  /** Ogg CRC-32 (polynomial 0x04C11DB7, no final XOR). */
  private oggCrc32(data: Uint8Array, offset: number, length: number): number {
    // Precompute table on first call
    if (!VoxtralListener.oggCrcTable) {
      VoxtralListener.oggCrcTable = new Uint32Array(256);
      for (let i = 0; i < 256; i++) {
        let r = i << 24;
        for (let j = 0; j < 8; j++) {
          r = (r & 0x80000000) ? ((r << 1) ^ 0x04C11DB7) : (r << 1);
        }
        VoxtralListener.oggCrcTable[i] = r >>> 0;
      }
    }

    let crc = 0;
    const table = VoxtralListener.oggCrcTable;
    for (let i = 0; i < length; i++) {
      crc = ((crc << 8) ^ table[((crc >>> 24) ^ data[offset + i]) & 0xFF]) >>> 0;
    }
    return crc;
  }

  private static oggCrcTable: Uint32Array | null = null;

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
