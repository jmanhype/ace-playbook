/**
 * Talker — PersonaPlex WebSocket client (System 1).
 *
 * Manages the WebSocket connection to PersonaPlex on the 3090 server.
 * PersonaPlex uses the Moshi protocol:
 *   - Binary frames with type byte prefix
 *   - 0x02 = text token messages (the model's text output)
 *   - Audio frames contain Opus-encoded chunks
 *
 * The Talker is always on, fast, and intuitive.
 * It gets smarter over time via dynamic text_prompt injection from the Reasoner.
 */

import { createLogger } from './lib/logger.js';
import { getEnv } from './lib/config.js';

const logger = createLogger('talker');

/** PersonaPlex binary message types (Moshi protocol). */
const MSG_TYPE = {
  HANDSHAKE: 0x00,
  AUDIO: 0x01,
  TEXT: 0x02,
  CONTROL: 0x03,
  METADATA: 0x04,
} as const;

export interface TalkerEvents {
  /** Fired when PersonaPlex produces a text token. */
  onText: (text: string) => void;
  /** Fired when a complete turn is detected (silence after speech). */
  onTurnComplete: (fullText: string) => void;
  /** Fired when PersonaPlex sends audio data. */
  onAudio: (data: Uint8Array) => void;
  /** Fired on connection state change. */
  onStateChange: (state: 'connecting' | 'connected' | 'disconnected') => void;
}

export class Talker {
  private ws: WebSocket | null = null;
  private currentTurnText = '';
  private turnTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelayMs = 2000;
  private events: Partial<TalkerEvents> = {};
  private textPrompt = '';
  private _handshakeComplete = false;
  /**
   * Ogg header page cache. PersonaPlex's opus decoder needs the Ogg
   * BOS (OpusHead) and comment (OpusTags) pages before any audio data.
   * These often arrive from the browser BEFORE the PersonaPlex handshake
   * completes, so sendAudio() drops them. We cache them here and replay
   * them right after the handshake.
   */
  private oggHeaderCache: Uint8Array[] = [];
  private oggHeadersSent = false;

  /** True once PersonaPlex handshake is complete and audio can flow. */
  get handshakeComplete(): boolean {
    return this._handshakeComplete;
  }

  /** Current dynamic text prompt injected by the Reasoner. */
  get currentPrompt(): string {
    return this.textPrompt;
  }

  /** Register event handlers. */
  on<K extends keyof TalkerEvents>(event: K, handler: TalkerEvents[K]): void {
    this.events[event] = handler;
  }

  /** Connect to PersonaPlex WebSocket. */
  async connect(): Promise<void> {
    const env = getEnv();
    // PersonaPlex requires text_prompt and voice_prompt as query params
    const prompt = encodeURIComponent(this.textPrompt || 'You are a helpful voice assistant. Be concise and natural.');
    const voicePrompt = encodeURIComponent(env.VOICE_PROMPT_PATH);
    const url = `wss://${env.PERSONAPLEX_HOST}:${env.PERSONAPLEX_PORT}${env.PERSONAPLEX_WS_PATH}?text_prompt=${prompt}&voice_prompt=${voicePrompt}`;

    logger.info({ url }, 'Connecting to PersonaPlex');
    this._handshakeComplete = false;
    this.oggHeadersSent = false;
    this.oggHeaderCache = [];
    this._sendCount = 0;
    this.events.onStateChange?.('connecting');

    try {
      this.ws = new WebSocket(url);
      this.ws.binaryType = 'arraybuffer';

      this.ws.addEventListener('open', () => {
        logger.info('WebSocket open to PersonaPlex (awaiting handshake)');
        this.reconnectAttempts = 0;
        // Don't emit 'connected' yet — wait for handshake exchange.
        // PersonaPlex loads system prompts before sending handshake (can take seconds).
      });

      this.ws.addEventListener('message', (event) => {
        this.handleMessage(event.data).catch(err => {
          logger.error({ err }, 'Unhandled error in handleMessage');
        });
      });

      this.ws.addEventListener('close', (event) => {
        logger.warn({ code: event.code, reason: event.reason }, 'PersonaPlex disconnected');
        this.events.onStateChange?.('disconnected');
        this.scheduleReconnect();
      });

      this.ws.addEventListener('error', (event) => {
        logger.error({ error: event }, 'PersonaPlex WebSocket error');
      });
    } catch (err) {
      logger.error({ err }, 'Failed to connect to PersonaPlex');
      this.scheduleReconnect();
    }
  }

  /** Update the dynamic text prompt (enriched by Reasoner's belief state). */
  updateTextPrompt(prompt: string): void {
    this.textPrompt = prompt;
    logger.debug({ promptLength: prompt.length }, 'Text prompt updated (applied on next reconnect)');
  }

  /**
   * Reconnect to PersonaPlex with the current text prompt.
   * Used when the Reasoner updates the belief state and the Talker needs
   * to incorporate the new context. Per the paper: "System 2 taking over
   * and overruling the impulses of System 1."
   */
  async reconnectWithNewPrompt(): Promise<void> {
    logger.info('Reconnecting PersonaPlex with updated prompt (System 2 override)');
    this.disconnect();
    // Small delay to let PersonaPlex release the session lock
    await new Promise(resolve => setTimeout(resolve, 500));
    await this.connect();
  }

  private _sendCount = 0;

  /**
   * Check if an ArrayBuffer contains an Ogg page and extract its flags.
   * Returns the Ogg header_type flags byte, or -1 if not an Ogg page.
   * Ogg page structure: "OggS" (4 bytes) + version (1) + header_type (1)
   * header_type: 0x02 = BOS (beginning of stream = OpusHead header)
   */
  private static getOggFlags(data: Uint8Array): number {
    if (data.length >= 6 && data[0] === 0x4f && data[1] === 0x67 &&
        data[2] === 0x67 && data[3] === 0x53) {
      return data[5]; // header_type byte
    }
    return -1;
  }

  /** Forward user audio to PersonaPlex (adds 0x01 prefix). Caches Ogg headers pre-handshake. */
  sendAudio(data: ArrayBuffer): void {
    const audioBytes = new Uint8Array(data);

    if (!this._handshakeComplete) {
      // Before handshake: cache Ogg header pages (BOS + comment) for replay after handshake
      const flags = Talker.getOggFlags(audioBytes);
      if (flags >= 0 && !this.oggHeadersSent) {
        // Cache the first 2 Ogg pages (OpusHead BOS page + OpusTags page)
        if (this.oggHeaderCache.length < 2) {
          this.oggHeaderCache.push(new Uint8Array(audioBytes));
          logger.info({ cachedPages: this.oggHeaderCache.length, flags: `0x${flags.toString(16)}`, size: audioBytes.length }, 'Cached Ogg header page (pre-handshake)');
        }
      }
      return; // Drop all audio until handshake done
    }

    if (this.ws?.readyState === WebSocket.OPEN) {
      const frame = new Uint8Array(1 + audioBytes.length);
      frame[0] = MSG_TYPE.AUDIO;
      frame.set(audioBytes, 1);
      this._sendCount++;
      this.ws.send(frame);
    }
  }

  /** Replay cached Ogg header pages to PersonaPlex after handshake. */
  private replayOggHeaders(): void {
    if (this.oggHeadersSent || this.oggHeaderCache.length === 0) return;
    if (this.ws?.readyState !== WebSocket.OPEN) return;

    logger.info({ cachedPages: this.oggHeaderCache.length }, 'Replaying cached Ogg headers to PersonaPlex');
    for (const header of this.oggHeaderCache) {
      const frame = new Uint8Array(1 + header.length);
      frame[0] = MSG_TYPE.AUDIO;
      frame.set(header, 1);
      this._sendCount++;
      this.ws.send(frame);
    }
    this.oggHeadersSent = true;
  }

  /** Send the Moshi handshake response (single 0x00 byte, matching native client). */
  private sendHandshake(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const msg = new Uint8Array([MSG_TYPE.HANDSHAKE]);
      this._sendCount++;
      logger.info('Sent handshake response to PersonaPlex');
      this.ws.send(msg);
    }
  }

  /** Handle incoming PersonaPlex messages. */
  private async handleMessage(data: unknown): Promise<void> {

    // Bun may deliver as Blob — convert to ArrayBuffer
    if (data instanceof Blob) {
      data = await data.arrayBuffer();
    }
    if (data instanceof ArrayBuffer) {
      const view = new Uint8Array(data);
      if (view.length === 0) return;

      const msgType = view[0];
      const payload = view.slice(1);

      try {
        switch (msgType) {
          case MSG_TYPE.HANDSHAKE: {
            logger.info({ payloadSize: payload.length, cachedOggPages: this.oggHeaderCache.length }, 'Received handshake from PersonaPlex');
            this.sendHandshake();
            this._handshakeComplete = true;
            // Replay any cached Ogg headers BEFORE signaling connected
            // (so PersonaPlex's opus decoder is initialized before live audio flows)
            this.replayOggHeaders();
            // Signal that PersonaPlex is truly ready for audio
            try { this.events.onStateChange?.('connected'); } catch (e) { logger.error({ err: e }, 'onStateChange handler error'); }
            break;
          }
          case MSG_TYPE.TEXT: {
            const text = new TextDecoder().decode(payload);
            this.currentTurnText += text;
            try { this.events.onText?.(text); } catch (e) { logger.error({ err: e }, 'onText handler error'); }

            // Reset turn-complete timer (350ms of silence = turn complete)
            if (this.turnTimer) clearTimeout(this.turnTimer);
            this.turnTimer = setTimeout(() => {
              if (this.currentTurnText.trim()) {
                try { this.events.onTurnComplete?.(this.currentTurnText.trim()); } catch (e) { logger.error({ err: e }, 'onTurnComplete handler error'); }
                this.currentTurnText = '';
              }
            }, 350);
            break;
          }
          case MSG_TYPE.AUDIO: {
            try { this.events.onAudio?.(payload); } catch (e) { logger.error({ err: e }, 'onAudio handler error'); }
            break;
          }
          default:
            logger.debug({ msgType, size: payload.length }, 'Unknown message type from PersonaPlex');
        }
      } catch (err) {
        logger.error({ err }, 'Error in handleMessage switch');
      }
    } else if (typeof data === 'string') {
      // Some PersonaPlex versions send JSON text frames
      try {
        const parsed = JSON.parse(data);
        if (parsed.text) {
          this.currentTurnText += parsed.text;
          this.events.onText?.(parsed.text);
        }
      } catch {
        // Plain text frame
        this.currentTurnText += data;
        this.events.onText?.(data);
      }
    }
  }

  /** Schedule reconnection with exponential backoff. */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      logger.error('Max reconnection attempts reached');
      return;
    }

    const delay = this.reconnectDelayMs * Math.pow(1.5, this.reconnectAttempts);
    this.reconnectAttempts++;
    logger.info({ attempt: this.reconnectAttempts, delayMs: delay }, 'Scheduling reconnect');

    setTimeout(() => this.connect(), delay);
  }

  /** Disconnect from PersonaPlex. */
  disconnect(): void {
    if (this.turnTimer) clearTimeout(this.turnTimer);
    if (this.ws) {
      this.ws.close(1000, 'Voice bridge shutting down');
      this.ws = null;
    }
  }

  /** Check if connected. */
  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
