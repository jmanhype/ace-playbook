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
  /** Timestamp when handshake completed — used to detect instant disconnects. */
  private _handshakeTime = 0;
  /** When true, suppress auto-reconnect on close. Set by disconnect(). */
  private _intentionalDisconnect = false;
  // NOTE: No Ogg header caching. PersonaPlex's sphn.OpusStreamReader expects
  // raw Opus packets, NOT Ogg page containers. Sending Ogg pages crashes the
  // server's recv_loop with "ValueError: sending on a closed channel".

  /** Text drought detection — auto-reconnect when PersonaPlex goes silent. */
  private lastTextTime = 0;
  private droughtCheckTimer: ReturnType<typeof setInterval> | null = null;
  /** Seconds without text before triggering reconnect (while audio still flows). */
  private static readonly TEXT_DROUGHT_SECS = 45;

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
    // Guard against duplicate connects (e.g., reconnect races)
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      logger.warn('Connect called while already connected/connecting — skipping');
      return;
    }
    // Stop any existing drought timer from previous connection
    this.stopDroughtCheck();

    const env = getEnv();
    // PersonaPlex requires text_prompt and voice_prompt as query params
    const prompt = encodeURIComponent(this.textPrompt || 'You are a helpful voice assistant. Be concise and natural.');
    const voicePrompt = encodeURIComponent(env.VOICE_PROMPT_PATH);
    const url = `wss://${env.PERSONAPLEX_HOST}:${env.PERSONAPLEX_PORT}${env.PERSONAPLEX_WS_PATH}?text_prompt=${prompt}&voice_prompt=${voicePrompt}`;

    logger.info({ url }, 'Connecting to PersonaPlex');
    this._handshakeComplete = false;
    this._sendCount = 0;
    this.events.onStateChange?.('connecting');

    try {
      this.ws = new WebSocket(url);
      this.ws.binaryType = 'arraybuffer';

      this.ws.addEventListener('open', () => {
        logger.info('WebSocket open to PersonaPlex (awaiting handshake)');
        // NOTE: Don't reset reconnectAttempts here — PersonaPlex may accept the
        // WS connection but immediately disconnect after handshake (single-session lock).
        // Reset only after a SUCCESSFUL handshake (in handleMessage → HANDSHAKE case).
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
    logger.info({ promptLength: this.textPrompt.length }, 'Reconnecting PersonaPlex with updated prompt');
    this.disconnect(); // sets _intentionalDisconnect = true, ws.close()
    // Wait for PersonaPlex to fully release its single-session lock.
    // 500ms is too fast — PersonaPlex closes the new connection immediately
    // (code 1000) if the old session hasn't fully cleaned up.
    await new Promise(resolve => setTimeout(resolve, 2000));
    this._intentionalDisconnect = false;
    await this.connect();
  }

  private _sendCount = 0;
  private _recvCount = 0;



  /** Forward user audio to PersonaPlex (adds 0x01 prefix). */
  sendAudio(data: ArrayBuffer): void {
    if (!this._handshakeComplete) return; // Drop audio until handshake done

    if (this.ws?.readyState === WebSocket.OPEN) {
      const audioBytes = new Uint8Array(data);
      const frame = new Uint8Array(1 + audioBytes.length);
      frame[0] = MSG_TYPE.AUDIO;
      frame.set(audioBytes, 1);
      this._sendCount++;
      this.ws.send(frame);
    }
  }

  // NOTE: No handshake response needed. PersonaPlex server only expects audio
  // frames (kind=1) from the client. Sending 0x00 causes "unknown kind 0" warning.

  /** Handle incoming PersonaPlex messages. */
  private async handleMessage(data: unknown): Promise<void> {
    this._recvCount = (this._recvCount ?? 0) + 1;
    if (this._recvCount <= 5 || this._recvCount % 100 === 0) {
      logger.debug({ recvCount: this._recvCount, dataType: typeof data, isBlob: data instanceof Blob, isAB: data instanceof ArrayBuffer, isBuffer: Buffer.isBuffer(data) }, 'handleMessage received');
    }

    // Bun may deliver as Blob — convert to ArrayBuffer
    if (data instanceof Blob) {
      data = await data.arrayBuffer();
    }
    // Bun may also deliver as Buffer
    if (Buffer.isBuffer(data)) {
      data = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    }
    if (data instanceof ArrayBuffer) {
      const view = new Uint8Array(data);
      if (view.length === 0) return;

      const msgType = view[0];
      const payload = view.slice(1);

      // Log non-audio message types (text=0x02, handshake=0x01, control=0x04+)
      if (msgType !== MSG_TYPE.AUDIO) {
        logger.debug({ msgType, payloadSize: payload.length }, 'Non-audio PersonaPlex message');
      }

      try {
        switch (msgType) {
          case MSG_TYPE.HANDSHAKE: {
            logger.info({ payloadSize: payload.length }, 'Received handshake from PersonaPlex — ready for audio');
            // Don't send anything back — server only accepts audio (kind=1).
            this._handshakeComplete = true;
            this._handshakeTime = Date.now();
            this.reconnectAttempts = 0;
            this.lastTextTime = Date.now();
            this.startDroughtCheck();
            try { this.events.onStateChange?.('connected'); } catch (e) { logger.error({ err: e }, 'onStateChange handler error'); }
            break;
          }
          case MSG_TYPE.TEXT: {
            const text = new TextDecoder().decode(payload);
            this.currentTurnText += text;
            this.lastTextTime = Date.now();
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
    if (this._intentionalDisconnect) {
      logger.debug('Skipping reconnect (intentional disconnect)');
      return;
    }
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      logger.error('Max reconnection attempts reached — giving up (restart bridge to retry)');
      return;
    }

    // Detect instant-disconnect pattern: PersonaPlex closes <500ms after handshake.
    // This means its single-session lock hasn't cleared. Use longer base delay.
    const instantDisconnect = this._handshakeTime > 0 && (Date.now() - this._handshakeTime) < 500;
    const baseDelay = instantDisconnect ? 5000 : this.reconnectDelayMs;
    const delay = baseDelay * Math.pow(1.5, this.reconnectAttempts);
    this.reconnectAttempts++;
    logger.info({ attempt: this.reconnectAttempts, delayMs: Math.round(delay), instantDisconnect }, 'Scheduling reconnect');

    setTimeout(() => this.connect(), delay);
  }

  /** Disconnect from PersonaPlex. Suppresses auto-reconnect. */
  disconnect(): void {
    this._intentionalDisconnect = true;
    this.stopDroughtCheck();
    if (this.turnTimer) clearTimeout(this.turnTimer);
    if (this.ws) {
      this.ws.close(1000, 'Voice bridge shutting down');
      this.ws = null;
    }
  }

  // ─── Text Drought Detection ─────────────────────────────────

  /** Start periodic check for PersonaPlex text silence. */
  private startDroughtCheck(): void {
    this.stopDroughtCheck();
    this.droughtCheckTimer = setInterval(() => {
      if (!this._handshakeComplete || !this.connected) return;
      const silenceSecs = (Date.now() - this.lastTextTime) / 1000;
      if (silenceSecs >= Talker.TEXT_DROUGHT_SECS) {
        logger.warn({ silenceSecs: Math.round(silenceSecs) }, 'PersonaPlex text drought detected — auto-reconnecting');
        this.stopDroughtCheck();
        // Reconnect with current prompt (non-intentional, so scheduleReconnect would work,
        // but we do an explicit reconnect to use the latest text_prompt)
        this._intentionalDisconnect = false;
        if (this.ws) {
          this.ws.close(1000, 'Text drought auto-reconnect');
          this.ws = null;
        }
        // Small delay then reconnect
        setTimeout(() => this.connect(), 1000);
      }
    }, 10_000); // Check every 10 seconds
  }

  /** Stop the drought check timer. */
  private stopDroughtCheck(): void {
    if (this.droughtCheckTimer) {
      clearInterval(this.droughtCheckTimer);
      this.droughtCheckTimer = null;
    }
  }

  /** Check if connected. */
  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
