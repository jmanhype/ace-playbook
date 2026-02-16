/**
 * Event Bus — X-Talk-inspired pub-sub for the voice pipeline.
 *
 * Adapted from X-Talk's EventBus (xcc-zach/xtalk):
 *   - Priority-sorted handlers (higher priority executes first)
 *   - Async handler spawning with error isolation
 *   - Dotted namespace event types (e.g., 'talker.turn', 'reasoner.interjection')
 *   - Session-scoped (one bus per voice session)
 *
 * Component communication flow:
 *   PersonaPlex → talker.* events
 *   Trigger     → subscribes talker.turn, publishes trigger.activate
 *   Reasoner    → subscribes trigger.activate, publishes reasoner.*
 *   Memory      → subscribes reasoner.belief, publishes memory.update
 *   Bridge      → subscribes *, forwards to browser WebSocket
 */

import { createLogger } from './lib/logger.js';

const logger = createLogger('events');

// ─── Event Definitions ──────────────────────────────────────────

export interface BaseEvent {
  type: string;
  sessionId: string;
  timestamp: number;
}

function ts(): number { return Date.now(); }

// --- Talker (PersonaPlex / System 1) events ---

export interface TalkerTextEvent extends BaseEvent {
  type: 'talker.text';
  text: string;
}

export interface TalkerTurnEvent extends BaseEvent {
  type: 'talker.turn';
  text: string;
  turnNumber: number;
}

export interface TalkerAudioEvent extends BaseEvent {
  type: 'talker.audio';
  data: Uint8Array;
}

export interface TalkerStateEvent extends BaseEvent {
  type: 'talker.state';
  state: 'connecting' | 'connected' | 'disconnected';
}

// --- Trigger (Semantic Gate) events ---

export interface TriggerActivateEvent extends BaseEvent {
  type: 'trigger.activate';
  reason: 'semantic' | 'deflection' | 'periodic' | 'user_request';
  confidence: number;
  /** Recent PersonaPlex text that triggered activation. */
  context: string;
}

// --- Reasoner (System 2) events ---

export interface ReasonerInterjectionEvent extends BaseEvent {
  type: 'reasoner.interjection';
  text: string;
  decision: string;
}

export interface ReasonerBeliefEvent extends BaseEvent {
  type: 'reasoner.belief';
  phase: string;
  /** Which memory blocks were updated. */
  blocks: string[];
}

export interface ReasonerThinkingEvent extends BaseEvent {
  type: 'reasoner.thinking';
  active: boolean;
}

// --- Memory events ---

export interface MemoryUpdateEvent extends BaseEvent {
  type: 'memory.update';
  block: string;
  source: 'api' | 'agent';
}

export interface MemoryCompressedEvent extends BaseEvent {
  type: 'memory.compressed';
  /** Compressed prompt for PersonaPlex. */
  prompt: string;
  tokenEstimate: number;
}

// --- User (Browser) events ---

export interface UserTextEvent extends BaseEvent {
  type: 'user.text';
  text: string;
}

export interface UserAudioEvent extends BaseEvent {
  type: 'user.audio';
  data: ArrayBuffer;
}

// --- Error events ---

export interface ErrorEvent extends BaseEvent {
  type: 'error.occurred';
  source: string;
  error: string;
}

// Union of all events
export type BusEvent =
  | TalkerTextEvent | TalkerTurnEvent | TalkerAudioEvent | TalkerStateEvent
  | TriggerActivateEvent
  | ReasonerInterjectionEvent | ReasonerBeliefEvent | ReasonerThinkingEvent
  | MemoryUpdateEvent | MemoryCompressedEvent
  | UserTextEvent | UserAudioEvent
  | ErrorEvent;

export type EventType = BusEvent['type'];

// Extract event by type string
type EventOfType<T extends EventType> = Extract<BusEvent, { type: T }>;
type Handler<T extends EventType> = (event: EventOfType<T>) => void | Promise<void>;

// ─── Event Factories ────────────────────────────────────────────

/** Helper to create events with auto-timestamp. */
export const E = {
  talkerText: (sessionId: string, text: string): TalkerTextEvent =>
    ({ type: 'talker.text', sessionId, text, timestamp: ts() }),

  talkerTurn: (sessionId: string, text: string, turnNumber: number): TalkerTurnEvent =>
    ({ type: 'talker.turn', sessionId, text, turnNumber, timestamp: ts() }),

  talkerAudio: (sessionId: string, data: Uint8Array): TalkerAudioEvent =>
    ({ type: 'talker.audio', sessionId, data, timestamp: ts() }),

  talkerState: (sessionId: string, state: TalkerStateEvent['state']): TalkerStateEvent =>
    ({ type: 'talker.state', sessionId, state, timestamp: ts() }),

  triggerActivate: (sessionId: string, reason: TriggerActivateEvent['reason'], confidence: number, context: string): TriggerActivateEvent =>
    ({ type: 'trigger.activate', sessionId, reason, confidence, context, timestamp: ts() }),

  reasonerInterjection: (sessionId: string, text: string, decision: string): ReasonerInterjectionEvent =>
    ({ type: 'reasoner.interjection', sessionId, text, decision, timestamp: ts() }),

  reasonerBelief: (sessionId: string, phase: string, blocks: string[]): ReasonerBeliefEvent =>
    ({ type: 'reasoner.belief', sessionId, phase, blocks, timestamp: ts() }),

  reasonerThinking: (sessionId: string, active: boolean): ReasonerThinkingEvent =>
    ({ type: 'reasoner.thinking', sessionId, active, timestamp: ts() }),

  memoryUpdate: (sessionId: string, block: string, source: 'api' | 'agent'): MemoryUpdateEvent =>
    ({ type: 'memory.update', sessionId, block, source, timestamp: ts() }),

  memoryCompressed: (sessionId: string, prompt: string, tokenEstimate: number): MemoryCompressedEvent =>
    ({ type: 'memory.compressed', sessionId, prompt, tokenEstimate, timestamp: ts() }),

  userText: (sessionId: string, text: string): UserTextEvent =>
    ({ type: 'user.text', sessionId, text, timestamp: ts() }),

  userAudio: (sessionId: string, data: ArrayBuffer): UserAudioEvent =>
    ({ type: 'user.audio', sessionId, data, timestamp: ts() }),

  error: (sessionId: string, source: string, error: string): ErrorEvent =>
    ({ type: 'error.occurred', sessionId, source, error, timestamp: ts() }),
};

// ─── Event Bus ──────────────────────────────────────────────────

interface RegisteredHandler {
  handler: Handler<any>;
  priority: number;
}

export class EventBus {
  private handlers = new Map<string, RegisteredHandler[]>();
  private eventCount = 0;
  private errorCount = 0;
  /** Prevent infinite recursion on error.occurred handlers. */
  private publishingError = false;

  /**
   * Subscribe to an event type with optional priority.
   * Higher priority handlers execute first (default 50).
   * Returns unsubscribe function.
   *
   * X-Talk convention: ASR=100, LLM=98, TTS=95, Output=5
   */
  on<T extends EventType>(type: T, handler: Handler<T>, priority = 50): () => void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, []);
    }
    const list = this.handlers.get(type)!;
    list.push({ handler, priority });
    // Sort descending by priority (highest first)
    list.sort((a, b) => b.priority - a.priority);

    return () => {
      const idx = list.findIndex(h => h.handler === handler);
      if (idx >= 0) list.splice(idx, 1);
    };
  }

  /**
   * Publish an event. All handlers are invoked concurrently (fire-and-forget).
   * Errors are caught and published as error.occurred events.
   *
   * Set waitForCompletion=true for ordering-sensitive transitions
   * (X-Talk uses this for ASR final → verification → LLM stop).
   */
  async emit(event: BusEvent, waitForCompletion = false): Promise<void> {
    this.eventCount++;
    const list = this.handlers.get(event.type);
    if (!list || list.length === 0) return;

    if (waitForCompletion) {
      // Sequential execution in priority order
      for (const { handler } of list) {
        try {
          await handler(event);
        } catch (err) {
          this.handleError(event, err);
        }
      }
    } else {
      // Concurrent execution (fire-and-forget, X-Talk default)
      for (const { handler } of list) {
        try {
          const result = handler(event);
          if (result instanceof Promise) {
            result.catch(err => this.handleError(event, err));
          }
        } catch (err) {
          this.handleError(event, err);
        }
      }
    }
  }

  private handleError(event: BusEvent, err: unknown): void {
    this.errorCount++;
    const msg = err instanceof Error ? err.message : String(err);
    logger.error({ eventType: event.type, err: msg }, 'Event handler error');

    // Publish error event (with recursion guard, matching X-Talk pattern)
    if (!this.publishingError && event.type !== 'error.occurred') {
      this.publishingError = true;
      this.emit(E.error(event.sessionId, event.type, msg));
      this.publishingError = false;
    }
  }

  get count(): number { return this.eventCount; }
  get errors(): number { return this.errorCount; }

  /** Shutdown: clear all handlers. */
  shutdown(): void {
    this.handlers.clear();
  }
}
