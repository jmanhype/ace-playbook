/**
 * Trigger — Semantic gate between System 1 (PersonaPlex) and System 2 (Reasoner).
 *
 * Inspired by LTS-VoiceAgent's Dynamic Semantic Trigger:
 *   - Detects when PersonaPlex (System 1) is failing to handle a request
 *   - Based on deflection patterns, action keywords, and confusion spirals
 *   - Emits trigger.activate events with confidence scores and reasons
 *
 * Trigger modes:
 *   'deflection'   — PersonaPlex admitting it can't do something
 *   'semantic'     — Action keywords detected in PersonaPlex output
 *   'periodic'     — Every N turns for proactive Reasoner evaluation
 *   'user_request' — Explicit user text requesting System 2 action
 *
 * Subscribes to: talker.turn
 * Publishes: trigger.activate
 */

import { createLogger } from './lib/logger.js';
import { type EventBus, E, type TalkerTurnEvent } from './events.js';

const logger = createLogger('trigger');

// ─── Pattern Definitions ────────────────────────────────────────

/** PersonaPlex admitting it can't do something. */
const DEFLECTION_PATTERNS = [
  "i can't do that", "i'm not able to", "i can't search", "i can't access",
  "i don't have access", "i can't make", "i can't check", "i can only share",
  "i can't really", "i'm unable to", "that's not something i can",
  "i don't have the ability", "beyond my capabilities",
  "not sure what you mean", "i don't think i follow", "i'm not sure what",
  "i don't understand", "could you clarify", "what do you mean by",
  "i'm confused", "i didn't catch that",
  "can we stick to", "let's focus on", "let's talk about something",
];

/** Action keywords suggesting the user wants a tool-augmented response. */
const ACTION_KEYWORDS = [
  'search', 'web search', 'look up', 'look it up',
  'memory', 'core memory', 'remember', 'what do you remember',
  'check', 'check on', 'status',
  'create a video', 'make a video', 'generate',
  'run code', 'execute', 'build', 'deploy',
  'ask the director', 'ask the writer', 'tell the director',
  'send a message',
];

/** User explicitly requesting System 2 reasoning. */
const USER_REQUEST_PATTERNS = [
  'can you actually', 'go ahead and', "what's the plan",
  'help me figure', 'i need you to', 'can you help me with',
  'think about this', 'what do you think',
  'figure out', 'make me', 'write me', 'analyze',
];

/** Social turns — greetings, fillers, small talk. No System 2 needed. */
const SOCIAL_PATTERNS = [
  /^(hi|hey|hello|yo|sup|what'?s up|howdy)\b/,
  /^(good morning|good afternoon|good evening|good night)\b/,
  /^(thanks|thank you|cool|ok|okay|sure|right|yeah|yep|nope|no)\b/,
  /^(bye|goodbye|see you|later|gotta go)\b/,
];

/** Frustration — highest priority trigger. */
const FRUSTRATION_PATTERNS = [
  'i already told you', 'i just said', 'are you listening',
  'pay attention', 'like i mentioned',
];

// ─── Trigger Class ──────────────────────────────────────────────

export interface TriggerConfig {
  /** Turns between proactive Reasoner evaluations (default 6). */
  proactiveInterval?: number;
  /** Minimum confidence to fire (0-1, default 0.5). */
  confidenceThreshold?: number;
  /** Number of recent turns to examine. */
  windowSize?: number;
}

export class Trigger {
  private bus: EventBus;
  private sessionId: string;
  private recentTurns: string[] = [];
  private turnCounter = 0;
  private config: Required<TriggerConfig>;

  /** Set by Reasoner when System 2 is actively processing. */
  private _system2Active = false;

  constructor(bus: EventBus, sessionId: string, config?: TriggerConfig) {
    this.bus = bus;
    this.sessionId = sessionId;
    this.config = {
      proactiveInterval: config?.proactiveInterval ?? 6,
      confidenceThreshold: config?.confidenceThreshold ?? 0.5,
      windowSize: config?.windowSize ?? 5,
    };

    // Subscribe to talker.turn at high priority (before Reasoner's handler)
    bus.on('talker.turn', (event) => this.evaluateTurn(event), 90);
  }

  get system2Active(): boolean { return this._system2Active; }
  set system2Active(v: boolean) { this._system2Active = v; }

  /** Reset turn history (after System 2 processes a batch). */
  resetHistory(): void {
    this.recentTurns = [];
  }

  // ─── Turn Evaluation (PersonaPlex output) ────────────────────

  private evaluateTurn(event: TalkerTurnEvent): void {
    this.turnCounter++;
    this.recentTurns.push(event.text);
    if (this.recentTurns.length > this.config.windowSize) {
      this.recentTurns.shift();
    }

    // Skip if System 2 already processing
    if (this._system2Active) {
      logger.debug('System 2 active — skipping trigger evaluation');
      return;
    }

    // 1. Periodic proactive check
    if (this.turnCounter % this.config.proactiveInterval === 0) {
      const context = this.recentTurns.join(' ').trim().slice(-1000);
      logger.info({ turnCounter: this.turnCounter }, 'Periodic proactive trigger');
      this.bus.emit(E.triggerActivate(this.sessionId, 'periodic', 0.7, context));
      return;
    }

    // 2. Pattern-based detection
    const result = this.detectPatterns();
    if (result && result.confidence >= this.config.confidenceThreshold) {
      const context = this.recentTurns.join(' ').trim().slice(-1000);
      logger.info({ reason: result.reason, confidence: result.confidence }, 'Trigger activated');
      this.bus.emit(E.triggerActivate(this.sessionId, result.reason, result.confidence, context));
    }
  }

  /** Pattern matching across the recent turn window. */
  private detectPatterns(): { reason: 'deflection' | 'semantic'; confidence: number } | null {
    if (this.recentTurns.length < 2) return null;

    const window = this.recentTurns.slice(-this.config.windowSize);
    const recentText = window.join(' ').toLowerCase();

    // Count deflections
    let deflectionCount = 0;
    for (const turn of window) {
      const lower = turn.toLowerCase();
      if (DEFLECTION_PATTERNS.some(p => lower.includes(p))) {
        deflectionCount++;
      }
    }

    // Count action keywords
    const actionCount = ACTION_KEYWORDS.filter(k => recentText.includes(k)).length;

    // Deflection + action keyword → strong trigger
    if (deflectionCount >= 1 && actionCount >= 1) {
      return { reason: 'deflection', confidence: 0.85 };
    }

    // Repeated deflection (confusion spiral)
    if (deflectionCount >= 2) {
      return { reason: 'deflection', confidence: 0.75 };
    }

    // Dense action keywords without deflection
    if (actionCount >= 2) {
      return { reason: 'semantic', confidence: 0.6 };
    }

    return null;
  }

  // ─── User Text Evaluation (typed input) ──────────────────────

  /**
   * Evaluate user text input (typed, not from PersonaPlex).
   * Returns true if trigger.activate was emitted.
   */
  evaluateUserText(text: string): boolean {
    if (this._system2Active) return false;

    const lower = text.toLowerCase().trim();
    const words = lower.split(/\s+/);

    // Social / short — no trigger
    if (words.length <= 2) return false;
    if (SOCIAL_PATTERNS.some(p => p.test(lower))) return false;

    // Frustration — highest priority
    if (FRUSTRATION_PATTERNS.some(p => lower.includes(p))) {
      this.bus.emit(E.triggerActivate(this.sessionId, 'user_request', 0.95, text));
      return true;
    }

    // User explicitly requesting action
    if (USER_REQUEST_PATTERNS.some(p => lower.includes(p))) {
      this.bus.emit(E.triggerActivate(this.sessionId, 'user_request', 0.9, text));
      return true;
    }

    // Action keywords in imperative context
    if (ACTION_KEYWORDS.some(k => lower.includes(k))) {
      // Skip past tense (describing, not requesting)
      if (/\b(i\s+\w+ed|i\s+built|already\s+\w+ed|have\s+\w+ed)\b/.test(lower)) {
        return false;
      }
      this.bus.emit(E.triggerActivate(this.sessionId, 'semantic', 0.7, text));
      return true;
    }

    return false;
  }

  /** Check if text is a social turn (no Reasoner needed). */
  static isSocialTurn(text: string): boolean {
    const lower = text.toLowerCase().trim();
    if (lower.split(/\s+/).length > 4) return false;
    return SOCIAL_PATTERNS.some(p => p.test(lower));
  }
}
