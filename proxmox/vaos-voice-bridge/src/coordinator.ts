/**
 * Coordinator — System 1 vs System 2 decision logic.
 *
 * Decides whether the Talker (PersonaPlex) should respond immediately
 * or wait for the Reasoner to process the turn first.
 *
 * Criteria:
 * 1. Action triggers — "build", "create", "plan", etc.
 * 2. Phrase triggers — "can you actually", "go ahead and"
 * 3. Belief phase override — planning/action phases
 * 4. Frustration detection — "I already told you"
 */

import { createLogger } from './lib/logger.js';
import type { Belief } from './belief.js';

const logger = createLogger('coordinator');

export interface Decision {
  waitForReasoner: boolean;
  reason: string;
  trigger?: string;
}

// Triggers that indicate the user wants action (System 2 override)
const ACTION_TRIGGERS = [
  'build', 'create', 'execute', 'plan', 'deploy', 'launch',
  'implement', 'design', 'analyze', 'research', 'set up',
  'figure out', 'make me', 'write me',
];

// Multi-word phrases that override to System 2
const PHRASE_TRIGGERS = [
  'can you actually',
  'go ahead and',
  "what's the plan",
  'what do you think about',
  'help me figure',
  'i need you to',
  'can you help me with',
];

// Belief phases that force System 2 override
const OVERRIDE_PHASES = new Set(['planning', 'action']);

// Frustration patterns (highest priority)
const FRUSTRATION_PATTERNS = [
  'i already told you',
  'i just said',
  'are you listening',
  'pay attention',
  'i said',
  'like i mentioned',
];

// Past-tense exclusions: if the keyword appears in past tense, don't trigger
const PAST_TENSE_PATTERNS = [
  /i\s+\w+ed\b/,     // "i created", "i built"
  /i\s+built\b/,
  /already\s+\w+ed\b/,
  /have\s+\w+ed\b/,
  /had\s+\w+ed\b/,
  /i\s+was\s+\w+ing\b/,
];

export function decide(text: string, belief: Belief): Decision {
  const lower = text.toLowerCase().trim();
  const words = lower.split(/\s+/);

  // Skip very short utterances (greetings, fillers)
  if (words.length <= 2) {
    return { waitForReasoner: false, reason: 'short_utterance' };
  }

  // 1. Frustration detection (highest priority)
  for (const pattern of FRUSTRATION_PATTERNS) {
    if (lower.includes(pattern)) {
      logger.info({ trigger: pattern }, 'Frustration detected → System 2');
      return { waitForReasoner: true, reason: 'frustration', trigger: pattern };
    }
  }

  // 2. Belief phase override
  if (OVERRIDE_PHASES.has(belief.conversation.phase)) {
    return {
      waitForReasoner: true,
      reason: 'belief_phase',
      trigger: belief.conversation.phase,
    };
  }

  // 3. Phrase triggers (multi-word, more specific)
  for (const phrase of PHRASE_TRIGGERS) {
    if (lower.includes(phrase)) {
      logger.info({ trigger: phrase }, 'Phrase trigger → System 2');
      return { waitForReasoner: true, reason: 'phrase_trigger', trigger: phrase };
    }
  }

  // 4. Action keyword triggers (only in imperative context)
  for (const trigger of ACTION_TRIGGERS) {
    if (lower.includes(trigger)) {
      // Check it's not past tense / descriptive
      const isPastTense = PAST_TENSE_PATTERNS.some(p => p.test(lower));
      if (!isPastTense) {
        logger.info({ trigger }, 'Action trigger → System 2');
        return { waitForReasoner: true, reason: 'action_trigger', trigger };
      }
    }
  }

  // Default: System 1 handles it (fast path)
  return { waitForReasoner: false, reason: 'no_trigger' };
}

// Social turn detection — greetings, fillers, small talk
const SOCIAL_PATTERNS = [
  /^(hi|hey|hello|yo|sup|what'?s up|howdy)\b/,
  /^(good morning|good afternoon|good evening|good night)\b/,
  /^(thanks|thank you|cool|ok|okay|sure|right|yeah|yep|nope|no)\b/,
  /^(bye|goodbye|see you|later|gotta go)\b/,
];

export function isSocialTurn(text: string): boolean {
  const lower = text.toLowerCase().trim();
  const words = lower.split(/\s+/);
  if (words.length > 4) return false;
  return SOCIAL_PATTERNS.some(p => p.test(lower));
}

export function shouldWaitForReasoner(text: string, belief: Belief): Decision {
  return decide(text, belief);
}
