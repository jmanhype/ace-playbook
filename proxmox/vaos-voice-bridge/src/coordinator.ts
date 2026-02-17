/**
 * Coordinator: decides whether the Talker handles a turn (System 1)
 * or waits for the Reasoner (System 2 override).
 *
 * Based on "Agents Thinking Fast and Slow" (arXiv:2410.08328):
 * - System 1 (Talker/PersonaPlex): fast, intuitive, always-on
 * - System 2 (Reasoner/Letta+Claude): slow, deliberate, plans and forms beliefs
 */

import type { BeliefState } from './belief.js';
import { createLogger } from './logger.js';

const logger = createLogger('coordinator');

export type RoutingMode = 'direct' | 'agent';

/** Keywords that force System 2 (Reasoner) engagement */
const AGENT_TRIGGERS = [
  'build me', 'build a', 'create a project', 'start a mission',
  'execute', 'deploy', 'analyze my', 'what have we',
  'remember when', 'plan for', 'design a', 'implement',
  'run the pipeline', 'trigger', 'ops loop', 'ops-loop',
];

/** Keywords that indicate frustration / Reasoner needed */
const FRUSTRATION_SIGNALS = [
  'i already told you', 'i said', 'no that\'s wrong',
  'you\'re not listening', 'pay attention',
];

export function detectComplexity(text: string): number {
  const lower = text.toLowerCase().trim();
  let score = 0;

  for (const trigger of AGENT_TRIGGERS) {
    if (lower.includes(trigger)) {
      score += 0.6;
      break; // One trigger is enough for System 2
    }
  }

  for (const signal of FRUSTRATION_SIGNALS) {
    if (lower.includes(signal)) {
      score += 0.3;
    }
  }

  // Length heuristic — longer requests tend to need more reasoning
  const words = lower.split(/\s+/).length;
  if (words > 25) score += 0.2;
  if (words > 50) score += 0.1;

  // Question complexity
  if (/\b(why|how|compare|evaluate|assess)\b/.test(lower) && lower.includes('?')) {
    score += 0.15;
  }

  return Math.min(score, 1.0);
}

export function route(text: string, belief: BeliefState, threshold: number): RoutingMode {
  // If the conversation is in planning/action phase, always use Reasoner
  if (belief.conversation.phase === 'planning' || belief.conversation.phase === 'action') {
    logger.info({ phase: belief.conversation.phase }, 'System 2 override: belief phase requires Reasoner');
    return 'agent';
  }

  const complexity = detectComplexity(text);
  const mode = complexity >= threshold ? 'agent' : 'direct';

  logger.debug({ text: text.slice(0, 60), complexity, threshold, mode }, 'Routing decision');
  return mode;
}

/**
 * Evaluate whether the conversation phase should transition.
 * Called after each Reasoner interaction to cycle phases appropriately.
 *
 *   understanding → planning (when goals detected)
 *   planning → action (when plan executed/mission submitted)
 *   action → understanding (when actions complete or new topic starts)
 */
export function evaluatePhaseTransition(belief: BeliefState): BeliefState['conversation']['phase'] | null {
  const { phase, turnsInPhase } = belief.conversation;
  const hasGoals = belief.userModel.goals.length > 0;
  const hasProject = belief.userModel.currentProject !== null;
  const hasPendingActions = belief.pendingActions.length > 0;

  switch (phase) {
    case 'understanding':
      // Move to planning when user has expressed goals
      if (hasGoals && turnsInPhase >= 2) return 'planning';
      break;

    case 'planning':
      // Move to action when missions are submitted
      if (hasPendingActions) return 'action';
      // Fall back to understanding if planning stalls
      if (turnsInPhase > 5) return 'understanding';
      break;

    case 'action':
      // Return to understanding when actions are done or topic shifts
      if (!hasPendingActions && turnsInPhase >= 2) return 'understanding';
      // Also reset if stuck in action too long
      if (turnsInPhase > 8) return 'understanding';
      break;
  }

  return null; // No transition
}
