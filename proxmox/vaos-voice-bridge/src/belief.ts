/**
 * Belief State — Shared memory between Talker (System 1) and Reasoner (System 2).
 *
 * Stored in Letta memory blocks and read/written by both systems.
 * Based on DeepMind "Agents Thinking Fast and Slow" (arXiv:2410.08328).
 */

import { z } from 'zod';

export const BeliefSchema = z.object({
  user_model: z.object({
    goals: z.array(z.string()).default([]),
    current_project: z.string().nullable().default(null),
    preferences: z.object({
      voice: z.string().default('NATF0'),
      verbosity: z.enum(['concise', 'detailed', 'verbose']).default('concise'),
    }).default({}),
    barriers: z.array(z.string()).default([]),
    expertise_level: z.enum(['beginner', 'intermediate', 'advanced']).default('advanced'),
  }).default({}),
  conversation: z.object({
    phase: z.enum(['understanding', 'planning', 'action', 'reflection']).default('understanding'),
    topic: z.string().nullable().default(null),
    turns_in_phase: z.number().default(0),
    summary: z.string().default(''),
  }).default({}),
  pending_actions: z.array(z.object({
    type: z.string(),
    description: z.string(),
    status: z.enum(['pending', 'running', 'completed', 'failed']).default('pending'),
    created_at: z.string(),
  })).default([]),
  last_reasoner_update: z.string().nullable().default(null),
});

export type Belief = z.infer<typeof BeliefSchema>;

/** Create an empty default belief state. */
export function createDefaultBelief(): Belief {
  return BeliefSchema.parse({});
}

/**
 * Generate a dynamic text_prompt for PersonaPlex from the current belief state.
 * This is how System 2's knowledge enriches System 1's fast responses.
 */
export function beliefToPrompt(belief: Belief): string {
  const parts: string[] = [
    'You are a voice assistant. Be natural and conversational.',
  ];

  if (belief.user_model.current_project) {
    parts.push(`The user is working on: ${belief.user_model.current_project}.`);
  }

  if (belief.user_model.goals.length > 0) {
    parts.push(`Their goals are: ${belief.user_model.goals.join(', ')}.`);
  }

  if (belief.conversation.topic) {
    parts.push(`Current topic: ${belief.conversation.topic}.`);
  }

  if (belief.conversation.summary) {
    parts.push(`Context: ${belief.conversation.summary}`);
  }

  if (belief.pending_actions.length > 0) {
    const active = belief.pending_actions.filter(a => a.status === 'running');
    if (active.length > 0) {
      parts.push(`Active tasks: ${active.map(a => a.description).join(', ')}.`);
    }
  }

  parts.push(`Keep responses ${belief.user_model.preferences.verbosity}.`);

  return parts.join(' ');
}

/**
 * Merge partial belief updates into existing belief state.
 * Used by the Reasoner to incrementally update beliefs.
 */
export function mergeBelief(current: Belief, patch: Partial<Belief>): Belief {
  const merged = {
    ...current,
    ...patch,
    user_model: {
      ...current.user_model,
      ...(patch.user_model ?? {}),
      preferences: {
        ...current.user_model.preferences,
        ...(patch.user_model?.preferences ?? {}),
      },
    },
    conversation: {
      ...current.conversation,
      ...(patch.conversation ?? {}),
    },
    last_reasoner_update: new Date().toISOString(),
  };

  // Validate the merged result
  return BeliefSchema.parse(merged);
}
