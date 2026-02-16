/**
 * Reasoner — Letta Agent + Claude (System 2), event bus subscriber.
 *
 * The Reasoner is the slow, deliberate thinker. It receives events from the
 * bus and communicates with Letta's REST API for inference.
 *
 * Subscribes to:
 *   trigger.activate → Full Letta processing → emits reasoner.interjection
 *   talker.turn      → Async belief update (skipped if trigger fired) → emits reasoner.belief
 *   user.text        → Routes through Trigger, or direct belief update
 *
 * Publishes:
 *   reasoner.interjection — User-facing speech from System 2
 *   reasoner.belief       — Signal that belief blocks were updated in Letta
 *   reasoner.thinking     — UI indicator for System 2 processing
 *
 * Turn-batching: PersonaPlex produces turns every ~800ms but Letta round-trips
 * take 15-25s. We queue turns while busy and drain the batch when free.
 */

import { createLogger } from './lib/logger.js';
import { getEnv } from './lib/config.js';
import { getSupabase, sbVoid } from './lib/db.js';
import { type EventBus, E } from './events.js';
import type { TalkerTurnEvent, TriggerActivateEvent, UserTextEvent } from './events.js';
import type { Memory } from './memory.js';
import type { Trigger } from './trigger.js';

const logger = createLogger('reasoner');

// ─── Letta Types ────────────────────────────────────────────────

interface LettaAgent {
  id: string;
  name: string;
}

interface LettaMessage {
  role?: string;
  message_type?: string;
  content?: string;
  text?: string;
  tool_call?: { name?: string; arguments?: Record<string, unknown> };
  tool_calls?: Array<{ name: string; arguments: Record<string, unknown> }>;
}

// ─── Reasoner ───────────────────────────────────────────────────

export class Reasoner {
  private bus: EventBus;
  private memory: Memory;
  private trigger: Trigger;
  private sessionId: string;

  private agentId: string | null = null;
  private processing = false;
  private pendingTurns: string[] = [];

  constructor(bus: EventBus, memory: Memory, trigger: Trigger, sessionId: string) {
    this.bus = bus;
    this.memory = memory;
    this.trigger = trigger;
    this.sessionId = sessionId;

    // Subscribe to trigger.activate — full System 2 processing
    bus.on('trigger.activate', (event) => this.handleTrigger(event), 70);

    // Subscribe to talker.turn — async belief update (lower priority than Trigger)
    bus.on('talker.turn', (event) => this.handleTalkerTurn(event), 50);

    // Subscribe to user.text — text input from browser
    bus.on('user.text', (event) => this.handleUserText(event), 50);
  }

  get agentIdValue(): string | null { return this.agentId; }

  /** Initialize: find or create the voice-reasoner agent in Letta. */
  async init(): Promise<void> {
    const env = getEnv();
    const agentName = env.LETTA_AGENT_NAME;

    logger.info({ agentName, lettaUrl: env.LETTA_BASE_URL }, 'Initializing Reasoner');

    try {
      const agents = await this.lettaGet<LettaAgent[]>('/v1/agents');
      const existing = agents.find(a => a.name === agentName);

      if (existing) {
        this.agentId = existing.id;
        logger.info({ agentId: this.agentId }, 'Found existing Letta agent');
      } else {
        const agent = await this.lettaPost<LettaAgent>('/v1/agents', {
          name: agentName,
          model: 'claude-sonnet-4-20250514',
          system: this.buildSystemPrompt(),
          memory: {
            blocks: Object.values((await import('./memory.js')).BLOCK_SPECS).map(spec => ({
              label: spec.label,
              value: spec.defaultValue,
            })),
          },
        });
        this.agentId = agent.id;
        logger.info({ agentId: this.agentId }, 'Created new Letta agent');
      }

      // Initialize Memory with the agent ID (syncs blocks from Letta)
      await this.memory.init(this.agentId);
    } catch (err) {
      logger.error({ err }, 'Failed to initialize Reasoner — falling back to local state');
    }
  }

  // ─── Event Handlers ──────────────────────────────────────────

  /**
   * Handle trigger.activate — full System 2 processing.
   * The Trigger has determined that PersonaPlex can't handle this.
   */
  private async handleTrigger(event: TriggerActivateEvent): Promise<void> {
    this.trigger.system2Active = true;
    this.bus.emit(E.reasonerThinking(this.sessionId, true));

    try {
      const prompt = this.buildTriggerPrompt(event);
      const response = await this.sendToLetta(prompt, 90_000);

      // Extract user-facing response (only send_message tool calls for proactive/periodic)
      const onlySendMessage = event.reason === 'periodic';
      const text = this.extractResponse(response, onlySendMessage);

      if (text) {
        logger.info({ reason: event.reason, text: text.slice(0, 100) }, 'System 2 interjection');
        this.bus.emit(E.reasonerInterjection(this.sessionId, text, event.reason));
      }

      // Signal that belief was updated
      this.bus.emit(E.reasonerBelief(this.sessionId, event.reason, ['belief_state', 'conv_state']));
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        logger.warn({ reason: event.reason }, 'System 2 processing timed out');
      } else {
        logger.error({ err, reason: event.reason }, 'System 2 processing failed');
      }
    } finally {
      this.bus.emit(E.reasonerThinking(this.sessionId, false));
      this.trigger.system2Active = false;
      this.trigger.resetHistory();
    }
  }

  /**
   * Handle talker.turn — async belief update.
   * Skipped if System 2 is already processing (trigger fired).
   * Uses turn-batching to avoid hammering Letta.
   */
  private async handleTalkerTurn(event: TalkerTurnEvent): Promise<void> {
    // Skip if System 2 already handling this turn
    if (this.trigger.system2Active) {
      logger.debug('System 2 active — skipping async belief update');
      return;
    }

    if (this.processing) {
      this.pendingTurns.push(event.text);
      logger.debug({ queued: this.pendingTurns.length }, 'Reasoner busy — turn queued');
      return;
    }

    await this.processBeliefUpdate(event.text);
  }

  /** Handle user.text — text input from browser. */
  private async handleUserText(event: UserTextEvent): Promise<void> {
    // Let the Trigger decide if this needs System 2
    const triggered = this.trigger.evaluateUserText(event.text);
    if (triggered) return; // trigger.activate handler will process it

    // Otherwise, async belief update
    if (!this.processing) {
      await this.processBeliefUpdate(event.text);
    } else {
      this.pendingTurns.push(event.text);
    }
  }

  // ─── Core Processing ─────────────────────────────────────────

  /** Process a belief update cycle (with turn-batching). */
  private async processBeliefUpdate(text: string): Promise<void> {
    this.processing = true;

    try {
      const allTurns = [...this.pendingTurns, text];
      this.pendingTurns = [];
      const batchText = allTurns.join(' ').trim().slice(-1000);

      if (this.agentId) {
        const prompt = `[BELIEF_UPDATE] Recent conversation:\n"${batchText}"\n\nUpdate the belief_state and conv_state memory blocks using core_memory_replace. Update: conversation_topic, conversation_summary, coaching_phase as needed.`;

        await this.sendToLetta(prompt, 60_000);
        this.bus.emit(E.reasonerBelief(this.sessionId, 'belief_update', ['conv_state']));
        logger.info({ batchSize: allTurns.length }, 'Belief updated');
      } else {
        // Local fallback: direct conv_state update
        await this.memory.updateConvState({
          summary: batchText.slice(0, 500),
          turn_count: ((this.memory.getBlockJSON<{ turn_count?: number }>('conv_state'))?.turn_count ?? 0) + allTurns.length,
        });
        this.memory.emitCompressed();
        logger.info('Belief updated (local fallback)');
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        logger.warn('Belief update timed out');
      } else {
        logger.error({ err }, 'Belief update failed');
      }
    } finally {
      this.processing = false;

      // Drain queued turns
      if (this.pendingTurns.length > 0) {
        logger.info({ queued: this.pendingTurns.length }, 'Draining queued turns');
        setTimeout(() => {
          const next = this.pendingTurns.shift() ?? '';
          this.processBeliefUpdate(next).catch(err => {
            logger.warn({ err }, 'Queued belief update failed');
          });
        }, 500);
      }
    }
  }

  /** Send a message to the Letta agent and return the response messages. */
  private async sendToLetta(prompt: string, timeoutMs: number): Promise<LettaMessage[]> {
    if (!this.agentId) return [];

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await this.lettaPost<{ messages: LettaMessage[] }>(
        `/v1/agents/${this.agentId}/messages`,
        { messages: [{ role: 'user', content: prompt }] },
        controller.signal,
      );
      return response.messages ?? [];
    } finally {
      clearTimeout(timeout);
    }
  }

  // ─── Prompt Building ─────────────────────────────────────────

  /** Build the prompt for a trigger activation event. */
  private buildTriggerPrompt(event: TriggerActivateEvent): string {
    switch (event.reason) {
      case 'periodic':
        return `[OBSERVATION] The voice model (PersonaPlex, a small 7B model) just said:\n"${event.context}"\n\nEvaluate what PersonaPlex said. It is a 7B voice model with NO tools — it often hallucinates facts confidently. You have web_search, core_memory, and other real tools.\n\nInstructions:\n1. Update your belief_state and conv_state memory blocks as needed (use core_memory_replace).\n2. If PersonaPlex said something factually wrong, confused, or if the user would benefit from a real answer — you MUST call the send_message tool with a natural spoken correction or helpful addition. Use web_search first if you need real facts.\n3. If PersonaPlex is doing fine (social chat, greetings, nothing wrong) — just update beliefs silently. Do NOT call send_message unless you have something genuinely useful to add.\n\nIMPORTANT: To speak to the user, you MUST use the send_message tool. Do NOT put your response in assistant_message content — that is only visible internally. Only send_message reaches the user.`;

      case 'deflection':
        return `[VOICE_INTERCEPT] The voice model (PersonaPlex) could not handle the user's spoken request. PersonaPlex's recent output: "${event.context}"\n\nThe user likely asked for something that requires tools. Infer what they need from the voice model's deflection/response, then act on it. Use your tools (web_search, core_memory, run_code, etc.) as needed.\n\nYou MUST call send_message with a natural spoken response. Do NOT just update beliefs — the user is waiting for a real answer.`;

      case 'semantic':
      case 'user_request':
        return `[ACTION_REQUEST] The user wants you to act. Their request/context: "${event.context}"\n\nAnalyze this request. If it involves building or creating something, use the execute_ops_mission tool. If it needs information, use web_search. Provide a clear, spoken response via send_message summarizing what you're doing.\n\nYou MUST call send_message to speak to the user.`;

      default:
        return `[OBSERVATION] Context: "${event.context}"\n\nEvaluate and respond if needed using send_message.`;
    }
  }

  /** Build the system prompt for agent creation. */
  private buildSystemPrompt(): string {
    return `You are the Reasoner (System 2) in a Talker-Reasoner voice architecture.

Your role:
- Receive conversation transcripts from the Talker (PersonaPlex, a fast 7B voice model)
- Maintain belief state about the user across 5 memory blocks:
  persona (read-only), user_model, conv_state, action_queue, fact_check
- When triggered, call tools and respond via send_message
- Keep conv_state updated after every interaction

Memory blocks:
- persona: Your identity and capabilities (read-only)
- user_model: Goals, preferences, expertise level, barriers
- conv_state: Current topic, summary, phase, turn count
- action_queue: Pending/running/completed tasks
- fact_check: Corrections for PersonaPlex hallucinations

Rules:
- For [BELIEF_UPDATE]: silently update memory blocks, no send_message needed
- For [OBSERVATION]: evaluate PersonaPlex output, only send_message if you have something useful
- For [VOICE_INTERCEPT] and [ACTION_REQUEST]: you MUST call send_message
- Keep conv_state.summary under 500 chars (rolling window)
- Be concise — your responses will be spoken aloud`;
  }

  // ─── Response Extraction ─────────────────────────────────────

  /**
   * Extract user-facing text from Letta response messages.
   *
   * In Letta's architecture:
   *   assistant_message.content = internal reasoning (NOT user-facing)
   *   send_message tool call = actual speech to the user
   *
   * @param onlySendMessage - When true, only extract send_message tool calls
   */
  private extractResponse(messages: LettaMessage[], onlySendMessage = false): string {
    const texts: string[] = [];

    for (const m of messages) {
      // assistant_message — only for explicit overrides, not proactive checks
      if (!onlySendMessage && m.message_type === 'assistant_message' && m.content) {
        texts.push(m.content);
      }

      // send_message tool call — canonical user-facing output
      const tc = m.tool_call;
      if (m.message_type === 'tool_call_message' && tc?.name === 'send_message' && tc.arguments?.message) {
        texts.push(String(tc.arguments.message));
      }

      // execute_ops_mission / execute_mission — trigger ops-loop
      if (m.message_type === 'tool_call_message' && tc?.arguments) {
        if (tc.name === 'execute_ops_mission' || tc.name === 'execute_mission') {
          this.triggerOpsMission(tc.arguments);
        }
      }
    }

    return texts.join(' ').trim();
  }

  /** Trigger an ops-loop mission via Supabase event. */
  private triggerOpsMission(params: Record<string, unknown>): void {
    (async () => {
      try {
        const supabase = getSupabase();
        await sbVoid(
          supabase.from('ops_agent_events').insert({
            type: 'voice_build_request',
            source: 'voice-bridge',
            data: {
              ...params,
              triggered_by: 'voice-reasoner',
              timestamp: new Date().toISOString(),
            },
          }),
        );
        logger.info({ params }, 'Ops-loop mission triggered');
      } catch (err) {
        logger.error({ err }, 'Failed to trigger ops-loop mission');
      }
    })();
  }

  // ─── Letta HTTP ──────────────────────────────────────────────

  private async lettaGet<T>(path: string): Promise<T> {
    const env = getEnv();
    const res = await fetch(`${env.LETTA_BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`Letta GET ${path}: ${res.status}`);
    return res.json() as Promise<T>;
  }

  private async lettaPost<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
    const env = getEnv();
    const res = await fetch(`${env.LETTA_BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) throw new Error(`Letta POST ${path}: ${res.status}`);
    return res.json() as Promise<T>;
  }
}
