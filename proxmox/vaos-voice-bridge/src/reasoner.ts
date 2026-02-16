/**
 * Reasoner — Letta Agent + Claude (System 2).
 *
 * The Reasoner is the slow, deliberate thinker. It:
 *   - Receives conversation transcripts asynchronously
 *   - Maintains a belief state about the user (goals, context, barriers)
 *   - Calls tools when needed (ops-loop missions, knowledge search, etc.)
 *   - Returns structured responses when System 2 override is triggered
 *
 * Communication with Letta is via its REST API (port 8283).
 * Belief state is stored in Letta memory blocks.
 */

import { createLogger } from './lib/logger.js';
import { getEnv } from './lib/config.js';
import { getSupabase, sb, sbVoid } from './lib/db.js';
import { type Belief, BeliefSchema, createDefaultBelief, mergeBelief } from './belief.js';

const logger = createLogger('reasoner');

interface LettaAgent {
  id: string;
  name: string;
}

interface LettaMemoryBlock {
  id: string;
  label: string;
  value: string;
}

interface LettaMessage {
  role: string;
  text?: string;
  tool_calls?: Array<{ name: string; arguments: Record<string, unknown> }>;
}

export class Reasoner {
  private agentId: string | null = null;
  private belief: Belief = createDefaultBelief();
  private processing = false;

  /** Get the current belief state (cached in memory, synced from Letta). */
  getBelief(): Belief {
    return this.belief;
  }

  /** Initialize: find or create the voice-reasoner agent in Letta. */
  async init(): Promise<void> {
    const env = getEnv();
    const agentName = env.LETTA_AGENT_NAME;

    logger.info({ agentName, lettaUrl: env.LETTA_BASE_URL }, 'Initializing Reasoner');

    try {
      // List agents and find ours
      const agents = await this.lettaGet<LettaAgent[]>('/v1/agents');
      const existing = agents.find(a => a.name === agentName);

      if (existing) {
        this.agentId = existing.id;
        logger.info({ agentId: this.agentId }, 'Found existing Letta agent');
      } else {
        // Create the agent
        const agent = await this.lettaPost<LettaAgent>('/v1/agents', {
          name: agentName,
          model: 'claude-sonnet-4-20250514',
          system: this.buildSystemPrompt(),
          memory: {
            blocks: [
              {
                label: 'belief_state',
                value: JSON.stringify(createDefaultBelief()),
              },
              {
                label: 'conversation_log',
                value: '',
              },
            ],
          },
        });
        this.agentId = agent.id;
        logger.info({ agentId: this.agentId }, 'Created new Letta agent');
      }

      // Load current belief from Letta memory
      await this.syncBeliefFromLetta();
    } catch (err) {
      logger.error({ err }, 'Failed to initialize Reasoner — falling back to local belief');
      // Continue with local belief; Letta may be unavailable
    }
  }

  /**
   * Async belief update (non-blocking, System 1 path).
   * Called after each turn to keep the Reasoner informed.
   */
  async updateBelief(userText: string, talkerResponse: string): Promise<void> {
    if (this.processing) {
      logger.debug('Reasoner busy, skipping belief update');
      return;
    }

    this.processing = true;
    const env = getEnv();

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), env.BELIEF_UPDATE_TIMEOUT_MS);

      if (this.agentId) {
        // Send to Letta agent for processing
        const response = await this.lettaPost<{ messages: LettaMessage[] }>(
          `/v1/agents/${this.agentId}/messages`,
          {
            role: 'user',
            text: `[BELIEF_UPDATE] User said: "${userText}"\nAssistant responded: "${talkerResponse}"\n\nUpdate the belief_state memory block based on this exchange. Focus on: user goals, current topic, conversation phase, and any barriers mentioned.`,
          },
          controller.signal,
        );

        clearTimeout(timeout);

        // Sync updated belief from Letta
        await this.syncBeliefFromLetta();
      } else {
        clearTimeout(timeout);
        // Local fallback: basic heuristic belief update
        this.belief = mergeBelief(this.belief, {
          conversation: {
            ...this.belief.conversation,
            turns_in_phase: this.belief.conversation.turns_in_phase + 1,
            summary: `${this.belief.conversation.summary} User: ${userText.slice(0, 100)}`.trim().slice(-500),
          },
        });
      }

      logger.debug({ phase: this.belief.conversation.phase }, 'Belief updated');
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        logger.warn('Belief update timed out');
      } else {
        logger.error({ err }, 'Belief update failed');
      }
    } finally {
      this.processing = false;
    }
  }

  /**
   * System 2 override: process a turn and return a structured response.
   * Called when the Coordinator decides the Talker should wait.
   */
  async processAndRespond(userText: string): Promise<string> {
    const env = getEnv();

    logger.info({ text: userText.slice(0, 100) }, 'System 2 override — processing');

    try {
      if (this.agentId) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), env.REASONER_TIMEOUT_MS);

        const response = await this.lettaPost<{ messages: LettaMessage[] }>(
          `/v1/agents/${this.agentId}/messages`,
          {
            role: 'user',
            text: `[ACTION_REQUEST] The user wants you to act. Their request: "${userText}"\n\nAnalyze this request. If it involves building or creating something, use the execute_ops_mission tool. Provide a clear, spoken response summarizing what you're doing.`,
          },
          controller.signal,
        );

        clearTimeout(timeout);

        // Extract text response from Letta messages
        const textMessages = response.messages?.filter(m => m.text) ?? [];
        const responseText = textMessages.map(m => m.text).join(' ').trim();

        // Check for tool calls that trigger ops-loop
        const toolCalls = response.messages?.flatMap(m => m.tool_calls ?? []) ?? [];
        for (const call of toolCalls) {
          if (call.name === 'execute_ops_mission') {
            await this.triggerOpsMission(call.arguments);
          }
        }

        // Sync belief after processing
        await this.syncBeliefFromLetta();

        return responseText || "I'm working on that for you. Give me a moment.";
      }

      // Fallback: use Claude directly via Anthropic API
      return await this.claudeFallback(userText);
    } catch (err) {
      logger.error({ err }, 'System 2 processing failed');
      return "I ran into an issue processing that. Let me try a different approach.";
    }
  }

  /** Trigger an ops-loop mission via Supabase event. */
  private async triggerOpsMission(params: Record<string, unknown>): Promise<void> {
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
      logger.info({ params }, 'Ops-loop mission triggered via event');
    } catch (err) {
      logger.error({ err }, 'Failed to trigger ops-loop mission');
    }
  }

  /** Read belief state from Letta memory blocks. */
  private async syncBeliefFromLetta(): Promise<void> {
    if (!this.agentId) return;

    try {
      const blocks = await this.lettaGet<LettaMemoryBlock[]>(
        `/v1/agents/${this.agentId}/memory/blocks`,
      );

      const beliefBlock = blocks.find(b => b.label === 'belief_state');
      if (beliefBlock?.value) {
        try {
          const parsed = JSON.parse(beliefBlock.value);
          this.belief = BeliefSchema.parse(parsed);
          logger.debug({ phase: this.belief.conversation.phase }, 'Belief synced from Letta');
        } catch (parseErr) {
          logger.warn({ err: parseErr }, 'Failed to parse belief from Letta — using current');
        }
      }
    } catch (err) {
      logger.warn({ err }, 'Failed to sync belief from Letta');
    }
  }

  /** Claude direct fallback when Letta is unavailable. */
  private async claudeFallback(userText: string): Promise<string> {
    const env = getEnv();

    try {
      const response = await fetch(`${env.ANTHROPIC_BASE_URL}/v1/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': env.ANTHROPIC_AUTH_TOKEN,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 300,
          system: 'You are a voice assistant Reasoner (System 2). Given the user request, provide a concise spoken response. If they want something built, describe what you would do.',
          messages: [
            {
              role: 'user',
              content: `Current context: ${JSON.stringify(this.belief.conversation)}\n\nUser request: ${userText}`,
            },
          ],
        }),
      });

      if (!response.ok) {
        throw new Error(`Claude API ${response.status}`);
      }

      const body = await response.json() as { content: Array<{ text?: string }> };
      return body.content?.[0]?.text ?? "I'm processing your request.";
    } catch (err) {
      logger.error({ err }, 'Claude fallback failed');
      return "I'm having trouble connecting to my reasoning system. Let me try again in a moment.";
    }
  }

  /** Build the system prompt for the Letta agent. */
  private buildSystemPrompt(): string {
    return `You are the Reasoner (System 2) in a Talker-Reasoner voice architecture.

Your role:
- Receive conversation transcripts from the Talker (PersonaPlex, a fast 7B voice model)
- Maintain a belief state about the user in your belief_state memory block
- When asked to act, call tools and return structured responses
- Keep your belief_state JSON updated after every interaction

Belief state schema:
{
  "user_model": { "goals": [], "current_project": null, "preferences": {...}, "barriers": [], "expertise_level": "advanced" },
  "conversation": { "phase": "understanding|planning|action|reflection", "topic": null, "turns_in_phase": 0, "summary": "" },
  "pending_actions": [],
  "last_reasoner_update": null
}

Rules:
- For [BELIEF_UPDATE] messages: silently update belief_state, no verbose response needed
- For [ACTION_REQUEST] messages: analyze, call tools if needed, provide a concise spoken response
- Always update the conversation phase appropriately
- Keep conversation.summary under 500 chars (rolling window)
- Be concise — your responses will be spoken aloud via TTS`;
  }

  /** HTTP GET to Letta. */
  private async lettaGet<T>(path: string): Promise<T> {
    const env = getEnv();
    const res = await fetch(`${env.LETTA_BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`Letta GET ${path}: ${res.status}`);
    return res.json() as Promise<T>;
  }

  /** HTTP POST to Letta. */
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
