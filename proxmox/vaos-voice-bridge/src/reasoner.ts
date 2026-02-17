/**
 * Reasoner client: async interface to Letta agent (System 2).
 *
 * The Reasoner receives conversation transcripts, updates belief state,
 * and returns structured responses when the Coordinator triggers System 2.
 */

import { createLogger } from './logger.js';
import { emptyBelief, type BeliefState } from './belief.js';
import { evaluatePhaseTransition } from './coordinator.js';
import type { Config } from './config.js';

const logger = createLogger('reasoner');

export class Reasoner {
  private config: Config;
  private belief: BeliefState;
  private agentId: string | null;

  constructor(config: Config) {
    this.config = config;
    this.belief = emptyBelief();
    this.agentId = config.letta.agentId;
  }

  private beliefChangeListeners: Array<(belief: BeliefState) => void> = [];

  /** Register a callback for when belief state changes. */
  onBeliefChange(fn: (belief: BeliefState) => void): void {
    this.beliefChangeListeners.push(fn);
  }

  private notifyBeliefChange(): void {
    for (const fn of this.beliefChangeListeners) {
      try { fn(this.belief); } catch { /* non-critical */ }
    }
  }

  /** Resolve agent ID from name if not set. */
  async init(): Promise<void> {
    if (this.agentId) return;

    try {
      const res = await fetch(`${this.config.letta.baseUrl}/v1/agents/`, {
        headers: { Authorization: `Bearer ${this.config.letta.token}` },
      });
      if (!res.ok) {
        logger.warn({ status: res.status }, 'Failed to list Letta agents');
        return;
      }
      const agents = (await res.json()) as Array<{ id: string; name: string }>;
      const match = agents.find((a) => a.name === this.config.letta.agentName);
      if (match) {
        this.agentId = match.id;
        logger.info({ agentId: this.agentId, name: match.name }, 'Resolved Letta Reasoner agent');
      } else {
        logger.warn({ name: this.config.letta.agentName }, 'Reasoner agent not found in Letta — will create on first use');
      }
    } catch (err) {
      logger.error({ err }, 'Failed to connect to Letta');
    }
  }

  /**
   * Sync belief state from Letta agent's core memory blocks.
   * The Reasoner (Claude) updates these via memory_replace tool calls.
   * This keeps the local belief in sync with what Claude actually stored.
   *
   * Letta core memory uses /v1/agents/{id}/core-memory with a blocks[] array.
   * The belief_state block may be JSON or key-value text format.
   */
  async syncBeliefFromLetta(): Promise<boolean> {
    if (!this.agentId) return false;

    try {
      const res = await fetch(
        `${this.config.letta.baseUrl}/v1/agents/${this.agentId}/core-memory`,
        { headers: { Authorization: `Bearer ${this.config.letta.token}` } },
      );

      if (!res.ok) {
        logger.debug({ status: res.status }, 'Failed to read Letta memory');
        return false;
      }

      const memory = (await res.json()) as {
        blocks?: Array<{ label: string; value: string }>;
      };

      const beliefBlock = memory.blocks?.find((b) => b.label === 'belief_state');
      if (!beliefBlock?.value) return false;

      // Parse belief — handles both JSON and key-value text format
      const raw = this.parseLettaBelief(beliefBlock.value);
      if (!raw) return false;

      let changed = false;

      // Merge user_goals
      if (raw.user_goals && raw.user_goals.length > 0) {
        const newGoals = Array.isArray(raw.user_goals) ? raw.user_goals : [raw.user_goals];
        if (JSON.stringify(newGoals) !== JSON.stringify(this.belief.userModel.goals)) {
          this.belief.userModel.goals = newGoals;
          changed = true;
        }
      }
      // Merge current_project
      if (raw.current_project && raw.current_project !== this.belief.userModel.currentProject) {
        this.belief.userModel.currentProject = raw.current_project;
        changed = true;
      }
      // Merge coaching_phase → conversation.phase
      if (raw.coaching_phase && ['understanding', 'planning', 'action'].includes(raw.coaching_phase)) {
        if (raw.coaching_phase !== this.belief.conversation.phase) {
          this.belief.conversation.phase = raw.coaching_phase as BeliefState['conversation']['phase'];
          changed = true;
        }
      }
      // Merge barriers
      if (raw.barriers) {
        const newBarriers = Array.isArray(raw.barriers) ? raw.barriers : [];
        if (JSON.stringify(newBarriers) !== JSON.stringify(this.belief.userModel.barriers)) {
          this.belief.userModel.barriers = newBarriers;
          changed = true;
        }
      }
      // Merge conversation_summary
      if (raw.conversation_summary && raw.conversation_summary !== this.belief.conversation.summary) {
        this.belief.conversation.summary = raw.conversation_summary;
        changed = true;
      }
      // Merge topic
      if (raw.conversation_topic && raw.conversation_topic !== this.belief.conversation.topic) {
        this.belief.conversation.topic = raw.conversation_topic;
        changed = true;
      }

      if (changed) {
        this.belief.lastReasonerUpdate = new Date().toISOString();
        logger.info(
          { goals: this.belief.userModel.goals, phase: this.belief.conversation.phase, topic: this.belief.conversation.topic },
          'Synced belief from Letta memory',
        );
        this.notifyBeliefChange();
      }

      return changed;
    } catch (err) {
      logger.debug({ err }, 'Letta memory sync failed');
      return false;
    }
  }

  /**
   * Parse Letta belief block — handles JSON or key-value text format.
   *
   * Claude's memory_replace often produces key-value text like:
   *   user_goals: [build a workout tracker]
   *   coaching_phase: planning
   *   conversation_topic: fitness app
   */
  private parseLettaBelief(text: string): Record<string, unknown> | null {
    // Try JSON first
    try {
      return JSON.parse(text);
    } catch { /* not JSON */ }

    // Parse key-value text format
    const result: Record<string, unknown> = {};
    const lines = text.split('\n');
    for (const line of lines) {
      const colonIdx = line.indexOf(':');
      if (colonIdx === -1) continue;
      const key = line.slice(0, colonIdx).trim();
      let val = line.slice(colonIdx + 1).trim();
      if (!key) continue;

      // Parse array-like values: [item1, item2]
      if (val.startsWith('[') && val.endsWith(']')) {
        const inner = val.slice(1, -1).trim();
        if (inner) {
          result[key] = inner.split(',').map((s) => s.trim()).filter(Boolean);
        } else {
          result[key] = [];
        }
      } else {
        result[key] = val;
      }
    }

    return Object.keys(result).length > 0 ? result : null;
  }

  /** Get current belief state. */
  getBelief(): BeliefState {
    return this.belief;
  }

  /**
   * Async belief update (System 1 path — non-blocking).
   * Sends transcript to Reasoner agent, which updates its memory.
   * We parse any belief updates from the response.
   */
  async updateBelief(userText: string, assistantText: string): Promise<void> {
    if (!this.agentId) {
      logger.debug('No Reasoner agent — skipping belief update');
      return;
    }

    const prompt = [
      'Update your belief state based on this conversation turn.',
      `User said: "${userText}"`,
      assistantText ? `Assistant said: "${assistantText}"` : '',
      'Respond with a JSON belief update if anything changed, otherwise say "no update".',
    ]
      .filter(Boolean)
      .join('\n');

    try {
      const response = await this.sendToAgent(prompt);
      this.parseBeliefUpdate(response);
      this.belief.lastReasonerUpdate = new Date().toISOString();
      this.belief.conversation.turnsInPhase++;
      logger.debug({ belief: this.belief.conversation }, 'Belief updated');

      // Also sync from Letta memory (Claude may have updated via memory_replace)
      await this.syncBeliefFromLetta();

      // Evaluate phase transitions
      const newPhase = evaluatePhaseTransition(this.belief);
      if (newPhase) {
        logger.info(
          { from: this.belief.conversation.phase, to: newPhase },
          'Phase transition',
        );
        this.belief.conversation.phase = newPhase;
        this.belief.conversation.turnsInPhase = 0;
        this.notifyBeliefChange();
      }
    } catch (err) {
      logger.error({ err }, 'Belief update failed');
    }
  }

  /**
   * Synchronous reasoning (System 2 path — blocks Talker).
   * Sends the user's request to the Reasoner for planning/action.
   * Returns a text response for the Talker to speak.
   */
  async processAndRespond(userText: string): Promise<string> {
    if (!this.agentId) {
      return "I don't have my reasoning module connected right now. Let me handle this with what I know.";
    }

    const prompt = [
      'The user needs a thoughtful response. Think step by step.',
      `User said: "${userText}"`,
      `Current context: ${JSON.stringify(this.belief.conversation)}`,
      'Provide a concise spoken response (2-3 sentences max for voice delivery).',
    ].join('\n');

    try {
      const response = await this.sendToAgent(prompt);
      // Update belief to reflect the planning action
      this.belief.conversation.phase = 'action';
      this.belief.lastReasonerUpdate = new Date().toISOString();

      // Sync from Letta memory (Claude may have updated via memory_replace)
      await this.syncBeliefFromLetta();
      this.notifyBeliefChange();

      return response;
    } catch (err) {
      logger.error({ err }, 'Reasoner processing failed');
      return 'I ran into an issue thinking about that. Could you rephrase?';
    }
  }

  /** Send message to Letta agent and extract assistant response. */
  private async sendToAgent(message: string): Promise<string> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.config.routing.agentTimeoutMs);

    try {
      const res = await fetch(
        `${this.config.letta.baseUrl}/v1/agents/${this.agentId}/messages`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.config.letta.token}`,
          },
          body: JSON.stringify({
            messages: [{ role: 'user', content: message }],
          }),
          signal: controller.signal,
        },
      );

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Letta ${res.status}: ${body}`);
      }

      interface LettaMessage {
        message_type?: string;
        role?: string;
        content?: string;
        text?: string;
        assistant_message?: string;
        tool_call?: { name?: string; arguments?: string };
        tool_return?: string;
      }

      const data = (await res.json()) as { messages: LettaMessage[] };

      // Log tool calls for observability
      for (const msg of data.messages ?? []) {
        if (msg.message_type === 'tool_call_message' && msg.tool_call) {
          const toolName = msg.tool_call.name ?? 'unknown';
          logger.info({ tool: toolName }, 'Reasoner called tool');

          // Track mission submissions
          if (toolName === 'execute_mission') {
            this.belief.conversation.phase = 'action';
          }
        }
        if (msg.message_type === 'tool_return_message' && msg.tool_return) {
          const ret = msg.tool_return;
          logger.debug({ returnPreview: ret.slice(0, 200) }, 'Tool returned');

          // If execute_mission succeeded, extract mission_id
          if (ret.includes('mission_id') && !ret.startsWith('ERROR')) {
            try {
              const parsed = JSON.parse(ret);
              if (parsed.mission_id) {
                this.belief.pendingActions.push({
                  type: 'mission',
                  id: parsed.mission_id,
                  status: parsed.status ?? 'submitted',
                });
                logger.info({ missionId: parsed.mission_id }, 'Mission submitted via Reasoner');
              }
            } catch { /* non-JSON return */ }
          }
        }
      }

      // Letta v1 uses message_type='assistant_message' with content field
      for (const msg of data.messages ?? []) {
        if (
          (msg.message_type === 'assistant_message' || msg.role === 'assistant') &&
          (msg.content || msg.text || msg.assistant_message)
        ) {
          return msg.content || msg.text || msg.assistant_message || '';
        }
      }
      return '';
    } finally {
      clearTimeout(timeout);
    }
  }

  /** Parse JSON belief updates from Reasoner response text. */
  private parseBeliefUpdate(text: string): void {
    if (!text || text.toLowerCase().includes('no update')) return;

    // Try to extract JSON from the response
    const jsonMatch = text.match(/\{[\s\S]*?\}/);
    if (!jsonMatch) return;

    try {
      const update = JSON.parse(jsonMatch[0]);

      if (update.topic) this.belief.conversation.topic = update.topic;
      if (update.phase) this.belief.conversation.phase = update.phase;
      if (update.goals && Array.isArray(update.goals)) {
        this.belief.userModel.goals = update.goals;
      }
      if (update.currentProject) {
        this.belief.userModel.currentProject = update.currentProject;
      }
      if (update.summary) this.belief.conversation.summary = update.summary;
    } catch {
      // Not valid JSON — that's fine, Reasoner just didn't output structured data
    }
  }
}
