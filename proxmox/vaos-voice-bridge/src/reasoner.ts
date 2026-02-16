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
  private ledger = '';
  /** Queued turn texts accumulated while Reasoner is busy. */
  private pendingTurns: string[] = [];

  /** Get the current belief state (cached in memory, synced from Letta). */
  getBelief(): Belief {
    return this.belief;
  }

  /** Get the assembled ledger of memory (from other agents + Supabase). */
  getLedger(): string {
    return this.ledger;
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

      // Build the ledger of memory from peer agents + Supabase
      await this.buildLedger();
    } catch (err) {
      logger.error({ err }, 'Failed to initialize Reasoner — falling back to local belief');
      // Continue with local belief; Letta may be unavailable
    }
  }

  /**
   * Async belief update (non-blocking, System 1 path).
   * Called after each turn to keep the Reasoner informed.
   * Uses turn-batching: if the Reasoner is busy, queues the turn
   * and processes the batch when the current call finishes.
   */
  async updateBelief(userText: string, talkerResponse: string): Promise<void> {
    if (this.processing) {
      // Queue the turn for batch processing
      this.pendingTurns.push(talkerResponse || userText);
      logger.debug({ queued: this.pendingTurns.length }, 'Reasoner busy — turn queued');
      return;
    }

    this.processing = true;
    const env = getEnv();

    try {
      // Drain any queued turns + current turn into a single batch
      const allTurns = [...this.pendingTurns, talkerResponse || userText];
      this.pendingTurns = [];
      const batchText = allTurns.join(' ').trim().slice(-1000); // Cap at 1000 chars

      const controller = new AbortController();
      // Letta→Claude→tool_call round-trips chain (15-25s typical, 40s cold start)
      // Use 60s to avoid premature timeouts
      const timeout = setTimeout(() => controller.abort(), 60_000);

      if (this.agentId) {
        // Send to Letta agent for processing
        const response = await this.lettaPost<{ messages: LettaMessage[] }>(
          `/v1/agents/${this.agentId}/messages`,
          {
            messages: [{
              role: 'user',
              content: `[BELIEF_UPDATE] Recent conversation:\n"${batchText}"\n\nUpdate the belief_state memory block using core_memory_replace. Update: conversation_topic, conversation_summary, coaching_phase as needed.`,
            }],
          },
          controller.signal,
        );

        clearTimeout(timeout);

        // Sync updated belief from Letta
        await this.syncBeliefFromLetta();
        logger.info({ phase: this.belief.conversation.phase, batchSize: allTurns.length }, 'Belief updated from batch');
      } else {
        clearTimeout(timeout);
        // Local fallback: basic heuristic belief update
        this.belief = mergeBelief(this.belief, {
          conversation: {
            ...this.belief.conversation,
            turns_in_phase: this.belief.conversation.turns_in_phase + 1,
            summary: `${this.belief.conversation.summary} ${batchText.slice(0, 200)}`.trim().slice(-500),
          },
        });
        logger.info({ phase: this.belief.conversation.phase }, 'Belief updated (local)');
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        logger.warn('Belief update timed out');
      } else {
        logger.error({ err }, 'Belief update failed');
      }
    } finally {
      this.processing = false;

      // If turns accumulated while we were processing, schedule another batch
      if (this.pendingTurns.length > 0) {
        logger.info({ queued: this.pendingTurns.length }, 'Processing queued turns');
        // Small delay to avoid hammering Letta
        setTimeout(() => {
          const next = this.pendingTurns.shift() || '';
          this.updateBelief(next, next).catch(err => {
            logger.warn({ err }, 'Queued belief update failed');
          });
        }, 500);
      }
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
            messages: [{
              role: 'user',
              content: `[ACTION_REQUEST] The user wants you to act. Their request: "${userText}"\n\nAnalyze this request. If it involves building or creating something, use the execute_ops_mission tool. Provide a clear, spoken response summarizing what you're doing.`,
            }],
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

  /**
   * Build the "Ledger of Memory" — aggregate context from peer Letta agents
   * and Supabase mission history. This gives PersonaPlex deep awareness of
   * the user's history, what agents have done, and what's been built.
   */
  private async buildLedger(): Promise<void> {
    const parts: string[] = [];

    // 1. Pull key memory blocks from peer Letta agents
    try {
      const peerContext = await this.readPeerAgentMemory();
      if (peerContext) parts.push(peerContext);
    } catch (err) {
      logger.warn({ err }, 'Failed to read peer agent memory');
    }

    // 2. Pull recent mission history from Supabase
    try {
      const missionContext = await this.readMissionHistory();
      if (missionContext) parts.push(missionContext);
    } catch (err) {
      logger.warn({ err }, 'Failed to read mission history');
    }

    this.ledger = parts.join(' ');
    logger.info({ ledgerLength: this.ledger.length }, 'Ledger of memory assembled');
  }

  /** Read useful memory blocks from peer Letta agents (Director, Writer). */
  private async readPeerAgentMemory(): Promise<string | null> {
    const env = getEnv();
    const peerAgents: Array<{ name: string; blocks: string[] }> = [
      { name: 'Director', blocks: ['persona', 'quality_standards', 'lessons_learned', 'user_style'] },
      { name: 'Writer', blocks: ['persona', 'lessons_learned'] },
    ];

    const summaries: string[] = [];

    try {
      const agents = await this.lettaGet<LettaAgent[]>('/v1/agents');

      for (const peer of peerAgents) {
        const agent = agents.find(a => a.name === peer.name);
        if (!agent) continue;

        try {
          const full = await this.lettaGet<{ memory?: { blocks?: LettaMemoryBlock[] } }>(
            `/v1/agents/${agent.id}`,
          );
          const blocks = full.memory?.blocks ?? [];

          for (const wantLabel of peer.blocks) {
            const block = blocks.find(b => b.label === wantLabel);
            if (!block?.value) continue;

            // Extract just the useful content, skip incident/hallucination noise
            const cleaned = this.cleanMemoryBlock(block.value);
            if (cleaned.length > 20) {
              summaries.push(`[${peer.name}/${wantLabel}] ${cleaned}`);
            }
          }
        } catch (err) {
          logger.debug({ agent: peer.name, err }, 'Failed to read peer agent');
        }
      }
    } catch (err) {
      logger.warn({ err }, 'Failed to list Letta agents for peer memory');
      return null;
    }

    if (summaries.length === 0) return null;

    // Condense to fit in a URL-safe prompt (target ~800 chars)
    let result = summaries.join(' | ');
    if (result.length > 800) result = result.slice(0, 797) + '...';
    return result;
  }

  /** Clean a Letta memory block: strip incident/hallucination noise, emoji spam, etc. */
  private cleanMemoryBlock(text: string): string {
    return text
      .split('\n')
      .filter(line => {
        const l = line.toLowerCase();
        // Skip hallucination incident reports and lock spam
        if (l.includes('hallucination') || l.includes('near-miss')) return false;
        if (l.includes('quadruple-locked') || l.includes('automated loop')) return false;
        if (l.includes('🚨') || l.includes('🔒')) return false;
        if (l.startsWith('---')) return false;
        if (l.startsWith('**february') || l.startsWith('**last_activity')) return false;
        return true;
      })
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 500);
  }

  /** Read recent mission history from Supabase ops_missions. */
  private async readMissionHistory(): Promise<string | null> {
    try {
      const supabase = getSupabase();
      const missions = await sb<Array<{
        id: string;
        status: string;
        created_at: string;
        policy_snapshot: { event_data?: { prompt?: string } } | null;
      }>>(
        supabase
          .from('ops_missions')
          .select('id,status,created_at,policy_snapshot')
          .order('created_at', { ascending: false })
          .limit(10),
      );

      if (!missions || missions.length === 0) return null;

      const lines = missions.map(m => {
        const prompt = m.policy_snapshot?.event_data?.prompt ?? 'unknown';
        // Extract product name from prompt like "Build 'vox-radar' — ..."
        const nameMatch = prompt.match(/['']([^'']+)['']/);
        const name = nameMatch?.[1] ?? prompt.slice(0, 40);
        const date = m.created_at?.slice(0, 10) ?? '?';
        return `${name} (${m.status}, ${date})`;
      });

      // Deduplicate same product names (keep latest status)
      const seen = new Set<string>();
      const deduped = lines.filter(l => {
        const key = l.split(' (')[0];
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });

      return `Products built: ${deduped.join('; ')}`;
    } catch (err) {
      logger.warn({ err }, 'Failed to read mission history');
      return null;
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
      // Letta returns memory blocks nested in the agent response at .memory.blocks
      // (the /memory/blocks sub-endpoint doesn't exist in all Letta versions)
      const agent = await this.lettaGet<{ memory?: { blocks?: LettaMemoryBlock[] } }>(
        `/v1/agents/${this.agentId}`,
      );
      const blocks = agent.memory?.blocks ?? [];

      const beliefBlock = blocks.find(b => b.label === 'belief_state');
      if (beliefBlock?.value) {
        // Belief may be stored as JSON or as key: value text
        try {
          const parsed = JSON.parse(beliefBlock.value);
          this.belief = BeliefSchema.parse(parsed);
          logger.info({ phase: this.belief.conversation.phase }, 'Belief synced from Letta (JSON)');
        } catch {
          // Parse key: value text format (e.g., "user_goals: [build X]\ncurrent_project: Y")
          const kv = this.parseBeliefText(beliefBlock.value);
          if (kv) {
            this.belief = kv;
            logger.info({ phase: this.belief.conversation.phase }, 'Belief synced from Letta (text)');
          } else {
            logger.warn('Failed to parse belief — using default');
          }
        }
      }
    } catch (err) {
      logger.warn({ err }, 'Failed to sync belief from Letta');
    }
  }

  /** Parse key: value text format from Letta belief block into Belief. */
  private parseBeliefText(text: string): Belief | null {
    try {
      const lines = text.split('\n').filter(l => l.includes(':'));
      const kv: Record<string, string> = {};
      for (const line of lines) {
        const idx = line.indexOf(':');
        if (idx > 0) kv[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
      }
      const parseList = (v?: string) => v ? v.replace(/[\[\]]/g, '').split(',').map(s => s.trim()).filter(Boolean) : [];
      return BeliefSchema.parse({
        user_model: {
          goals: parseList(kv.user_goals),
          current_project: kv.current_project || null,
          barriers: parseList(kv.barriers),
          expertise_level: kv.expertise_level || 'advanced',
          preferences: {
            voice: kv.preferred_voice || 'NATF0',
            verbosity: kv.preferred_verbosity || 'concise',
          },
        },
        conversation: {
          phase: kv.coaching_phase || kv.conversation_phase || 'understanding',
          topic: kv.conversation_topic || null,
          summary: kv.conversation_summary || '',
        },
        pending_actions: [],
      });
    } catch {
      return null;
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
