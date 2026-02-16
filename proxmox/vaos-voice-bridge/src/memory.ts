/**
 * Memory — CoALA-inspired memory manager for the voice pipeline.
 *
 * Implements the 4-module CoALA memory taxonomy (Sumers, Yao et al.):
 *   Working:    5 core blocks (persona, user_model, conv_state, action_queue, fact_check)
 *   Episodic:   Letta recall (auto-managed conversation history)
 *   Semantic:   Letta archival (vector-searchable long-term facts)
 *   Procedural: Agent tools (web_search, core_memory_*, execute_mission, etc.)
 *
 * Key optimization: Direct API block writes bypass Letta inference for routine
 * updates like conv_state (~0.1s RTT vs 15-25s for full agent message).
 *
 * Includes a compressor that distills all blocks into PersonaPlex's
 * ~200-token text_prompt budget (lossy, priority-ordered).
 */

import { createLogger } from './lib/logger.js';
import { getEnv } from './lib/config.js';
import { getSupabase, sb } from './lib/db.js';
import { type EventBus, E, type ReasonerBeliefEvent } from './events.js';

const logger = createLogger('memory');

// ─── Block Definitions ──────────────────────────────────────────

export interface BlockSpec {
  label: string;
  limit: number;
  readOnly?: boolean;
  defaultValue: string;
}

/**
 * Block specs matching the existing voice-reasoner Letta agent.
 * Labels MUST match what Letta has — the agent uses 'belief_state' and
 * 'conversation_context' (not 'user_model'/'conv_state').
 */
export const BLOCK_SPECS: Record<string, BlockSpec> = {
  persona: {
    label: 'persona',
    limit: 2000,
    readOnly: true,
    defaultValue: 'You are the Reasoner (System 2) in a Talker-Reasoner voice architecture. You maintain beliefs about the user and provide tool-augmented responses when the fast voice model (PersonaPlex, System 1) cannot handle a request. You have web_search, core_memory tools, and can dispatch build missions.',
  },
  belief_state: {
    label: 'belief_state',
    limit: 2000,
    defaultValue: JSON.stringify({
      user_goals: [],
      expertise_level: 'advanced',
      preferences: { voice: 'NATF0', verbosity: 'concise' },
      barriers: [],
      current_project: null,
    }),
  },
  conversation_context: {
    label: 'conversation_context',
    limit: 2000,
    defaultValue: JSON.stringify({
      phase: 'understanding',
      topic: null,
      summary: '',
      turn_count: 0,
      last_update: null,
    }),
  },
  action_queue: {
    label: 'action_queue',
    limit: 2000,
    defaultValue: JSON.stringify({ pending: [], running: [], completed_recent: [] }),
  },
  fact_check: {
    label: 'fact_check',
    limit: 2000,
    defaultValue: JSON.stringify({ corrections: [], verified_facts: [] }),
  },
};

// ─── Memory Manager ─────────────────────────────────────────────

export class Memory {
  private bus: EventBus;
  private agentId: string | null = null;
  private sessionId: string;
  /** Local cache of block content (label → value). */
  private blocks = new Map<string, string>();
  /** Letta block IDs for direct API writes (label → id). */
  private blockIds = new Map<string, string>();
  /** Assembled ledger from peer agents + Supabase missions. */
  private ledger = '';
  /** Last System 2 response — injected into PersonaPlex prompt so it knows what was just said. */
  private _lastSystem2Response = '';
  private _lastSystem2Time = 0;

  constructor(bus: EventBus, sessionId: string) {
    this.bus = bus;
    this.sessionId = sessionId;

    // Initialize blocks with defaults
    for (const spec of Object.values(BLOCK_SPECS)) {
      this.blocks.set(spec.label, spec.defaultValue);
    }

    // Subscribe to reasoner.belief events → sync + recompress
    bus.on('reasoner.belief', (event) => this.handleBeliefUpdate(event), 80);
  }

  /** Initialize: sync blocks from Letta and build the ledger. */
  async init(agentId: string): Promise<void> {
    this.agentId = agentId;
    await this.syncFromLetta();
    await this.buildLedger();
  }

  get currentLedger(): string { return this.ledger; }

  // ─── Block Access ────────────────────────────────────────────

  getBlock(label: string): string {
    return this.blocks.get(label) ?? '';
  }

  getBlockJSON<T = unknown>(label: string): T | null {
    const raw = this.blocks.get(label);
    if (!raw) return null;
    try { return JSON.parse(raw) as T; } catch { return null; }
  }

  getAllBlocks(): Map<string, string> {
    return new Map(this.blocks);
  }

  /** Get the Letta block ID for a label (for direct API writes). */
  getBlockId(label: string): string | null {
    return this.blockIds.get(label) ?? null;
  }

  /** Record what System 2 just said (for PersonaPlex prompt injection). */
  setLastSystem2Response(text: string): void {
    this._lastSystem2Response = text.slice(0, 300);
    this._lastSystem2Time = Date.now();
  }

  /** Get last System 2 response (expires after 60s). */
  getLastSystem2Response(): string {
    if (Date.now() - this._lastSystem2Time > 60_000) return '';
    return this._lastSystem2Response;
  }

  /** Clear last System 2 response (call after answer reconnect to prevent loops). */
  clearLastSystem2Response(): void {
    this._lastSystem2Response = '';
    this._lastSystem2Time = 0;
  }

  /** Update the local block cache without writing to Letta. */
  setBlockLocal(label: string, value: string): void {
    const spec = Object.values(BLOCK_SPECS).find(s => s.label === label);
    if (spec) {
      this.blocks.set(label, value.slice(0, spec.limit));
    }
  }

  // ─── Direct API Writes (bypass inference) ────────────────────

  /**
   * Write a block directly via Letta REST API.
   * ~0.1s RTT vs 15-25s for full agent message round-trip.
   */
  async writeBlock(label: string, value: string): Promise<void> {
    const spec = Object.values(BLOCK_SPECS).find(s => s.label === label);
    if (!spec) throw new Error(`Unknown block: ${label}`);
    if (spec.readOnly) throw new Error(`Block ${label} is read-only`);

    const truncated = value.slice(0, spec.limit);
    this.blocks.set(label, truncated);

    // Use the agent-specific core-memory endpoint so the agent's in-memory
    // copy is updated (global /v1/blocks/{id} only updates the DB, not the
    // agent's cached memory, causing stale reads from /v1/agents/{id}).
    if (this.agentId) {
      try {
        const env = getEnv();
        const res = await fetch(`${env.LETTA_BASE_URL}/v1/agents/${this.agentId}/core-memory/blocks/${label}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value: truncated }),
        });
        if (!res.ok) throw new Error(`PATCH agent block: ${res.status}`);
        logger.debug({ label, size: truncated.length }, 'Block written via agent core-memory API');
      } catch (err) {
        logger.warn({ err, label }, 'Direct block write failed — cached locally');
      }
    }

    this.bus.emit(E.memoryUpdate(this.sessionId, label, 'api'));
  }

  /** Merge partial JSON update into conversation_context. */
  async updateConvState(patch: Record<string, unknown>): Promise<void> {
    const current = this.getBlockJSON<Record<string, unknown>>('conversation_context') ?? {};
    const merged = { ...current, ...patch, last_update: new Date().toISOString() };
    await this.writeBlock('conversation_context', JSON.stringify(merged));
  }

  /** Append a fact correction (PersonaPlex hallucination detected by Reasoner). */
  async addCorrection(wrong: string, right: string): Promise<void> {
    const current = this.getBlockJSON<{ corrections: Array<{ wrong: string; right: string; ts: string }>; verified_facts: unknown[] }>('fact_check')
      ?? { corrections: [], verified_facts: [] };
    current.corrections.push({ wrong, right, ts: new Date().toISOString() });
    if (current.corrections.length > 10) current.corrections = current.corrections.slice(-10);
    await this.writeBlock('fact_check', JSON.stringify(current));
  }

  // ─── Archival Memory ─────────────────────────────────────────

  /** Search archival (Letta vector-indexed long-term store). */
  async searchArchival(query: string, limit = 5): Promise<string[]> {
    if (!this.agentId) return [];
    try {
      const env = getEnv();
      const res = await fetch(
        `${env.LETTA_BASE_URL}/v1/agents/${this.agentId}/archival?query=${encodeURIComponent(query)}&count=${limit}`,
        { headers: { 'Content-Type': 'application/json' } },
      );
      if (!res.ok) return [];
      const passages = await res.json() as Array<{ text: string }>;
      return passages.map(p => p.text);
    } catch (err) {
      logger.warn({ err }, 'Archival search failed');
      return [];
    }
  }

  /** Insert tagged passage into archival memory. */
  async insertArchival(text: string, tag: string): Promise<void> {
    if (!this.agentId) return;
    try {
      const env = getEnv();
      await fetch(`${env.LETTA_BASE_URL}/v1/agents/${this.agentId}/archival`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: `[${tag}] ${text}` }),
      });
      logger.debug({ tag, textLength: text.length }, 'Inserted into archival memory');
    } catch (err) {
      logger.warn({ err }, 'Archival insert failed');
    }
  }

  // ─── Compression (PersonaPlex ~200-token budget) ─────────────

  /**
   * Compress all memory blocks into PersonaPlex's text_prompt.
   * Budget: ~200 tokens (~150 words). Priority-based allocation:
   *   1. persona — identity, non-negotiable (~30 words)
   *   2. conv_state — current context (up to 50 words)
   *   3. user_model — who they are (up to 30 words)
   *   4. fact_check — corrections to prevent hallucination loops (up to 25 words)
   *   5. action_queue — active tasks (up to 15 words)
   */
  compress(): string {
    const parts: string[] = [];

    // 1. Persona (fixed, agnostic — no project identity)
    parts.push('You are a helpful voice assistant. Be natural, conversational, and concise. Adapt to whatever the user wants to discuss.');

    // 2. Conversation state (Letta label: conversation_context)
    //    Only inject if there's actual content (not "none" / empty)
    const conv = this.getBlockJSON<Record<string, unknown>>('conversation_context');
    if (conv) {
      const topic = String(conv.topic ?? '');
      if (topic && topic !== 'none' && topic !== 'null') parts.push(`Topic: ${topic}.`);
      const summary = String(conv.summary ?? '');
      if (summary && summary !== 'New session started.' && summary.length > 10) {
        const words = summary.split(/\s+/).slice(0, 40);
        parts.push(`Context: ${words.join(' ')}`);
      }
    } else {
      // Legacy plaintext format
      const convRaw = this.getBlock('conversation_context');
      const topicMatch = convRaw.match(/conversation_topic:\s*(.+)/);
      if (topicMatch?.[1] && topicMatch[1].trim() !== 'none') {
        parts.push(`Topic: ${topicMatch[1].trim()}.`);
      }
    }

    // 3. Belief state / user model — dynamic, only inject real values
    const userRaw = this.getBlock('belief_state');
    const user = this.getBlockJSON<Record<string, unknown>>('belief_state');
    if (user) {
      const project = user.current_project ?? user.currentProject;
      if (project && project !== 'none' && project !== null) parts.push(`User project: ${project}.`);
      const goals = (user.user_goals ?? user.goals) as string[] | undefined;
      if (goals?.length) parts.push(`Goals: ${goals.slice(0, 3).join(', ')}.`);
    } else if (userRaw) {
      // Legacy plaintext: only inject if value is real (not "none", "[]", etc.)
      const projectMatch = userRaw.match(/current_project:\s*(.+)/);
      const projectVal = projectMatch?.[1]?.trim();
      if (projectVal && projectVal !== 'none' && projectVal !== 'null') {
        parts.push(`User project: ${projectVal}.`);
      }
      const goalsMatch = userRaw.match(/user_goals:\s*\[([^\]]+)\]/);
      const goalsVal = goalsMatch?.[1]?.trim();
      if (goalsVal && goalsVal !== '') {
        parts.push(`Goals: ${goalsVal}.`);
      }
    }

    // 4. Fact corrections (prevent hallucination loops)
    const facts = this.getBlockJSON<{ corrections: Array<{ wrong: string; right: string }> }>('fact_check');
    if (facts?.corrections?.length) {
      const recent = facts.corrections.slice(-3);
      const fixes = recent.map(c => `NOT "${c.wrong}" → "${c.right}"`).join('; ');
      parts.push(`Corrections: ${fixes}`);
    }

    // 5. System 2 result — PersonaPlex should deliver this as its own knowledge.
    //    After reconnect, PersonaPlex is starting a fresh conversation turn.
    //    It should naturally incorporate the information as if it looked it up.
    const lastS2 = this.getLastSystem2Response();
    if (lastS2) {
      parts.push(`[New information available to share with the user: "${lastS2.slice(0, 150)}"] Naturally incorporate this into your next response.`);
    }

    // 6. Action queue
    const actions = this.getBlockJSON<{ running: Array<{ description?: string }> }>('action_queue');
    if (actions?.running?.length) {
      parts.push(`Active: ${actions.running.map(a => a.description ?? 'task').join(', ')}.`);
    }

    // 7. Ledger — DO NOT inject into PersonaPlex text_prompt.
    //    PersonaPlex interprets ANY context as its identity/purpose.
    //    "Products built: vox-radarv5" → it thinks it IS vox-radar.
    //    The ledger is available to the Reasoner (System 2) via this.currentLedger.

    // Enforce ~150 word budget
    let prompt = parts.join(' ');
    const words = prompt.split(/\s+/);
    if (words.length > 150) {
      prompt = words.slice(0, 150).join(' ');
    }

    return prompt;
  }

  /** Compress and emit memory.compressed event. */
  emitCompressed(): void {
    const prompt = this.compress();
    const tokenEstimate = Math.ceil(prompt.split(/\s+/).length / 0.75);
    this.bus.emit(E.memoryCompressed(this.sessionId, prompt, tokenEstimate));
  }

  // ─── Ledger of Memory ────────────────────────────────────────

  /**
   * Build the "Ledger of Memory" — aggregate context from peer Letta
   * agents (Director, Writer) and Supabase mission history.
   */
  private async buildLedger(): Promise<void> {
    const parts: string[] = [];

    try {
      const peer = await this.readPeerAgentMemory();
      if (peer) parts.push(peer);
    } catch (err) {
      logger.warn({ err }, 'Failed to read peer agent memory');
    }

    try {
      const missions = await this.readMissionHistory();
      if (missions) parts.push(missions);
    } catch (err) {
      logger.warn({ err }, 'Failed to read mission history');
    }

    this.ledger = parts.join(' ');
    logger.info({ ledgerLength: this.ledger.length }, 'Ledger of memory assembled');
  }

  /** Read memory blocks from peer Letta agents (Director, Writer). */
  private async readPeerAgentMemory(): Promise<string | null> {
    const env = getEnv();
    // NOTE: Do NOT include 'persona' — peer agent identities leak into
    // PersonaPlex's text_prompt and cause it to adopt their identity
    // ("I am the Director of a video production crew").
    const peers = [
      { name: 'Director', blocks: ['quality_standards', 'lessons_learned', 'user_style'] },
      { name: 'Writer', blocks: ['lessons_learned'] },
    ];

    const summaries: string[] = [];

    try {
      const res = await fetch(`${env.LETTA_BASE_URL}/v1/agents`, {
        headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) return null;
      const agents = await res.json() as Array<{ id: string; name: string }>;

      for (const peer of peers) {
        const agent = agents.find(a => a.name === peer.name);
        if (!agent) continue;

        try {
          const agentRes = await fetch(`${env.LETTA_BASE_URL}/v1/agents/${agent.id}`, {
            headers: { 'Content-Type': 'application/json' },
          });
          if (!agentRes.ok) continue;
          const full = await agentRes.json() as { memory?: { blocks?: Array<{ label: string; value: string }> } };
          const blocks = full.memory?.blocks ?? [];

          for (const wantLabel of peer.blocks) {
            const block = blocks.find(b => b.label === wantLabel);
            if (!block?.value) continue;
            const cleaned = this.cleanBlockText(block.value);
            if (cleaned.length > 20) {
              summaries.push(`[${peer.name}/${wantLabel}] ${cleaned}`);
            }
          }
        } catch {
          // Skip this peer
        }
      }
    } catch {
      return null;
    }

    if (summaries.length === 0) return null;
    let result = summaries.join(' | ');
    if (result.length > 800) result = result.slice(0, 797) + '...';
    return result;
  }

  /** Clean Letta memory block text: strip noise, emoji spam, incident reports. */
  private cleanBlockText(text: string): string {
    return text
      .split('\n')
      .filter(line => {
        const l = line.toLowerCase();
        if (l.includes('hallucination') || l.includes('near-miss')) return false;
        if (l.includes('quadruple-locked') || l.includes('automated loop')) return false;
        if (l.startsWith('---')) return false;
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

      if (!missions?.length) return null;

      const seen = new Set<string>();
      const lines: string[] = [];
      for (const m of missions) {
        const prompt = m.policy_snapshot?.event_data?.prompt ?? 'unknown';
        const nameMatch = prompt.match(/['']([^'']+)['']/);
        const name = nameMatch?.[1] ?? prompt.slice(0, 40);
        if (seen.has(name)) continue;
        seen.add(name);
        lines.push(`${name} (${m.status}, ${m.created_at?.slice(0, 10) ?? '?'})`);
      }

      return `Products built: ${lines.join('; ')}`;
    } catch (err) {
      logger.warn({ err }, 'Failed to read mission history');
      return null;
    }
  }

  // ─── Sync ────────────────────────────────────────────────────

  /** Pull all blocks from Letta into local cache. */
  async syncFromLetta(): Promise<void> {
    if (!this.agentId) return;
    try {
      const env = getEnv();
      const res = await fetch(`${env.LETTA_BASE_URL}/v1/agents/${this.agentId}`, {
        headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) throw new Error(`GET agent: ${res.status}`);
      const agent = await res.json() as { memory?: { blocks?: Array<{ id: string; label: string; value: string }> } };
      const blocks = agent.memory?.blocks ?? [];

      for (const block of blocks) {
        this.blocks.set(block.label, block.value);
        this.blockIds.set(block.label, block.id);
      }

      logger.info({ blockCount: blocks.length, labels: blocks.map(b => b.label) }, 'Memory synced from Letta');
    } catch (err) {
      logger.warn({ err }, 'Failed to sync memory from Letta');
    }
  }

  // ─── Event Handlers ──────────────────────────────────────────

  private async handleBeliefUpdate(_event: ReasonerBeliefEvent): Promise<void> {
    await this.syncFromLetta();
    this.emitCompressed();
  }
}
