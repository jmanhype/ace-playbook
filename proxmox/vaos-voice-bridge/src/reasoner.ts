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
  /** Cooldown after Letta timeout — skip processing until this time. */
  private cooldownUntil = 0;
  /** Rate-limit belief updates: minimum seconds between Letta calls. */
  private lastBeliefUpdate = 0;
  private static readonly BELIEF_UPDATE_INTERVAL_MS = 45_000; // 45s between belief updates

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
   * Handle trigger.activate — fast Ollama response + async Letta memory.
   *
   * Architecture: The Letta agentic loop takes 30-45s which is unacceptable
   * for voice UX. Instead we:
   *   1. Fast path: Direct Ollama call (~3-4s) for immediate user-facing response
   *   2. Background: Async Letta call for memory/tool updates (non-blocking)
   */
  private async handleTrigger(event: TriggerActivateEvent): Promise<void> {
    // After a fast Ollama response, ALL triggers respect cooldown (10s).
    // Only bypass cooldown for deflection/user_request when cooldown came from
    // a timeout/error (20s+), not from a successful fast response (10s).
    const isLongCooldown = (this.cooldownUntil - Date.now()) > 12_000;
    const bypassCooldown = isLongCooldown && (event.reason === 'deflection' || event.reason === 'user_request');
    if (!bypassCooldown && Date.now() < this.cooldownUntil) {
      logger.debug({ cooldownRemainingSecs: Math.round((this.cooldownUntil - Date.now()) / 1000), reason: event.reason }, 'Skipping trigger (Letta cooldown)');
      return;
    }
    this.trigger.system2Active = true;
    this.bus.emit(E.reasonerThinking(this.sessionId, true));

    try {
      // ── Fast path: Ollama direct (~2-4s) ──
      const t0 = Date.now();
      const fastText = await this.fastOllamaRespond(event);
      const fastMs = Date.now() - t0;

      if (fastText) {
        logger.info({ reason: event.reason, latencyMs: fastMs, text: fastText.slice(0, 100) }, 'System 2 fast interjection (Ollama)');
        this.bus.emit(E.reasonerInterjection(this.sessionId, fastText, event.reason));
        // Short cooldown after fast response to prevent rapid-fire triggers
        // (PersonaPlex keeps echoing keywords like "search", "results", etc.)
        this.cooldownUntil = Date.now() + 10_000;
      } else {
        logger.warn({ reason: event.reason, latencyMs: fastMs }, 'Ollama fast path produced no response');
      }

      // ── Background: Letta memory sync (non-blocking) ──
      this.asyncLettaMemoryUpdate(event).catch(err => {
        logger.warn({ err: err instanceof Error ? err.message : String(err), reason: event.reason }, 'Background Letta memory update failed');
      });
    } catch (err) {
      // Ollama fast path failed — fall back to Letta direct
      logger.warn({ err: err instanceof Error ? err.message : String(err), reason: event.reason }, 'Ollama fast path failed, trying Letta fallback');
      try {
        const prompt = this.buildTriggerPrompt(event);
        const response = await this.sendToLetta(prompt, 90_000);
        await this.memory.syncFromLetta();
        const text = this.extractResponse(response, false);
        if (text) {
          logger.info({ reason: event.reason, text: text.slice(0, 100) }, 'System 2 interjection (Letta fallback)');
          this.bus.emit(E.reasonerInterjection(this.sessionId, text, event.reason));
        }
        this.bus.emit(E.reasonerBelief(this.sessionId, event.reason, ['belief_state', 'conversation_context']));
      } catch (fallbackErr) {
        if (fallbackErr instanceof Error && fallbackErr.name === 'AbortError') {
          logger.warn({ reason: event.reason }, 'System 2 Letta fallback timed out');
          this.cooldownUntil = Date.now() + 20_000;
        } else {
          logger.error({ err: fallbackErr, reason: event.reason }, 'System 2 Letta fallback failed');
          this.cooldownUntil = Date.now() + 15_000;
        }
      }
    } finally {
      this.bus.emit(E.reasonerThinking(this.sessionId, false));
      this.trigger.system2Active = false;
      this.trigger.resetHistory();
    }
  }

  /**
   * Fast-path: Direct Ollama call for immediate user-facing response.
   * Bypasses Letta's multi-step agentic loop entirely.
   * Typical latency: 3-5s vs Letta's 30-45s.
   */
  private async fastOllamaRespond(event: TriggerActivateEvent): Promise<string> {
    const env = getEnv();
    const ollamaUrl = env.OLLAMA_URL;
    const model = env.OLLAMA_MODEL;

    // Build an acknowledgment prompt — NOT a full answer.
    // The fast path just tells the user we heard them and are working on it.
    // The actual answer comes from Letta (with real tools) in the background.
    const systemContext = 'You are a helpful voice assistant. The main voice model could not handle this request. Your job is to give a BRIEF acknowledgment (1 sentence max) that you understood the request and are working on it. Do NOT try to answer the question yourself. Do NOT suggest the user do it themselves. Just acknowledge.';

    let userPrompt: string;
    switch (event.reason) {
      case 'deflection':
        userPrompt = `The voice assistant said: "${event.context?.slice(0, 200)}"\n\nThe user asked for something requiring tools (search, memory, etc). Give a brief acknowledgment like "Let me look into that for you" or "One moment, I'll check on that." Do NOT try to answer — just acknowledge.`;
        break;
      case 'user_request':
      case 'semantic':
        userPrompt = `Context: "${event.context?.slice(0, 200)}"\n\nAcknowledge the request briefly. Say something like "Sure, let me look that up" or "On it, give me a moment." Do NOT answer the question.`;
        break;
      default:
        userPrompt = `Context: "${event.context?.slice(0, 200)}"\n\nBriefly acknowledge if appropriate.`;
    }

    // Include memory context if available
    const belief = this.memory.getBlock('belief_state');
    const conv = this.memory.getBlock('conversation_context');
    let memoryContext = '';
    if (belief && belief.length > 10) memoryContext += `\nUser context: ${belief.slice(0, 300)}`;
    if (conv && conv.length > 10) memoryContext += `\nConversation: ${conv.slice(0, 200)}`;

    const fullPrompt = `${systemContext}${memoryContext}\n\n${userPrompt}`;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15_000);

    try {
      const res = await fetch(`${ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          prompt: fullPrompt,
          stream: false,
          options: { num_predict: 150, temperature: 0.7 },
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`Ollama ${res.status}: ${await res.text()}`);
      const data = await res.json() as { response?: string; total_duration?: number };
      logger.debug({ model, totalDurationMs: Math.round((data.total_duration ?? 0) / 1e6) }, 'Ollama response received');
      return data.response?.trim() ?? '';
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Background Letta call — runs async after fast Ollama acknowledgment.
   * This is the REAL System 2: uses tools (web_search, memory, etc.)
   * and delivers the actual answer when ready.
   */
  private async asyncLettaMemoryUpdate(event: TriggerActivateEvent): Promise<void> {
    if (!this.agentId) return;

    // Use the full trigger prompt — not a belief update.
    // Letta has real tools and should actually answer the user's question.
    const prompt = this.buildTriggerPrompt(event);

    try {
      const t0 = Date.now();
      const response = await this.sendToLetta(prompt, 90_000);
      const ms = Date.now() - t0;
      await this.memory.syncFromLetta();

      // Extract the real answer from Letta (with tool results)
      const text = this.extractResponse(response, false);

      if (text) {
        logger.info({ reason: event.reason, latencyMs: ms, text: text.slice(0, 100) }, 'System 2 Letta response (background)');
        this.bus.emit(E.reasonerInterjection(this.sessionId, text, event.reason));
      } else {
        logger.debug({ reason: event.reason, latencyMs: ms, msgCount: response.length }, 'Background Letta produced no user-facing output');
      }

      this.bus.emit(E.reasonerBelief(this.sessionId, event.reason, ['belief_state', 'conversation_context']));
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        logger.warn({ reason: event.reason }, 'Background Letta call timed out');
      } else {
        logger.error({ err: err instanceof Error ? err.message : String(err), reason: event.reason }, 'Background Letta call failed');
      }
    }
  }

  /**
   * Handle talker.turn — async belief update.
   * Rate-limited: only sends to Letta every BELIEF_UPDATE_INTERVAL_MS.
   * Between updates, turns are accumulated locally (no Letta call).
   */
  private async handleTalkerTurn(event: TalkerTurnEvent): Promise<void> {
    // Skip if System 2 already handling this turn
    if (this.trigger.system2Active) {
      return; // Don't even queue — trigger handler will handle context
    }
    // Skip if in cooldown
    if (Date.now() < this.cooldownUntil) return;

    // Always accumulate text locally
    this.pendingTurns.push(event.text);

    // Rate-limit: only do Letta belief updates every 45s
    const sinceLastUpdate = Date.now() - this.lastBeliefUpdate;
    if (sinceLastUpdate < Reasoner.BELIEF_UPDATE_INTERVAL_MS) {
      return; // Accumulate silently — no Letta call yet
    }

    // Skip if already processing a belief update
    if (this.processing) return;

    await this.processBeliefUpdate('');
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

  /** Process a belief update cycle — rate-limited, drains accumulated turns. */
  private async processBeliefUpdate(text: string): Promise<void> {
    this.processing = true;
    this.lastBeliefUpdate = Date.now();

    try {
      const allTurns = [...this.pendingTurns];
      if (text) allTurns.push(text);
      this.pendingTurns = [];
      const batchText = allTurns.join(' ').trim().slice(-1000);

      if (!batchText) {
        logger.debug('No turns to process for belief update');
        return;
      }

      if (this.agentId) {
        const prompt = `[BELIEF_UPDATE] Recent conversation:\n"${batchText}"\n\nUpdate the belief_state and conversation_context memory blocks using core_memory_replace. Update: conversation_topic, conversation_summary, coaching_phase as needed.`;

        const response = await this.sendToLetta(prompt, 90_000);
        await this.memory.syncFromLetta();
        this.extractResponse(response, true);
        this.bus.emit(E.reasonerBelief(this.sessionId, 'belief_update', ['belief_state', 'conversation_context']));
        logger.info({ batchSize: allTurns.length, responseMsgs: response.length }, 'Belief updated');
      } else {
        await this.memory.updateConvState({
          summary: batchText.slice(0, 500),
          turn_count: ((this.memory.getBlockJSON<{ turn_count?: number }>('conversation_context'))?.turn_count ?? 0) + allTurns.length,
        });
        this.memory.emitCompressed();
        logger.info('Belief updated (local fallback)');
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        logger.warn('Belief update timed out');
        this.cooldownUntil = Date.now() + 20_000;
      } else {
        logger.error({ err }, 'Belief update failed');
        this.cooldownUntil = Date.now() + 15_000;
      }
    } finally {
      this.processing = false;
      // No more drain loop — next belief update happens when the rate-limit
      // interval expires and a new turn comes in.
    }
  }

  /** Send a message to the Letta agent and return the response messages. */
  private async sendToLetta(prompt: string, timeoutMs: number): Promise<LettaMessage[]> {
    if (!this.agentId) return [];

    const controller = new AbortController();
    const abortTimer = setTimeout(() => controller.abort(), timeoutMs);

    // Use Promise.race as belt-and-suspenders — Bun's AbortController
    // may not reliably abort hung fetch connections.
    const timeoutPromise = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new DOMException('Letta timeout', 'AbortError')), timeoutMs + 1000),
    );

    try {
      const response = await Promise.race([
        this.lettaPost<{ messages: LettaMessage[] }>(
          `/v1/agents/${this.agentId}/messages`,
          { messages: [{ role: 'user', content: prompt }] },
          controller.signal,
        ),
        timeoutPromise,
      ]);
      return response.messages ?? [];
    } finally {
      clearTimeout(abortTimer);
    }
  }

  // ─── Prompt Building ─────────────────────────────────────────

  /** Build the prompt for a trigger activation event. */
  private buildTriggerPrompt(event: TriggerActivateEvent): string {
    switch (event.reason) {
      case 'periodic':
        return `[OBSERVATION] The voice model (PersonaPlex, a small 7B model) just said:\n"${event.context}"\n\nEvaluate what PersonaPlex said. It is a 7B voice model with NO tools — it often hallucinates facts confidently. You have web_search, core_memory, and other real tools.\n\nInstructions:\n1. Update your belief_state and conversation_context memory blocks as needed (use core_memory_replace).\n2. If PersonaPlex said something factually wrong, confused, or if the user would benefit from a real answer — you MUST call the send_message tool with a natural spoken correction or helpful addition. Use web_search first if you need real facts.\n3. If PersonaPlex is doing fine (social chat, greetings, nothing wrong) — just update beliefs silently. Do NOT call send_message unless you have something genuinely useful to add.\n\nIMPORTANT: To speak to the user, you MUST use the send_message tool. Do NOT put your response in assistant_message content — that is only visible internally. Only send_message reaches the user.`;

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
  persona (read-only), belief_state, conversation_context, action_queue, fact_check
- When triggered, call tools and respond via send_message
- Keep conversation_context updated after every interaction

Memory blocks:
- persona: Your identity and capabilities (read-only)
- belief_state: User goals, preferences, expertise level, barriers, current project
- conversation_context: Current topic, summary, phase, turn count
- action_queue: Pending/running/completed tasks
- fact_check: Corrections for PersonaPlex hallucinations

Rules:
- For [BELIEF_UPDATE]: silently update memory blocks, no send_message needed
- For [OBSERVATION]: evaluate PersonaPlex output, only send_message if you have something useful
- For [VOICE_INTERCEPT] and [ACTION_REQUEST]: you MUST call send_message
- Keep conversation_context.summary under 500 chars (rolling window)
- Be concise — your responses will be spoken aloud`;
  }

  // ─── Response Extraction ─────────────────────────────────────

  /**
   * Parse tool_call arguments — Letta API returns them as a JSON string,
   * not a parsed object.
   */
  private parseArgs(args: unknown): Record<string, unknown> {
    if (typeof args === 'string') {
      try { return JSON.parse(args); } catch { return {}; }
    }
    if (args && typeof args === 'object') return args as Record<string, unknown>;
    return {};
  }

  /**
   * Extract user-facing text from Letta response messages.
   *
   * Handles TWO response formats:
   *   1. Native tool calls: message_type='tool_call_message' with tool_call object
   *   2. Text tool calls: message_type='assistant_message' with JSON in content
   *      (used by models like qwen2.5-coder, GLM-4.7 that don't emit native tool_use)
   *
   * For text tool calls, memory operations (core_memory_replace/append) are
   * executed directly via the Letta blocks API.
   *
   * @param onlySendMessage - When true, only extract send_message tool calls
   */
  private extractResponse(messages: LettaMessage[], onlySendMessage = false): string {
    const texts: string[] = [];
    logger.debug({ msgCount: messages.length, onlySendMessage }, 'Extracting response from Letta messages');

    for (const m of messages) {
      // 1. Native tool_call_message (Claude, GPT-4, etc.)
      //    IMPORTANT: Letta already executed these tool calls internally.
      //    We extract send_message text and dispatch missions, but DO NOT
      //    re-execute memory writes — that causes double-write corruption
      //    where old_content mismatches and the fallback overwrites the block.
      const tc = m.tool_call;
      if (m.message_type === 'tool_call_message' && tc?.name === 'send_message') {
        const parsed = this.parseArgs(tc.arguments);
        if (parsed.message) {
          texts.push(String(parsed.message));
        }
      }
      if (m.message_type === 'tool_call_message' && tc?.arguments) {
        const parsed = this.parseArgs(tc.arguments);
        if (tc.name === 'execute_ops_mission' || tc.name === 'execute_mission') {
          this.triggerOpsMission(parsed);
        }
        // NOTE: core_memory_replace/append are NOT re-executed here.
        // Letta already processed them. We sync blocks after processing.
        if (tc.name === 'core_memory_replace' || tc.name === 'core_memory_append') {
          logger.debug({ name: tc.name, label: parsed.label ?? parsed.block_label }, 'Skipping native memory tool call (Letta already executed)');
        }
      }

      // 2. Text-based tool calls in assistant_message content (qwen, GLM-4.7, etc.)
      if (m.message_type === 'assistant_message' && m.content) {
        const calls = this.parseTextToolCalls(m.content);
        if (calls.length > 0) {
          for (const call of calls) {
            if (call.name === 'send_message' && call.arguments?.message) {
              texts.push(String(call.arguments.message));
            } else if (call.name === 'core_memory_replace' || call.name === 'core_memory_append') {
              this.executeMemoryToolCall(call.name, call.arguments);
            } else if (call.name === 'execute_ops_mission' || call.name === 'execute_mission') {
              this.triggerOpsMission(call.arguments);
            }
          }
        } else if (!onlySendMessage) {
          texts.push(m.content);
        }
      }
    }

    return texts.join(' ').trim();
  }

  /**
   * Parse one or more JSON tool calls from assistant_message text content.
   * Models that don't support native tool calling concatenate JSON objects like:
   *   {"name": "core_memory_replace", "arguments": {...}}
   *   {"name": "send_message", "arguments": {"message": "Hello"}}
   *
   * Uses brace-depth parsing to extract individual JSON objects.
   */
  private parseTextToolCalls(content: string): Array<{ name: string; arguments: Record<string, unknown> }> {
    const results: Array<{ name: string; arguments: Record<string, unknown> }> = [];

    // Brace-depth extraction: find all top-level JSON objects
    const objects = this.extractJsonObjects(content);
    for (const obj of objects) {
      try {
        const parsed = JSON.parse(obj);
        if (parsed?.name && typeof parsed.name === 'string' && parsed.arguments) {
          results.push({ name: parsed.name, arguments: parsed.arguments });
        }
      } catch { /* skip malformed */ }
    }

    // Fallback: try code blocks
    if (results.length === 0) {
      const codeBlockMatch = content.match(/```(?:json)?\s*\n?([\s\S]*?)\n?\s*```/g);
      if (codeBlockMatch) {
        for (const block of codeBlockMatch) {
          const inner = block.replace(/```(?:json)?\s*\n?/, '').replace(/\n?\s*```/, '');
          for (const obj of this.extractJsonObjects(inner)) {
            try {
              const parsed = JSON.parse(obj);
              if (parsed?.name && typeof parsed.name === 'string' && parsed.arguments) {
                results.push({ name: parsed.name, arguments: parsed.arguments });
              }
            } catch { /* skip */ }
          }
        }
      }
    }

    if (results.length > 0) {
      logger.debug({ count: results.length, names: results.map(r => r.name) }, 'Parsed text tool calls');
    }

    return results;
  }

  /** Extract top-level JSON objects from a string using brace-depth counting. */
  private extractJsonObjects(text: string): string[] {
    const objects: string[] = [];
    let depth = 0;
    let start = -1;
    let inString = false;
    let escape = false;

    for (let i = 0; i < text.length; i++) {
      const ch = text[i];

      if (escape) { escape = false; continue; }
      if (ch === '\\' && inString) { escape = true; continue; }
      if (ch === '"' && !escape) { inString = !inString; continue; }
      if (inString) continue;

      if (ch === '{') {
        if (depth === 0) start = i;
        depth++;
      } else if (ch === '}') {
        depth--;
        if (depth === 0 && start >= 0) {
          objects.push(text.slice(start, i + 1));
          start = -1;
        }
      }
    }

    return objects;
  }

  /**
   * Execute core_memory_replace or core_memory_append via direct block writes.
   * Bypasses Letta inference — writes directly to the block via PATCH API.
   */
  private executeMemoryToolCall(name: string, args: Record<string, unknown>): void {
    const label = String(args.label ?? args.block_label ?? '');
    const content = String(args.content ?? args.new_content ?? '');
    const oldContent = String(args.old_content ?? '');

    if (!label) {
      logger.warn({ name, args }, 'Memory tool call missing label');
      return;
    }

    (async () => {
      try {
        // Get the block ID from memory
        const blockId = this.memory.getBlockId(label);
        if (!blockId) {
          logger.warn({ label }, 'No block ID found for label');
          return;
        }

        const env = getEnv();
        let newValue: string;

        if (name === 'core_memory_append') {
          // Append: get current value, add content
          const current = this.memory.getBlock(label) ?? '';
          newValue = current + '\n' + content;
        } else {
          // Replace: try substring replace first, fall back to full replace
          const current = this.memory.getBlock(label) ?? '';
          if (oldContent && current.includes(oldContent)) {
            newValue = current.replace(oldContent, content);
          } else {
            // Model's old_content doesn't match (common with non-native tool callers)
            // Use new_content as the full block value
            newValue = content;
            logger.debug({ label, oldContentLen: oldContent.length, matched: false }, 'old_content not found, using full replace');
          }
        }

        // Write directly to Letta block API
        const res = await fetch(`${env.LETTA_BASE_URL}/v1/blocks/${blockId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value: newValue }),
        });

        if (res.ok) {
          // Update local cache
          this.memory.setBlockLocal(label, newValue);
          logger.info({ name, label }, 'Memory tool call executed via direct write');
        } else {
          logger.error({ status: res.status, label }, 'Direct block write failed');
        }
      } catch (err) {
        logger.error({ err, name, label }, 'Memory tool call execution failed');
      }
    })();
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
