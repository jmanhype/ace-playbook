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
   * Handle trigger.activate — bridge-side search + Ollama summarize.
   *
   * Architecture v3: Letta takes 30-90s (serialized). Unusable for voice.
   * Instead the bridge owns the real-time path:
   *   1. Bridge does DuckDuckGo search (~1-2s)
   *   2. Ollama summarizes results (~3-5s)
   *   3. Total: ~5-8s with REAL answers
   *   4. Background: Letta persists memory (non-blocking, no user output)
   */
  private async handleTrigger(event: TriggerActivateEvent): Promise<void> {
    const isLongCooldown = (this.cooldownUntil - Date.now()) > 12_000;
    const bypassCooldown = isLongCooldown && (event.reason === 'deflection' || event.reason === 'user_request');
    if (!bypassCooldown && Date.now() < this.cooldownUntil) {
      logger.debug({ cooldownRemainingSecs: Math.round((this.cooldownUntil - Date.now()) / 1000), reason: event.reason }, 'Skipping trigger (Letta cooldown)');
      return;
    }
    this.trigger.system2Active = true;
    this.bus.emit(E.reasonerThinking(this.sessionId, true));

    try {
      const t0 = Date.now();

      // ── Infer what the user wants from the context ──
      const query = this.inferSearchQuery(event);

      // ── Phase 1: Bridge-side DuckDuckGo search (~1-2s) ──
      let searchResults = '';
      if (query) {
        searchResults = await this.bridgeWebSearch(query);
        logger.info({ query, resultLength: searchResults.length, ms: Date.now() - t0 }, 'Bridge web search completed');
      }

      // ── Phase 2: Ollama summarizes with search context (~3-5s) ──
      const fastText = await this.fastOllamaRespond(event, searchResults);
      const totalMs = Date.now() - t0;

      if (fastText) {
        logger.info({ reason: event.reason, latencyMs: totalMs, hasSearch: !!searchResults, text: fastText.slice(0, 100) }, 'System 2 fast response');

        // Store in memory → compress includes it → PersonaPlex reconnects
        // with the answer in its prompt. System 2 never speaks to the user
        // directly. PersonaPlex (System 1) delivers the answer in its own voice.
        this.memory.setLastSystem2Response(fastText);

        // Emit interjection (browser debug panel only — NOT spoken aloud)
        this.bus.emit(E.reasonerInterjection(this.sessionId, fastText, event.reason));

        // Trigger memory.compressed → answer reconnect → PersonaPlex speaks it
        this.memory.emitCompressed();

        this.cooldownUntil = Date.now() + 15_000;
      } else {
        logger.warn({ reason: event.reason, latencyMs: totalMs }, 'System 2 fast path produced no response');
      }

      // ── Background: Letta memory persistence only (no user output) ──
      this.asyncLettaPersist(event, fastText ?? '').catch(err => {
        logger.warn({ err: err instanceof Error ? err.message : String(err) }, 'Background Letta persist failed');
      });
    } catch (err) {
      logger.error({ err: err instanceof Error ? err.message : String(err), reason: event.reason }, 'System 2 fast path failed');
      this.cooldownUntil = Date.now() + 15_000;
    } finally {
      this.bus.emit(E.reasonerThinking(this.sessionId, false));
      this.trigger.system2Active = false;
      this.trigger.resetHistory();
    }
  }

  /**
   * Infer a search query from the trigger context.
   *
   * Now that PersonaPlex-output triggers are disabled, the context is the
   * user's actual text (from Speech Recognition or typed input). Much simpler
   * extraction — just strip the command prefix and use the rest as the query.
   */
  private inferSearchQuery(event: TriggerActivateEvent): string {
    const ctx = (event.context ?? '').trim();
    if (!ctx || ctx.length < 3) return '';

    const lower = ctx.toLowerCase();

    // Strip command prefixes to get the actual topic
    const prefixPatterns = [
      /^(?:can you |please |could you |hey |okay )?(?:search|look up|find|check|google)\s+(?:for\s+)?(?:the\s+)?/i,
      /^(?:what(?:'s| is| are)\s+(?:the\s+)?)/i,
      /^(?:tell me about|what about|how about|look into)\s+/i,
      /^(?:i need|i want|get me|show me)\s+(?:info(?:rmation)?\s+(?:on|about)\s+)?/i,
    ];

    for (const pattern of prefixPatterns) {
      const stripped = ctx.replace(pattern, '').trim();
      if (stripped.length >= 3 && stripped.length < ctx.length) {
        logger.debug({ original: ctx.slice(0, 80), query: stripped.slice(0, 60) }, 'Search query extracted');
        return stripped.slice(0, 80);
      }
    }

    // If the text mentions search-related keywords, use the whole thing
    if (/\b(news|latest|weather|stock|price|score|update|who is|what happened)\b/i.test(lower)) {
      logger.debug({ query: ctx.slice(0, 60) }, 'Search query: full text');
      return ctx.slice(0, 80);
    }

    // For non-search triggers (user_request with 4+ words), still try to search
    // if the text looks like a question or request
    if (lower.startsWith('what') || lower.startsWith('who') || lower.startsWith('how') ||
        lower.startsWith('when') || lower.startsWith('where') || lower.startsWith('why')) {
      logger.debug({ query: ctx.slice(0, 60) }, 'Search query: question');
      return ctx.slice(0, 80);
    }

    // Default: use the raw text as the query
    logger.debug({ query: ctx.slice(0, 60) }, 'Search query: raw fallback');
    return ctx.slice(0, 80);
  }

  /**
   * Bridge-side DuckDuckGo web search. Runs directly in the bridge process.
   * No API key needed. ~1-2s latency.
   */
  private async bridgeWebSearch(query: string): Promise<string> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8_000);

    try {
      const encoded = encodeURIComponent(query);
      const res = await fetch(`https://html.duckduckgo.com/html/?q=${encoded}`, {
        headers: { 'User-Agent': 'Mozilla/5.0 (compatible; VAOSBridge/1.0)' },
        signal: controller.signal,
      });

      if (!res.ok) return '';
      const html = await res.text();

      // Parse search results from DuckDuckGo HTML
      const results: string[] = [];
      const linkRegex = /<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)<\/a>/gs;
      const snippetRegex = /<a[^>]*class="result__snippet"[^>]*>(.*?)<\/a>/gs;

      const links: Array<{ url: string; title: string }> = [];
      let match;
      while ((match = linkRegex.exec(html)) && links.length < 5) {
        const rawUrl = match[1];
        const title = match[2].replace(/<[^>]+>/g, '').trim();
        // Extract actual URL from DuckDuckGo redirect
        const uddg = rawUrl.match(/uddg=([^&]+)/);
        const url = uddg ? decodeURIComponent(uddg[1]) : rawUrl;
        links.push({ url, title });
      }

      const snippets: string[] = [];
      while ((match = snippetRegex.exec(html)) && snippets.length < 5) {
        snippets.push(match[1].replace(/<[^>]+>/g, '').trim());
      }

      for (let i = 0; i < links.length; i++) {
        const snippet = i < snippets.length ? snippets[i] : '';
        results.push(`${i + 1}. ${links[i].title}\n   ${snippet}\n   ${links[i].url}`);
      }

      return results.join('\n\n');
    } catch (err) {
      logger.warn({ err: err instanceof Error ? err.message : String(err), query }, 'Bridge web search failed');
      return '';
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Fast Ollama response with optional search context.
   * If search results are provided, Ollama summarizes them.
   * If not, Ollama gives a direct conversational response.
   * Typical latency: 3-5s.
   */
  private async fastOllamaRespond(event: TriggerActivateEvent, searchResults: string): Promise<string> {
    const env = getEnv();
    const ollamaUrl = env.OLLAMA_URL;
    const model = env.OLLAMA_MODEL;

    // Include memory context
    const belief = this.memory.getBlock('belief_state');
    const conv = this.memory.getBlock('conversation_context');
    let memoryCtx = '';
    if (belief && belief.length > 10) memoryCtx += `User context: ${belief.slice(0, 300)}\n`;
    if (conv && conv.length > 10) memoryCtx += `Conversation: ${conv.slice(0, 200)}\n`;

    let systemPrompt: string;
    let userPrompt: string;

    if (searchResults) {
      // Search + summarize mode
      systemPrompt = 'You are a helpful voice assistant. Summarize the search results below in 2-3 natural spoken sentences. Be conversational and concise — this will be read aloud. Focus on the most relevant/interesting findings.';
      userPrompt = `${memoryCtx}The user asked about: "${event.context?.slice(0, 200)}"\n\nSearch results:\n${searchResults.slice(0, 2000)}\n\nGive a natural spoken summary (2-3 sentences max).`;
    } else {
      // Direct response mode (no search needed)
      systemPrompt = 'You are a helpful voice assistant. The main voice model could not handle this request adequately. Give a brief, helpful response (2-3 sentences max). Be conversational — this will be read aloud.';
      userPrompt = `${memoryCtx}Context: "${event.context?.slice(0, 300)}"\n\nProvide a helpful response.`;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15_000);

    try {
      const res = await fetch(`${ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          system: systemPrompt,
          prompt: userPrompt,
          stream: false,
          options: { num_predict: 300, temperature: 0.7 },
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
   * Background Letta persistence — memory updates only, no user output.
   * Letta takes 30-90s so it NEVER produces user-facing responses.
   * It only persists what happened into memory blocks.
   */
  private async asyncLettaPersist(event: TriggerActivateEvent, bridgeResponse: string): Promise<void> {
    if (!this.agentId) return;

    // Tell Letta what happened so it can update memory
    const prompt = `[MEMORY_PERSIST] The voice bridge handled a trigger (reason: ${event.reason}).
Context: "${(event.context ?? '').slice(0, 300)}"
Bridge response to user: "${bridgeResponse.slice(0, 500)}"

Update belief_state and conversation_context memory blocks to reflect this interaction. Use core_memory_replace to update relevant fields. Do NOT call send_message — the user already received a response.`;

    try {
      const t0 = Date.now();
      const response = await this.sendToLetta(prompt, 90_000);
      await this.memory.syncFromLetta();
      logger.info({ reason: event.reason, latencyMs: Date.now() - t0, msgCount: response.length }, 'Letta memory persisted (background)');
      this.bus.emit(E.reasonerBelief(this.sessionId, event.reason, ['belief_state', 'conversation_context']));
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        logger.warn({ reason: event.reason }, 'Background Letta persist timed out');
      } else {
        logger.error({ err: err instanceof Error ? err.message : String(err) }, 'Background Letta persist failed');
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
    const sendMessageTexts: string[] = [];
    const fallbackTexts: string[] = [];
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
          sendMessageTexts.push(String(parsed.message));
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
              sendMessageTexts.push(String(call.arguments.message));
            } else if (call.name === 'core_memory_replace' || call.name === 'core_memory_append') {
              this.executeMemoryToolCall(call.name, call.arguments);
            } else if (call.name === 'execute_ops_mission' || call.name === 'execute_mission') {
              this.triggerOpsMission(call.arguments);
            }
          }
        } else if (!onlySendMessage) {
          // Only use raw assistant_message as fallback — these are internal
          // thoughts and should NOT be shown if send_message calls exist.
          fallbackTexts.push(m.content);
        }
      }
    }

    // Prefer send_message text (explicit user-facing output).
    // Only fall back to raw assistant_message if no send_message was found.
    if (sendMessageTexts.length > 0) {
      return sendMessageTexts.join(' ').trim();
    }
    return fallbackTexts.join(' ').trim();
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
