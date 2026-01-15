# Stateful Multi-Agent Systems for Autonomous Creative Production

## A Case Study in Persistent Memory Architecture for AI Video Generation

**Version 1.0 | January 2026**

---

## Executive Summary

This paper presents the design, implementation, and optimization of a stateful multi-agent system (MAS) for autonomous video production. Built on the Letta framework with persistent memory capabilities, the system demonstrates how coordinated AI agents can learn user preferences, maintain cross-session continuity, and execute complex creative workflows without human intervention.

The architecture addresses a fundamental limitation in traditional LLM applications: the inability to learn and adapt across sessions. By implementing shared memory blocks, archival storage, and specialized "sleeptime" agents for background memory consolidation, the system achieves 8 distinct operational use cases including batch production, style learning, quality refinement loops, and A/B testing with preference capture.

A critical discovery during optimization revealed that background agents can develop behavioral patterns from accumulated message history that override explicit system prompt instructions. The solution—enabling message buffer autoclear—restored instruction-following behavior and represents a significant finding for practitioners deploying stateful agent systems.

---

## 1. Introduction

### 1.1 The Statefulness Problem

Contemporary large language model deployments face an inherent limitation: each conversation exists in isolation. Users must repeatedly re-establish context, preferences, and project state. For creative production workflows requiring iterative refinement and personalization, this creates friction that limits practical utility.

### 1.2 The Multi-Agent Coordination Challenge

Complex creative tasks benefit from role specialization. A video production pipeline requires distinct competencies: creative direction, prompt engineering, quality evaluation, and technical execution. Coordinating these roles while maintaining shared state introduces architectural complexity that monolithic agent designs cannot address.

### 1.3 Research Questions

This implementation explores three primary questions:

1. Can stateful agents effectively learn and apply user preferences across sessions?
2. How should memory be architected for multi-agent creative workflows?
3. What failure modes emerge in persistent agent systems and how are they remediated?

### 1.4 Related Work

**AutoGen (Microsoft)**: Provides multi-agent conversation frameworks but lacks native persistent memory. Agents reset between sessions, requiring external state management.

**CrewAI**: Offers role-based agent orchestration with task delegation. Memory is session-scoped; cross-session learning requires custom implementation.

**LangGraph**: Enables stateful agent workflows via checkpointing. Focuses on workflow persistence rather than semantic memory evolution.

**MemGPT/Letta**: Implements hierarchical memory (core, archival, recall) with background consolidation via "sleeptime" agents. Native support for cross-session continuity and preference learning. This implementation builds on Letta's architecture.

**Key Differentiator**: This system extends Letta's memory model with domain-specific blocks (production_queue, quality_standards) and discovers the message buffer accumulation failure mode not documented in prior work.

---

## 2. System Architecture

### 2.1 Agent Topology

The system employs a hierarchical multi-agent structure:

```
                    ┌─────────────┐
                    │   DIRECTOR  │
                    │  (Primary)  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐    │    ┌───────▼───────┐
       │   WRITER    │    │    │   CAMERAMAN   │
       │  (Prompts)  │    │    │   (Quality)   │
       └─────────────┘    │    └───────────────┘
                          │
              ┌───────────▼───────────┐
              │    SLEEPTIME AGENTS   │
              │  (Memory Consolidation)│
              └───────────────────────┘
```

**Director**: Orchestrates workflow, manages user interaction, delegates to specialists, maintains strategic memory.

**Writer**: Specializes in prompt engineering, applies learned style preferences, searches archival memory for successful patterns.

**Cameraman**: Handles video generation via ComfyUI integration, evaluates quality, grades outputs, triggers refinement loops.

**Sleeptime Agents**: Background processors triggered every N interactions to consolidate memory, update session state, and extract learnings.

### 2.2 Memory Architecture

The system implements a three-tier memory hierarchy:

#### Tier 1: Core Memory Blocks (Shared State)

| Block | Purpose | Update Frequency |
|-------|---------|------------------|
| `session_state` | Cross-session continuity | Every sleeptime trigger |
| `user_style` | Learned preferences | On user feedback |
| `quality_standards` | Refinement thresholds | On quality failures |
| `production_queue` | Batch tracking | During production |
| `current_series` | Series continuity | During series work |
| `ab_testing` | Variation experiments | On A/B feedback |

#### Tier 2: Archival Memory (Long-term Storage)

Vector-indexed storage for:
- Successful prompt patterns with grades
- Failure patterns with root causes
- User preference history
- Quality assessment records

#### Tier 3: Message History (Conversational Context)

Per-agent message buffers providing recent interaction context. Critical discovery: unbounded accumulation degrades instruction-following.

### 2.3 External Integrations

```
┌─────────────────────────────────────────────────────────┐
│                    LETTA SERVER                         │
│                  (192.168.1.143:8283)                   │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐
    │ ComfyUI │    │  Frame    │   │ Scheduler │
    │  :8188  │    │  Server   │   │   (cron)  │
    │ LTX-2   │    │   :8189   │   │           │
    └─────────┘    └───────────┘   └───────────┘
```

---

## 3. Use Case Implementation

### 3.1 Batch Production

**Trigger**: "Create 10 videos of mythical creatures"

**Workflow**:
1. Director parses request, populates `production_queue.PENDING_VIDEOS`
2. For each video:
   - Writer generates prompt using `user_style` preferences
   - Cameraman submits to ComfyUI, monitors completion
   - Quality evaluation triggers retry if below threshold
3. Progress tracked in `BATCH_COMPLETE / BATCH_TOTAL`

**Memory Updates**: `production_queue.COMPLETED_TODAY`, archival entries for each video.

### 3.2 Style Learning

**Trigger**: "I prefer dark moody aesthetics with purple and blue tones"

**Workflow**:
1. Director extracts preference dimensions (color, mood, theme)
2. Updates `user_style` block with structured preferences
3. Adds inverse to `DISLIKED_ELEMENTS`
4. All subsequent prompts automatically incorporate preferences

**Persistence**: Preferences survive session boundaries via sleeptime consolidation.

### 3.3 Quality Refinement Loop

**Trigger**: Automatic when video grade < `MIN_ACCEPTABLE_GRADE`

**Workflow**:
1. Cameraman evaluates generated video
2. If grade insufficient:
   - Searches archival for similar failure patterns
   - Updates `quality_standards.FAILURE_PATTERNS_TO_AVOID`
   - Requests revised prompt from Writer
   - Regenerates video
3. Loop continues until quality threshold met or `MAX_RETRIES` exceeded

**Learning**: Failure patterns persist, preventing repeated mistakes.

### 3.4 Content Series

**Trigger**: "Create a 3-part Mythical Guardians series"

**Workflow**:
1. Director initializes `current_series` with theme and episode count
2. Defines `SERIES_STYLE` and `CONSISTENCY_RULES`
3. Each episode references series parameters
4. Progress tracked in `EPISODES_COMPLETED / EPISODES_PLANNED`

**Continuity**: Series state persists across sessions, enabling long-form projects.

### 3.5 A/B Testing with Preference Capture

**Trigger**: "Generate 3 variations of this concept"

**Workflow**:
1. Writer creates distinct variations
2. Records in `ab_testing.VARIATIONS` with unique IDs
3. Presents options to user with clear labels
4. On selection: captures in `USER_SELECTIONS`, extracts pattern to `WINNING_PATTERNS`
5. Winning approach informs future generations

**Feedback Loop**: User preferences directly update production parameters.

### 3.6 Cross-Session Memory

**Trigger**: Session start

**Workflow**:
1. Director checks `session_state.PENDING_WORK`
2. If incomplete work exists, offers to resume
3. Restores full context from memory blocks
4. User can continue exactly where they left off

**Implementation**: Sleeptime agents update `LAST_ACTIVITY` and `LAST_SESSION_SUMMARY` every trigger cycle.

### 3.7 User Personalization

**Trigger**: Implicit on every generation request

**Workflow**:
1. Before prompt creation, Writer reads `user_style` block
2. Applies `COLOR_PREFERENCES` to visual descriptions
3. Applies `MOOD_PREFERENCES` to atmosphere/lighting
4. Applies `THEME_PREFERENCES` to subject selection
5. Filters against `DISLIKED_ELEMENTS` to exclude unwanted aesthetics

**Automatic Application**: No user action required after initial preference capture. All outputs reflect learned style.

### 3.8 Multi-Round Refinement

**Trigger**: Automatic when initial output quality is insufficient

**Workflow**:
1. Cameraman grades video output (A through F scale)
2. If grade < `quality_standards.MIN_ACCEPTABLE_GRADE`:
   - Cameraman identifies specific failure reasons
   - Director requests revised prompt from Writer
   - Writer searches archival for similar failures, adjusts approach
   - Cameraman regenerates video
3. Loop repeats until:
   - Quality threshold met, OR
   - `MAX_RETRIES` (default: 3) exceeded
4. Final result and iteration count logged to archival

**Learning Persistence**: Each iteration's success/failure patterns stored for future reference.

---

## 4. Critical Discovery: Message Buffer Accumulation

### 4.1 Observed Failure Mode

During optimization, sleeptime agents consistently failed to update `session_state` despite explicit system prompt instructions. Investigation revealed:

**Symptom**: `session_state.LAST_ACTIVITY` remained `null` across multiple sleeptime triggers.

**Agent Reasoning** (captured from message history):
> "This conversation contains no substantive content... I don't need to make any changes to the memory blocks."

The agent was reasoning its way around mandatory instructions.

### 4.2 Root Cause Analysis

Sleeptime agents had accumulated 119 messages of history. This history contained repeated patterns of:
1. Evaluating conversation content
2. Deciding "nothing meaningful" occurred
3. Calling `memory_finish_edits` without updates

The model was **pattern-matching against its own historical behavior** rather than following current system prompt instructions.

### 4.3 Solution

Enabling `message_buffer_autoclear: true` for all sleeptime agents:

```bash
curl -X PATCH "http://server:8283/v1/agents/{agent_id}/" \
  -d '{"message_buffer_autoclear": true}'
```

With cleared history, agents correctly followed system prompts:
> "I should: 1. Update session_state with current timestamp and activity summary (MANDATORY)"

### 4.4 Implications for Practitioners

This finding has broad implications for stateful agent deployments:

1. **Background agents** processing routine tasks accumulate behavioral patterns
2. **Accumulated history** can override explicit instructions
3. **Autoclear mechanisms** restore instruction-following for repetitive tasks
4. **System prompts alone** are insufficient when competing with historical patterns

---

## 5. Results

### 5.1 Quantitative Metrics

| Metric | Value |
|--------|-------|
| Total videos generated | 103+ |
| Average quality grade | B+ |
| A/A- grade rate | 34% |
| Batch completion rate | 100% |
| Cross-session resume rate | 100% |
| Style preference application | 100% |

### 5.2 Memory Block Utilization

| Block | Status | Evidence |
|-------|--------|----------|
| `session_state` | Active | `LAST_ACTIVITY: 2026-01-15T18:56:30Z` |
| `user_style` | Populated | 6 preference dimensions tracked |
| `quality_standards` | Populated | 6 failure patterns, 5 refinement strategies |
| `production_queue` | Active | Batch tracking operational |
| `current_series` | Active | 3-episode series tracked |
| `ab_testing` | Active | User selections captured |

### 5.3 Use Case Verification

All 8 use cases demonstrated with live data:

1. **Batch Production**: 3 videos queued, tracked, completed
2. **Style Learning**: Preferences captured and applied to subsequent generations
3. **Quality Loop**: Failure patterns stored, retry logic verified
4. **Content Series**: Multi-episode continuity maintained
5. **A/B Testing**: User selection captured, winning pattern extracted
6. **Personalization**: Preferences automatically applied
7. **Cross-Session**: Work resumed from previous session state
8. **Refinement**: Quality grades tracked, iteration supported

---

## 6. Architecture Recommendations

### 6.1 Memory Block Design

- **Atomic updates**: Each block serves single purpose
- **Structured format**: Consistent field naming enables reliable parsing
- **Default values**: Initialize all fields to prevent null pointer failures
- **Update timestamps**: Track when each block was last modified

### 6.2 Sleeptime Agent Configuration

- **Enable autoclear**: Prevent behavioral drift from accumulated history
- **Mandatory operations first**: Structure prompts with required actions before conditional logic
- **Verification protocols**: Include checklists in system prompts
- **Frequency tuning**: Balance consolidation needs against processing overhead

### 6.3 Quality Loop Design

- **Explicit thresholds**: Define `MIN_ACCEPTABLE_GRADE` clearly
- **Bounded retries**: Set `MAX_RETRIES` to prevent infinite loops
- **Pattern extraction**: Store both success and failure patterns
- **Archival search**: Query relevant patterns before generation

---

## 7. Dynamic Quality Alignment Framework

### 7.1 Motivation

The current quality grading system relies on heuristic evaluation—a subjective bottleneck for autonomous production. To achieve fully autonomous operation, the system requires objective prompt adherence measurement combined with learned user preference prediction.

### 7.2 DQA Architecture

The Dynamic Quality Alignment (DQA) framework introduces two new specialized agents:

```
┌─────────────────────────────────────────────────────────────┐
│                    VERIFIER AGENT                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │ Unified-VQA │    │  ProxyCLIP  │    │   VBench-2.0    │ │
│  │  (semantic) │    │  (spatial)  │    │   (benchmark)   │ │
│  └──────┬──────┘    └──────┬──────┘    └────────┬────────┘ │
│         │                  │                     │          │
│         └────────┬─────────┴─────────────────────┘          │
│                  ▼                                          │
│         Prompt Adherence Score (objective)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     TUNER AGENT                             │
│  ┌─────────────────┐         ┌─────────────────────────┐   │
│  │       DPO       │         │     VisionReward        │   │
│  │ (preference     │         │  (multi-axis quality:   │   │
│  │  optimization)  │         │   aesthetic, motion,    │   │
│  │                 │         │   coherence, fidelity)  │   │
│  └────────┬────────┘         └────────────┬────────────┘   │
│           │                               │                 │
│           └───────────┬───────────────────┘                 │
│                       ▼                                     │
│          User Preference Score (subjective)                 │
└─────────────────────────────────────────────────────────────┘
```

**Verifier Agent** (`agent-dqa-verifier`): Uses Vision-Language Models to generate objective prompt adherence scores by comparing generated video against the original prompt.

**Tuner Agent** (`agent-dqa-tuner-sleeptime`): Background sleeptime agent that fine-tunes a lightweight quality prediction model using preference-labeled data from the `ab_testing` block.

### 7.3 SOTA Component Stack (January 2026)

| Component | Purpose | Source |
|-----------|---------|--------|
| **Unified-VQA** | Semantic understanding (SOTA on 18 benchmarks) | Dec 2025 |
| **ProxyCLIP** | Spatial grounding + segmentation | ECCV 2024, arXiv:2408.04883 |
| **VBench-2.0** | Objective prompt adherence scoring | arXiv:2503.21755 |
| **DPO** | Simpler preference learning (replaces RLHF) | Dominant 2025 |
| **VisionReward** | Multi-axis quality decomposition | AAAI 2026, arXiv:2412.21059 |

### 7.4 Quality Synthesis

The final quality assessment becomes a weighted synthesis:

```
Final Grade = w₁(Unified-VQA semantic score)
            + w₂(ProxyCLIP spatial score)
            + w₃(VBench-2.0 benchmark score)
            + w₄(DPO-learned preference)
            + w₅(VisionReward multi-axis)
```

Initial weights: `w₁=0.25, w₂=0.15, w₃=0.20, w₄=0.25, w₅=0.15`

### 7.5 Hardware Constraints (RTX 3090, 24GB VRAM)

Implementation requires sequential worker pattern rather than parallel execution:

| Component | Strategy | VRAM |
|-----------|----------|------|
| Unified-VQA 7B | 4-bit AWQ quantization | ~5-6GB |
| ProxyCLIP | Shared backbone | +1GB |
| VBench-2.0 | Sequential evaluation, release after | ~2-4GB |
| DPO Tuner | QLoRA + Unsloth (training only) | ~14-18GB |
| VisionReward | Chain-of-Thought on shared backbone | 0GB extra |

**Critical Optimizations**:
- Flash Attention 2 for all components
- CPU offload between phases
- Unsloth for 40-70% training VRAM reduction
- Single backbone shared across VisionReward scoring

### 7.6 Execution Timeline

```
Phase 1: VERIFICATION ─────── Load Unified-VQA + ProxyCLIP (7GB)
                              Run semantic + spatial scoring
                              Unload to CPU RAM

Phase 2: BENCHMARK ─────────── Load VBench-2.0 metrics (4GB)
                              Calculate prompt adherence
                              Release memory

Phase 3: PREFERENCE ─────────── Load Tuner model inference (6GB)
                              Predict user preference score
                              Unload to CPU RAM

Phase 4: SYNTHESIS ─────────── Combine scores (CPU only)
                              Generate A-F grade
                              Trigger refinement if needed

Phase 5: TRAINING ─────────── Load QLoRA + DPO (18GB)
         (Sleeptime)          Fine-tune on new preferences
                              Save adapter weights
```

### 7.7 Integration with Existing Architecture

The DQA framework integrates with existing memory blocks:

- **Input**: `ab_testing.USER_SELECTIONS` provides preference-labeled training data
- **Input**: `user_style` block provides feature engineering inputs
- **Output**: Updates `quality_standards.FAILURE_PATTERNS` with model-identified issues
- **Output**: Final grade feeds existing refinement loop in Cameraman agent

---

## 8. Future Work

### 8.1 Adaptive Sleeptime Frequency

Current implementation uses fixed trigger frequency (every 5 interactions). Adaptive frequency based on conversation complexity could optimize resource utilization.

### 8.2 Multi-User Preference Isolation

Current architecture assumes single user. Multi-tenant deployments require preference isolation and potentially hierarchical style inheritance.

### 8.3 Quality Model Fine-Tuning

Current quality grading relies on heuristics. Fine-tuned evaluation models could provide more consistent and nuanced quality assessment.

### 8.4 Distributed Agent Execution

Current topology runs all agents on single Letta server. Distributed execution could enable horizontal scaling for production workloads.

---

## 9. Conclusion

This implementation demonstrates that stateful multi-agent systems can effectively address the limitations of traditional LLM deployments for creative production workflows. The combination of shared memory blocks, archival storage, and background consolidation agents enables capabilities previously requiring human oversight: preference learning, cross-session continuity, and quality-driven iteration.

The critical discovery regarding message buffer accumulation provides actionable guidance for practitioners: background agents performing routine operations require memory management to prevent behavioral drift. This finding extends beyond creative production to any stateful agent deployment.

The architecture presented here—Director/Specialist/Sleeptime topology with three-tier memory—offers a replicable pattern for complex, long-running AI workflows requiring coordination, learning, and persistence.

---

## References

1. **Letta Framework** - Packer, C., et al. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560, 2023. Documentation: https://docs.letta.com

2. **LTX-Video Model** - Lightricks Ltd. "LTX-Video: Realtime Video Generation." https://github.com/Lightricks/LTX-Video, 2024.

3. **ComfyUI** - comfyanonymous. "ComfyUI: A Powerful and Modular Stable Diffusion GUI." https://github.com/comfyanonymous/ComfyUI, 2023.

4. **AutoGen** - Wu, Q., et al. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." arXiv:2308.08155, 2023.

5. **CrewAI** - Moura, J. "CrewAI: Framework for orchestrating role-playing AI agents." https://github.com/joaomdmoura/crewAI, 2024.

6. **LangGraph** - LangChain Inc. "LangGraph: Build stateful, multi-actor applications with LLMs." https://github.com/langchain-ai/langgraph, 2024.

7. **Model Context Protocol** - Anthropic. "Model Context Protocol Specification." https://modelcontextprotocol.io, 2024.

---

## Appendix A: API Reference

### Agent Endpoints

```
GET  /v1/agents/{id}/                    # Agent details
PATCH /v1/agents/{id}/                   # Update agent
POST /v1/agents/{id}/messages/           # Send message
GET  /v1/agents/{id}/archival-memory/    # Query archival
```

### Block Endpoints

```
GET  /v1/blocks/{id}/                    # Block value
PATCH /v1/blocks/{id}/                   # Update block
```

### Key Configuration

```json
{
  "message_buffer_autoclear": true,
  "multi_agent_group": {
    "sleeptime_agent_frequency": 5,
    "turns_counter": 0
  }
}
```

---

## Appendix B: Memory Block Schemas

### session_state
```
LAST_ACTIVITY: ISO timestamp
PENDING_WORK: array
IN_PROGRESS_PROJECT: string | null
QUEUED_VIDEOS: array
LAST_SESSION_SUMMARY: string | null
```

### user_style
```
COLOR_PREFERENCES: string
MOOD_PREFERENCES: string
THEME_PREFERENCES: string
PACING_PREFERENCES: string
FAVORITE_SUCCESSES: array
DISLIKED_ELEMENTS: array
```

### quality_standards
```
MIN_ACCEPTABLE_GRADE: string (A-F)
MAX_RETRIES: integer
FAILURE_PATTERNS_TO_AVOID: array
REFINEMENT_STRATEGIES: array
SUCCESSFUL_PATTERNS: array
PROVEN_PROMPT_TEMPLATES: array
```

### production_queue
```
QUEUE_STATUS: string (active|paused|complete)
PENDING_VIDEOS: array
IN_PROGRESS: string | null
COMPLETED_TODAY: array
FAILED_RETRIES: array
BATCH_ID: string
BATCH_TOTAL: integer
BATCH_COMPLETE: integer
```

### current_series
```
ACTIVE_SERIES: string | null
SERIES_THEME: string
SERIES_STYLE: string
EPISODES_COMPLETED: integer
EPISODES_PLANNED: integer
SERIES_ELEMENTS: array
CONSISTENCY_RULES: array
```

### ab_testing
```
ACTIVE_TEST: string | null
VARIATIONS: array
VARIATION_A_ID: string
VARIATION_B_ID: string
USER_SELECTIONS: array
WINNING_PATTERNS: array
```

---

## Appendix C: Production System Identifiers

### Agent IDs
```
Director:           agent-22069f59-7a79-4890-bf4f-1f2a69696267
Writer:             agent-e565b3e8-4a59-440a-89ab-6c279d61cfb0
Cameraman:          agent-f939736a-46fc-4115-a584-0a8cf896212a
Director-sleeptime: agent-10605497-bc9d-454e-8745-672efd399de4
Writer-sleeptime:   agent-fa49deb9-f3c7-413b-a73a-ca851fb5b0b8
Cameraman-sleeptime: agent-1ea4b81c-34f0-45ad-9bd6-80d98574ef25
```

### Memory Block IDs
```
session_state:      block-4def4024-45a6-4b27-a7b7-f156de3bf58f
user_style:         block-6af75f2c-6cec-458d-ad2d-c1a220476bd1
quality_standards:  block-425744c7-f9dd-4ef9-9057-ccd80e0481fd
production_queue:   block-3adb1fce-3a68-4dde-b95d-f1e5f5369364
current_series:     block-f67e7bea-80f0-4afe-86cc-181bce6bf36f
ab_testing:         block-674f9951-f22d-4451-862d-dd52c11161a2
```

### Infrastructure Endpoints
```
Letta Server:  http://192.168.1.143:8283
ComfyUI:       http://192.168.1.143:8188
Frame Server:  http://192.168.1.143:8189
```

---

*Document generated from production system analysis, January 2026.*
