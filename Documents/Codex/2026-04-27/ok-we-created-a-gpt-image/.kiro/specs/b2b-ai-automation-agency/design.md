# Technical Design Document

## Overview

The B2B AI Automation Agency is a self-hosted, multi-tenant SMS automation platform that provides local businesses with AI-powered customer communication, lead management, and reactivation campaigns. The system runs entirely on existing infrastructure (Zimaboard + 3090 GPU server) connected via Tailscale mesh, achieving near-zero operating costs (~$0.007/text) while delivering $300-500/month value per tenant.

The platform orchestrates open-source components (Twenty CRM, Chatwoot, n8n) with a local Hermes AI agent to process inbound SMS, generate contextual responses, manage lead pipelines, and execute bulk reactivation campaigns — all without cloud API dependencies for core operations.

### Design Philosophy: Applied AI Skills Framework

This platform embodies the **9 AI Skills for Business Success** framework in practice:

1. **Ask AI First Default Reaction**: Every inbound SMS immediately triggers AI analysis before human intervention
2. **Skepticism & Verification**: AI responses tagged with confidence scores, escalation when uncertain
3. **Context Mastery (TICA Framework)**: Task-Information-Constraints-Ask prompts injected into every AI request
4. **Team Augmentation**: AI handles 90% of routine queries, humans focus on complex negotiations
5. **Treat AI Like New Hire**: Continuous feedback loops via sentiment analysis and response quality metrics
6. **Feedback Loops**: AI grades its own responses against quality rubrics before sending
7. **Documentation as Context**: Business profiles, FAQs, and conversation history stored as structured context
8. **AI Agents with Tools**: Hermes Agent integrated with CRM, SMS, and knowledge base tools
9. **Proven Business Model + AI**: Traditional SMS marketing automation enhanced with AI, not a novel "AI business"

This isn't trying to build an "AI business" from scratch — it's applying AI to a proven business model (SMS marketing for local businesses) to make it 10x more efficient.

## Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL BOUNDARY                             │
│                                                                      │
│  [Twilio/Bandwidth]  ←→  [Cloudflare Tunnel]  ←→  [Chatwoot]       │
│       (SMS Carrier)        (Public Ingress)        (Internal)        │
│                                                                      │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   │ Tailscale Mesh (100.x.x.x)
                                   │
┌──────────────────────────────────┴──────────────────────────────────┐
│                     INTERNAL INFRASTRUCTURE                           │
│                                                                      │
│  ┌─────────────────────┐          ┌──────────────────────────┐      │
│  │   ZIMABOARD          │          │   3090 GPU SERVER         │      │
│  │   (Services Layer)   │◄────────►│   (Compute Layer)         │      │
│  │                      │ Tailscale │                           │      │
│  │  - Chatwoot          │          │  - Hermes Agent Runtime   │      │
│  │  - Twenty CRM        │          │  - Paperclip AI           │      │
│  │  - n8n               │          │  - Knowledge Store        │      │
│  │  - Knowledge Store   │          │  - Ollama (Qwen3/Hermes)  │      │
│  │  - MCP Gateway       │          │                           │      │
│  │  - Coolify           │          │                           │      │
│  └─────────────────────┘          └──────────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Container Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ LXC 112: B2B Platform (Zimaboard - 192.168.1.180)              │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Chatwoot  │  │ Twenty   │  │   n8n    │  │  PostgreSQL   │  │
│  │ :3000     │  │ CRM      │  │  :5678   │  │  :5432        │  │
│  │           │  │ :3001    │  │          │  │               │  │
│  │ Webhooks  │  │ GraphQL  │  │ Workflows│  │ Shared DB     │  │
│  │ Twilio    │  │ API      │  │ Engine   │  │ (per-tenant   │  │
│  │ Routing   │  │ Pipeline │  │ Retry    │  │  schemas)     │  │
│  └─────┬────┘  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
│        │             │             │                 │           │
│        └─────────────┴─────────────┴─────────────────┘           │
│                            │                                     │
│  ┌─────────────────────────┴───────────────────────────────┐    │
│  │              Redis :6379 (Queue + Cache)                  │    │
│  │  - Campaign message queue                                │    │
│  │  - Rate limiter state                                    │    │
│  │  - Session cache                                         │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │         Cloudflare Tunnel (cloudflared)                    │    │
│  │  - Routes: sms-webhook.yourdomain.com → Chatwoot:3000    │    │
│  │  - Only exposes webhook endpoint, nothing else            │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3090 GPU Server (192.168.1.143 / 100.x.x.x via Tailscale)     │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Hermes Agent │  │ Paperclip AI │  │ Knowledge Store       │  │
│  │ Runtime      │  │ (Management) │  │ (pgvector)            │  │
│  │              │  │              │  │                        │  │
│  │ - Prompt     │  │ - Heartbeats │  │ - Business profiles   │  │
│  │   assembly   │  │ - Issues     │  │ - Conversation hist   │  │
│  │ - Context    │  │ - Monitoring │  │ - Semantic search     │  │
│  │   injection  │  │ - Approvals  │  │ - Embeddings          │  │
│  │ - Response   │  │              │  │                        │  │
│  │   generation │  │              │  │                        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              Ollama (Local LLM Inference)                  │    │
│  │  - Primary: Hermes-3-Llama-3.1-8B (fast, conversational) │    │
│  │  - Fallback: Qwen3-7B (reasoning, complex queries)        │    │
│  │  - Embeddings: nomic-embed-text (384-dim)                 │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Topology — Tri-Node Asymmetric Cluster

**Design Rationale:** The ZimaBoard (8GB RAM, Celeron) cannot handle heavy apps like Twenty CRM (Node.js) and Chatwoot (Ruby on Rails) at 50 tenants. The Mac Mini (Apple Silicon, unified memory, NVMe) is purpose-built for databases and heavy web apps. The 3090 is reserved exclusively for tensor math.

| Node | Role | SSH | Tailscale IP | LAN IP |
|------|------|-----|-------------|--------|
| **Zimaboard** | Edge Router & Queue | `ssh zima` | 100.112.106.69 | 192.168.1.123 |
| **Mac Mini** | State & DB Core | `ssh mini` | 100.94.237.121 | 192.168.1.196 |
| **3090** | Tensor Compute | `ssh 3090` | 100.77.225.85 | 192.168.1.143 |

#### Node 1: Zimaboard (Edge — Lightweight Only)

| Service | Port | Purpose |
|---------|------|---------|
| Cloudflare Tunnel (cloudflared) | — | Public ingress for Twilio webhooks |
| n8n | 5678 | Workflow orchestration, webhook handling, HMAC validation |
| Redis | 6379 | Message queue, rate limiter state, session cache |

#### Node 2: Mac Mini (State — Heavy Apps & Data)

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL + pgvector | 5432 | All databases (n8n, Twenty, Chatwoot, Knowledge Store) |
| Twenty CRM | 3000 | Lead pipeline, contact management, GraphQL API |
| Chatwoot | 3001 | Omnichannel inbox, SMS routing, Twilio integration |

#### Node 3: 3090 GPU Server (Compute — Tensor Only)

| Service | Port | Purpose |
|---------|------|---------|
| Ollama | 11434 | LLM inference (Hermes-3-8B, Qwen3-7B, nomic-embed-text) |
| Hermes Agent | 8080 | Prompt assembly, context injection, response generation |
| Paperclip AI | 8888 | Agent management, heartbeats, issue tracking |

#### Inter-Node Communication (All via Tailscale mesh — encrypted WireGuard)

```
Twilio → Cloudflare Tunnel → Zimaboard:n8n (webhook + HMAC validation)
                                    │
                                    ├──► Mac Mini:5432 (PostgreSQL queries via Tailscale)
                                    ├──► Mac Mini:3000 (Twenty CRM API via Tailscale)
                                    ├──► Mac Mini:3001 (Chatwoot API via Tailscale)
                                    └──► 3090:8080 (Hermes Agent inference via Tailscale)
```

| Existing Service | Host | Port | Notes |
|---------|------|------|-------|
| MCP Gateway | Zimaboard | 192.168.1.197:3001-3005 | LXC 106 (unchanged) |
| Media Automation | Zimaboard | 192.168.1.178 | LXC 110 (unchanged) |

## Data Models

### Tenant Configuration Schema

```json
{
  "tenant_id": "uuid",
  "business_name": "string",
  "phone_number": "+1XXXXXXXXXX",
  "twilio_sid": "string (encrypted)",
  "services": ["string"],
  "pricing_guidelines": "text",
  "operating_hours": {
    "timezone": "America/Chicago",
    "schedule": {
      "mon": {"open": "08:00", "close": "17:00"},
      "tue": {"open": "08:00", "close": "17:00"}
    }
  },
  "tone_preferences": "friendly|professional|casual",
  "calendar_rules": {
    "slot_duration_minutes": 60,
    "buffer_minutes": 15,
    "max_daily_bookings": 8
  },
  "escalation_contacts": [
    {"name": "string", "phone": "+1XXXXXXXXXX", "role": "owner"}
  ],
  "rate_limits": {
    "messages_per_second": 0.2,
    "daily_campaign_max": 200
  },
  "status": "active|paused|archived",
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "config_version": "integer"
}
```

### Contact/Lead Schema

```json
{
  "contact_id": "uuid",
  "tenant_id": "uuid (FK)",
  "phone_number": "+1XXXXXXXXXX",
  "name": "string (nullable)",
  "email": "string (nullable)",
  "pipeline_stage": "cold|contacted|negotiating|won|lost",
  "stage_changed_at": "ISO8601",
  "first_contact_at": "ISO8601",
  "last_activity_at": "ISO8601",
  "opt_out": false,
  "opt_out_at": "ISO8601 (nullable)",
  "suppressed": false,
  "tags": ["string"],
  "custom_fields": {},
  "twenty_crm_id": "string (external reference)",
  "chatwoot_contact_id": "integer (external reference)",
  "sentiment_score": "float (-1.0 to 1.0)",
  "total_messages_sent": "integer",
  "total_messages_received": "integer",
  "created_at": "ISO8601",
  "updated_at": "ISO8601"
}
```

### Conversation History Schema

```json
{
  "message_id": "uuid",
  "tenant_id": "uuid (FK)",
  "contact_id": "uuid (FK)",
  "direction": "inbound|outbound",
  "content": "text",
  "content_embedding": "vector(384)",
  "timestamp": "ISO8601",
  "channel": "sms|email|whatsapp",
  "carrier_status": "sent|delivered|failed|queued",
  "carrier_message_id": "string (nullable)",
  "ai_generated": "boolean",
  "escalated": "boolean",
  "sentiment": "positive|neutral|negative",
  "intent_classification": "scheduling|pricing|general|complaint|opt_out",
  "response_latency_ms": "integer (nullable)",
  "campaign_id": "uuid (nullable, FK)",
  "created_at": "ISO8601"
}
```

### Campaign Schema

```json
{
  "campaign_id": "uuid",
  "tenant_id": "uuid (FK)",
  "name": "string",
  "status": "draft|active|paused|completed|cancelled",
  "type": "reactivation|follow_up|announcement",
  "criteria": {
    "last_activity_before": "ISO8601",
    "pipeline_stages": ["lost", "cold"],
    "tags_include": ["string"],
    "tags_exclude": ["string"]
  },
  "message_template": "text (with {{variables}})",
  "personalization_enabled": true,
  "rate_limit_per_second": 0.33,
  "daily_send_limit": 200,
  "total_contacts": "integer",
  "sent_count": "integer",
  "response_count": "integer",
  "opt_out_count": "integer",
  "scheduled_start": "ISO8601",
  "started_at": "ISO8601 (nullable)",
  "completed_at": "ISO8601 (nullable)",
  "created_at": "ISO8601",
  "updated_at": "ISO8601"
}
```

## Component Design

### Component 1: SMS Gateway Layer (Chatwoot)

**Addresses:** Requirement 2 (Inbound SMS Processing), Requirement 11 (SMS Compliance)

**Deployment:**
- Chatwoot deployed as Docker Compose stack within LXC 112
- PostgreSQL shared instance (separate database: `chatwoot_production`)
- Redis shared instance (separate DB index: 2)
- Sidekiq workers for async message processing

**Configuration:**
```yaml
# docker-compose.chatwoot.yml
services:
  chatwoot-web:
    image: chatwoot/chatwoot:latest
    environment:
      RAILS_ENV: production
      SECRET_KEY_BASE: ${CHATWOOT_SECRET}
      FRONTEND_URL: http://100.x.x.10:3000
      DATABASE_URL: postgres://chatwoot:${DB_PASS}@postgres:5432/chatwoot_production
      REDIS_URL: redis://redis:6379/2
    ports:
      - "3000:3000"
    depends_on:
      - postgres
      - redis

  chatwoot-worker:
    image: chatwoot/chatwoot:latest
    command: bundle exec sidekiq
    environment:
      <<: *chatwoot-env
```

**Twilio Integration:**
- Each tenant gets a dedicated Twilio phone number ($1/month + $0.0079/SMS)
- Chatwoot configured with "API Channel" per tenant (not native Twilio integration — gives us webhook control)
- Inbound webhook URL: `https://sms-webhook.yourdomain.com/webhooks/twilio/{tenant_phone}`
- Cloudflare Tunnel routes this single public endpoint to Chatwoot internally

**Webhook Setup:**
```
Twilio Console → Phone Number → Messaging:
  Webhook URL: https://sms-webhook.yourdomain.com/webhooks/twilio/+1XXXXXXXXXX
  Method: POST
  Fallback URL: https://sms-webhook.yourdomain.com/webhooks/twilio/fallback
```

**Multi-Tenant Inbox Strategy:**
- One Chatwoot "inbox" per tenant phone number
- Inbox ID maps to tenant_id in our configuration store
- Chatwoot webhook fires on every new message → n8n receives it
- Webhook payload includes inbox_id for tenant identification

**Outbound Flow:**
- n8n calls Chatwoot API to send outbound messages
- `POST /api/v1/accounts/{account_id}/conversations/{conv_id}/messages`
- Chatwoot routes through configured Twilio channel
- Delivery status callbacks update carrier_status in conversation history

### Component 2: CRM Layer (Twenty)

**Addresses:** Requirement 1 (Multi-Tenant Isolation), Requirement 4 (Lead Pipeline Management)

**Deployment:**
- Twenty CRM deployed as Docker Compose stack within LXC 112
- Separate PostgreSQL database: `twenty_production`
- Twenty's built-in workspace isolation used for multi-tenancy

**GraphQL Schema Extensions:**
```graphql
# Twenty CRM custom objects per tenant workspace
type LeadPipeline {
  id: ID!
  contact: Contact!
  stage: PipelineStage!
  stageChangedAt: DateTime!
  source: String
  notes: String
  tenant: Tenant!
}

enum PipelineStage {
  COLD
  CONTACTED
  NEGOTIATING
  WON
  LOST
}

type PipelineMetrics {
  tenantId: ID!
  totalLeads: Int!
  coldCount: Int!
  contactedCount: Int!
  negotiatingCount: Int!
  wonCount: Int!
  lostCount: Int!
  conversionRate: Float!
  periodStart: DateTime!
  periodEnd: DateTime!
}
```

**Pipeline Configuration:**
- Each tenant workspace gets a pre-configured pipeline with 5 stages
- Stage transitions triggered by n8n workflows (not manual CRM actions)
- Twenty's GraphQL API exposed internally on port 3001
- n8n uses Twenty's API tokens (one per tenant workspace) for mutations

**Multi-Tenant Approach:**
- Twenty CRM supports native "workspaces" — each tenant = one workspace
- Workspace isolation is enforced at the application layer
- Single Twenty instance serves all tenants (resource efficient)
- API tokens scoped per workspace prevent cross-tenant access
- Onboarding creates workspace + pipeline + API token atomically

**Pipeline Stage Automation Rules:**
| Trigger | From Stage | To Stage | Actor |
|---------|-----------|----------|-------|
| First inbound SMS from unknown number | — | COLD | n8n workflow |
| First outbound SMS sent | COLD | CONTACTED | n8n workflow |
| Positive intent detected by AI | CONTACTED | NEGOTIATING | Hermes Agent |
| Booking/purchase confirmed | NEGOTIATING | WON | Hermes Agent |
| Explicit decline or 14-day silence | CONTACTED/NEGOTIATING | LOST | n8n scheduled job |

### Component 3: Orchestration Layer (n8n)

**Addresses:** Requirement 7 (Workflow Orchestration), Requirement 2 (Inbound SMS Processing)

**Deployment:**
- n8n deployed as Docker container within LXC 112
- Persistent storage: `/data/n8n/` mounted volume
- PostgreSQL backend for execution history: `n8n_production` database
- Queue mode enabled with Redis for reliable execution

**Configuration:**
```yaml
# docker-compose.n8n.yml
services:
  n8n:
    image: n8nio/n8n:latest
    environment:
      N8N_BASIC_AUTH_ACTIVE: "true"
      N8N_BASIC_AUTH_USER: ${N8N_USER}
      N8N_BASIC_AUTH_PASSWORD: ${N8N_PASS}
      DB_TYPE: postgresdb
      DB_POSTGRESDB_HOST: postgres
      DB_POSTGRESDB_DATABASE: n8n_production
      EXECUTIONS_MODE: queue
      QUEUE_BULL_REDIS_HOST: redis
      N8N_METRICS: "true"
      WEBHOOK_URL: http://100.x.x.10:5678
    ports:
      - "5678:5678"
    volumes:
      - /data/n8n:/home/node/.n8n
```

**Core Workflows:**

1. **Inbound SMS Handler** (webhook-triggered)
   ```
   Webhook Receive → Extract inbox_id → Lookup tenant_id
   → Query Knowledge Store (contact history + business context)
   → POST to Hermes Agent (message + context + tenant config)
   → Receive AI response
   → POST to Chatwoot (send outbound SMS)
   → Update Twenty CRM (pipeline stage if needed)
   → Update Knowledge Store (append conversation)
   → Log execution metrics
   ```

2. **Campaign Executor** (cron/manual-triggered)
   ```
   Fetch campaign config → Query Twenty CRM (matching contacts)
   → Filter suppression list → Rate-limit queue (Redis)
   → For each contact: Generate personalized message via Hermes
   → Send via Chatwoot → Update campaign stats
   → Log delivery status
   ```

3. **Tenant Onboarding** (API-triggered)
   ```
   Receive config payload → Validate phone number
   → Create Twenty workspace + pipeline
   → Create Chatwoot inbox + channel
   → Create Knowledge Store namespace
   → Clone workflow template with tenant params
   → Send test SMS → Verify delivery
   → Mark onboarding complete / rollback on failure
   ```

4. **Stale Lead Checker** (daily cron)
   ```
   For each active tenant:
   → Query Twenty CRM for leads in CONTACTED/NEGOTIATING
   → Filter: last_activity > 14 days
   → Move to LOST stage
   → Log transitions
   ```

**Webhook Handlers:**
- `POST /webhook/inbound-sms` — Chatwoot forwards inbound messages here
- `POST /webhook/delivery-status` — Twilio delivery receipts
- `POST /webhook/onboard-tenant` — Programmatic tenant provisioning
- `POST /webhook/campaign-trigger` — Start/pause/resume campaigns

**Tenant Routing:**
- Chatwoot webhook includes `inbox_id` in payload
- n8n maintains a lookup table: `inbox_id → tenant_id → tenant_config`
- Lookup table cached in Redis with 5-minute TTL
- On cache miss, queries PostgreSQL tenant configuration table

**Retry Logic:**
- All HTTP nodes configured with retry: 3 attempts
- Exponential backoff: 2s → 4s → 8s
- Dead letter queue in Redis for permanently failed messages
- Failed executions create Paperclip AI issues automatically

### Component 4: AI Layer (Hermes Agent)

**Addresses:** Requirement 3 (AI Response Generation), Requirement 4 (Lead Pipeline Management)

**Deployment:**
- Hermes agent runtime already running on 3090 (existing infrastructure)
- New "sms-responder" agent registered in Paperclip AI
- Communicates with Ollama on localhost:11434 for inference
- Exposes HTTP API on port 8080 for n8n to call

**Prompt Engineering (TICA Framework Applied):**

All prompts use the **TICA framework** (Task-Information-Constraints-Ask) for optimal AI responses:

**T - Task**: Clear objective definition
**I - Information**: Relevant context and background  
**C - Constraints**: Boundaries and limitations
**A - Ask**: Request for clarifying questions when needed

System prompt template (per-tenant, assembled at runtime):
```
You are an AI receptionist for {{business_name}}.

TASK (T):
Generate helpful, contextually appropriate SMS responses to customer inquiries.
Maximize booking/scheduling conversions while maintaining professional tone.

INFORMATION (I):
BUSINESS CONTEXT:
- Services: {{services_list}}
- Pricing: {{pricing_guidelines}}
- Hours: {{operating_hours}}
- Service Area: {{service_area}}

CONVERSATION HISTORY:
{{recent_messages}}

RELEVANT KNOWLEDGE:
{{semantic_search_results}}

CALENDAR RULES:
{{calendar_rules}}

CONSTRAINTS (C):
- Response length: ≤320 characters (single SMS segment)
- Never reveal you are AI
- Tone: {{tone_preferences}}
- Scope: Only answer questions within business knowledge
- Safety: Escalate uncertain or complex requests
- Compliance: No medical/legal/tax advice
- Pipeline: Suggest stage advances when appropriate

ASK (A):
If the customer request is unclear or outside your knowledge:
1. Ask clarifying questions before responding
2. Use [ESCALATE] tag when human intervention needed
3. Request additional context when confidence < 0.7

RESPONSE FORMAT:
ESCALATION: If you cannot confidently answer, respond with [ESCALATE] prefix
INTENT TAGS: Tag your response with one of: [SCHEDULING] [PRICING] [GENERAL] [COMPLAINT]
SENTIMENT: Tag the customer message as: [POSITIVE] [NEUTRAL] [NEGATIVE]
PIPELINE: If appropriate, suggest stage change: [ADVANCE:stage_name] or [NO_CHANGE]
CONFIDENCE: Include confidence score (0.0-1.0) for response quality

FEEDBACK LOOP:
Grade your response against these criteria:
- Relevance to customer query (0-1)
- Tone appropriateness (0-1)
- Action clarity (0-1)
- Length compliance (0-1)
If total score < 0.7, revise and regrade before responding.
```

**Context Injection:**
1. n8n sends request with: `{message, sender_phone, tenant_id}`
2. Hermes fetches from Knowledge Store:
   - Last 10 messages in conversation (for continuity)
   - Tenant business profile (for context)
   - Semantic search results for relevant FAQ/knowledge (top 3 matches)
3. Assembles full prompt with all context
4. Sends to Ollama (Hermes-3-Llama-3.1-8B)
5. Parses response for tags: intent, sentiment, pipeline action, escalation flag

**Response Generation Flow:**
```
Input: {message, context, tenant_config}
  │
  ├─ Assemble system prompt (tenant-specific)
  ├─ Inject conversation history (last 10 messages)
  ├─ Inject semantic search results (relevant knowledge)
  ├─ Call Ollama /api/generate (stream: false, temperature: 0.7)
  │
  ├─ Parse response metadata tags
  ├─ Validate response length (≤320 chars for SMS)
  ├─ Check for [ESCALATE] flag
  │
  └─ Return: {response_text, intent, sentiment, pipeline_action, escalated}
```

**Escalation Logic:**
- If response contains `[ESCALATE]`: do NOT send AI response
- Instead: notify tenant's escalation contact via Chatwoot internal note
- Send customer a holding message: "Let me check on that and get back to you shortly."
- Create Paperclip issue for human follow-up tracking

**Fallback Behavior:**
- If Ollama doesn't respond within 15 seconds: return configurable fallback
- Default fallback: "Thanks for reaching out! We received your message and will get back to you shortly."
- If 3090 is completely down: messages queue in Redis (4-hour retention)
- On recovery: process queued messages in FIFO order

### Component 5: Knowledge Store

**Addresses:** Requirement 8 (Knowledge Graph Management), Requirement 3 (AI Response Generation)

**Storage Engine: PostgreSQL + pgvector**

Deployed as a dedicated PostgreSQL 16 instance with pgvector extension on the 3090 server. This co-locates the knowledge store with the AI inference layer to minimize latency on embedding lookups.

**Schema:**
```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Tenant business profiles
CREATE TABLE tenant_profiles (
  tenant_id UUID PRIMARY KEY,
  business_name TEXT NOT NULL,
  services JSONB NOT NULL,
  pricing_guidelines TEXT,
  operating_hours JSONB,
  service_area TEXT,
  tone_preferences TEXT DEFAULT 'professional',
  calendar_rules JSONB,
  faq_entries JSONB DEFAULT '[]',
  custom_knowledge TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Conversation history with embeddings
CREATE TABLE conversations (
  message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenant_profiles(tenant_id),
  contact_phone TEXT NOT NULL,
  direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
  content TEXT NOT NULL,
  content_embedding vector(384),
  sentiment TEXT CHECK (sentiment IN ('positive', 'neutral', 'negative')),
  intent TEXT,
  timestamp TIMESTAMPTZ DEFAULT NOW(),
  campaign_id UUID,
  ai_generated BOOLEAN DEFAULT false
);

-- Knowledge base entries (per-tenant FAQ, procedures, etc.)
CREATE TABLE knowledge_entries (
  entry_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenant_profiles(tenant_id),
  category TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  content_embedding vector(384),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_conversations_tenant_phone 
  ON conversations(tenant_id, contact_phone, timestamp DESC);
CREATE INDEX idx_conversations_embedding 
  ON conversations USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_knowledge_embedding 
  ON knowledge_entries USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX idx_conversations_timestamp 
  ON conversations(tenant_id, timestamp DESC);
```

**Indexing Strategy:**
- IVFFlat index on embeddings (good balance of speed vs accuracy for <1M vectors)
- B-tree indexes on tenant_id + phone + timestamp for conversation retrieval
- Partition conversations table by month after 6 months of operation

**Semantic Search Approach:**
```sql
-- Find relevant knowledge for a customer query
SELECT title, content, 
       1 - (content_embedding <=> $1) AS similarity
FROM knowledge_entries
WHERE tenant_id = $2
ORDER BY content_embedding <=> $1
LIMIT 3;
```

**Embedding Generation:**
- Uses Ollama's `nomic-embed-text` model (384 dimensions)
- Embeddings generated on the 3090 at message ingestion time
- Batch embedding for knowledge base entries during onboarding
- Average embedding latency: ~50ms per text chunk

**API (REST, exposed via MCP Gateway):**
- `GET /knowledge/{tenant_id}/profile` — Fetch tenant business profile
- `GET /knowledge/{tenant_id}/conversations/{phone}?limit=10` — Recent conversation history
- `POST /knowledge/{tenant_id}/search` — Semantic search across knowledge base
- `POST /knowledge/{tenant_id}/conversations` — Append new message
- `PUT /knowledge/{tenant_id}/profile` — Update tenant profile
- `POST /knowledge/{tenant_id}/entries` — Add knowledge base entry

### Component 6: Onboarding Service

**Addresses:** Requirement 10 (Tenant Onboarding and Configuration), Requirement 1 (Multi-Tenant Isolation)

**Provisioning Flow:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    ONBOARDING SEQUENCE                            │
│                                                                  │
│  1. Receive config payload (via API or Paperclip agent)          │
│     │                                                            │
│  2. Validate payload schema + phone number format                │
│     │                                                            │
│  3. Create tenant record in PostgreSQL                           │
│     │  ← CHECKPOINT 1 (rollback: delete tenant record)          │
│     │                                                            │
│  4. Create Twenty CRM workspace + pipeline + API token           │
│     │  ← CHECKPOINT 2 (rollback: delete workspace)              │
│     │                                                            │
│  5. Create Chatwoot inbox + API channel + webhook config         │
│     │  ← CHECKPOINT 3 (rollback: delete inbox)                  │
│     │                                                            │
│  6. Create Knowledge Store namespace (profile + initial entries) │
│     │  ← CHECKPOINT 4 (rollback: delete namespace data)         │
│     │                                                            │
│  7. Clone n8n workflow template with tenant parameters           │
│     │  ← CHECKPOINT 5 (rollback: delete workflow)               │
│     │                                                            │
│  8. Configure Twilio webhook for tenant phone number             │
│     │  ← CHECKPOINT 6 (rollback: remove webhook)                │
│     │                                                            │
│  9. Send test SMS to tenant's phone → verify delivery            │
│     │  ← CHECKPOINT 7 (rollback: N/A, just report failure)      │
│     │                                                            │
│  10. Mark tenant status = "active"                               │
│      │                                                           │
│  11. Report success to Paperclip AI                              │
└─────────────────────────────────────────────────────────────────┘
```

**Rollback Logic:**
- Each provisioning step records its completion in a `provisioning_log` table
- On failure at any step, the onboarding service walks backward through completed steps
- Each step has a corresponding `undo_` function:
  - `undo_twenty_workspace()` — DELETE workspace via Twenty API
  - `undo_chatwoot_inbox()` — DELETE inbox via Chatwoot API
  - `undo_knowledge_namespace()` — DELETE FROM tenant_profiles WHERE tenant_id = X
  - `undo_n8n_workflow()` — DELETE workflow via n8n API
  - `undo_twilio_webhook()` — Remove webhook from Twilio phone number config
- Rollback results logged to Paperclip AI as an issue with full diagnostic details

**Validation:**
- Phone number: E.164 format validation + Twilio lookup API (carrier check)
- Business name: non-empty, ≤100 characters
- Services list: at least 1 service, each ≤200 characters
- Operating hours: valid timezone, valid time ranges
- Tone preferences: must be one of allowed values
- Rate limits: within platform maximums (≤1 msg/sec, ≤500 daily)

**Implementation:** n8n workflow (Tenant Onboarding workflow) with error handling nodes at each step. Exposed as webhook endpoint accessible via MCP Gateway for Paperclip agents to trigger programmatically.

**Time Target:** Full provisioning completes within 10 minutes (most time spent on Twilio phone number verification and test SMS round-trip).

### Component 7: Campaign Engine

**Addresses:** Requirement 5 (Reactivation Campaigns), Requirement 11 (SMS Compliance and Rate Limiting)

**Architecture:**
- Campaign engine implemented as n8n workflows + Redis queue
- Redis Sorted Set used as a time-delayed message queue
- Separate worker process (n8n sub-workflow) drains queue at controlled rate

**Rate Limiting Implementation:**
```
┌─────────────────────────────────────────────────────────────┐
│                   CAMPAIGN EXECUTION                          │
│                                                              │
│  Campaign Start                                              │
│    │                                                         │
│    ├─ Query Twenty CRM: contacts matching criteria           │
│    ├─ Filter: remove suppressed/opted-out contacts           │
│    ├─ For each contact:                                      │
│    │    ├─ Generate personalized message (Hermes Agent)      │
│    │    └─ ZADD to Redis sorted set (score = send_time)      │
│    │                                                         │
│  Queue Drainer (runs every 1 second):                        │
│    ├─ ZRANGEBYSCORE: get messages where score ≤ NOW          │
│    ├─ Rate check: ≤1 msg per 3 seconds per campaign         │
│    ├─ Daily check: ≤200 msgs per phone number today          │
│    ├─ Send via Chatwoot API                                  │
│    ├─ Update campaign stats                                  │
│    └─ ZREM processed messages                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Queue Management:**
- Redis key: `campaign:{campaign_id}:queue` (Sorted Set)
- Score: Unix timestamp of intended send time (spaced 3+ seconds apart)
- Value: JSON `{contact_id, phone, personalized_message}`
- TTL on queue key: 48 hours (auto-cleanup for completed/stale campaigns)

**Scheduling:**
- Campaigns can be scheduled for future start (stored in campaign config)
- n8n cron job checks for campaigns where `scheduled_start ≤ NOW` and `status = draft`
- Transitions campaign to `active` and begins queue population
- Campaigns respect tenant operating hours (no sends outside business hours)

**Opt-Out Handling:**
```
Inbound message received
  │
  ├─ Check content against opt-out keywords:
  │   STOP, CANCEL, UNSUBSCRIBE, END, QUIT, REMOVE
  │   (case-insensitive, exact match or starts-with)
  │
  ├─ If match:
  │   ├─ Set contact.opt_out = true, contact.opt_out_at = NOW
  │   ├─ Update Twenty CRM contact record
  │   ├─ Remove contact from ALL active campaign queues (ZREM)
  │   ├─ Add to tenant suppression list
  │   ├─ Send confirmation: "You've been unsubscribed. Reply START to re-subscribe."
  │   └─ Do NOT process through AI pipeline
  │
  └─ If no match:
      └─ Continue normal inbound processing
```

**First-Contact Compliance:**
- First outbound SMS to any new contact includes footer:
  `"Reply STOP to opt out"`
- Appended automatically by the campaign engine (not by AI)
- Tracked per contact: `first_outbound_sent` flag

**Daily Limits:**
- Redis counter: `ratelimit:{phone_number}:{YYYY-MM-DD}` with TTL 86400
- INCR on each send, reject if > 200
- Per-tenant configurable (default 200, some carriers allow more)

### Component 8: Observability

**Addresses:** Requirement 9 (Observability and Monitoring)

**Metrics Collection:**

The platform collects metrics at three levels:

1. **Agent-Level (Paperclip Integration)**
   - Hermes "sms-responder" agent sends heartbeat every 30 seconds
   - Heartbeat payload: `{agent_id, status, last_processed_at, queue_depth, uptime_seconds}`
   - If heartbeat missed for 60s, Paperclip marks agent as unhealthy

2. **Service-Level (n8n + Redis)**
   - n8n built-in execution metrics (enabled via `N8N_METRICS=true`)
   - Prometheus-compatible endpoint at `http://100.x.x.10:5678/metrics`
   - Key metrics exported:
     - `n8n_workflow_executions_total{workflow, status}`
     - `n8n_workflow_execution_duration_seconds`
     - Custom metrics via n8n "Set" nodes in workflows

3. **Business-Level (Custom Dashboard)**
   - Stored in PostgreSQL `platform_metrics` table
   - Aggregated per tenant, per hour:
     ```sql
     CREATE TABLE platform_metrics (
       metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
       tenant_id UUID NOT NULL,
       hour_bucket TIMESTAMPTZ NOT NULL,
       messages_sent INTEGER DEFAULT 0,
       messages_received INTEGER DEFAULT 0,
       avg_response_latency_ms INTEGER,
       ai_success_count INTEGER DEFAULT 0,
       ai_failure_count INTEGER DEFAULT 0,
       escalation_count INTEGER DEFAULT 0,
       pipeline_advances JSONB DEFAULT '{}',
       created_at TIMESTAMPTZ DEFAULT NOW()
     );
     CREATE INDEX idx_metrics_tenant_hour 
       ON platform_metrics(tenant_id, hour_bucket DESC);
     ```

**Paperclip Integration:**
- Agent registration: `sms-responder` agent in Paperclip company "B2B Automation"
- Issue creation triggers:
  - SMS delivery failure (carrier rejection)
  - Response latency > 30s (5-minute rolling average)
  - Agent heartbeat missed
  - Campaign completion (summary report)
  - Onboarding failure (with rollback details)
- Issue format: `{title, description, severity, tenant_id, metadata}`

**Alerting Rules:**
| Condition | Severity | Action |
|-----------|----------|--------|
| Heartbeat missed > 60s | Critical | Paperclip issue + attempt agent restart |
| Avg latency > 30s (5min window) | Warning | Paperclip issue |
| SMS delivery failure | Medium | Paperclip issue + retry |
| Daily send limit approaching (>180/200) | Info | Paperclip issue (advisory) |
| Campaign error rate > 5% | Warning | Pause campaign + Paperclip issue |
| 3090 GPU unreachable | Critical | Queue messages + Paperclip issue |

**Message Delivery Logs:**
- All SMS events stored in `delivery_logs` table
- Retention: 30 days minimum (configurable per tenant)
- Fields: message_id, carrier_message_id, status, status_timestamp, error_code
- Twilio delivery status webhooks update these records in real-time

## Key Design Decisions

### Decision 1: Knowledge Store Engine

**Options Evaluated:**

| Option | Pros | Cons |
|--------|------|------|
| PostgreSQL + pgvector | Single DB engine, mature, ACID, co-locates with structured data | Slightly slower than dedicated vector DBs at scale |
| SQLite + embeddings | Zero-config, file-based, simple | No concurrent writes, no native vector ops, won't scale |
| Dedicated Vector DB (Qdrant/Milvus) | Purpose-built, fastest similarity search | Another service to manage, overkill for <1M vectors |

**Chosen: PostgreSQL 16 + pgvector**

Rationale:
- We already run PostgreSQL for Chatwoot, Twenty, and n8n — adding pgvector is just an extension
- At 50 tenants × 1000 contacts × 50 messages = 2.5M vectors max — well within pgvector's sweet spot
- IVFFlat indexing provides sub-10ms similarity search at this scale
- Single backup strategy, single monitoring target, single operational model
- Zero additional SaaS cost (self-hosted)
- If we outgrow pgvector (unlikely before 10M+ vectors), migration to Qdrant is straightforward

### Decision 2: Multi-Tenant Strategy

**Options Evaluated:**

| Option | Pros | Cons |
|--------|------|------|
| Shared DB with tenant_id column | Simple, resource efficient, easy queries | Risk of data leaks if WHERE clause missed |
| Separate DB per tenant | Perfect isolation, easy backup/restore per tenant | Connection pool explosion at 50+ tenants, complex migrations |
| Namespace isolation (schemas/workspaces) | Good isolation, manageable connections | Slightly more complex queries |

**Chosen: Hybrid — Application-Level Workspace Isolation**

Rationale:
- Twenty CRM: Uses native workspace isolation (built-in, battle-tested)
- Chatwoot: Uses inbox-per-tenant (built-in isolation model)
- Knowledge Store: Shared tables with `tenant_id` column + row-level security (RLS)
- n8n: Parameterized workflow templates (tenant_id injected at runtime)

This hybrid approach leverages each tool's native isolation model rather than fighting against it. The Knowledge Store uses PostgreSQL Row-Level Security as a safety net:
```sql
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON conversations
  USING (tenant_id = current_setting('app.current_tenant')::uuid);
```

At 50 tenants, this approach uses ~3 DB connections per tenant (150 total) — well within PostgreSQL's capacity with PgBouncer connection pooling.

### Decision 3: AI Model Selection

**Options Evaluated:**

| Option | Pros | Cons |
|--------|------|------|
| Local Hermes-3-Llama-3.1-8B via Ollama | Free, fast (~2s), good conversational quality, runs on 3090 | Less capable than frontier models |
| Qwen 3.7 (local via Ollama) | Strong reasoning, multilingual | Slower (~4s), overkill for SMS responses |
| GPT-5.4 via API | Best quality, most capable | $0.01-0.03/request, defeats zero-cost goal, adds latency |
| Claude via API | Excellent instruction following | Same cost/latency issues as GPT |

**Chosen: Hermes-3-Llama-3.1-8B (primary) + Qwen3-7B (fallback/complex)**

Rationale:
- Hermes-3 is specifically fine-tuned for conversational AI and function calling — perfect for SMS
- 8B parameter model generates responses in ~1.5-2s on 3090 (well within 15s budget)
- Zero per-request cost (only electricity: ~$0.02/day for inference)
- Qwen3-7B available as fallback for complex reasoning tasks (scheduling logic, multi-step queries)
- If quality proves insufficient for specific tenants, can selectively route to cloud API (per-tenant config flag)
- Embedding model: nomic-embed-text (384-dim, fast, good quality for semantic search)

### Decision 4: SMS Provider

**Options Evaluated:**

| Option | Pros | Cons |
|--------|------|------|
| Twilio | Most documented, best Chatwoot integration, reliable | $0.0079/SMS + $1/month/number, slightly pricier |
| Bandwidth | Cheaper ($0.004/SMS), good API | Less documentation, fewer community integrations |
| Telnyx | Cheapest ($0.003/SMS), modern API | Newer company, less proven at scale |

**Chosen: Twilio (initial) with Telnyx migration path**

Rationale:
- Chatwoot has first-class Twilio integration (least custom code to write)
- Twilio's webhook reliability is industry-leading (critical for real-time SMS)
- At 50 tenants × 200 msgs/day = 10,000 msgs/day = ~$79/day in SMS costs
- Revenue at 50 tenants × $400/month = $20,000/month vs ~$2,400/month SMS cost = 88% margin
- Twilio's Messaging Service feature handles 10DLC compliance automatically
- Migration to Telnyx planned for Phase 2 (after proving the model) to improve margins further
- Abstraction layer in n8n workflows makes provider swap a config change, not a rewrite

### Decision 5: Deployment Location — Tri-Node Asymmetric Cluster

**Options Evaluated:**

| Option | Pros | Cons |
|--------|------|------|
| All on Zimaboard | Single machine, simple ops | 8GB RAM Celeron will choke on Twenty+Chatwoot+pgvector at 10+ tenants |
| All on 3090 | GPU available, powerful | Mixing services with compute, VRAM contention |
| Split: Zimaboard (all services) + 3090 (AI) | Clean separation | Zimaboard still overloaded with heavy apps |
| **Tri-Node: Zimaboard (edge) + Mac Mini (state) + 3090 (compute)** | Each node does what it's built for | 3 machines to manage, network hops |

**Chosen: Tri-Node Asymmetric Cluster**

Rationale:
- ZimaBoard (8GB Celeron) is perfect for lightweight edge tasks (webhooks, queues, routing) but will catastrophically fail under Twenty CRM + Chatwoot + pgvector at scale
- Mac Mini (Apple Silicon, unified memory, NVMe) is purpose-built for databases and heavy web applications — handles PostgreSQL vector searches and Ruby/Node.js apps without breaking a sweat
- 3090 stays pure tensor compute — no databases, no webhooks, just CUDA inference
- Tailscale mesh makes inter-node communication seamless (~1ms LAN latency, encrypted)
- Mirrors AWS availability zone pattern: Edge → State → Compute
- Each node can be independently upgraded/replaced without affecting others

## API Contracts

### Onboarding API

```
POST /webhook/onboard-tenant
Host: 100.x.x.10:5678 (n8n, internal Tailscale only)

Request:
{
  "business_name": "Joe's Plumbing",
  "phone_number": "+15551234567",
  "services": ["Drain cleaning", "Water heater repair", "Pipe replacement"],
  "pricing_guidelines": "Drain cleaning starts at $99. Water heater install $800-1200.",
  "operating_hours": {
    "timezone": "America/Chicago",
    "schedule": {
      "mon": {"open": "07:00", "close": "18:00"},
      "tue": {"open": "07:00", "close": "18:00"},
      "wed": {"open": "07:00", "close": "18:00"},
      "thu": {"open": "07:00", "close": "18:00"},
      "fri": {"open": "07:00", "close": "17:00"},
      "sat": {"open": "08:00", "close": "12:00"},
      "sun": null
    }
  },
  "tone_preferences": "friendly",
  "escalation_contacts": [
    {"name": "Joe Smith", "phone": "+15559876543", "role": "owner"}
  ],
  "knowledge_entries": [
    {"category": "faq", "title": "Emergency service", "content": "We offer 24/7 emergency service for burst pipes. Call our emergency line."},
    {"category": "policy", "title": "Warranty", "content": "All work comes with a 1-year warranty on parts and labor."}
  ]
}

Response (Success - 201):
{
  "status": "success",
  "tenant_id": "uuid",
  "provisioned": {
    "twenty_workspace_id": "string",
    "chatwoot_inbox_id": 42,
    "knowledge_namespace": "uuid",
    "n8n_workflow_id": "string"
  },
  "test_sms_delivered": true,
  "onboarded_at": "ISO8601"
}

Response (Failure - 422):
{
  "status": "failed",
  "failed_at_step": "chatwoot_inbox_creation",
  "error": "Twilio number not SMS-capable",
  "rollback_completed": true,
  "rollback_details": ["deleted_tenant_record", "deleted_twenty_workspace"]
}
```

### Inbound SMS Webhook

```
POST /webhook/inbound-sms
Host: 100.x.x.10:5678 (n8n, receives from Chatwoot)

Request (Chatwoot webhook payload):
{
  "event": "message_created",
  "message_type": "incoming",
  "inbox": {
    "id": 42,
    "name": "Joe's Plumbing SMS"
  },
  "conversation": {
    "id": 1234,
    "contact_inbox": {
      "source_id": "+15557654321"
    }
  },
  "sender": {
    "id": 567,
    "phone_number": "+15557654321",
    "name": "Unknown"
  },
  "content": "Hi, do you guys do water heater installs? What's the cost?",
  "created_at": "2026-05-15T14:30:00Z"
}

Response: 200 OK (async processing, no body needed)
```

### Agent Response API

```
POST /api/generate-response
Host: 100.x.x.5:8080 (Hermes Agent on 3090, internal Tailscale only)

Request:
{
  "tenant_id": "uuid",
  "contact_phone": "+15557654321",
  "message": "Hi, do you guys do water heater installs? What's the cost?",
  "conversation_history": [
    {"role": "customer", "content": "...", "timestamp": "ISO8601"},
    {"role": "assistant", "content": "...", "timestamp": "ISO8601"}
  ],
  "tenant_context": {
    "business_name": "Joe's Plumbing",
    "services": ["Drain cleaning", "Water heater repair"],
    "pricing_guidelines": "Water heater install $800-1200",
    "tone": "friendly"
  },
  "knowledge_results": [
    {"title": "Water heater pricing", "content": "...", "similarity": 0.89}
  ]
}

Response (Success - 200):
{
  "response_text": "Hey! Yes we do water heater installs. Typically runs $800-1200 depending on the unit. Want me to set up a free estimate? We can usually get out same week.",
  "intent": "pricing",
  "sentiment": "positive",
  "pipeline_action": "ADVANCE:negotiating",
  "escalated": false,
  "confidence": 0.92,
  "generation_time_ms": 1847
}

Response (Escalation - 200):
{
  "response_text": "Let me check on that and get back to you shortly.",
  "intent": "general",
  "sentiment": "neutral",
  "pipeline_action": "NO_CHANGE",
  "escalated": true,
  "escalation_reason": "Customer asking about commercial project - outside standard residential scope",
  "confidence": 0.34,
  "generation_time_ms": 2103
}
```

### Campaign Trigger API

```
POST /webhook/campaign-trigger
Host: 100.x.x.10:5678 (n8n, internal Tailscale only)

Request:
{
  "action": "start|pause|resume|cancel",
  "tenant_id": "uuid",
  "campaign": {
    "name": "Q2 Reactivation - AC Tune-ups",
    "type": "reactivation",
    "criteria": {
      "last_activity_before": "2026-02-15T00:00:00Z",
      "pipeline_stages": ["lost", "cold"],
      "tags_include": ["ac-service"],
      "tags_exclude": ["do-not-contact"]
    },
    "message_template": "Hi {{name}}, it's been a while! Summer's coming - want to schedule an AC tune-up? We're offering $20 off for returning customers. Reply YES to book.",
    "personalization_enabled": true,
    "scheduled_start": "2026-05-16T09:00:00-05:00"
  }
}

Response (Success - 201):
{
  "status": "campaign_created",
  "campaign_id": "uuid",
  "matching_contacts": 47,
  "after_suppression_filter": 42,
  "estimated_completion": "2026-05-16T09:07:00-05:00",
  "estimated_cost": "$0.33"
}
```

## Sequence Diagrams

### Inbound SMS Flow

```
Customer        Twilio       CF Tunnel    Chatwoot      n8n         Knowledge    Hermes      Twenty
   │               │            │            │           │          Store(3090)  Agent(3090)  CRM
   │──SMS──────────►│            │            │           │              │            │         │
   │               │──webhook───►│            │           │              │            │         │
   │               │            │──forward───►│           │              │            │         │
   │               │            │            │──webhook──►│              │            │         │
   │               │            │            │           │              │            │         │
   │               │            │            │           │──get history─►│            │         │
   │               │            │            │           │◄─history─────│            │         │
   │               │            │            │           │              │            │         │
   │               │            │            │           │──semantic────►│            │         │
   │               │            │            │           │  search      │            │         │
   │               │            │            │           │◄─results─────│            │         │
   │               │            │            │           │              │            │         │
   │               │            │            │           │──generate────────────────►│         │
   │               │            │            │           │  request     │            │         │
   │               │            │            │           │◄─response────────────────│         │
   │               │            │            │           │              │            │         │
   │               │            │            │◄──send────│              │            │         │
   │               │            │            │  outbound │              │            │         │
   │               │◄───────────│◄───────────│           │              │            │         │
   │◄──SMS─────────│            │            │           │              │            │         │
   │               │            │            │           │              │            │         │
   │               │            │            │           │──update──────►│            │         │
   │               │            │            │           │  conversation│            │         │
   │               │            │            │           │              │            │         │
   │               │            │            │           │──update CRM────────────────────────►│
   │               │            │            │           │  (if stage   │            │         │
   │               │            │            │           │   change)    │            │         │
   │               │            │            │           │              │            │         │

Total target latency: < 30 seconds end-to-end
Typical latency: 3-5 seconds (network + inference + send)
```

### Reactivation Campaign Flow

```
Operator/       n8n            Twenty        Knowledge    Hermes       Redis        Chatwoot    Twilio
Paperclip       (orchestrator) CRM           Store        Agent        (queue)
   │               │              │              │           │            │            │          │
   │──trigger──────►│              │              │           │            │            │          │
   │  campaign     │              │              │           │            │            │          │
   │               │──query───────►│              │           │            │            │          │
   │               │  contacts    │              │           │            │            │          │
   │               │◄─results─────│              │           │            │            │          │
   │               │              │              │           │            │            │          │
   │               │──check suppression list─────────────────────────────►│            │          │
   │               │◄─filtered list──────────────────────────────────────│            │          │
   │               │              │              │           │            │            │          │
   │               │  FOR EACH CONTACT:          │           │            │            │          │
   │               │──get context─────────────────►│           │            │            │          │
   │               │◄─context─────────────────────│           │            │            │          │
   │               │              │              │           │            │            │          │
   │               │──personalize────────────────────────────►│            │            │          │
   │               │◄─message────────────────────────────────│            │            │          │
   │               │              │              │           │            │            │          │
   │               │──ZADD (scheduled send time)─────────────►│            │            │          │
   │               │              │              │           │            │            │          │
   │               │  QUEUE DRAINER (every 1s):  │           │            │            │          │
   │               │──ZRANGEBYSCORE──────────────────────────►│            │            │          │
   │               │◄─ready messages─────────────────────────│            │            │          │
   │               │              │              │           │            │            │          │
   │               │──send─────────────────────────────────────────────────►│            │          │
   │               │              │              │           │            │◄───────────►│          │
   │               │              │              │           │            │            │──SMS────►│
   │               │              │              │           │            │            │          │
   │               │──update stats────────────────────────────►│            │            │          │
   │               │              │              │           │            │            │          │
```

### Tenant Onboarding Flow

```
Paperclip/      n8n              PostgreSQL    Twenty       Chatwoot     Knowledge    Twilio
Operator        (onboarding)     (tenant DB)   CRM          (inbox)      Store
   │               │                │            │            │            │           │
   │──POST config──►│                │            │            │            │           │
   │               │                │            │            │            │           │
   │               │──validate──────►│            │            │            │           │
   │               │  payload       │            │            │            │           │
   │               │                │            │            │            │           │
   │               │──INSERT tenant─►│            │            │            │           │
   │               │◄─tenant_id─────│            │            │            │           │
   │               │                │            │            │            │           │
   │               │──create workspace───────────►│            │            │           │
   │               │◄─workspace_id + api_token───│            │            │           │
   │               │                │            │            │            │           │
   │               │──create inbox + channel──────────────────►│            │           │
   │               │◄─inbox_id─────────────────────────────────│            │           │
   │               │                │            │            │            │           │
   │               │──create namespace + profile──────────────────────────►│           │
   │               │  + embed knowledge entries  │            │            │           │
   │               │◄─confirmed────────────────────────────────────────────│           │
   │               │                │            │            │            │           │
   │               │──configure webhook──────────────────────────────────────────────►│
   │               │◄─confirmed────────────────────────────────────────────────────────│
   │               │                │            │            │            │           │
   │               │──send test SMS──────────────────────────────────────────────────►│
   │               │◄─delivery confirmed─────────────────────────────────────────────│
   │               │                │            │            │            │           │
   │               │──UPDATE status='active'─────►│            │            │           │
   │               │                │            │            │            │           │
   │◄──201 success─│                │            │            │            │           │
   │               │                │            │            │            │           │
   │  ON FAILURE AT ANY STEP:       │            │            │            │           │
   │               │──rollback all completed steps (reverse order)─────────────────────│
   │◄──422 failure─│                │            │            │            │           │
   │  + diagnostics│                │            │            │            │           │
```

## Security Considerations

**Network Security:**
- All inter-service communication over Tailscale mesh (WireGuard encrypted, mutual TLS)
- Only public endpoint: Cloudflare Tunnel → Chatwoot webhook (SMS inbound only)
- Cloudflare Tunnel configured with Access policies (Twilio IP allowlist)
- No services exposed on public internet directly
- PostgreSQL, Redis, n8n admin UI accessible only via Tailscale IPs

**Data Security:**
- Tenant API keys and Twilio credentials encrypted at rest (PostgreSQL `pgcrypto`)
- Row-Level Security on Knowledge Store tables (defense in depth)
- n8n credentials stored in n8n's encrypted credential store
- Conversation content stored in plaintext (required for AI processing) but access-controlled
- 90-day data retention for archived tenants, then permanent deletion

**Authentication & Authorization:**
- n8n webhook endpoints: validated via HMAC signature (Chatwoot signs webhooks)
- Hermes Agent API: Bearer token authentication (rotated monthly)
- Twenty CRM API: Per-workspace API tokens (scoped, non-transferable)
- MCP Gateway: Existing authentication mechanism (Paperclip agent tokens)
- Onboarding API: Requires Paperclip agent token or operator API key

**SMS Security:**
- Twilio webhook signature validation (prevents spoofed inbound messages)
- Rate limiting prevents abuse (per-number, per-tenant, platform-wide)
- Opt-out handling is immediate and irreversible (compliance requirement)
- No PII in logs beyond phone numbers (required for routing)

**Operational Security:**
- All Docker containers run as non-root users
- Container images pinned to specific SHA digests (not `:latest` in production)
- PostgreSQL connections via Unix socket where co-located, TLS where remote
- Redis protected with AUTH password + Tailscale network isolation
- Secrets managed via environment variables (not committed to any repo)

## Trade-offs and Risks

| Trade-off | Accepted Risk | Mitigation |
|-----------|--------------|------------|
| Single 3090 for all AI inference | GPU failure = no AI responses | 4-hour message queue + fallback messages + Paperclip alerting |
| Shared PostgreSQL instance | DB failure affects all services | Daily automated backups, WAL archiving, 15-min recovery target |
| Hermes-3-8B vs frontier models | Lower response quality for complex queries | Escalation logic + per-tenant cloud API override flag |
| Twilio as sole SMS provider | Provider outage = no SMS | Telnyx failover planned for Phase 2; Twilio has 99.95% SLA |
| Cloudflare Tunnel for ingress | CF outage = no inbound SMS | Fallback: direct Tailscale Funnel endpoint (manual switchover) |
| n8n for orchestration | Workflow complexity ceiling | n8n handles current complexity well; custom service if >100 tenants |
| Single Zimaboard for services | Hardware failure = full outage | Coolify enables rapid redeployment to backup hardware |

**Key Risks:**

1. **3090 GPU thermal throttling under sustained load**
   - 50 tenants × peak 10 msgs/min = ~500 inference calls/min
   - Hermes-3-8B at ~1.5s/response = ~12 concurrent requests max
   - Mitigation: Request queuing + batch processing + GPU monitoring

2. **Twilio 10DLC registration delays**
   - New phone numbers require 10DLC campaign registration (1-4 weeks)
   - Mitigation: Pre-register numbers in batches, maintain pool of ready numbers

3. **n8n workflow complexity at scale**
   - 50+ parameterized workflows may strain n8n's execution engine
   - Mitigation: Single workflow with tenant routing (not per-tenant workflows)

4. **Knowledge Store embedding drift**
   - Model updates change embedding space, invalidating existing vectors
   - Mitigation: Version embeddings, re-embed on model change (batch job)

## Future Extensibility

**Addresses:** Requirement 12 (Scalability and Extensibility)

**Channel Plugin Architecture:**
```
┌─────────────────────────────────────────────────┐
│              Channel Plugin Interface             │
│                                                  │
│  interface ChannelPlugin {                       │
│    name: string                                  │
│    receiveWebhook(payload): NormalizedMessage    │
│    sendMessage(contact, content): DeliveryResult │
│    getDeliveryStatus(messageId): Status          │
│    handleOptOut(contact): void                   │
│  }                                              │
│                                                  │
│  Implementations:                                │
│  ├── TwilioSMSPlugin (current)                  │
│  ├── TelnyxSMSPlugin (Phase 2)                  │
│  ├── WhatsAppPlugin (future)                    │
│  ├── EmailPlugin (future - via Chatwoot)        │
│  └── VoicePlugin (future - via Twilio Voice)    │
└─────────────────────────────────────────────────┘
```

**Horizontal Scaling Path:**
- Phase 1 (current): Zimaboard + 3090 (supports 5-50 tenants)
- Phase 2: Add Mac Mini as second service node via Tailscale
  - Move n8n + Redis to Mac Mini (offload Zimaboard)
  - Load balance AI requests across 3090 + Mac Mini (Apple Silicon MLX)
- Phase 3: Additional Zimaboard for geographic redundancy
  - Active-passive failover for services
  - Shared PostgreSQL via streaming replication

**Workload Redistribution:**
- All services deployed via Docker Compose (portable across nodes)
- Tailscale DNS provides stable hostnames regardless of physical location
- Configuration change = update Tailscale DNS + restart affected containers
- No code changes required to move services between nodes

**Configuration Versioning:**
- All tenant configs stored with `config_version` integer
- n8n workflow templates versioned in Git (local GitLab)
- Schema migrations tracked via standard PostgreSQL migration tooling
- Rollback: restore previous config version + re-apply workflow template

**API Versioning:**
- All API endpoints prefixed with `/v1/`
- Breaking changes deployed as `/v2/` with deprecation period
- Webhook payloads include `api_version` field for backward compatibility
