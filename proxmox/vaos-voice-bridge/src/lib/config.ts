import { z } from 'zod';

const EnvSchema = z.object({
  // Supabase (shared with VAOS control plane)
  SUPABASE_URL: z.string().url(),
  SUPABASE_SERVICE_ROLE_KEY: z.string().min(20),

  // LLM
  ANTHROPIC_AUTH_TOKEN: z.string().min(8),
  ANTHROPIC_BASE_URL: z.string().url().default('https://api.z.ai/api/anthropic'),

  // Voice Bridge
  PORT: z.coerce.number().default(9001),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),

  // PersonaPlex (Talker / System 1)
  PERSONAPLEX_HOST: z.string().default('10.0.0.3'),
  PERSONAPLEX_PORT: z.coerce.number().default(8998),
  PERSONAPLEX_WS_PATH: z.string().default('/ws'),

  // Letta (Reasoner / System 2)
  LETTA_BASE_URL: z.string().url().default('http://10.0.0.3:8283'),
  LETTA_AGENT_NAME: z.string().default('voice-reasoner'),
  REASONER_TIMEOUT_MS: z.coerce.number().default(30_000),
  BELIEF_UPDATE_TIMEOUT_MS: z.coerce.number().default(15_000),

  // Ollama (fast path for System 2 responses)
  OLLAMA_URL: z.string().url().default('http://192.168.1.143:11434'),
  OLLAMA_MODEL: z.string().default('qwen2.5:7b'),

  // TTS
  VOICE_PROMPT_PATH: z.string().default('/opt/moshi/voices/NATF0.pt'),
  TTS_SAMPLE_RATE: z.coerce.number().default(24000),
});

export type Env = z.infer<typeof EnvSchema>;

let cachedEnv: Env | null = null;

export function validateEnv(): Env {
  if (cachedEnv) return cachedEnv;

  const result = EnvSchema.safeParse(process.env);
  if (!result.success) {
    console.error('❌ Invalid environment configuration:');
    for (const issue of result.error.issues) {
      console.error(`   ${issue.path.join('.')}: ${issue.message}`);
    }
    process.exit(1);
  }

  cachedEnv = result.data;
  return cachedEnv;
}

export function getEnv(): Env {
  if (!cachedEnv) return validateEnv();
  return cachedEnv;
}
