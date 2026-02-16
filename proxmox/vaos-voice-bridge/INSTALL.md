# Voice Bridge Installation

## 1. Move to VAOS directory
```bash
mv /Users/speed/proxmox/vaos-voice-bridge /Users/speed/vaos/vaos-voice-bridge
cd /Users/speed/vaos/vaos-voice-bridge
bun install
```

## 2. Copy voice executor to vaos-control-plane
```bash
cp src/executors/voice.ts /Users/speed/vaos/vaos-control-plane/src/executors/voice.ts
```

Then add to `/Users/speed/vaos/vaos-control-plane/src/executors/index.ts`:
```typescript
import { executeVoice } from './voice.js';

// Add to EXECUTORS map:
const EXECUTORS: Record<string, Executor> = {
  // ... existing executors ...
  voice: executeVoice,
};
```

Rebuild control plane:
```bash
cd /Users/speed/vaos/vaos-control-plane
bun run build
pm2 restart vaos-worker
```

## 3. Add to PM2 ecosystem
Add to `/Users/speed/vaos/vaos-deploy/ecosystem.config.cjs`:
```javascript
{
  name: "vaos-voice-bridge",
  script: BUN,
  args: `run /Users/speed/vaos/vaos-voice-bridge/src/server.ts`,
  cwd: "/Users/speed/vaos/vaos-voice-bridge",
  interpreter: "none",
  env: {
    ...controlPlaneEnv,
    PORT: "9001",
    PERSONAPLEX_HOST: "10.0.0.3",
    PERSONAPLEX_PORT: "8998",
    LETTA_BASE_URL: "http://10.0.0.3:8283",
    LETTA_AGENT_NAME: "voice-reasoner",
    VOICE_PROMPT_PATH: "/opt/moshi/voices/NATF0.pt",
  },
},
```

## 4. Create the Letta agent
```bash
curl -X POST http://localhost:9001/api/v1/setup
```

## 5. Start
```bash
pm2 start ecosystem.config.cjs --only vaos-voice-bridge
```

## 6. Verify
```bash
curl http://localhost:9001/health
# Should show: talker_connected, reasoner_available, belief_phase
```
