# Voice Bridge Installation

## 1. Clone and install

```bash
git clone https://github.com/jmanhype/vaos-voice-bridge.git
cd vaos-voice-bridge
cp .env.example .env
# Edit .env with your own values (PersonaPlex host, Supabase keys, etc.)
bun install
```

## 2. Generate TLS certificates

Chrome requires a secure context (HTTPS) for SpeechRecognition. Generate
self-signed certs for local development:

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/key.pem -out certs/cert.pem \
  -days 365 -subj '/CN=localhost'
```

## 3. Start PersonaPlex on the GPU server

```bash
# On the machine with the NVIDIA GPU (~20GB VRAM)
pip install moshi
moshi-server --host 0.0.0.0 --port 8998 \
  --cert /path/to/cert.pem --key /path/to/key.pem
```

Set `PERSONAPLEX_HOST` and `PERSONAPLEX_PORT` in your `.env` to point at this
machine.

## 4. Start Letta (System 2 backend)

```bash
# Docker (recommended)
docker run -d --name letta-server -p 8283:8283 \
  -v letta-data:/root/.letta letta/letta:latest

# Or pip
pip install letta && letta server --port 8283
```

Set `LETTA_BASE_URL` in your `.env` (e.g., `http://192.168.1.100:8283`).

## 5. Start the bridge

```bash
bun run start          # production
bun run dev            # watch mode with auto-reload
```

## 6. Open in Chrome

Navigate to `https://localhost:9001`, accept the self-signed certificate
warning, and click the mic button to begin.

## 7. Verify

```bash
curl -k https://localhost:9001/api/v1/health
# Should show: talker_connected, reasoner_available, belief_phase
```

## Optional: VAOS Control Plane integration

If you're running the full VAOS stack with a control plane and PM2:

1. Copy the voice executor to your control plane:
   ```bash
   cp src/executors/voice.ts /path/to/vaos-control-plane/src/executors/voice.ts
   ```

2. Register it in your control plane's executor index:
   ```typescript
   import { executeVoice } from './voice.js';
   // Add to EXECUTORS map:
   voice: executeVoice,
   ```

3. Add to your PM2 ecosystem config:
   ```javascript
   {
     name: "vaos-voice-bridge",
     script: "bun",
     args: "run src/server.ts",
     cwd: "/path/to/vaos-voice-bridge",
     interpreter: "none",
     env: {
       PORT: "9001",
       PERSONAPLEX_HOST: "your-gpu-server-ip",
       PERSONAPLEX_PORT: "8998",
       LETTA_BASE_URL: "http://your-letta-server:8283",
       LETTA_AGENT_NAME: "voice-reasoner",
     },
   }
   ```
