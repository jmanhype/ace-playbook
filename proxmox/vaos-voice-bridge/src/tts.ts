/**
 * TTS via PersonaPlex WebSocket (uses the running server).
 *
 * Deploys a lightweight Python proxy script to the 3090 that connects
 * to the running PersonaPlex server's WebSocket API as a client.
 * Uses the same `sphn` audio library as PersonaPlex for format compatibility.
 *
 * Flow: text → SSH → Python proxy → WebSocket → PersonaPlex → WAV → SCP back
 *
 * Zero extra VRAM — the proxy is just a WebSocket client.
 * The model is already loaded in the running server (~19GB).
 */

import { spawn } from 'child_process';
import { createLogger } from './logger.js';
import type { Config } from './config.js';
import { randomUUID } from 'crypto';
import { readFile, unlink, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';

const logger = createLogger('tts');

/**
 * Python TTS proxy — sequential send/receive loop.
 * Uses sphn (same audio library as PersonaPlex) for Opus encoding/decoding.
 * Sends silence frames to drive the model, collects audio output as WAV.
 */
const TTS_PROXY_SCRIPT = `#!/usr/bin/env python3
"""PersonaPlex TTS proxy — connects to local WebSocket, sends silence, captures speech as WAV."""
import asyncio, sys, struct, time, urllib.parse
import numpy as np
import sphn
import aiohttp

async def tts(text, output_path, voice='NATF0.pt', host='localhost', port=8998):
    sr = 24000
    params = urllib.parse.urlencode({
        'text_prompt': text,
        'voice_prompt': voice,
        'text_temperature': '0.3',
        'text_topk': '10',
        'audio_temperature': '0.6',
        'audio_topk': '200',
        'pad_mult': '0',
        'repetition_penalty': '1.0',
        'repetition_penalty_context': '64',
    })
    url = f"http://{host}:{port}/api/chat?{params}"

    writer = sphn.OpusStreamWriter(sr)
    reader = sphn.OpusStreamReader(sr)
    audio_chunks = []
    got_audio = False
    last_audio_time = 0

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url, timeout=30) as ws:
            # Wait for handshake
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY and len(msg.data) > 0 and msg.data[0] == 0:
                    break

            # Sequential send/receive loop
            for i in range(250):  # ~20 seconds max
                # Send silence (80ms = 1920 samples at 24kHz)
                silence = np.zeros(1920, dtype=np.float32)
                writer.append_pcm(silence)
                data = writer.read_bytes()
                if len(data) > 0:
                    await ws.send_bytes(b"\\x01" + data)

                # Read responses (up to 5 messages per iteration)
                for _ in range(5):
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                        if msg.type == aiohttp.WSMsgType.BINARY and len(msg.data) > 0:
                            kind = msg.data[0]
                            if kind == 1:  # audio
                                reader.append_bytes(msg.data[1:])
                                pcm = reader.read_pcm()
                                if pcm is not None and len(pcm) > 0:
                                    audio_chunks.append(pcm)
                                    got_audio = True
                                    last_audio_time = time.time()
                            elif kind == 6:  # ping
                                await ws.send_bytes(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                            break
                    except asyncio.TimeoutError:
                        break

                # End detection: 2s of silence after receiving audio
                if got_audio and time.time() - last_audio_time > 2.0:
                    break

                await asyncio.sleep(0.06)

    if not audio_chunks:
        print("ERROR: no audio received", file=sys.stderr)
        sys.exit(1)

    audio = np.concatenate(audio_chunks)
    pcm16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    # Write WAV (24kHz mono 16-bit PCM)
    data_size = len(pcm16) * 2
    with open(output_path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))   # PCM
        f.write(struct.pack('<H', 1))   # mono
        f.write(struct.pack('<I', sr))
        f.write(struct.pack('<I', sr * 2))
        f.write(struct.pack('<H', 2))   # block align
        f.write(struct.pack('<H', 16))  # bits/sample
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(pcm16.tobytes())

    duration = len(pcm16) / sr
    print(f"OK {len(pcm16)} samples, {duration:.1f}s")

if __name__ == '__main__':
    text = sys.argv[1]
    output = sys.argv[2]
    voice = sys.argv[3] if len(sys.argv) > 3 else 'NATF0.pt'
    asyncio.run(tts(text, output, voice))
`;

export class PersonaplexTTS {
  private config: Config;
  private proxyReady = false;
  private remoteScript = '/tmp/personaplex-tts-proxy.py';

  constructor(config: Config) {
    this.config = config;
  }

  /** Deploy the TTS proxy script to the 3090. */
  async init(): Promise<void> {
    const { host, sshUser } = this.config.personaplex;

    // Write script locally, SCP to 3090
    const localScript = join(tmpdir(), 'personaplex-tts-proxy.py');
    await writeFile(localScript, TTS_PROXY_SCRIPT, { mode: 0o755 });

    try {
      await this.exec('scp', [
        '-o', 'StrictHostKeyChecking=no',
        localScript,
        `${sshUser}@${host}:${this.remoteScript}`,
      ]);
      await unlink(localScript).catch(() => {});
      this.proxyReady = true;
      logger.info('TTS proxy script deployed to 3090');
    } catch (err) {
      await unlink(localScript).catch(() => {});
      logger.warn({ err }, 'Failed to deploy TTS proxy — TTS may not work');
    }
  }

  /**
   * Synthesize text to WAV via PersonaPlex.
   * Returns WAV bytes (24kHz mono 16-bit PCM).
   */
  async speak(text: string, voice?: string): Promise<Buffer> {
    if (!this.proxyReady) {
      await this.init();
      if (!this.proxyReady) {
        throw new Error('TTS proxy not deployed');
      }
    }

    const id = randomUUID().slice(0, 8);
    const remoteOut = `/tmp/personaplex-tts-${id}.wav`;
    const voicePrompt = voice || this.config.personaplex.defaultVoice;
    const { host, sshUser, venvPath } = this.config.personaplex;

    // Sanitize text for shell
    const safeText = text.replace(/'/g, "'\\''");

    logger.info({ text: text.slice(0, 80), voice: voicePrompt, id }, 'Synthesizing speech');
    const start = Date.now();

    try {
      // Write a bash wrapper to avoid shell quoting issues
      const wrapper = [
        '#!/bin/bash',
        `source ${venvPath}/bin/activate`,
        `python3 ${this.remoteScript} "$1" "$2" "$3"`,
      ].join('\n');

      const remoteWrapper = `/tmp/personaplex-tts-run-${id}.sh`;
      const localWrapper = join(tmpdir(), `personaplex-tts-run-${id}.sh`);
      await writeFile(localWrapper, wrapper, { mode: 0o755 });

      // Upload wrapper + execute
      await this.exec('scp', [
        '-o', 'StrictHostKeyChecking=no',
        localWrapper,
        `${sshUser}@${host}:${remoteWrapper}`,
      ]);
      await unlink(localWrapper).catch(() => {});

      await this.exec('ssh', [
        '-o', 'StrictHostKeyChecking=no',
        `${sshUser}@${host}`,
        `bash ${remoteWrapper} '${safeText}' '${remoteOut}' '${voicePrompt}'`,
      ]);

      // Download the WAV
      const localOut = join(tmpdir(), `personaplex-tts-${id}.wav`);
      await this.exec('scp', [
        '-o', 'StrictHostKeyChecking=no',
        `${sshUser}@${host}:${remoteOut}`,
        localOut,
      ]);

      const wav = await readFile(localOut);
      const elapsed = Date.now() - start;
      logger.info({ id, elapsed, bytes: wav.length }, 'Speech synthesized');

      // Cleanup
      await Promise.all([
        unlink(localOut).catch(() => {}),
        this.exec('ssh', [
          '-o', 'StrictHostKeyChecking=no',
          `${sshUser}@${host}`,
          `rm -f ${remoteOut} ${remoteWrapper}`,
        ]).catch(() => {}),
      ]);

      return wav;
    } catch (err) {
      logger.error({ err, id }, 'TTS failed');
      throw err;
    }
  }

  private exec(cmd: string, args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
      let stdout = '';
      let stderr = '';
      proc.stdout.on('data', (d: Buffer) => { stdout += d.toString(); });
      proc.stderr.on('data', (d: Buffer) => { stderr += d.toString(); });
      proc.on('close', (code) => {
        if (code === 0) resolve(stdout);
        else reject(new Error(`${cmd} exited ${code}: ${stderr}`));
      });
      proc.on('error', reject);
    });
  }
}
