/**
 * TTS — PersonaPlex offline mode wrapper for Reasoner responses.
 *
 * When System 2 overrides, the Reasoner returns text that needs to be spoken.
 * We use PersonaPlex's offline mode to synthesize speech:
 *   python -m moshi.offline --text-prompt "<text>" --voice-prompt NATF0.pt \
 *     --input-wav silence.wav --output-wav response.wav
 *
 * The resulting WAV is streamed back to the user.
 */

import { createLogger } from './lib/logger.js';
import { getEnv } from './lib/config.js';
import { spawn } from 'node:child_process';
import { readFile, writeFile, access } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const logger = createLogger('tts');

const SILENCE_WAV_PATH = join(tmpdir(), 'voice-bridge-silence.wav');

/**
 * Generate a 3-second silence WAV file for PersonaPlex offline mode input.
 * Only generated once, cached on disk.
 */
async function ensureSilenceWav(): Promise<string> {
  try {
    await access(SILENCE_WAV_PATH);
    return SILENCE_WAV_PATH;
  } catch {
    // Generate silence WAV: 3 seconds at 24kHz, 16-bit mono
    const sampleRate = 24000;
    const duration = 3;
    const numSamples = sampleRate * duration;
    const dataSize = numSamples * 2; // 16-bit = 2 bytes per sample
    const fileSize = 44 + dataSize; // WAV header = 44 bytes

    const buffer = new ArrayBuffer(fileSize);
    const view = new DataView(buffer);

    // WAV header
    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, fileSize - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);       // fmt chunk size
    view.setUint16(20, 1, true);        // PCM format
    view.setUint16(22, 1, true);        // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true);        // block align
    view.setUint16(34, 16, true);       // bits per sample
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);
    // Samples are all 0 (silence) — ArrayBuffer is zero-initialized

    await writeFile(SILENCE_WAV_PATH, new Uint8Array(buffer));
    logger.info({ path: SILENCE_WAV_PATH }, 'Generated silence WAV');
    return SILENCE_WAV_PATH;
  }
}

/**
 * Synthesize text to speech using PersonaPlex offline mode.
 *
 * @param text - The text to speak (from Reasoner response)
 * @returns Path to the generated WAV file, or null on failure
 */
export async function synthesize(text: string): Promise<string | null> {
  const env = getEnv();
  const silenceWav = await ensureSilenceWav();

  const outputPath = join(tmpdir(), `voice-bridge-tts-${Date.now()}.wav`);

  logger.info({ textLength: text.length }, 'Synthesizing TTS');

  return new Promise((resolve) => {
    const proc = spawn('python', [
      '-m', 'moshi.offline',
      '--text-prompt', text,
      '--voice-prompt', env.VOICE_PROMPT_PATH,
      '--input-wav', silenceWav,
      '--output-wav', outputPath,
    ], {
      timeout: 30_000,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stderr = '';

    proc.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    proc.on('close', (code) => {
      if (code === 0) {
        logger.debug({ outputPath }, 'TTS synthesis complete');
        resolve(outputPath);
      } else {
        logger.error({ code, stderr: stderr.slice(0, 500) }, 'TTS synthesis failed');
        resolve(null);
      }
    });

    proc.on('error', (err) => {
      logger.error({ err }, 'TTS process error');
      resolve(null);
    });
  });
}

/**
 * Read a WAV file and return the raw PCM audio data (skipping header).
 */
export async function readWavPcm(wavPath: string): Promise<Uint8Array | null> {
  try {
    const data = await readFile(wavPath);
    // Skip 44-byte WAV header to get raw PCM
    return new Uint8Array(data.buffer, 44);
  } catch (err) {
    logger.error({ err, wavPath }, 'Failed to read WAV file');
    return null;
  }
}
