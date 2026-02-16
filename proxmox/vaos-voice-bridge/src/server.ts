/**
 * Voice Bridge Server — Coordination layer between Talker and Reasoner.
 *
 * Hono HTTP server (matching VAOS conventions) with WebSocket upgrade
 * for voice sessions. Also exposes REST endpoints for ops-loop integration.
 *
 * Architecture (DeepMind "Agents Thinking Fast and Slow"):
 *   User <-> Voice Bridge <-> PersonaPlex (System 1 / Talker)
 *                          <-> Letta+Claude (System 2 / Reasoner)
 *                          <-> Supabase (shared state / ops-loop events)
 */

import { Hono } from 'hono';
import { createLogger } from './lib/logger.js';
import { validateEnv, getEnv } from './lib/config.js';
import { Talker } from './talker.js';
import { Reasoner } from './reasoner.js';
import { shouldWaitForReasoner, isSocialTurn } from './coordinator.js';
import { beliefToPrompt } from './belief.js';
import { synthesize, readWavPcm } from './tts.js';

const logger = createLogger('bridge');

// ─── Session state ──────────────────────────────────────────────

interface VoiceSession {
  id: string;
  talker: Talker;
  reasoner: Reasoner;
  userWs: WebSocket | null;
  turnCount: number;
  createdAt: Date;
}

const sessions = new Map<string, VoiceSession>();

// ─── Client HTML ────────────────────────────────────────────────

const CLIENT_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VAOS Voice Bridge</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    color:#e0e0e0;min-height:100vh;display:flex;flex-direction:column;align-items:center;
    padding:2rem 1rem}
  h1{color:#a78bfa;font-size:1.8rem;margin-bottom:.25rem}
  .subtitle{color:#8b8b9e;font-size:.9rem;margin-bottom:1.5rem}
  #status{padding:.5rem 2rem;border-radius:8px;font-weight:600;font-size:.9rem;
    margin-bottom:1.5rem;transition:all .3s}
  .disconnected{background:#3a1c1c;color:#f87171;border:1px solid #7f1d1d}
  .connecting{background:#3a2e1c;color:#fbbf24;border:1px solid #78350f}
  .connected{background:#1c3a2e;color:#34d399;border:1px solid #064e3b}
  #mic-btn{width:100px;height:100px;border-radius:50%;border:3px solid #4a4a6a;
    background:#2a2a4a;cursor:pointer;display:flex;align-items:center;
    justify-content:center;margin:1.5rem 0;transition:all .2s;position:relative}
  #mic-btn:hover{border-color:#a78bfa;background:#3a3a5a}
  #mic-btn.active{border-color:#f87171;background:#4a2a2a;animation:pulse 1.5s infinite}
  #mic-btn svg{width:40px;height:40px;fill:#8b8b9e}
  #mic-btn.active svg{fill:#f87171}
  @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(248,113,113,.4)}50%{box-shadow:0 0 0 15px rgba(248,113,113,0)}}
  .panel{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:1.25rem;
    width:100%;max-width:600px;margin:.5rem 0}
  .panel-title{font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;
    color:#6b6b8d;margin-bottom:.75rem;font-weight:600}
  .msg{padding:.5rem 0;border-bottom:1px solid #2a2a4a;font-size:.9rem;line-height:1.4}
  .msg:last-child{border-bottom:none}
  .msg.system1{color:#60a5fa}
  .msg.system2{color:#a78bfa}
  .msg.user{color:#34d399}
  .msg.error{color:#f87171;font-style:italic}
  .belief-grid{display:grid;grid-template-columns:90px 1fr;gap:.4rem .75rem;font-size:.9rem}
  .belief-label{color:#6b6b8d;font-weight:500}
  .belief-value{color:#e0e0e0}
  .badge{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.8rem;
    background:#3730a3;color:#a78bfa;font-weight:600}
  #text-input{display:flex;gap:.5rem;width:100%;max-width:600px;margin-top:.5rem}
  #text-input input{flex:1;padding:.6rem 1rem;border-radius:8px;border:1px solid #2a2a4a;
    background:#1a1a2e;color:#e0e0e0;font-size:.9rem;outline:none}
  #text-input input:focus{border-color:#a78bfa}
  #text-input button{padding:.6rem 1.2rem;border-radius:8px;border:none;
    background:#4c1d95;color:#e0e0e0;cursor:pointer;font-weight:600}
  #text-input button:hover{background:#5b21b6}
</style>
</head>
<body>
<h1>VAOS Voice Bridge</h1>
<p class="subtitle">Talker-Reasoner Architecture</p>
<div id="status" class="disconnected">Disconnected</div>

<button id="mic-btn" title="Push to talk">
  <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V5zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
</button>

<div class="panel">
  <div class="panel-title">Conversation</div>
  <div id="convo"><div class="msg">Waiting for connection...</div></div>
</div>

<div class="panel">
  <div class="panel-title">Belief State (Reasoner)</div>
  <div class="belief-grid">
    <span class="belief-label">Phase</span><span class="belief-value" id="b-phase"><span class="badge">--</span></span>
    <span class="belief-label">Goals</span><span class="belief-value" id="b-goals">--</span>
    <span class="belief-label">Project</span><span class="belief-value" id="b-project">--</span>
    <span class="belief-label">Topic</span><span class="belief-value" id="b-topic">--</span>
    <span class="belief-label">Actions</span><span class="belief-value" id="b-actions">None</span>
  </div>
</div>

<div id="text-input">
  <input id="txt" placeholder="Type a message (or use mic)..." autocomplete="off">
  <button id="send-btn">Send</button>
</div>

<script>
const statusEl=document.getElementById('status');
const convoEl=document.getElementById('convo');
const micBtn=document.getElementById('mic-btn');
const txtInput=document.getElementById('txt');
const sendBtn=document.getElementById('send-btn');
let ws=null,opusRec=null,recording=false;
let audioCtx=null,encoderBlobUrl=null;
let decoderWorker=null,moshiWorklet=null;
let totalAudioBytes=0;
let audioSetupPromise=null; // Guard against parallel setupAudioPlayback calls

function setStatus(s){
  statusEl.textContent=s.charAt(0).toUpperCase()+s.slice(1);
  statusEl.className=s;
}

function addMsg(text,cls){
  const d=document.createElement('div');
  d.className='msg '+(cls||'');
  d.textContent=text;
  convoEl.appendChild(d);
  convoEl.scrollTop=convoEl.scrollHeight;
  if(convoEl.children.length>50)convoEl.removeChild(convoEl.firstChild);
}

function updateBelief(b){
  if(!b)return;
  const c=b.conversation||{};
  document.getElementById('b-phase').innerHTML='<span class="badge">'+(c.phase||'--')+'</span>';
  document.getElementById('b-goals').textContent=(c.goals||[]).join(', ')||'--';
  document.getElementById('b-project').textContent=c.project_context||'--';
  document.getElementById('b-topic').textContent=c.current_topic||'--';
  document.getElementById('b-actions').textContent=(b.pending_actions||[]).length?b.pending_actions.map(a=>a.description).join(', '):'None';
}

// Moshi AudioWorklet processor (plays decoded PCM with jitter buffer)
const MOSHI_PROCESSOR_CODE=\`(function(){"use strict";function r(f){return(f*1e3/sampleRate).toFixed(1)}function i(f){return Math.round(f*sampleRate/1e3)}class u extends AudioWorkletProcessor{constructor(){super();let l=i(80);this.initialBufferSamples=1*l;this.partialBufferSamples=i(10);this.maxBufferSamples=i(10);this.partialBufferIncrement=i(5);this.maxPartialWithIncrements=i(80);this.maxBufferSamplesIncrement=i(5);this.maxMaxBufferWithIncrements=i(80);this.initState();this.port.onmessage=a=>{if(a.data.type==="reset"){this.initState();return}let m=a.data.frame;this.frames.push(m);if(this.currentSamples()>=this.initialBufferSamples&&!this.started)this.start();if(this.currentSamples()>=this.totalMaxBufferSamples()){let h=this.initialBufferSamples+this.partialBufferSamples;while(this.currentSamples()>h&&this.frames.length){let e=this.frames[0],t=this.currentSamples()-h;t=Math.min(e.length-this.offsetInFirstBuffer,t);this.offsetInFirstBuffer+=t;if(this.offsetInFirstBuffer>=e.length){this.frames.shift();this.offsetInFirstBuffer=0}}this.maxBufferSamples+=this.maxBufferSamplesIncrement;this.maxBufferSamples=Math.min(this.maxMaxBufferWithIncrements,this.maxBufferSamples)}}}initState(){this.frames=[];this.offsetInFirstBuffer=0;this.started=false;this.remainingPartialBufferSamples=0;this.resetStart();this.partialBufferSamples=i(10);this.maxBufferSamples=i(10)}totalMaxBufferSamples(){return this.maxBufferSamples+this.partialBufferSamples+this.initialBufferSamples}currentSamples(){let l=0;for(let a=0;a<this.frames.length;a++)l+=this.frames[a].length;return l-this.offsetInFirstBuffer}resetStart(){this.started=false}start(){this.started=true;this.remainingPartialBufferSamples=this.partialBufferSamples}canPlay(){return this.started&&this.frames.length>0&&this.remainingPartialBufferSamples<=0}process(l,a,m){const e=a[0][0];if(!this.canPlay()){this.remainingPartialBufferSamples-=e.length;return true}let t=0;while(t<e.length&&this.frames.length){let s=this.frames[0];let n=Math.min(s.length-this.offsetInFirstBuffer,e.length-t);e.set(s.subarray(this.offsetInFirstBuffer,this.offsetInFirstBuffer+n),t);this.offsetInFirstBuffer+=n;t+=n;if(this.offsetInFirstBuffer>=s.length){this.offsetInFirstBuffer=0;this.frames.shift()}}if(t<e.length){this.partialBufferSamples+=this.partialBufferIncrement;this.partialBufferSamples=Math.min(this.partialBufferSamples,this.maxPartialWithIncrements);this.resetStart();for(let s=0;s<t;s++)e[s]*=(t-s)/t}return true}}registerProcessor("moshi-processor",u)})()\`;

// Setup audio playback pipeline: opus decoder worker → AudioWorklet
async function setupAudioPlayback(){
  audioCtx=new AudioContext({sampleRate:24000});
  await audioCtx.resume();
  // Load AudioWorklet (moshi-processor for jitter-buffered playback)
  const procBlob=new Blob([MOSHI_PROCESSOR_CODE],{type:'application/javascript'});
  await audioCtx.audioWorklet.addModule(URL.createObjectURL(procBlob));
  moshiWorklet=new AudioWorkletNode(audioCtx,'moshi-processor');
  moshiWorklet.connect(audioCtx.destination);
  // Load decoder worker from CDN with WASM fetch redirect
  const workerResp=await fetch(OPUS_CDN+'/decoderWorker.min.js');
  if(!workerResp.ok)throw new Error('CDN fetch failed: '+workerResp.status);
  const workerCode=await workerResp.text();
  const patchedCode='const _f=self.fetch;self.fetch=(u,o)=>{if(typeof u==="string"&&u.endsWith(".wasm"))return _f("'+OPUS_CDN+'/decoderWorker.min.wasm",o);return _f(u,o)};'+workerCode;
  decoderWorker=new Worker(URL.createObjectURL(new Blob([patchedCode],{type:'application/javascript'})));
  decoderWorker.onerror=(e)=>addMsg('Decoder error: '+e.message,'error');
  decoderWorker.onmessage=(e)=>{
    if(e.data&&moshiWorklet){
      const pcm=e.data instanceof Float32Array?e.data:(e.data.length?e.data[0]:null);
      if(pcm&&pcm.length>0) moshiWorklet.port.postMessage({frame:pcm,type:'audio',micDuration:0});
    }
  };
  decoderWorker.postMessage({
    command:'init',
    bufferLength:Math.round(960*audioCtx.sampleRate/24000),
    decoderSampleRate:24000,
    outputBufferSampleRate:audioCtx.sampleRate,
    resampleQuality:0
  });
  // Send warmup BOS page (same as native PersonaPlex client)
  const head=new Uint8Array([79,112,117,115,72,101,97,100,1,1,56,1,128,187,0,0,0,0,0]);
  const ogg=new Uint8Array([79,103,103,83,0,2,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,19]);
  const bos=new Uint8Array(ogg.length+head.length);
  bos.set(ogg,0);bos.set(head,ogg.length);
  decoderWorker.postMessage({command:'decode',pages:bos},[bos.buffer]);
  addMsg('Audio playback ready (24kHz Opus decoder)','system1');
  console.log('[audio] Setup complete');
}

async function playAudio(buf){
  totalAudioBytes+=buf.byteLength;
  if(!decoderWorker){
    if(!audioSetupPromise){
      audioSetupPromise=setupAudioPlayback().catch(e=>{
        console.error('Audio setup failed:',e);
        addMsg('Audio setup error: '+e.message,'error');
        audioSetupPromise=null;
      });
    }
    await audioSetupPromise;
  }
  if(audioCtx&&audioCtx.state==='suspended')await audioCtx.resume();
  if(decoderWorker){
    const data=new Uint8Array(buf);
    decoderWorker.postMessage({command:'decode',pages:data},[buf]);
  }
}

function connect(){
  if(ws&&ws.readyState<2)return;
  setStatus('connecting');
  const proto=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(proto+'//'+location.host);
  ws.binaryType='arraybuffer';
  ws.onopen=()=>{
    setStatus('connecting');
    convoEl.innerHTML='';
    addMsg('Connected to Voice Bridge, waiting for PersonaPlex...','system1');
    pollBelief();
    // Mic starts when PersonaPlex handshake completes (state_change → connected)
  };
  ws.onclose=(e)=>{
    setStatus('disconnected');
    stopMic();
    addMsg('Disconnected ('+e.code+')','error');
    setTimeout(connect,3000);
  };
  ws.onerror=()=>addMsg('Connection error','error');
  ws.onmessage=(e)=>{
    if(e.data instanceof ArrayBuffer){
      playAudio(e.data);
    }else{
      try{
        const msg=JSON.parse(e.data);
        if(msg.type==='reasoner_response'){
          addMsg('[System 2] '+msg.text,'system2');
        }else if(msg.type==='talker_text'){
          addMsg('[System 1] '+msg.text,'system1');
        }else if(msg.type==='state_change'){
          addMsg('State: '+msg.state+(msg.source==='talker'?' (PersonaPlex)':''),'system1');
          if(msg.state==='connected'&&msg.source==='talker'){
            // Force-restart mic so fresh Ogg headers are sent to PersonaPlex
            stopMic();
            setStatus('connected');
            addMsg('PersonaPlex ready! Starting mic...','system1');
            setTimeout(startMic,300);
          }else if(msg.state==='disconnected'){
            setStatus('connecting');
            stopMic(); // Stop mic on PersonaPlex disconnect
          }
        }else if(msg.type==='belief_update'){
          updateBelief(msg.belief);
        }
      }catch{
        addMsg(e.data);
      }
    }
  };
}

async function pollBelief(){
  try{
    const r=await fetch('/api/v1/sessions');
    const list=await r.json();
    if(list.length>0){
      const sr=await fetch('/api/v1/sessions/'+list[0].id+'/belief');
      updateBelief(await sr.json());
    }
  }catch{}
  if(ws&&ws.readyState===1)setTimeout(pollBelief,5000);
}

// opus-recorder: produces clean Ogg/Opus pages compatible with PersonaPlex
const OPUS_CDN='https://cdn.jsdelivr.net/npm/opus-recorder@8.0.5/dist';

async function loadOpusRecorder(){
  await new Promise((resolve,reject)=>{
    const s=document.createElement('script');
    s.src=OPUS_CDN+'/recorder.min.js';
    s.onload=resolve;
    s.onerror=()=>reject(new Error('Failed to load opus-recorder from CDN'));
    document.head.appendChild(s);
  });
  // Workers require same-origin — fetch and create blob URL
  const resp=await fetch(OPUS_CDN+'/encoderWorker.min.js');
  if(!resp.ok)throw new Error('Failed to fetch encoder worker');
  const blob=new Blob([await resp.text()],{type:'application/javascript'});
  encoderBlobUrl=URL.createObjectURL(blob);
}

async function startMic(){
  if(recording)return;
  try{
    if(!encoderBlobUrl){
      addMsg('Loading Opus encoder...','system1');
      await loadOpusRecorder();
      addMsg('Opus encoder ready','system1');
    }
    // Match PersonaPlex client config exactly
    opusRec=new Recorder({
      encoderPath:encoderBlobUrl,
      encoderSampleRate:24000,
      maxFramesPerPage:2,
      streamPages:true,
      encoderFrameSize:20,
      numberOfChannels:1,
      encoderComplexity:0,
      encoderApplication:2049,
      resampleQuality:3,
    });
    opusRec.ondataavailable=(typedArray)=>{
      if(ws&&ws.readyState===1){
        ws.send(typedArray.buffer);
      }
    };
    await opusRec.start();
    recording=true;
    micBtn.classList.add('active');
    addMsg('Mic active (Ogg/Opus 24kHz streamed)','system1');
  }catch(err){
    addMsg('Mic error: '+err.message,'error');
    console.error('Mic error:',err);
  }
}

function stopMic(){
  if(!recording)return;
  recording=false;
  micBtn.classList.remove('active');
  if(opusRec){opusRec.stop();opusRec=null;}
}

micBtn.addEventListener('click',()=>{
  if(recording)stopMic();
  else startMic();
});

sendBtn.addEventListener('click',sendText);
txtInput.addEventListener('keydown',(e)=>{if(e.key==='Enter')sendText();});
function sendText(){
  const t=txtInput.value.trim();
  if(!t||!ws||ws.readyState!==1)return;
  ws.send(t);
  addMsg('[You] '+t,'user');
  txtInput.value='';
}

connect();
</script>
</body>
</html>`;

// ─── Hono app ───────────────────────────────────────────────────

function createApp(): Hono {
  const app = new Hono();

  // ─── Client UI ──────────────────────────────────────────────
  app.get('/', (c) => {
    c.header('Cache-Control', 'no-cache, no-store, must-revalidate');
    c.header('Pragma', 'no-cache');
    c.header('Expires', '0');
    return c.html(CLIENT_HTML);
  });

  // Health check
  app.get('/api/v1/health', (c) => {
    return c.json({
      status: 'ok',
      service: 'vaos-voice-bridge',
      version: '1.0.0',
      sessions: sessions.size,
      uptime: process.uptime(),
    });
  });

  // Get current belief state for a session
  app.get('/api/v1/sessions/:id/belief', (c) => {
    const session = sessions.get(c.req.param('id'));
    if (!session) return c.json({ error: 'Session not found' }, 404);
    return c.json(session.reasoner.getBelief());
  });

  // List active sessions
  app.get('/api/v1/sessions', (c) => {
    const list = Array.from(sessions.values()).map(s => ({
      id: s.id,
      turnCount: s.turnCount,
      talkerConnected: s.talker.connected,
      belief: s.reasoner.getBelief().conversation,
      createdAt: s.createdAt.toISOString(),
    }));
    return c.json(list);
  });

  /**
   * POST /api/v1/speak — Ops-loop voice feedback endpoint.
   * Text → PersonaPlex TTS → play to active session.
   * Used by the voice executor to provide spoken feedback.
   */
  app.post('/api/v1/speak', async (c) => {
    const body = await c.req.json<{ text: string; session_id?: string }>();
    if (!body.text) return c.json({ error: 'text required' }, 400);

    // Find target session (specific or first active)
    let session: VoiceSession | undefined;
    if (body.session_id) {
      session = sessions.get(body.session_id);
    } else {
      session = sessions.values().next().value;
    }

    if (!session) return c.json({ error: 'No active voice session' }, 404);

    // Synthesize and stream to user
    const wavPath = await synthesize(body.text);
    if (!wavPath) return c.json({ error: 'TTS synthesis failed' }, 500);

    const pcm = await readWavPcm(wavPath);
    if (!pcm || !session.userWs) {
      return c.json({ error: 'Failed to deliver audio' }, 500);
    }

    // Send audio frame to user WebSocket
    session.userWs.send(pcm);
    logger.info({ text: body.text.slice(0, 80), sessionId: session.id }, 'Spoke to user via ops-loop');

    return c.json({ ok: true, spoken: body.text.length });
  });

  /**
   * POST /api/v1/sessions/:id/inject — Inject context into a session's belief.
   * Used by external systems to inform the Reasoner about events.
   */
  app.post('/api/v1/sessions/:id/inject', async (c) => {
    const session = sessions.get(c.req.param('id'));
    if (!session) return c.json({ error: 'Session not found' }, 404);

    const body = await c.req.json<{ context: string }>();
    if (!body.context) return c.json({ error: 'context required' }, 400);

    // Send context to Reasoner as a belief update
    await session.reasoner.updateBelief(`[SYSTEM] ${body.context}`, '');
    return c.json({ ok: true });
  });

  return app;
}

// ─── WebSocket session handler ──────────────────────────────────

async function handleVoiceSession(userWs: WebSocket): Promise<void> {
  const sessionId = crypto.randomUUID();
  const talker = new Talker();
  const reasoner = new Reasoner();

  const session: VoiceSession = {
    id: sessionId,
    talker,
    reasoner,
    userWs,
    turnCount: 0,
    createdAt: new Date(),
  };
  sessions.set(sessionId, session);

  logger.info({ sessionId }, 'New voice session started');

  // Initialize Reasoner (connects to Letta, loads belief)
  await reasoner.init();

  // Set initial text prompt from belief
  const belief = reasoner.getBelief();
  talker.updateTextPrompt(beliefToPrompt(belief));

  // Connect to PersonaPlex
  await talker.connect();

  // ─── Talker event handlers ──────────────────────────

  // Forward PersonaPlex audio to user
  talker.on('onAudio', (data) => {
    if (userWs.readyState === WebSocket.OPEN) {
      userWs.send(data);
    }
  });

  // Forward Talker state changes to user
  talker.on('onStateChange', (state) => {
    if (userWs.readyState === WebSocket.OPEN) {
      userWs.send(JSON.stringify({ type: 'state_change', state, source: 'talker' }));
    }
  });

  // On each complete turn from PersonaPlex
  talker.on('onTurnComplete', (talkerText) => {
    session.turnCount++;
    logger.debug({ turn: session.turnCount, text: talkerText.slice(0, 80) }, 'Talker turn complete');
    // Send text to client UI
    if (userWs.readyState === WebSocket.OPEN) {
      userWs.send(JSON.stringify({ type: 'talker_text', text: talkerText }));
    }
  });

  // ─── User message handler ──────────────────────────
  // Note: Bun ServerWebSocket uses the websocket handler callbacks
  // (message/close) defined in Bun.serve, not addEventListener.
  // We store the session handler so the Bun WS callbacks can dispatch.
  (userWs as any)._voiceSession = session;
  (userWs as any)._handleText = handleUserText;
}

/**
 * Handle a user's text turn through the Talker-Reasoner pipeline.
 */
async function handleUserText(session: VoiceSession, text: string): Promise<void> {
  const { talker, reasoner } = session;
  const belief = reasoner.getBelief();

  // Check if this is a social turn (no Reasoner needed)
  if (isSocialTurn(text)) {
    logger.debug({ text: text.slice(0, 40) }, 'Social turn — System 1 only');
    return;
  }

  // Ask Coordinator: should Talker wait for Reasoner?
  const decision = shouldWaitForReasoner(text, belief);

  if (decision.waitForReasoner) {
    // ─── System 2 Override ──────────────────────────
    logger.info({
      reason: decision.reason,
      confidence: decision.confidence,
      text: text.slice(0, 80),
    }, 'System 2 override triggered');

    // Get Reasoner's response (blocking)
    const response = await reasoner.processAndRespond(text);

    // Synthesize response via TTS
    const wavPath = await synthesize(response);
    if (wavPath && session.userWs?.readyState === WebSocket.OPEN) {
      const pcm = await readWavPcm(wavPath);
      if (pcm) {
        session.userWs.send(pcm);
      }
    }

    // Also send text response for UI display
    if (session.userWs?.readyState === WebSocket.OPEN) {
      session.userWs.send(JSON.stringify({
        type: 'reasoner_response',
        text: response,
        system: 2,
        decision: decision.reason,
      }));
    }

    // Update PersonaPlex text prompt with new belief
    const updatedBelief = reasoner.getBelief();
    talker.updateTextPrompt(beliefToPrompt(updatedBelief));
  } else {
    // ─── System 1 (Async Reasoner Update) ───────────
    // Don't block — let PersonaPlex handle the response naturally
    // Fire-and-forget belief update to Reasoner

    // We'll get the talker's response from the onTurnComplete event
    const talkerResponsePromise = new Promise<string>((resolve) => {
      const handler = (turnText: string) => {
        resolve(turnText);
      };
      talker.on('onTurnComplete', handler);
      // Timeout: if no response in 5s, resolve with empty
      setTimeout(() => resolve(''), 5000);
    });

    const talkerResponse = await talkerResponsePromise;

    // Async belief update (non-blocking)
    reasoner.updateBelief(text, talkerResponse).then(() => {
      // After belief update, refresh PersonaPlex's prompt
      const updatedBelief = reasoner.getBelief();
      talker.updateTextPrompt(beliefToPrompt(updatedBelief));
    }).catch((err) => {
      logger.error({ err }, 'Background belief update failed');
    });
  }
}

// ─── Start server ───────────────────────────────────────────────

if (import.meta.main) {
  validateEnv();
  const env = getEnv();
  const app = createApp();

  logger.info({ port: env.PORT }, 'Starting Voice Bridge server');

  const server = Bun.serve({
    fetch(req, server) {
      // WebSocket upgrade for voice sessions
      if (req.headers.get('upgrade')?.toLowerCase() === 'websocket') {
        const success = server.upgrade(req);
        if (success) return undefined;
        return new Response('WebSocket upgrade failed', { status: 400 });
      }

      // Regular HTTP → Hono
      return app.fetch(req);
    },
    websocket: {
      open(ws) {
        handleVoiceSession(ws as unknown as WebSocket);
      },
      async message(ws, message) {
        const session = (ws as any)._voiceSession as VoiceSession | undefined;
        if (!session) return;

        if (message instanceof ArrayBuffer || message instanceof Uint8Array) {
          // Audio from user → forward to PersonaPlex
          const buf = message instanceof Uint8Array ? message.buffer : message;
          session.talker.sendAudio(buf);
        } else if (typeof message === 'string') {
          // Text input (for testing or hybrid mode)
          await handleUserText(session, message);
        }
      },
      close(ws) {
        const session = (ws as any)._voiceSession as VoiceSession | undefined;
        if (session) {
          logger.info({ sessionId: session.id, turns: session.turnCount }, 'Voice session ended');
          session.talker.disconnect();
          sessions.delete(session.id);
        }
        logger.debug('WebSocket closed');
      },
    },
    port: env.PORT,
  });

  logger.info({ port: server.port }, 'Voice Bridge listening');
}

export { createApp };
