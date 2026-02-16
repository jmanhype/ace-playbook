/**
 * Voice Bridge Server — Event bus backbone for the Talker-Reasoner pipeline.
 *
 * Architecture (DeepMind "Agents Thinking Fast and Slow" + CoALA + X-Talk):
 *   User <-> Voice Bridge <-> PersonaPlex (System 1 / Talker)
 *                          <-> Letta+Claude (System 2 / Reasoner)
 *                          <-> Letta memory blocks (CoALA working memory)
 *                          <-> Letta archival (CoALA semantic memory)
 *                          <-> Supabase (shared state / ops-loop events)
 *
 * Event bus flow:
 *   Talker → talker.* events
 *   Trigger → subscribes talker.turn, publishes trigger.activate
 *   Reasoner → subscribes trigger.activate + talker.turn, publishes reasoner.*
 *   Memory → subscribes reasoner.belief, publishes memory.compressed
 *   Bridge → subscribes *, forwards to browser WebSocket
 */

import { Hono } from 'hono';
import { createLogger } from './lib/logger.js';
import { validateEnv, getEnv } from './lib/config.js';
import { EventBus, E } from './events.js';
import { Talker } from './talker.js';
import { Memory } from './memory.js';
import { Trigger } from './trigger.js';
import { Reasoner } from './reasoner.js';
import { synthesize, readWavPcm } from './tts.js';

const logger = createLogger('bridge');

// ─── Echo detection (server-side dedup) ─────────────────────────

function isLikelyEcho(
  userText: string,
  recentTurns: Array<{ text: string; timestamp: number }>,
): boolean {
  const now = Date.now();
  const userWords = new Set(
    userText.toLowerCase().split(/\s+/).filter(w => w.length > 2),
  );
  if (userWords.size === 0) return false;

  for (const turn of recentTurns) {
    if (now - turn.timestamp > 30_000) continue;
    const turnWords = new Set(
      turn.text.toLowerCase().split(/\s+/).filter(w => w.length > 2),
    );
    let overlap = 0;
    for (const w of userWords) {
      if (turnWords.has(w)) overlap++;
    }
    const ratio = overlap / userWords.size;
    // Lower threshold: 40% overlap with 2+ matching words (was 60%/3).
    // SpeechRecognition captures mixed user+speaker audio, so overlap
    // may be lower than pure echo.
    if (ratio >= 0.4 && overlap >= 2) return true;
  }
  return false;
}

/**
 * Length-based echo heuristic for speech input.
 * PersonaPlex monologues transcribed by SpeechRecognition are typically 100+ chars.
 * Genuine user speech that triggers System 2 is typically under 100 chars.
 * If SpeechRecognition produces a long transcript AND PersonaPlex was speaking recently,
 * it's almost certainly echo (mixed user + speaker audio in one transcript).
 */
function isLikelyEchoByLength(
  userText: string,
  recentTurns: Array<{ text: string; timestamp: number }>,
  streamingText: string,
): boolean {
  if (userText.length < 120) return false;
  const now = Date.now();
  // PersonaPlex spoke within the last 30 seconds
  const recentSpeech = recentTurns.some(t => now - t.timestamp <= 30_000);
  const hasStreamingText = streamingText.length > 20;
  return recentSpeech || hasStreamingText;
}

// ─── Session state ──────────────────────────────────────────────

interface VoiceSession {
  id: string;
  bus: EventBus;
  talker: Talker;
  memory: Memory;
  trigger: Trigger;
  reasoner: Reasoner;
  userWs: WebSocket | null;
  turnCount: number;
  createdAt: Date;
  recentTalkerTurns: Array<{ text: string; timestamp: number }>;
  /** Accumulates PersonaPlex's streaming text tokens for real-time echo comparison. */
  talkerStreamingText: string;
  /** Timestamp of last talker audio frame forwarded to browser (for echo gate). */
  lastTalkerAudioTs: number;
}

const sessions = new Map<string, VoiceSession>();

/**
 * Singleton active session.  PersonaPlex only allows ONE WebSocket session at
 * a time (async lock).  When a new browser tab connects we reuse the existing
 * Talker / PersonaPlex connection instead of creating a second one that would
 * deadlock on PersonaPlex's session lock.
 */
let activeSession: VoiceSession | null = null;
let teardownTimer: ReturnType<typeof setTimeout> | null = null;

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
  @keyframes s2pulse{0%,100%{opacity:1}50%{opacity:.5}}
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
<p class="subtitle">Talker-Reasoner Architecture (Event Bus)</p>
<div id="status" class="disconnected">Disconnected</div>

<button id="mic-btn" title="Push to talk">
  <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V5zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
</button>

<div class="panel">
  <div class="panel-title">Conversation</div>
  <div id="convo"><div class="msg">Waiting for connection...</div></div>
  <div id="s2thinking" style="display:none;padding:8px 12px;background:#2d1b69;border-radius:6px;margin-top:6px;color:#a78bfa;font-size:.85rem;animation:s2pulse 1.5s infinite">System 2 thinking...</div>
</div>

<div class="panel">
  <div class="panel-title">Memory (CoALA Blocks)</div>
  <div class="belief-grid">
    <span class="belief-label">Phase</span><span class="belief-value" id="b-phase"><span class="badge">--</span></span>
    <span class="belief-label">Goals</span><span class="belief-value" id="b-goals">--</span>
    <span class="belief-label">Project</span><span class="belief-value" id="b-project">--</span>
    <span class="belief-label">Topic</span><span class="belief-value" id="b-topic">--</span>
    <span class="belief-label">Actions</span><span class="belief-value" id="b-actions">None</span>
    <span class="belief-label">Corrections</span><span class="belief-value" id="b-corrections">None</span>
  </div>
  <div id="b-ledger" style="margin-top:8px;font-size:11px;color:#8888aa;max-height:80px;overflow-y:auto;word-break:break-word;"></div>
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
let audioSetupPromise=null;
let lastAudioPlaybackTs=0;
let lastTalkerSpeechTs=0;
const ECHO_GATE_MS=8000;

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

function updateMemory(blocks,ledger){
  if(!blocks)return;
  try{
    const conv=typeof blocks.conversation_context==='string'?JSON.parse(blocks.conversation_context):blocks.conversation_context;
    const user=typeof blocks.belief_state==='string'?JSON.parse(blocks.belief_state):blocks.belief_state;
    const actions=typeof blocks.action_queue==='string'?JSON.parse(blocks.action_queue):blocks.action_queue;
    const facts=typeof blocks.fact_check==='string'?JSON.parse(blocks.fact_check):blocks.fact_check;
    if(conv){
      document.getElementById('b-phase').innerHTML='<span class="badge">'+(conv.phase||'--')+'</span>';
      document.getElementById('b-topic').textContent=conv.topic||'--';
    }
    if(user){
      document.getElementById('b-goals').textContent=(user.user_goals||user.goals||[]).join(', ')||'--';
      document.getElementById('b-project').textContent=user.current_project||user.currentProject||'--';
    }
    if(actions){
      const running=(actions.running||[]);
      document.getElementById('b-actions').textContent=running.length?running.map(a=>a.description||a).join(', '):'None';
    }
    if(facts){
      const corr=(facts.corrections||[]);
      document.getElementById('b-corrections').textContent=corr.length?corr.slice(-3).map(c=>c.wrong+' -> '+c.right).join('; '):'None';
    }
  }catch{}
  if(ledger){document.getElementById('b-ledger').textContent=ledger;}
}

const MOSHI_PROCESSOR_CODE=\`(function(){"use strict";function r(f){return(f*1e3/sampleRate).toFixed(1)}function i(f){return Math.round(f*sampleRate/1e3)}class u extends AudioWorkletProcessor{constructor(){super();let l=i(80);this.initialBufferSamples=1*l;this.partialBufferSamples=i(10);this.maxBufferSamples=i(10);this.partialBufferIncrement=i(5);this.maxPartialWithIncrements=i(80);this.maxBufferSamplesIncrement=i(5);this.maxMaxBufferWithIncrements=i(80);this.initState();this.port.onmessage=a=>{if(a.data.type==="reset"){this.initState();return}let m=a.data.frame;this.frames.push(m);if(this.currentSamples()>=this.initialBufferSamples&&!this.started)this.start();if(this.currentSamples()>=this.totalMaxBufferSamples()){let h=this.initialBufferSamples+this.partialBufferSamples;while(this.currentSamples()>h&&this.frames.length){let e=this.frames[0],t=this.currentSamples()-h;t=Math.min(e.length-this.offsetInFirstBuffer,t);this.offsetInFirstBuffer+=t;if(this.offsetInFirstBuffer>=e.length){this.frames.shift();this.offsetInFirstBuffer=0}}this.maxBufferSamples+=this.maxBufferSamplesIncrement;this.maxBufferSamples=Math.min(this.maxMaxBufferWithIncrements,this.maxBufferSamples)}}}initState(){this.frames=[];this.offsetInFirstBuffer=0;this.started=false;this.remainingPartialBufferSamples=0;this.resetStart();this.partialBufferSamples=i(10);this.maxBufferSamples=i(10)}totalMaxBufferSamples(){return this.maxBufferSamples+this.partialBufferSamples+this.initialBufferSamples}currentSamples(){let l=0;for(let a=0;a<this.frames.length;a++)l+=this.frames[a].length;return l-this.offsetInFirstBuffer}resetStart(){this.started=false}start(){this.started=true;this.remainingPartialBufferSamples=this.partialBufferSamples}canPlay(){return this.started&&this.frames.length>0&&this.remainingPartialBufferSamples<=0}process(l,a,m){const e=a[0][0];if(!this.canPlay()){this.remainingPartialBufferSamples-=e.length;return true}let t=0;while(t<e.length&&this.frames.length){let s=this.frames[0];let n=Math.min(s.length-this.offsetInFirstBuffer,e.length-t);e.set(s.subarray(this.offsetInFirstBuffer,this.offsetInFirstBuffer+n),t);this.offsetInFirstBuffer+=n;t+=n;if(this.offsetInFirstBuffer>=s.length){this.offsetInFirstBuffer=0;this.frames.shift()}}if(t<e.length){this.partialBufferSamples+=this.partialBufferIncrement;this.partialBufferSamples=Math.min(this.partialBufferSamples,this.maxPartialWithIncrements);this.resetStart();for(let s=0;s<t;s++)e[s]*=(t-s)/t}return true}}registerProcessor("moshi-processor",u)})()\`;

async function setupAudioPlayback(){
  audioCtx=new AudioContext({sampleRate:24000});
  await audioCtx.resume();
  const procBlob=new Blob([MOSHI_PROCESSOR_CODE],{type:'application/javascript'});
  await audioCtx.audioWorklet.addModule(URL.createObjectURL(procBlob));
  moshiWorklet=new AudioWorkletNode(audioCtx,'moshi-processor');
  moshiWorklet.connect(audioCtx.destination);
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
  const head=new Uint8Array([79,112,117,115,72,101,97,100,1,1,56,1,128,187,0,0,0,0,0]);
  const ogg=new Uint8Array([79,103,103,83,0,2,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,19]);
  const bos=new Uint8Array(ogg.length+head.length);
  bos.set(ogg,0);bos.set(head,ogg.length);
  decoderWorker.postMessage({command:'decode',pages:bos},[bos.buffer]);
  addMsg('Audio playback ready (24kHz Opus decoder)','system1');
}

async function playAudio(buf){
  totalAudioBytes+=buf.byteLength;
  // Track audio activity as secondary echo signal.
  // PersonaPlex sends audio frames even during silence, but speech frames
  // are larger (>200 bytes) than silence/comfort-noise frames.
  // This catches the case where PersonaPlex sends audio without text tokens.
  if(buf.byteLength>200){lastTalkerSpeechTs=Date.now();}
  if(!decoderWorker){
    if(!audioSetupPromise){
      audioSetupPromise=setupAudioPlayback().catch(e=>{
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
    pollMemory();
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
        if(msg.type==='system2_thinking'){
          const el=document.getElementById('s2thinking');
          if(el)el.style.display=msg.active?'block':'none';
          // Reset Moshi audio buffer to cut off hallucinated speech mid-sentence
          if(msg.resetAudio&&moshiWorklet){moshiWorklet.port.postMessage({type:'reset'});}
        }else if(msg.type==='reasoner_response'){
          const el=document.getElementById('s2thinking');
          if(el)el.style.display='none';
          addMsg('[System 2 → System 1] '+msg.text,'system2');
        }else if(msg.type==='talker_speaking'){
          lastTalkerSpeechTs=Date.now();
        }else if(msg.type==='talker_text'){
          lastTalkerSpeechTs=Date.now();
          addMsg('[System 1] '+msg.text,'system1');
        }else if(msg.type==='state_change'){
          addMsg('State: '+msg.state+(msg.source==='talker'?' (PersonaPlex)':''),'system1');
          if(msg.state==='connected'&&msg.source==='talker'){
            stopMic();
            setStatus('connected');
            addMsg('PersonaPlex ready! Starting mic...','system1');
            setTimeout(startMic,300);
          }else if(msg.state==='disconnected'){
            setStatus('connecting');
            stopMic();
          }
        }else if(msg.type==='memory_update'){
          updateMemory(msg.blocks,msg.ledger);
        }
      }catch{
        addMsg(e.data);
      }
    }
  };
}

async function pollMemory(){
  try{
    const r=await fetch('/api/v1/sessions');
    const list=await r.json();
    if(list.length>0){
      const sr=await fetch('/api/v1/sessions/'+list[0].id+'/memory');
      updateMemory(await sr.json());
    }
  }catch{}
  if(ws&&ws.readyState===1)setTimeout(pollMemory,5000);
}

const OPUS_CDN='https://cdn.jsdelivr.net/npm/opus-recorder@8.0.5/dist';

async function loadOpusRecorder(){
  await new Promise((resolve,reject)=>{
    const s=document.createElement('script');
    s.src=OPUS_CDN+'/recorder.min.js';
    s.onload=resolve;
    s.onerror=()=>reject(new Error('Failed to load opus-recorder from CDN'));
    document.head.appendChild(s);
  });
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
  ws.send('text:'+t);
  addMsg('[You] '+t,'user');
  txtInput.value='';
}

// ── Web Speech API: live transcription of user's voice ──
// Runs in parallel with the Opus stream to PersonaPlex.
// Sends transcribed text as user.text events to the bridge.
let speechRec=null;
let speechActive=false;
function startSpeechRecognition(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR){console.warn('SpeechRecognition not supported');addMsg('[Speech API not supported in this browser]','error');return;}
  speechRec=new SR();
  speechRec.continuous=true;
  speechRec.interimResults=true;
  speechRec.lang='en-US';
  speechRec.maxAlternatives=1;
  let lastFinal='';
  speechRec.onresult=(e)=>{
    let interim='',final='';
    for(let i=e.resultIndex;i<e.results.length;i++){
      const t=e.results[i][0].transcript;
      if(e.results[i].isFinal){final+=t;}
      else{interim+=t;}
    }
    if(final&&final!==lastFinal){
      lastFinal=final;
      const trimmed=final.trim();
      if(Date.now()-lastTalkerSpeechTs<ECHO_GATE_MS){
        console.log('[SpeechRec] ECHO SUPPRESSED (final):',trimmed);
        return;
      }
      console.log('[SpeechRec] Final:',trimmed);
      addMsg('[You] '+trimmed,'user');
      // Send to bridge as user text for trigger evaluation
      if(ws&&ws.readyState===1){ws.send('speech:'+trimmed);}
    }
    if(interim){
      if(Date.now()-lastTalkerSpeechTs<ECHO_GATE_MS){return;}
      // Show interim results in a lighter style
      const existing=document.getElementById('interim-speech');
      if(existing){existing.textContent='[...] '+interim;}
      else{
        const el=document.createElement('div');
        el.id='interim-speech';
        el.style.cssText='color:#888;font-style:italic;padding:2px 6px;';
        el.textContent='[...] '+interim;
        msgContainer.appendChild(el);
        msgContainer.scrollTop=msgContainer.scrollHeight;
      }
    }
  };
  speechRec.onerror=(e)=>{
    console.warn('Speech recognition error:',e.error);
    if(e.error!=='no-speech'&&e.error!=='aborted'){
      addMsg('[Speech error: '+e.error+']','error');
    }
  };
  speechRec.onstart=()=>{
    console.log('[SpeechRec] Started');
    addMsg('[Speech recognition active]','system1');
  };
  speechRec.onend=()=>{
    console.log('[SpeechRec] Ended, speechActive='+speechActive+' recording='+recording);
    // Remove interim display
    const interim=document.getElementById('interim-speech');
    if(interim)interim.remove();
    // Auto-restart if mic is still active
    if(speechActive&&recording){
      setTimeout(()=>{try{speechRec.start();}catch(e){console.warn('Speech restart failed:',e);}},200);
    }
  };
  try{speechRec.start();speechActive=true;console.log('[SpeechRec] Initiating...');}catch(e){console.warn('Speech start failed:',e);addMsg('[Speech start failed: '+e.message+']','error');}
}
function stopSpeechRecognition(){
  speechActive=false;
  if(speechRec){try{speechRec.stop();}catch(e){}}
}

// Patch startMic/stopMic to also start/stop speech recognition
const _origStartMic=startMic;
startMic=async function(){await _origStartMic();startSpeechRecognition();};
const _origStopMic=stopMic;
stopMic=function(){stopSpeechRecognition();_origStopMic();};

connect();
</script>
</body>
</html>`;

// ─── Hono app ───────────────────────────────────────────────────

function createApp(): Hono {
  const app = new Hono();

  app.get('/', (c) => {
    c.header('Cache-Control', 'no-cache, no-store, must-revalidate');
    c.header('Pragma', 'no-cache');
    c.header('Expires', '0');
    return c.html(CLIENT_HTML);
  });

  app.get('/api/v1/health', (c) => {
    return c.json({
      status: 'ok',
      service: 'vaos-voice-bridge',
      version: '2.0.0',
      architecture: 'event-bus',
      sessions: sessions.size,
      uptime: process.uptime(),
    });
  });

  // Get memory blocks for a session
  app.get('/api/v1/sessions/:id/memory', (c) => {
    const session = sessions.get(c.req.param('id'));
    if (!session) return c.json({ error: 'Session not found' }, 404);
    const blocks: Record<string, string> = {};
    for (const [label, value] of session.memory.getAllBlocks()) {
      blocks[label] = value;
    }
    return c.json(blocks);
  });

  // List active sessions
  app.get('/api/v1/sessions', (c) => {
    const list = Array.from(sessions.values()).map(s => ({
      id: s.id,
      turnCount: s.turnCount,
      talkerConnected: s.talker.connected,
      eventCount: s.bus.count,
      errorCount: s.bus.errors,
      createdAt: s.createdAt.toISOString(),
    }));
    return c.json(list);
  });

  // POST /api/v1/speak — Ops-loop voice feedback endpoint
  app.post('/api/v1/speak', async (c) => {
    const body = await c.req.json<{ text: string; session_id?: string }>();
    if (!body.text) return c.json({ error: 'text required' }, 400);

    let session: VoiceSession | undefined;
    if (body.session_id) {
      session = sessions.get(body.session_id);
    } else {
      session = sessions.values().next().value;
    }
    if (!session) return c.json({ error: 'No active voice session' }, 404);

    const wavPath = await synthesize(body.text);
    if (!wavPath) return c.json({ error: 'TTS synthesis failed' }, 500);

    const pcm = await readWavPcm(wavPath);
    if (!pcm || !session.userWs) {
      return c.json({ error: 'Failed to deliver audio' }, 500);
    }

    session.userWs.send(pcm);
    logger.info({ text: body.text.slice(0, 80), sessionId: session.id }, 'Spoke to user via ops-loop');
    return c.json({ ok: true, spoken: body.text.length });
  });

  // POST /api/v1/sessions/:id/inject — Inject context into a session
  app.post('/api/v1/sessions/:id/inject', async (c) => {
    const session = sessions.get(c.req.param('id'));
    if (!session) return c.json({ error: 'Session not found' }, 404);

    const body = await c.req.json<{ context: string }>();
    if (!body.context) return c.json({ error: 'context required' }, 400);

    // Emit as user text event (will route through Trigger → Reasoner)
    session.bus.emit(E.userText(session.id, `[SYSTEM] ${body.context}`));
    return c.json({ ok: true });
  });

  return app;
}

// ─── Session Setup (Event Bus Wiring) ───────────────────────────

async function handleVoiceSession(userWs: WebSocket): Promise<void> {
  const sessionId = crypto.randomUUID();

  // 1. Create the event bus (session-scoped, X-Talk pattern)
  const bus = new EventBus();

  // 2. Create components
  const talker = new Talker();
  const memory = new Memory(bus, sessionId);
  const trigger = new Trigger(bus, sessionId, {
    proactiveInterval: 25,
    confidenceThreshold: 0.5,
  });
  const reasoner = new Reasoner(bus, memory, trigger, sessionId);

  const session: VoiceSession = {
    id: sessionId,
    bus, talker, memory, trigger, reasoner,
    userWs,
    turnCount: 0,
    createdAt: new Date(),
    recentTalkerTurns: [],
    talkerStreamingText: '',
    lastTalkerAudioTs: 0,
  };
  sessions.set(sessionId, session);

  // CRITICAL: Set activeSession + WS tag immediately (before any awaits) so that
  // a second browser WS open (which fires synchronously) sees it and
  // adopts into this session instead of creating a duplicate Talker.
  activeSession = session;
  (userWs as any)._voiceSession = session;

  logger.info({ sessionId }, 'New voice session (event bus architecture)');

  // 3. Initialize Reasoner (connects to Letta, syncs memory blocks)
  await reasoner.init();

  // 4. Wire Talker callbacks → event bus
  talker.on('onText', (text) => {
    bus.emit(E.talkerText(sessionId, text));
  });

  talker.on('onTurnComplete', (text) => {
    session.turnCount++;
    bus.emit(E.talkerTurn(sessionId, text, session.turnCount));
  });

  talker.on('onAudio', (data) => {
    bus.emit(E.talkerAudio(sessionId, data));
  });

  talker.on('onStateChange', (state) => {
    bus.emit(E.talkerState(sessionId, state));
  });

  // 5. Wire event bus → browser WebSocket (the Bridge subscriber)
  //    NOTE: All handlers read session.userWs (not the closure variable)
  //    so that browser WS swaps (reconnects) are picked up automatically.

  // ── System 2 flow ──────────────────────────────────────────────
  //
  // When trigger fires:
  //   1. Mute PersonaPlex audio/text to browser (stop hallucination delivery)
  //   2. Tell browser to reset Moshi audio buffer (cut off mid-sentence hallucination)
  //   3. Show thinking indicator in browser
  //   4. PersonaPlex stays connected (no disconnect — avoids handshake latency)
  //
  // When System 2 finishes:
  //   5. Unmute
  //   6. Single reconnect with answer prompt
  //   7. PersonaPlex delivers the result in its own voice
  //
  // One audio disruption (the answer reconnect), not two.

  let system2Muted = false;

  bus.on('reasoner.thinking', (event) => {
    system2Muted = event.active;
    if (event.active) {
      logger.info('System 2 active — muting PersonaPlex + resetting browser audio buffer');
      // Tell browser to clear buffered hallucinated audio and show thinking indicator
      if (session.userWs?.readyState === WebSocket.OPEN) {
        session.userWs.send(JSON.stringify({ type: 'system2_thinking', active: true, resetAudio: true }));
      }
    } else {
      logger.info('System 2 done — unmuting, answer reconnect follows');
    }
  }, 90); // High priority — runs before the audio forwarder

  // Forward PersonaPlex audio to browser (suppressed during System 2)
  bus.on('talker.audio', (event) => {
    if (system2Muted) return; // Suppress hallucinated audio
    if (session.userWs?.readyState === WebSocket.OPEN) {
      session.userWs.send(event.data);
    }
  }, 10); // Low priority — output layer

  // Accumulate PersonaPlex streaming text for real-time echo comparison.
  // This fires on every text token, BEFORE turn-complete (which waits 350ms).
  // Also notifies browser so it can gate SpeechRecognition (speech-based, not audio-frame-based).
  bus.on('talker.text', (event) => {
    session.talkerStreamingText += event.text;
    // Cap at 1000 chars to prevent unbounded growth (covers ~30s of speech)
    if (session.talkerStreamingText.length > 1000) {
      session.talkerStreamingText = session.talkerStreamingText.slice(-600);
    }
    // Notify browser that PersonaPlex is actively speaking (not just sending audio frames).
    // The browser uses this to gate SpeechRecognition — more reliable than raw audio timing
    // because Moshi sends continuous audio (including silence) in full-duplex mode.
    if (session.userWs?.readyState === WebSocket.OPEN) {
      session.userWs.send(JSON.stringify({ type: 'talker_speaking' }));
    }
  }, 50);

  // Forward Talker text to browser (suppressed during System 2)
  // Also record for echo dedup
  bus.on('talker.turn', (event) => {
    // Record for server-side echo detection (prune >15s)
    const now = Date.now();
    session.recentTalkerTurns.push({ text: event.text, timestamp: now });
    session.recentTalkerTurns = session.recentTalkerTurns.filter(
      t => now - t.timestamp <= 30_000,
    );
    // Don't clear streaming buffer — SpeechRecognition results arrive delayed,
    // so we need the text available for echo matching even after turn completes.

    if (system2Muted) return; // Suppress hallucinated text
    if (session.userWs?.readyState === WebSocket.OPEN) {
      session.userWs.send(JSON.stringify({ type: 'talker_text', text: event.text }));
    }
  }, 10);

  // Forward Talker state changes to browser
  bus.on('talker.state', (event) => {
    logger.info({ state: event.state, hasUserWs: !!session.userWs, wsReady: session.userWs?.readyState }, 'Talker state → browser');
    if (session.userWs?.readyState === WebSocket.OPEN) {
      session.userWs.send(JSON.stringify({ type: 'state_change', state: event.state, source: 'talker' }));
    }
  }, 10);

  // Forward Reasoner interjections to browser
  bus.on('reasoner.interjection', (event) => {
    if (session.userWs?.readyState === WebSocket.OPEN) {
      session.userWs.send(JSON.stringify({
        type: 'reasoner_response',
        text: event.text,
        system: 2,
        decision: event.decision,
      }));
    }
  }, 10);

  // Memory compressed → update Talker text prompt + push to browser.
  // When System 2 has an answer, do the SINGLE reconnect with the answer prompt.
  // This is the only audio disruption — PersonaPlex reconnects and delivers the result.
  let lastAnswerReconnect = 0;
  bus.on('memory.compressed', (event) => {
    // Update PersonaPlex's text prompt (stored for next connect)
    talker.updateTextPrompt(event.prompt);

    // Push updated memory blocks to browser
    if (session.userWs?.readyState === WebSocket.OPEN) {
      const blocks: Record<string, string> = {};
      for (const [label, value] of memory.getAllBlocks()) {
        blocks[label] = value;
      }
      session.userWs.send(JSON.stringify({
        type: 'memory_update',
        blocks,
        ledger: memory.currentLedger,
        tokenEstimate: event.tokenEstimate,
      }));
    }

    // If System 2 just responded, do the answer reconnect.
    // This fires ONCE — we clear the System 2 response after triggering
    // the reconnect to prevent re-triggering on subsequent compressed events.
    const hasSystem2Context = memory.getLastSystem2Response().length > 0;
    if (hasSystem2Context) {
      // Clear immediately — one shot only.
      memory.clearLastSystem2Response();
      lastAnswerReconnect = Date.now();
      logger.info({ promptLength: event.prompt.length }, 'Answer reconnect — PersonaPlex will deliver System 2 result');
      talker.reconnectWithNewPrompt().catch(err => {
        logger.warn({ err: err instanceof Error ? err.message : String(err) }, 'Answer reconnect failed');
      });
    }
  }, 10);

  // Forward error events to browser
  bus.on('error.occurred', (event) => {
    if (session.userWs?.readyState === WebSocket.OPEN) {
      session.userWs.send(JSON.stringify({
        type: 'error',
        source: event.source,
        error: event.error,
      }));
    }
  }, 5);

  // 6. Set initial text prompt from memory compression
  const initialPrompt = memory.compress();
  talker.updateTextPrompt(initialPrompt);

  // Push initial memory to browser
  pushMemoryToBrowser(session);

  // 7. Connect to PersonaPlex
  await talker.connect();
}

/** Push current memory blocks to the browser WS. */
function pushMemoryToBrowser(session: VoiceSession): void {
  if (session.userWs?.readyState !== WebSocket.OPEN) return;
  const blocks: Record<string, string> = {};
  for (const [label, value] of session.memory.getAllBlocks()) {
    blocks[label] = value;
  }
  session.userWs.send(JSON.stringify({
    type: 'memory_update',
    blocks,
    ledger: session.memory.currentLedger,
  }));
}

/**
 * Adopt a new browser WebSocket into an existing session.
 * Swaps the WS reference so all event-bus handlers automatically use it.
 */
function adoptSession(session: VoiceSession, newWs: WebSocket): void {
  // Suppress adopt spam — if the new WS is already the current one, skip
  if (session.userWs === newWs) return;

  logger.info({ sessionId: session.id, talkerConnected: session.talker.connected }, 'Adopting new browser WS into existing session');

  // Mark old WS as adopted (so its close handler doesn't trigger teardown)
  // but do NOT close it — closing triggers browser reconnect → adopt loop.
  if (session.userWs) {
    (session.userWs as any)._adopted = true;
  }

  // Swap in the new WS
  session.userWs = newWs;
  (newWs as any)._voiceSession = session;

  // Cancel any pending teardown timer
  if (teardownTimer) {
    clearTimeout(teardownTimer);
    teardownTimer = null;
  }

  // Push current state to the new browser
  pushMemoryToBrowser(session);

  // If Talker is already connected, tell the browser immediately
  if (session.talker.handshakeComplete) {
    logger.info({ sessionId: session.id }, 'Adopt: sending connected (handshake already done)');
    newWs.send(JSON.stringify({ type: 'state_change', state: 'connected', source: 'talker' }));
  } else if (session.talker.connected) {
    logger.info({ sessionId: session.id }, 'Adopt: sending connecting (handshake pending)');
    newWs.send(JSON.stringify({ type: 'state_change', state: 'connecting', source: 'talker' }));
  } else {
    logger.info({ sessionId: session.id }, 'Adopt: talker not yet connected — browser will receive state_change via bus when handshake completes');
  }
}

// ─── Start server ───────────────────────────────────────────────

if (import.meta.main) {
  validateEnv();
  const env = getEnv();
  const app = createApp();

  // TLS for SpeechRecognition (Chrome requires secure context)
  const certDir = new URL('../certs/', import.meta.url).pathname;
  let tls: { key: string; cert: string } | undefined;
  try {
    const keyFile = Bun.file(certDir + 'key.pem');
    const certFile = Bun.file(certDir + 'cert.pem');
    if (await keyFile.exists() && await certFile.exists()) {
      tls = {
        key: await keyFile.text(),
        cert: await certFile.text(),
      };
      logger.info('TLS enabled (self-signed cert)');
    }
  } catch { /* no certs, run plain HTTP */ }

  const proto = tls ? 'https' : 'http';
  logger.info({ port: env.PORT, proto }, 'Starting Voice Bridge server (event bus architecture)');

  const server = Bun.serve({
    tls,
    fetch(req, server) {
      if (req.headers.get('upgrade')?.toLowerCase() === 'websocket') {
        const success = server.upgrade(req);
        if (success) return undefined;
        return new Response('WebSocket upgrade failed', { status: 400 });
      }
      return app.fetch(req);
    },
    websocket: {
      open(ws) {
        const browserWs = ws as unknown as WebSocket;
        logger.info({ hasActiveSession: !!activeSession, wsReadyState: ws.readyState }, 'Browser WS open event');

        // Singleton: if we already have an active session, adopt this new
        // browser WS into it instead of creating a duplicate PersonaPlex
        // connection (which would deadlock on its single-session lock).
        if (activeSession) {
          adoptSession(activeSession, browserWs);
          return;
        }
        handleVoiceSession(browserWs);
      },
      async message(ws, message) {
        const session = (ws as any)._voiceSession as VoiceSession | undefined;
        if (!session) {
          logger.warn({ hasSession: false, msgType: typeof message }, 'Message from browser with no session');
          return;
        }

        if (message instanceof ArrayBuffer || message instanceof Uint8Array) {
          // Audio from user → forward to PersonaPlex
          const buf = message instanceof Uint8Array ? message.buffer : message;
          session._browserMsgCount = (session._browserMsgCount ?? 0) + 1;
          if (session._browserMsgCount <= 3 || session._browserMsgCount % 500 === 0) {
            logger.debug({ count: session._browserMsgCount, size: buf.byteLength, talkerConnected: session.talker.connected }, 'Browser audio → PersonaPlex');
          }
          session.talker.sendAudio(buf);
        } else if (typeof message === 'string') {
          // Parse prefix: "speech:..." = SpeechRecognition, "text:..." = typed input
          const isSpeech = message.startsWith('speech:');
          const isTyped = message.startsWith('text:');
          const text = isSpeech ? message.slice(7) : isTyped ? message.slice(5) : message;

          if (isSpeech) {
            // === Echo suppression (speech only — never blocks typed text) ===

            // Layer 1: Length heuristic — long transcripts during active PersonaPlex
            // conversation are almost certainly echo (mixed user + speaker audio).
            if (isLikelyEchoByLength(text, session.recentTalkerTurns, session.talkerStreamingText)) {
              logger.info({ text: text.slice(0, 200), len: text.length }, 'Echo suppressed (length heuristic) — long speech during active conversation');
              return;
            }

            // Layer 2: Content match against PersonaPlex's streaming text buffer
            if (session.talkerStreamingText && isLikelyEcho(text, [{ text: session.talkerStreamingText, timestamp: Date.now() }])) {
              logger.info({ text: text.slice(0, 200) }, 'Echo suppressed (streaming match)');
              return;
            }

            // Layer 3: Text dedup against recent completed PersonaPlex turns
            if (isLikelyEcho(text, session.recentTalkerTurns)) {
              logger.info({ text: text.slice(0, 200) }, 'Echo suppressed (turn dedup)');
              return;
            }
          }

          logger.info({ text: text.slice(0, 200), len: text.length, source: isSpeech ? 'speech' : isTyped ? 'typed' : 'raw' }, 'User text received from browser');
          session.bus.emit(E.userText(session.id, text));
        }
      },
      close(ws) {
        // If this WS was replaced by adoptSession(), don't tear down
        if ((ws as any)._adopted) return;

        const session = (ws as any)._voiceSession as VoiceSession | undefined;
        if (!session) return;

        // Null out the browser WS ref but keep the session alive.
        // Browser auto-reconnects in ~3s; give 10s grace period before teardown.
        session.userWs = null;
        logger.info({ sessionId: session.id }, 'Browser disconnected — grace period before teardown');

        if (teardownTimer) clearTimeout(teardownTimer);
        teardownTimer = setTimeout(() => {
          // If browser reconnected (adoptSession ran), userWs will be non-null
          if (session.userWs !== null) return;

          logger.info({
            sessionId: session.id,
            turns: session.turnCount,
            events: session.bus.count,
            errors: session.bus.errors,
          }, 'Voice session ended (no browser reconnect)');
          session.talker.disconnect();
          session.bus.shutdown();
          sessions.delete(session.id);
          if (activeSession === session) activeSession = null;
          teardownTimer = null;
        }, 10_000);
      },
    },
    port: env.PORT,
  });

  logger.info({ port: server.port, proto, url: `${proto}://localhost:${server.port}` }, 'Voice Bridge listening (event bus architecture)');
}

export { createApp };
