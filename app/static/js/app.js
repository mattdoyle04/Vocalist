// Game/app logic extracted from index.html

// --- Auth shim (unchanged) ---
if (typeof window.ensureAuth !== 'function') {
  window.ensureAuth = async function ensureAuth(options = {}) {
    const force = !!options.force;
    try {
      if (window.sb?.auth?.getSession) {
        const { data:{ session } } = await window.sb.auth.getSession();
        if (session?.access_token) { window.__supabase_token = session.access_token; return session; }
      }
    } catch (e) {}
    if (window.__supabase_token) return { access_token: window.__supabase_token };
    if (force) document.getElementById('authDialog')?.showModal();
    return null;
  };
}

const state = { voice:false,listening:false,running:false,remaining:60,words:new Set(),onTheme:new Set(),offTheme:new Set(),invalid:0,sr:null,timerId:null,startedAt:null,buffer:'' };

function showStartPlaceholder(show){
  const ph=document.getElementById('startPlaceholder'), ti=document.getElementById('typeInput');
  if(!ph||!ti) return; ph.style.display=show?'grid':'none'; ti.style.visibility=show?'hidden':'visible';
  try{ document.body.classList.toggle('prestart', !!show); }catch{}
}
async function bootPlaceholder(){
  const session = await window.ensureAuth({force:false});
  showStartPlaceholder(!!session && !state.running);
  try{ document.body.classList.remove('running'); }catch{}
  window.sb?.auth?.onAuthStateChange((_e,sess)=>{ if(!state.running) showStartPlaceholder(!!sess); });
}
document.addEventListener('DOMContentLoaded', bootPlaceholder);

function showTinyHint(msg){ const el=document.getElementById('tinyHint'); el.textContent=String(msg||"").toUpperCase(); el.classList.add("show"); clearTimeout(showTinyHint._t); showTinyHint._t=setTimeout(()=> el.classList.remove("show"), 1200); }
function setMicVisual(on){ document.getElementById("micMini").setAttribute("aria-pressed", String(on)); }

// --- On-screen keyboard helpers ---
function isSmallScreen(){ return window.matchMedia('(max-width: 480px)').matches; }
function fitGiantInput(){
  const ti = document.getElementById('typeInput'); if (!ti) return;
  const buf = (state.buffer || ti.value || '');
  const base = 48; const min = 24;
  // If 10 or fewer chars, lock at base size and exit
  if (buf.length <= 10){ ti.style.fontSize = base + 'px'; return; }
  // After 10 chars, reduce by 2px per extra char
  let target = Math.max(min, base - (buf.length - 10) * 2);
  ti.style.fontSize = target + 'px';
  // Safety: if still overflowing, shrink stepwise
  while (ti.scrollWidth > ti.clientWidth - 8 && target > min){ target -= 1; ti.style.fontSize = target + 'px'; }
}
function renderBuffer(){ const ti=document.getElementById('typeInput'); if (ti) { ti.value=(state.buffer||'').toUpperCase(); fitGiantInput(); } }
function showKeyboard(show){ const kb=document.getElementById('osk'); if (!kb) return; const on=!!show; kb.classList.toggle('show', on); kb.setAttribute('aria-hidden', on? 'false':'true'); try{ document.body.classList.toggle('osk-open', on); }catch{} }
function clearBuffer(){ state.buffer=''; renderBuffer(); }

async function ensureMicPermission(){ if (!navigator.mediaDevices?.getUserMedia) return true; try { const s = await navigator.mediaDevices.getUserMedia({ audio: true }); s.getTracks().forEach(t=>t.stop()); return true; } catch { return false; } }
document.getElementById('micMini').addEventListener("click", async ()=>{ if (!state.running) { showTinyHint("PRESS START"); return; } if(!state.voice){ const ok=await ensureMicPermission(); if(!ok) return; } state.voice=!state.voice; if(state.voice) enterVoiceMode(); else exitVoiceMode(); });

function ensureSpeech(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition; if(!SR) return null;
  const r = new SR(); r.continuous = true; r.interimResults = true; r.lang = (navigator.language || "en-US");
  r.onresult = (e) => {
    if (!state.running) return;
    for (let i = e.resultIndex; i < e.results.length; i++){
      const res = e.results[i];
      const t = (res[0]?.transcript || "").trim(); if (!t) continue;
      const parts = t.toLowerCase().split(/\s+/);
      // For interim results, avoid submitting the last (possibly partial) token
      const take = res.isFinal ? parts : parts.slice(0, Math.max(0, parts.length - 1));
      take.forEach(w => acceptCandidate(w, false));
      const last = parts[parts.length-1] || "";
      const live = document.getElementById('liveTranscript');
      if (live) { if (live.style.display === "none") live.style.display = "block"; live.textContent = last.toUpperCase(); }
    }
  };
  r.onerror = _=>{};
  r.onend = () => { if (state.voice && state.running && state.remaining > 0) try { r.start(); } catch {} };
  return r;
}
function enterVoiceMode(){ const live=document.getElementById('liveTranscript'); if (live){ live.textContent=""; live.style.display="none"; } document.getElementById('typeInput').style.display="none"; document.getElementById('modePill').style.display="inline-block"; showKeyboard(false); try{ document.body.classList.add('voice'); }catch{} if(!state.sr) state.sr=ensureSpeech(); if(!state.sr){ showTinyHint("VOICE NOT SUPPORTED"); state.voice=false; return exitVoiceMode(); } state.listening=true; setMicVisual(true); try{ state.sr.start(); }catch{}; }
function exitVoiceMode(){ document.getElementById('typeInput').style.display="block"; document.getElementById('liveTranscript').style.display="none"; document.getElementById('modePill').style.display="none"; try{ document.body.classList.remove('voice'); }catch{} if(state.listening){ state.listening=false; setMicVisual(false); try{ state.sr.stop(); }catch{} } if (state.running && isSmallScreen()) showKeyboard(true); }

(function wrapFetch(){ const _fetch = window.fetch.bind(window); window.fetch = async (input, init = {}) => { const url = (typeof input === 'string') ? input : (input?.url || ''); const sameOrigin = !/^https?:\/\//i.test(url) || url.startsWith(location.origin); if (sameOrigin) { if (!window.__supabase_token) await (window.syncSupabaseToken?.() || Promise.resolve()); init.headers = new Headers(init.headers || {}); if (window.__supabase_token && !init.headers.has('Authorization')) { init.headers.set('Authorization', 'Bearer ' + window.__supabase_token); } } return _fetch(input, init); }; })();

async function syncToday(advance=false){ try{ const r=await fetch('/api/today' + (advance ? '?advance=1' : '')); if(!r.ok) return; const d=await r.json(); if(d.letter) document.getElementById('letterBubble').textContent=d.letter.toUpperCase(); if(d.theme) document.getElementById('themeHeading').textContent=d.theme.toUpperCase(); window.__letterBonus = (d.letterBonus|0); window.__roundSeconds  = (d.roundSeconds|0)}catch{} }

document.getElementById('typeInput').addEventListener("keydown",(e)=>{ if(e.key!=="Enter") return; e.preventDefault(); if (!state.running) { showTinyHint("PRESS START"); return; } const v=(isSmallScreen()? state.buffer : e.target.value).trim(); if(!v) return; acceptCandidate(v,true); clearBuffer(); e.target.value=""; });
document.getElementById('typeInput').addEventListener('input', ()=>{ if (!isSmallScreen()) fitGiantInput(); });
window.addEventListener('resize', ()=>{ fitGiantInput(); });

document.getElementById('startBtn').addEventListener("click", async () => {
  const session = await window.ensureAuth({ force: true });
  if (!session) return;
  if (session.access_token) window.__supabase_token = session.access_token;
  // Default to typed input on start (no mic pre-prompt)
  state.voice = false;
  startRound();
});

// Finish button removed (no early end)

function ensureAcceptedUI(){ if (startRound._ul) return; const section=document.createElement("section"); section.className="section"; const ul=document.createElement("ul"); ul.className="wordstream"; section.append(ul); document.getElementById('acceptedSectionAnchor').replaceWith(section); startRound._ul=ul; }

async function acceptCandidate(raw, fromTyping){
  if (!state.running) { if (fromTyping) showTinyHint("PRESS START"); return; }
  const w = String(raw||"").toLowerCase().replace(/[^a-z\u00C0-\u024f'-]/g,""); if (!w) return;
  const dailyLetter = (document.getElementById('letterBubble').textContent||'').toLowerCase();
  if (state.words.has(w)) return fromTyping && showTinyHint("DUPLICATE");
  if (w.length < 3)      return fromTyping && showTinyHint("MIN LENGTH");
  // Ensure JS string API is used (not Python's startswith)
  if (!w.startsWith(dailyLetter)) return fromTyping && showTinyHint(`MUST START “${dailyLetter.toUpperCase()}”`);
  let server;
  try { const r = await fetch('/api/check-word',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ word:w })}); server = await r.json(); }
  catch { return fromTyping && showTinyHint("CHECK FAILED"); }
  if (!server?.ok) return fromTyping && showTinyHint("NOT A WORD");
  const strict = (server.theme_mode || '').toLowerCase() === 'strict';
  if (!server.theme_ok) { showTinyHint(`OFF THEME: ${(server.theme||'').toUpperCase()}`); if (strict) return; }
  state.words.add(w); if (server.theme_ok) state.onTheme.add(w); else state.offTheme.add(w);
  ensureAcceptedUI(); const li=document.createElement("li"); if (!server.theme_ok) li.classList.add("off-theme"); li.textContent=w.toUpperCase(); startRound._ul.prepend(li); if (fromTyping) { clearBuffer(); renderBuffer(); }
}

async function syncAndRevealToday(){ await syncToday(true); const s=document.getElementById('todaySection'); if (s && s.style.display==='none') s.style.display='block'; }

function setTimer(secs){ state.remaining=secs; document.getElementById("timerMini").textContent=new Date(secs*1000).toISOString().substr(14,5); }

async function startRound(){
  showStartPlaceholder(false);
  await syncAndRevealToday();
  // If still no letter (e.g., network hiccup), try a non-advancing refresh
  const bubble = document.getElementById('letterBubble');
  if (!bubble.textContent || bubble.textContent.trim() === '?') {
    try { await syncToday(false); } catch {}
  }
  const ti=document.getElementById('typeInput'); ti.placeholder=document.getElementById('letterBubble').textContent; ti.disabled=false; document.getElementById('micMini').disabled=false;
  if(startRound._ul) startRound._ul.innerHTML="";
  state.words.clear(); state.onTheme.clear(); state.offTheme.clear(); state.invalid=0; state.startedAt=Date.now(); state.running=true; try{ document.body.classList.add('running'); }catch{} clearBuffer(); renderBuffer();
  document.getElementById('startBtn').style.display="none"; setTimer(30);
  if(state.voice) { enterVoiceMode(); }
  else {
    exitVoiceMode();
    if (isSmallScreen()) { showKeyboard(true); /* do not focus to avoid native kb */ }
    else { ti.focus(); }
  }
  clearInterval(state.timerId); state.timerId=setInterval(()=>{ state.remaining=Math.max(0,state.remaining-1); document.getElementById("timerMini").textContent=new Date(state.remaining*1000).toISOString().substr(14,5); if(state.remaining<=0) endRound(); },1000);
}

async function endRound(){
  clearInterval(state.timerId); state.running=false; try{ document.body.classList.remove('running'); }catch{} document.getElementById('startBtn').style.display="inline-block";
  const ti=document.getElementById('typeInput'); ti.disabled=true;
  if(state.listening){ state.listening=false; setMicVisual(false); try{ state.sr.stop(); }catch{} }
  showKeyboard(false);
  try{ document.body.classList.remove('voice'); }catch{}
  const gameSeconds = Math.max(1, Math.round((Date.now()-(state.startedAt||Date.now()))/1000));
  const letter = document.getElementById('letterBubble').textContent || '—';
  const theme  = document.getElementById('themeHeading').textContent || '—';
  // Summary tiles (no Off Theme, no Duration)
  const letterUp = letter.toUpperCase();
  const themeUp  = theme.toUpperCase();
  document.getElementById('sumLetterHdr')?.textContent = letterUp;
  document.getElementById('sumThemeHdr')?.textContent  = themeUp;
  document.getElementById('sumValid').textContent  = state.onTheme.size;
  document.getElementById('sumWords').textContent  = state.words.size;
  // Bonus per word removed in new scoring; nothing to show here
  document.getElementById('sumScore').textContent  = '…';
  const list = document.getElementById('sumWordList');
  if (list){ list.innerHTML=''; Array.from(state.onTheme).sort().forEach(w=>{const s=document.createElement('span'); s.textContent=w.toUpperCase(); list.appendChild(s);}); Array.from(state.offTheme).sort().forEach(w=>{const s=document.createElement('span'); s.textContent=w.toUpperCase(); s.classList.add('off-theme'); list.appendChild(s);}); }
  document.getElementById('summaryModal').showModal();

  // Submit to server
  const payload = { words: Array.from(state.words), inputs: state.words.size + state.invalid, duration: gameSeconds };
  const session = await window.ensureAuth({ force: true });
  if (!session) { window.__pendingRun = payload; return; }
  await submitRun(payload);
}

function refreshLeaderboardIfOpen(){ const dlg = document.querySelector('dialog[data-role="leaderboard"]'); if (dlg && window.htmx) htmx.ajax('GET', '/leaderboard', { target: dlg, swap: 'outerHTML' }); }
async function submitRun(payload){
  try { const resp = await fetch('/api/submit-run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); if (!resp.ok) { document.getElementById('sumScore').textContent='—'; return; } const d = await resp.json(); if (typeof d.score === 'number') { document.getElementById('sumScore').textContent = String(d.score); } refreshLeaderboardIfOpen(); } catch { document.getElementById('sumScore').textContent='—'; }
}

// Wire OSK events
(function(){
  const kb = document.getElementById('osk');
  if (!kb) return;
  kb.addEventListener('click', (e)=>{
    const btn = e.target.closest('button'); if(!btn) return;
    if (!state.running) { showTinyHint('PRESS START'); return; }
    const key = btn.getAttribute('data-key');
    const action = btn.getAttribute('data-action');
    if (key){ state.buffer += key; renderBuffer(); return; }
    if (action === 'back'){ state.buffer = state.buffer.slice(0,-1); renderBuffer(); return; }
    if (action === 'enter'){
      const v = (state.buffer||'').trim(); if(!v) return; acceptCandidate(v, true); return;
    }
  });
})();
