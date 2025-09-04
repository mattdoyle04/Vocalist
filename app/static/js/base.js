// === Supabase bootstrap via data-* from server ===
const SUPABASE_URL  = document.body.dataset.supabaseUrl || "";
const SUPABASE_ANON = document.body.dataset.supabaseAnon || "";
const SITE_URL_HINT = document.body.dataset.siteUrl || ""; // e.g. https://vocalist-xxxx.onrender.com

function haveKeys(){ return !!(SUPABASE_URL && SUPABASE_ANON); }
function initSB(){
  if (!haveKeys()) return null;
  if (!window.sb || !window.sb.auth || !String(window.sb?.supabaseUrl).startsWith(SUPABASE_URL)) {
    window.sb = supabase.createClient(SUPABASE_URL, SUPABASE_ANON);
  }
  return window.sb;
}
initSB();

async function getSessionSafe(){
  const sb = initSB();
  if (!sb?.auth) return null;
  const { data:{ session } } = await sb.auth.getSession();
  return session || null;
}
async function syncSupabaseToken(){
  const session = await getSessionSafe();
  window.__supabase_token = session?.access_token || null;
  toggleAuthUI(!!session, session?.user);
  updateHxAuthHeaders(); // keep hx-headers in sync with token
}
function toggleAuthUI(authed, _user){
  const loginBtn  = document.getElementById('authLoginBtn');
  const logoutBtn = document.getElementById('authLogoutBtn');
  const userEl    = document.getElementById('navUser');
  if (loginBtn)  loginBtn.style.display  = authed ? 'none' : '';
  if (logoutBtn) logoutBtn.style.display = authed ? '' : 'none';
  if (userEl) {
    // Privacy: never show any user-identifying text in the navbar
    userEl.style.display = 'none';
    userEl.textContent = '';
  }
}

// Initial sync (don’t await here)
syncSupabaseToken();
// React to auth changes
window.sb?.auth?.onAuthStateChange((_evt, session) => {
  window.__supabase_token = session?.access_token || null;
  toggleAuthUI(!!session, session?.user);
  updateHxAuthHeaders();
});

// === HTMX auth header (synchronous) ===
document.body.addEventListener('htmx:configRequest', (evt) => {
  const t = window.__supabase_token;
  if (t) evt.detail.headers['Authorization'] = 'Bearer ' + t;
});

// Keep hx-headers set on nav buttons (so token travels with click)
function updateHxAuthHeaders() {
  const t = window.__supabase_token;
  const hdr = t ? JSON.stringify({ Authorization: 'Bearer ' + t }) : null;
  document.querySelectorAll('button.nav-link[hx-get]').forEach(btn => {
    if (hdr) btn.setAttribute('hx-headers', hdr);
    else btn.removeAttribute('hx-headers');
  });
}
// Gate clicks when not signed in: show login dialog instead of sending a 401
function requireAuthFor(selector) {
  const el = document.querySelector(selector);
  if (!el) return;
  el.addEventListener('click', async (e) => {
    const session = await getSessionSafe();
    if (!session) {
      e.preventDefault(); e.stopPropagation();
      document.getElementById('authDialog')?.showModal();
    }
  }, { capture: true });
}
requireAuthFor('button.nav-link[hx-get="/my-stats"]');
requireAuthFor('button.nav-link[hx-get="/history"]');
requireAuthFor('button.nav-link[hx-get="/leaderboard"]');

// Mobile nav toggle
document.getElementById('navMenuBtn')?.addEventListener('click', () => {
  document.querySelector('.nav-right')?.classList.toggle('open');
});

// Only open the newly inserted dialog in #modalHost; remove on close
const modalHost = document.getElementById('modalHost');
document.body.addEventListener('htmx:afterSwap', (evt) => {
  if (evt.detail.target !== modalHost) return;
  const dlg = modalHost.lastElementChild;
  if (dlg && dlg.nodeName === 'DIALOG' && dlg.hasAttribute('data-autoshow')) {
    try { dlg.showModal(); } catch {}
  }
});
modalHost.addEventListener('close', (e) => {
  const dlg = e.target;
  if (dlg?.nodeName === 'DIALOG') dlg.remove();
}, true);

// Login / Logout
document.getElementById('authLoginBtn')?.addEventListener('click', () => {
  document.getElementById('authDialog')?.showModal();
});
document.getElementById('authCloseBtn')?.addEventListener('click', () => {
  document.getElementById('authDialog')?.close();
});
document.getElementById('authSendBtn')?.addEventListener('click', async () => {
  const emailEl = document.getElementById('authEmailInput');
  const msgEl   = document.getElementById('authMsg');
  const email   = (emailEl.value || '').trim();

  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)){
    msgEl.textContent = 'Please enter a valid email.'; emailEl.focus(); return;
  }
  if (!haveKeys()){
    msgEl.textContent = 'Auth not configured (server didn’t pass SUPABASE_URL/ANON).';
    return;
  }
  msgEl.textContent = 'Sending…';
  document.getElementById('authSendBtn').disabled = true;

  const redirectTo = SITE_URL_HINT || window.location.origin;
  try{
    const sb = initSB();
    const { error } = await sb.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: redirectTo }
    });
    msgEl.textContent = error ? ('Error: ' + error.message) : 'Check your inbox for the magic link.';
  }catch(e){
    msgEl.textContent = 'Error: ' + (e?.message || 'unknown');
  }finally{
    document.getElementById('authSendBtn').disabled = false;
  }
});

document.getElementById('authLogoutBtn')?.addEventListener('click', async () => {
  try { const sb = initSB(); await sb?.auth?.signOut(); } finally { window.__supabase_token = null; toggleAuthUI(false, null); updateHxAuthHeaders(); }
});

// Optional: same-origin fetch wrapper (adds Authorization automatically)
(function wrapFetch(){
  const _fetch = window.fetch.bind(window);
  window.fetch = async (input, init = {}) => {
    const url = (typeof input === 'string') ? input : (input?.url || '');
    const sameOrigin = !/^https?:\/\//i.test(url) || url.startsWith(location.origin);
    if (sameOrigin) {
      init.headers = new Headers(init.headers || {});
      if (window.__supabase_token && !init.headers.has('Authorization')) {
        init.headers.set('Authorization', 'Bearer ' + window.__supabase_token);
      }
    }
    return _fetch(input, init);
  };
})();

