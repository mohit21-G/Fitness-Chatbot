/**
 * API Client
 *
 * API_BASE resolution order:
 *   1. localStorage override  — set FITNESS_API_BASE to point at any backend.
 *   2. localhost              — any localhost origin targets http://localhost:8001.
 *   3. Everything else        — always uses the Render backend (production).
 *
 * Session: the logged-in user is stored in localStorage as SESSION_KEY.
 * All user-specific calls use the session user_id automatically.
 * If no session exists the calls silently fall back to 'default_user'
 * (backward-compat for benchmarks / direct API use).
 */
const _RENDER_BACKEND = 'https://fitness-chatbot-kneq.onrender.com';
const _LOCAL_BACKEND  = 'http://localhost:8001';

const _localOverride = (typeof localStorage !== 'undefined') && localStorage.getItem('FITNESS_API_BASE');
const _isLocalhost   = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

const API_BASE = _localOverride || (_isLocalhost ? _LOCAL_BACKEND : _RENDER_BACKEND);

// ── Session helpers ──────────────────────────────────────────────────────────
const SESSION_KEY = 'fitness_session';

const session = {
  get() {
    try { return JSON.parse(localStorage.getItem(SESSION_KEY)) || null; } catch { return null; }
  },
  set(data) {
    localStorage.setItem(SESSION_KEY, JSON.stringify(data));
  },
  clear() {
    localStorage.removeItem(SESSION_KEY);
  },
  userId() {
    return this.get()?.user_id || 'default_user';
  },
  name() {
    return this.get()?.name || 'User';
  },
  isLoggedIn() {
    return !!this.get()?.user_id;
  },
};

// ── API object ───────────────────────────────────────────────────────────────
const api = {

  // ---- Auth ----
  async login(username, password) {
    return this._post('/api/auth/login', { username, password });
  },

  // ---- Chat ----
  async sendMessage(message, autoLog = false) {
    const uid = session.userId();
    const qs  = uid !== 'default_user' ? `?user_id=${encodeURIComponent(uid)}` : '';
    return this._post(`/api/chat/message${qs}`, { message, auto_log: autoLog });
  },

  async confirmAction(confirm, pendingAction) {
    const uid = session.userId();
    const qs  = uid !== 'default_user' ? `?user_id=${encodeURIComponent(uid)}` : '';
    return this._post(`/api/chat/confirm${qs}`, { confirm, pending_action: pendingAction });
  },

  async sendVoice(audioBlob, ext = 'webm', language = null, autoLog = false) {
    const formData = new FormData();
    formData.append('audio', audioBlob, `recording.${ext}`);
    const uid = session.userId();
    let url = `/api/chat/voice?auto_log=${autoLog}`;
    if (uid !== 'default_user') url += `&user_id=${encodeURIComponent(uid)}`;
    if (language) url += '&language=' + language;

    try {
      const resp = await fetch(API_BASE + url, { method: 'POST', body: formData });
      const data = await resp.json();
      return { ok: resp.ok, status: resp.status, data };
    } catch (e) {
      return { ok: false, status: 0, data: { message: 'Network error', success: false } };
    }
  },

  // ---- Profile + Summary ----
  async getProfile() {
    return this._get(`/api/users/${session.userId()}`);
  },

  async getCalorieTarget() {
    return this._get(`/api/users/${session.userId()}/calorie-target`);
  },

  async updateProfile(fields) {
    return this._put(`/api/users/${session.userId()}`, fields);
  },

  async completeOnboarding(fields) {
    return this._put(`/api/users/${session.userId()}/onboard`, fields);
  },

  async getDailySummary(date = null) {
    const d = date || new Date().toISOString().split('T')[0];
    return this._get(`/api/logs/${session.userId()}/summary/${d}`);
  },

  // ---- Internal ----
  async _get(path) {
    try {
      const resp = await fetch(API_BASE + path, { headers: { 'Accept': 'application/json' } });
      const data = await resp.json();
      return { ok: resp.ok, status: resp.status, data };
    } catch (e) {
      return { ok: false, status: 0, data: { error: 'Network error' } };
    }
  },

  async _post(path, body) {
    try {
      const resp = await fetch(API_BASE + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      return { ok: resp.ok, status: resp.status, data };
    } catch (e) {
      // On first network failure (e.g. Render free-tier cold start), retry once
      try {
        await new Promise(r => setTimeout(r, 3000));
        const resp2 = await fetch(API_BASE + path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
          body: JSON.stringify(body),
        });
        const data2 = await resp2.json();
        return { ok: resp2.ok, status: resp2.status, data: data2 };
      } catch (e2) {
        return { ok: false, status: 0, data: { error: 'Network error' } };
      }
    }
  },

  async _put(path, body) {
    try {
      const resp = await fetch(API_BASE + path, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      return { ok: resp.ok, status: resp.status, data };
    } catch (e) {
      return { ok: false, status: 0, data: { error: 'Network error' } };
    }
  },
};
