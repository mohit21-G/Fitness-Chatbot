/**
 * API Client — No authentication required.
 *
 * API_BASE resolution order:
 *   1. localStorage override  — set FITNESS_API_BASE to point at any backend.
 *   2. localhost              — any localhost origin targets http://localhost:8001.
 *   3. Everything else        — always uses the Render backend (production).
 *
 * To override for a custom backend URL, set:
 *   localStorage.setItem('FITNESS_API_BASE', 'https://your-host.onrender.com')
 * before the page loads.
 */
const _RENDER_BACKEND = 'https://fitness-chatbot-kneq.onrender.com';
const _LOCAL_BACKEND  = 'http://localhost:8001';

const _localOverride = (typeof localStorage !== 'undefined') && localStorage.getItem('FITNESS_API_BASE');
const _isLocalhost   = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

const API_BASE = _localOverride || (_isLocalhost ? _LOCAL_BACKEND : _RENDER_BACKEND);

const api = {
  // ---- Chat ----
  async sendMessage(message, autoLog = false) {
    return this._post('/api/chat/message', { message, auto_log: autoLog });
  },

  async confirmAction(confirm, pendingAction) {
    return this._post('/api/chat/confirm', { confirm, pending_action: pendingAction });
  },

  async sendVoice(audioBlob, ext = 'webm', language = null, autoLog = false) {
    const formData = new FormData();
    formData.append('audio', audioBlob, `recording.${ext}`);
    let url = '/api/chat/voice?auto_log=' + autoLog;
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
    return this._get('/api/users/default_user');
  },

  async getCalorieTarget() {
    return this._get('/api/users/default_user/calorie-target');
  },

  async updateProfile(fields) {
    // Persist the user's actual profile values (age, weight, height, gender,
    // activity level, goal, …). BMR/TDEE/targets recompute from these.
    return this._put('/api/users/default_user', fields);
  },

  async getDailySummary(date = null) {
    const d = date || new Date().toISOString().split('T')[0];
    return this._get('/api/logs/default_user/summary/' + d);
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
      // after a short delay before giving up.
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
