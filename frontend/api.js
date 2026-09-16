/**
 * API Client — No authentication required.
 *
 * API_BASE resolution:
 *   - Served by FastAPI (port 8001 or any backend port): use window.location.origin
 *     so all API calls go to the same origin (no CORS needed).
 *   - Standalone dev server (e.g. `npm run dev` on port 3000): use the explicit
 *     backend URL so the frontend can reach FastAPI across ports.
 *
 * To override for a custom backend URL, set:
 *   localStorage.setItem('FITNESS_API_BASE', 'http://your-host:8001')
 * before the page loads (useful for staging/production deployments).
 */
const _BACKEND_PORT = 8001;
const _localOverride = (typeof localStorage !== 'undefined') && localStorage.getItem('FITNESS_API_BASE');
const _isStandalone = window.location.port !== String(_BACKEND_PORT) && window.location.port !== '';
const API_BASE = _localOverride
  || (_isStandalone ? `http://localhost:${_BACKEND_PORT}` : window.location.origin);

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
      return { ok: false, status: 0, data: { error: 'Network error' } };
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
