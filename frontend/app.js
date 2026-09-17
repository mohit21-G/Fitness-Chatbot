/**
 * Fitness AI Chatbot — Main App
 * Handles: login, onboarding wizard, chat, profile, summary, voice.
 */

const $ = (sel) => document.querySelector(sel);
const chatMessages  = () => $('#chat-messages');
const chatInput     = () => $('#chat-input');

let pendingAction   = null;
let mediaRecorder   = null;
let audioChunks     = [];
let isRecording     = false;

// ============================================================
// BOOTSTRAP — decide which screen to show on load
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  if (session.isLoggedIn()) {
    enterChat();
  } else {
    showLogin();
  }
});

// ── Tab switcher (called from HTML onclick) ──────────────────
function switchTab(tab) {
  const isSignin = tab === 'signin';
  document.getElementById('tab-signin').classList.toggle('active',  isSignin);
  document.getElementById('tab-signup').classList.toggle('active', !isSignin);
  document.getElementById('form-signin').classList.toggle('hidden', !isSignin);
  document.getElementById('form-signup').classList.toggle('hidden',  isSignin);
  document.getElementById('signin-error').textContent = '';
  document.getElementById('signup-error').textContent = '';
}
window.switchTab = switchTab;

function showLogin() {
  $('#screen-login').classList.add('active');
  $('#screen-chat').classList.remove('active');
  _bindAuthEvents();
}

let _authBound = false;
function _bindAuthEvents() {
  if (_authBound) return;
  _authBound = true;

  // ── Sign In ──────────────────────────────────────────────
  const siBtn = $('#btn-signin');
  const siErr = $('#signin-error');
  ['si-username','si-password'].forEach(id => {
    document.getElementById(id)?.addEventListener('keydown', e => {
      if (e.key === 'Enter') siBtn.click();
    });
  });

  siBtn.addEventListener('click', async () => {
    const username = (document.getElementById('si-username').value || '').trim();
    const password = (document.getElementById('si-password').value || '').trim();
    siErr.textContent = '';
    if (!username) { siErr.textContent = 'Please enter your username.'; return; }
    if (!password) { siErr.textContent = 'Please enter your password.'; return; }

    siBtn.disabled = true; siBtn.textContent = 'Signing in…';
    const res = await api.signin(username, password);   // ← dedicated signin endpoint
    siBtn.disabled = false; siBtn.textContent = 'Sign In';

    if (!res.ok) {
      // 403 = account exists but onboarding incomplete → redirect to Sign Up
      if (res.status === 403) {
        siErr.textContent = 'Account setup incomplete. Switching to Sign Up…';
        setTimeout(() => {
          switchTab('signup');
          document.getElementById('su-username').value = username;
          document.getElementById('su-password').value = password;
          siErr.textContent = '';
        }, 1200);
        return;
      }
      siErr.textContent = res.data?.detail || 'Sign in failed. Check username / password.';
      return;
    }
    const d = res.data;
    session.set({ user_id: d.user_id, name: d.name, username: d.username });
    enterChat();
  });

  // ── Sign Up ──────────────────────────────────────────────
  const suBtn = $('#btn-signup');
  const suErr = $('#signup-error');

  suBtn.addEventListener('click', async () => {
    suErr.textContent = '';

    const g = id => (document.getElementById(id)?.value || '').trim();
    const username  = g('su-username');
    const password  = g('su-password');
    const name      = g('su-name');
    const age       = parseInt(g('su-age'), 10);
    const gender    = g('su-gender');
    const height_cm = parseFloat(g('su-height'));
    const weight_kg = parseFloat(g('su-weight'));
    const activity  = g('su-activity');
    const goal      = g('su-goal');
    const diet      = g('su-diet');
    const calRaw    = g('su-calorie');
    const calorie   = calRaw ? parseFloat(calRaw) : null;

    // Validation
    if (username.length < 2)  { suErr.textContent = 'Username must be at least 2 characters.'; return; }
    if (password.length < 4)  { suErr.textContent = 'Password must be at least 4 characters.'; return; }
    if (!name)                 { suErr.textContent = 'Full name is required.'; return; }
    if (!(age >= 10 && age <= 120))        { suErr.textContent = 'Age must be between 10 and 120.'; return; }
    if (!gender)               { suErr.textContent = 'Please select gender.'; return; }
    if (!(height_cm >= 50 && height_cm <= 300)) { suErr.textContent = 'Height must be 50–300 cm.'; return; }
    if (!(weight_kg >= 20 && weight_kg <= 500)) { suErr.textContent = 'Weight must be 20–500 kg.'; return; }
    if (!activity)             { suErr.textContent = 'Please select activity level.'; return; }
    if (!goal)                 { suErr.textContent = 'Please select fitness goal.'; return; }
    if (!diet)                 { suErr.textContent = 'Please select diet type.'; return; }
    if (calorie !== null && (isNaN(calorie) || calorie < 500 || calorie > 10000)) {
      suErr.textContent = 'Calorie goal must be between 500 and 10 000.'; return;
    }

    suBtn.disabled = true; suBtn.textContent = 'Creating account…';

    // Step 1: register the account (dedicated signup endpoint — never checks existing password)
    const signupRes = await api.signup(username, password);
    if (!signupRes.ok) {
      suBtn.disabled = false; suBtn.textContent = 'Create Account & Start';
      // 409 = username already fully registered → tell user to sign in
      if (signupRes.status === 409) {
        suErr.textContent = 'Username already registered. Please use Sign In.';
        setTimeout(() => {
          switchTab('signin');
          document.getElementById('si-username').value = username;
          document.getElementById('si-password').value = password;
          suErr.textContent = '';
        }, 1500);
        return;
      }
      suErr.textContent = signupRes.data?.detail || 'Could not create account. Please try again.';
      return;
    }
    const d = signupRes.data;
    session.set({ user_id: d.user_id, name: name, username: d.username });

    // Step 2: save full profile via onboard endpoint (sets onboarding_complete=true)
    const profileRes = await api.completeOnboarding({
      name, age, gender, height_cm, weight_kg,
      activity_level: activity, fitness_goal: goal, diet_type: diet,
      custom_calorie_goal: calorie,
    });

    suBtn.disabled = false; suBtn.textContent = 'Create Account & Start';

    if (!profileRes.ok) {
      suErr.textContent = profileRes.data?.detail || 'Profile save failed — please try again.';
      return;
    }

    // Update session name from saved profile
    if (profileRes.data?.name) {
      const s = session.get();
      session.set({ ...s, name: profileRes.data.name });
    }

    enterChat();
  });
}

function enterChat() {
  $('#screen-login').classList.remove('active');
  $('#screen-chat').classList.add('active');
  bindChatEvents();
  updateUserBadge();
  loadGreeting();
}

function updateUserBadge() {
  const badge = $('#user-badge');
  if (badge) badge.textContent = '👤 ' + session.name();
}

// ============================================================
// CHAT INIT
// ============================================================

function bindChatEvents() {
  // Prevent double-binding
  if (window._chatEventsBound) return;
  window._chatEventsBound = true;

  $('#btn-send').addEventListener('click', handleSendText);
  chatInput().addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendText(); }
  });
  $('#btn-voice').addEventListener('click', toggleRecording);
  $('#btn-summary').addEventListener('click', showSummary);
  $('#btn-profile').addEventListener('click', showProfile);
  $('#btn-logout').addEventListener('click', handleLogout);
  $('#close-summary').addEventListener('click', () => $('#modal-summary').classList.add('hidden'));
  $('#close-profile').addEventListener('click', () => $('#modal-profile').classList.add('hidden'));
}

async function loadGreeting() {
  showTyping();
  const res = await api.sendMessage('hello');
  hideTyping();
  if (res.ok) {
    addBotMessage(res.data);
  } else {
    addBotError(
      res.status === 0
        ? 'Backend is starting up — please send a message in a few seconds.'
        : (res.data?.detail || 'Something went wrong')
    );
  }
}

function handleLogout() {
  session.clear();
  window._chatEventsBound = false;
  _authBound = false;
  pendingAction = null;
  const cm = chatMessages();
  if (cm) cm.innerHTML = '';
  showLogin();
}

// ============================================================
// CHAT SEND
// ============================================================

async function handleSendText() {
  const inp  = chatInput();
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  addUserMessage(text);
  showTyping();

  const res = await api.sendMessage(text, false);
  hideTyping();

  if (res.ok) {
    addBotMessage(res.data);
  } else {
    addBotError(
      res.status === 0
        ? 'Could not reach the server — please try again in a moment.'
        : (res.data?.detail || 'Something went wrong')
    );
  }
}

// ============================================================
// VOICE — Click to Start / Click to Stop Toggle
// ============================================================

function toggleRecording() {
  if (isRecording) stopRecording();
  else startRecording();
}

async function startRecording() {
  if (isRecording) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    let mimeType = 'audio/webm';
    if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) mimeType = 'audio/ogg;codecs=opus';
    else if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) mimeType = 'audio/webm;codecs=opus';
    mediaRecorder = new MediaRecorder(stream, { mimeType });
    audioChunks = [];
    mediaRecorder.ondataavailable = (e) => { if (e.data?.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = sendVoiceMessage;
    mediaRecorder.start(250);
    isRecording = true;
    $('#btn-voice').classList.add('recording');
    $('#voice-indicator').classList.remove('hidden');
  } catch {
    addBotError('Microphone access denied. Please allow microphone.');
  }
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  isRecording = false;
  $('#btn-voice').classList.remove('recording');
  $('#voice-indicator').classList.add('hidden');
  setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(t => t.stop());
    }
  }, 300);
}

async function sendVoiceMessage() {
  if (audioChunks.length === 0) { addBotError('No audio captured. Please try again.'); return; }
  const mimeType = mediaRecorder?.mimeType || 'audio/webm';
  const blob     = new Blob(audioChunks, { type: mimeType });
  if (blob.size < 100) { addBotError('No audio captured. Please hold and speak.'); return; }

  let ext = 'webm';
  if (mimeType.includes('ogg')) ext = 'ogg';
  else if (mimeType.includes('mp4')) ext = 'mp4';

  addUserMessage('[Voice message]');
  showTyping();
  const res = await api.sendVoice(blob, ext);
  hideTyping();

  if (res.ok) {
    if (res.data?.data?.transcribed_text) {
      const lang      = res.data.data.stt_language || '';
      const conf      = res.data.data.stt_confidence;
      const confLabel = conf ? ` ${(conf * 100).toFixed(0)}%` : '';
      const langLabel = lang ? ` [${lang.toUpperCase()}${confLabel}]` : '';
      const normalised = res.data.data.normalised_text || '';
      let heardMsg = `🎤 Heard${langLabel}: "${res.data.data.transcribed_text}"`;
      if (normalised && normalised !== res.data.data.transcribed_text)
        heardMsg += `\n→ Understood: "${normalised}"`;
      addSystemMessage(heardMsg);
    }
    addBotMessage(res.data);
  } else {
    addBotError(res.data?.detail || res.data?.message || 'Voice processing failed');
  }
}

// ============================================================
// MESSAGE RENDERING
// ============================================================

function addUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg msg-user';
  div.innerHTML = `<div class="msg-text">${escapeHtml(text)}</div><div class="msg-time">${timeNow()}</div>`;
  chatMessages().appendChild(div);
  scrollToBottom();
}

function addBotMessage(data) {
  const div = document.createElement('div');
  div.className = 'msg msg-bot';
  let html = `<div class="msg-text">${formatBotMarkdown(data.message)}</div>`;

  if (data.data?.nutrition?.calories) {
    const n = data.data.nutrition;
    html += `<div class="nutrition-card">
      <div class="food-name">${escapeHtml(n.food_name_display || '')}</div>
      <div class="cal">${n.calories.toFixed(0)} kcal</div>
      <div class="macros">
        <span>P: ${n.protein_g.toFixed(0)}g</span>
        <span>C: ${n.carbs_g.toFixed(0)}g</span>
        <span>F: ${n.fat_g.toFixed(0)}g</span>
      </div>
    </div>`;
  }

  if (data.needs_confirmation && data.pending_action) {
    pendingAction = data.pending_action;
    html += `<div class="confirm-card"><div class="confirm-btns">
      <button class="btn-yes" onclick="handleConfirm(true)">Haa, Save karo</button>
      <button class="btn-no" onclick="handleConfirm(false)">Na, Cancel</button>
    </div></div>`;
  }

  if (data.options?.length > 0) {
    html += `<div class="options-row">`;
    for (const opt of data.options) {
      html += `<button class="option-btn" onclick="handleOptionClick(this,'${escapeAttr(opt)}')">${escapeHtml(opt)}</button>`;
    }
    html += `</div>`;
  }

  html += `<div class="msg-time">${timeNow()}</div>`;
  div.innerHTML = html;
  chatMessages().appendChild(div);
  scrollToBottom();

  if (data.action_taken === 'food_logged' || data.action_taken === 'exercise_logged' ||
      data.data?.action_taken === 'food_logged') {
    const msg = data.data?.toast_message || '✓ Log saved successfully';
    showToast('success', msg);
  }
}

function addBotError(msg) {
  const div = document.createElement('div');
  div.className = 'msg msg-bot';
  div.innerHTML = `<div class="msg-text" style="color:var(--error)">${escapeHtml(msg)}</div><div class="msg-time">${timeNow()}</div>`;
  chatMessages().appendChild(div);
  scrollToBottom();
}

function addSystemMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg msg-bot';
  div.style.opacity = '0.7';
  div.style.fontSize = '0.82rem';
  div.innerHTML = `<div class="msg-text">${escapeHtml(text)}</div>`;
  chatMessages().appendChild(div);
  scrollToBottom();
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'msg msg-typing';
  div.id = 'typing-indicator';
  div.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
  chatMessages().appendChild(div);
  scrollToBottom();
}

function hideTyping() {
  document.getElementById('typing-indicator')?.remove();
}

// ============================================================
// CONFIRMATION
// ============================================================

async function handleConfirm(confirm) {
  document.querySelectorAll('.confirm-card').forEach(el => el.remove());
  if (!pendingAction) return;
  showTyping();
  const res = await api.confirmAction(confirm, pendingAction);
  hideTyping();
  pendingAction = null;
  if (res.ok) addBotMessage(res.data);
  else { addBotError('Failed to process confirmation'); showToast('error', 'Failed to save log.'); }
}
window.handleConfirm = handleConfirm;

async function handleOptionClick(btn, optionText) {
  const row = btn.closest('.options-row');
  if (row) {
    row.querySelectorAll('.option-btn').forEach(b => { b.disabled = true; b.classList.remove('selected'); });
    btn.classList.add('selected');
    btn.disabled = true;
  }
  const text = optionText.split('(')[0].trim();
  addUserMessage(text);
  showTyping();
  const res = await api.sendMessage(text, false);
  hideTyping();
  if (res.ok) addBotMessage(res.data);
  else addBotError(res.status === 0 ? 'Could not reach the server.' : (res.data?.detail || 'Something went wrong'));
}
window.handleOptionClick = handleOptionClick;

// ============================================================
// SUMMARY MODAL
// ============================================================

async function showSummary() {
  const modal = $('#modal-summary');
  const body  = $('#summary-body');
  body.innerHTML = '<p>Loading...</p>';
  modal.classList.remove('hidden');

  const res = await api.getDailySummary();
  if (!res.ok) { body.innerHTML = '<p style="color:var(--error)">Failed to load summary</p>'; return; }

  const d   = res.data;
  const pct = d.calorie_target > 0 ? Math.min((d.net_calories / d.calorie_target) * 100, 100) : 0;
  const isOver = d.net_calories > d.calorie_target;

  body.innerHTML = `
    <div class="summary-card">
      <h4>Today's Progress</h4>
      <div class="progress-bar"><div class="progress-fill ${isOver ? 'over' : ''}" style="width:${Math.min(pct,100)}%"></div></div>
      <div class="summary-row"><span>Target</span><span>${d.calorie_target.toFixed(0)} kcal</span></div>
      <div class="summary-row"><span><strong>Total Consumed</strong></span><span><strong>${d.total_calories_consumed.toFixed(0)} kcal</strong></span></div>
      <div class="summary-row"><span>Burned</span><span>${d.total_calories_burned.toFixed(0)} kcal</span></div>
      <div class="summary-row highlight"><span><strong>Total Remaining</strong></span><span><strong>${d.remaining_calories.toFixed(0)} kcal</strong></span></div>
    </div>
    ${renderDailyLog(d.meal_breakdown)}
    <div style="margin-top:1rem">
      <h4 style="margin-bottom:0.5rem">Macros</h4>
      <div class="summary-row"><span><strong>Protein</strong></span><span><strong>${d.total_protein_g.toFixed(0)}g</strong></span></div>
      <div class="summary-row"><span><strong>Carbs</strong></span><span><strong>${d.total_carbs_g.toFixed(0)}g</strong></span></div>
      <div class="summary-row"><span><strong>Fat</strong></span><span><strong>${d.total_fat_g.toFixed(0)}g</strong></span></div>
      <div class="summary-row"><span>Fiber</span><span>${d.total_fiber_g.toFixed(0)}g</span></div>
    </div>
    <div style="margin-top:1rem;font-size:0.82rem;color:var(--text-muted)">
      ${d.food_entries} food items | ${d.exercise_entries} exercises logged today
    </div>`;
}

const MEAL_DISPLAY_ORDER = ['breakfast','morning','lunch','afternoon','snack','evening','dinner'];
const MEAL_LABELS = {
  breakfast:'🍳 Breakfast', morning:'🌞 Morning', lunch:'🍱 Lunch', afternoon:'🍱 Afternoon',
  snack:'☕ Snack', evening:'🌇 Evening', dinner:'🍽️ Dinner',
};

function _mealLabel(meal) {
  return MEAL_LABELS[meal] || (meal ? `${getMealEmoji(meal)} ${meal.charAt(0).toUpperCase()+meal.slice(1)}` : '🍽️ Other');
}

function renderDailyLog(mealBreakdown) {
  if (!mealBreakdown || !Object.keys(mealBreakdown).length) return '';
  const allMeals  = [...MEAL_DISPLAY_ORDER, ...Object.keys(mealBreakdown).filter(m => !MEAL_DISPLAY_ORDER.includes(m))];
  const presented = allMeals.filter(m => mealBreakdown[m]);
  if (!presented.length) return '';

  let html = `<div style="margin-top:1rem"><h4 style="margin-bottom:0.5rem">Meal Log</h4>`;
  for (const meal of presented) {
    const entry       = mealBreakdown[meal];
    const items       = entry.items       || [];
    const itemDetails = entry.item_details || [];
    if (items.length === 0 && itemDetails.length === 0) continue;
    const cals = typeof entry.calories === 'number' ? entry.calories : 0;
    let itemsHtml = '';
    if (itemDetails.length > 0) {
      itemsHtml = itemDetails.map(item => {
        const name = escapeHtml(item.food_name || '?');
        const qty  = item.quantity ? ` — <strong>${escapeHtml(item.quantity)}</strong>` : '';
        const cal  = typeof item.calories === 'number' ? ` — <strong>${item.calories.toFixed(0)} kcal</strong>` : '';
        return `<div class="log-item">• ${name}${qty}${cal}</div>`;
      }).join('');
    } else if (items.length > 0) {
      itemsHtml = items.map(i => `<div class="log-item">• ${escapeHtml(i)}</div>`).join('');
    }
    html += `<div class="meal-section">
      <div class="summary-row"><strong>${_mealLabel(meal)}</strong><span>${cals.toFixed(0)} kcal</span></div>
      ${itemsHtml}
    </div>`;
  }
  html += `</div>`;
  return html;
}

function getMealEmoji(meal) {
  const map = { breakfast:'🍳', morning:'🌞', lunch:'🍱', afternoon:'☀️', snack:'☕', evening:'🌇', dinner:'🍽️' };
  return map[meal] || '🍽️';
}

// ============================================================
// PROFILE MODAL
// ============================================================

const PROFILE_OPTIONS = {
  gender:         ['male','female'],
  activity_level: ['sedentary','light','moderate','active','very_active'],
  fitness_goal:   ['lose_weight','maintain','gain_muscle','gain_weight'],
  diet_type:      ['veg','non_veg','eggetarian','vegan'],
};

function _optionsHtml(field, selected) {
  return PROFILE_OPTIONS[field].map(v =>
    `<option value="${v}"${v === selected ? ' selected' : ''}>${v.replace(/_/g,' ')}</option>`
  ).join('');
}

async function showProfile() {
  const modal = $('#modal-profile');
  const body  = $('#profile-body');
  body.innerHTML = '<p>Loading...</p>';
  modal.classList.remove('hidden');

  const [profileRes, targetRes] = await Promise.all([api.getProfile(), api.getCalorieTarget()]);
  if (!profileRes.ok) { body.innerHTML = '<p style="color:var(--error)">Failed to load profile</p>'; return; }

  renderProfileForm(profileRes.data, targetRes.ok ? targetRes.data : {});
}

function renderProfileForm(p, t) {
  const body          = $('#profile-body');
  const computedTarget = t?.tdee ? Math.round(t.tdee + (
    p.fitness_goal === 'lose_weight' ? -500 :
    p.fitness_goal === 'gain_muscle' ?  300 :
    p.fitness_goal === 'gain_weight' ?  500 : 0
  )) : 2000;
  const customGoalVal  = (p.custom_calorie_goal != null && p.custom_calorie_goal > 0) ? p.custom_calorie_goal : '';
  const isCustomActive = customGoalVal !== '';

  body.innerHTML = `
    <form id="profile-form" class="profile-grid">
      <div class="profile-item"><label>Name</label>
        <input type="text" name="name" value="${escapeAttr(p.name||'')}"></div>
      <div class="profile-item"><label>Age</label>
        <input type="number" name="age" min="10" max="120" step="1" value="${p.age??''}"></div>
      <div class="profile-item"><label>Gender</label>
        <select name="gender">${_optionsHtml('gender',p.gender)}</select></div>
      <div class="profile-item"><label>Height (cm)</label>
        <input type="number" name="height_cm" min="50" max="300" step="0.1" value="${p.height_cm??''}"></div>
      <div class="profile-item"><label>Weight (kg)</label>
        <input type="number" name="weight_kg" min="20" max="500" step="0.1" value="${p.weight_kg??''}"></div>
      <div class="profile-item"><label>Activity</label>
        <select name="activity_level">${_optionsHtml('activity_level',p.activity_level)}</select></div>
      <div class="profile-item"><label>Goal</label>
        <select name="fitness_goal">${_optionsHtml('fitness_goal',p.fitness_goal)}</select></div>
      <div class="profile-item"><label>Diet</label>
        <select name="diet_type">${_optionsHtml('diet_type',p.diet_type)}</select></div>

      <div class="profile-item" style="grid-column:1/-1;border-top:1px solid var(--border);padding-top:0.75rem;margin-top:0.25rem">
        <label>Daily Calorie Goal (kcal)</label>
        <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;margin-top:0.35rem">
          <input type="number" name="custom_calorie_goal" id="custom-calorie-input"
            min="500" max="10000" step="50"
            value="${escapeAttr(String(customGoalVal))}"
            placeholder="${computedTarget} (auto-computed)"
            style="flex:1;min-width:140px;padding:0.4rem 0.5rem;border:1px solid var(--border);border-radius:6px;font-size:0.95rem">
          <span style="font-size:0.82rem;color:var(--text-muted)">Leave blank to use auto (${computedTarget} kcal)</span>
          ${isCustomActive ? `<button type="button" id="clear-custom-goal"
            style="font-size:0.8rem;padding:0.25rem 0.6rem;border:1px solid var(--border);border-radius:5px;background:none;cursor:pointer;color:var(--text-muted)">
            ✕ Use auto</button>` : ''}
        </div>
        ${isCustomActive ? `<div style="margin-top:0.3rem;font-size:0.82rem;color:var(--success)">
          ✅ Custom goal active: <strong>${customGoalVal} kcal</strong></div>` : ''}
      </div>
    </form>

    <div id="profile-targets">
      ${t?.bmr ? `<div style="margin-top:1rem">
        <h4 style="margin-bottom:0.5rem">Calorie Targets</h4>
        <div class="summary-row"><span>BMR</span><span>${t.bmr.toFixed(0)} kcal</span></div>
        <div class="summary-row"><span>TDEE (auto-computed)</span><span>${t.tdee.toFixed(0)} kcal</span></div>
        <div class="summary-row highlight"><span><strong>Daily Target ${isCustomActive ? '(custom)' : '(auto)'}</strong></span>
          <span><strong>${t.calorie_target.toFixed(0)} kcal</strong></span></div>
      </div>` : ''}
    </div>
    <div style="margin-top:1rem;display:flex;gap:0.5rem;align-items:center">
      <button id="profile-save-btn" class="btn-primary" type="button">Save Profile</button>
      <span id="profile-save-status" style="font-size:0.85rem"></span>
    </div>`;

  $('#profile-save-btn').addEventListener('click', saveProfile);
  const clearBtn = $('#clear-custom-goal');
  if (clearBtn) clearBtn.addEventListener('click', () => {
    const inp = $('#custom-calorie-input');
    if (inp) inp.value = '';
  });
}

async function saveProfile() {
  const form   = $('#profile-form');
  const status = $('#profile-save-status');
  if (!form) return;

  const fd = new FormData(form);
  const payload = {
    name:           (fd.get('name')||'').trim(),
    age:            parseInt(fd.get('age'), 10),
    gender:         fd.get('gender'),
    height_cm:      parseFloat(fd.get('height_cm')),
    weight_kg:      parseFloat(fd.get('weight_kg')),
    activity_level: fd.get('activity_level'),
    fitness_goal:   fd.get('fitness_goal'),
    diet_type:      fd.get('diet_type'),
  };

  // Custom calorie goal — now inside the form so FormData captures it correctly.
  // Empty field = "no change" when a custom goal is already saved (user didn't
  // touch the field). Only sends null (clears) when the field was explicitly
  // blanked via the "Use auto" button, which the user must click deliberately.
  const rawCustom = (fd.get('custom_calorie_goal') || '').trim();
  if (rawCustom === '') {
    // Field is blank. Check whether a custom goal was previously active:
    // if the input was pre-populated and the user cleared it manually, send null.
    // We use a data attribute set on the input to distinguish "pre-filled then
    // cleared" from "was already blank". Simpler: always send null when blank —
    // the user can re-enter if they change their mind. This is the documented
    // "clear" behaviour.
    payload.custom_calorie_goal = null;
  } else {
    const parsed = parseFloat(rawCustom);
    if (isNaN(parsed) || parsed < 500 || parsed > 10000) {
      status.textContent = 'Custom calorie goal must be 500–10 000 kcal';
      status.style.color = 'var(--error)';
      return;
    }
    payload.custom_calorie_goal = parsed;
  }

  if (!payload.name)  { status.textContent = 'Name is required'; status.style.color = 'var(--error)'; return; }
  if (!(payload.age >= 10 && payload.age <= 120)) { status.textContent = 'Age 10–120'; status.style.color = 'var(--error)'; return; }
  if (!(payload.height_cm >= 50 && payload.height_cm <= 300)) { status.textContent = 'Height 50–300 cm'; status.style.color = 'var(--error)'; return; }
  if (!(payload.weight_kg >= 20 && payload.weight_kg <= 500)) { status.textContent = 'Weight 20–500 kg'; status.style.color = 'var(--error)'; return; }

  status.style.color = 'var(--text-muted,#888)';
  status.textContent = 'Saving…';

  const res = await api.updateProfile(payload);
  if (!res.ok) {
    const detail = res.data?.detail;
    status.textContent = (typeof detail === 'string' ? detail : 'Save failed');
    status.style.color = 'var(--error)';
    return;
  }

  // Update session name immediately — no refresh needed
  if (res.data?.name) {
    const s = session.get();
    session.set({ ...s, name: res.data.name });
    updateUserBadge();
  }

  // Re-fetch and re-render so targets reflect the saved values instantly
  const [profileRes, targetRes] = await Promise.all([api.getProfile(), api.getCalorieTarget()]);
  if (profileRes.ok) renderProfileForm(profileRes.data, targetRes.ok ? targetRes.data : {});
  const st = $('#profile-save-status');
  if (st) { st.textContent = '✓ Saved — targets updated'; st.style.color = 'var(--success,#2e7d32)'; }
  showToast('success', '✓ Profile saved');
}

// ============================================================
// UTILITIES
// ============================================================

function scrollToBottom() { const cm = chatMessages(); if (cm) cm.scrollTop = cm.scrollHeight; }
function timeNow() { return new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' }); }

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML.replace(/\n/g,'<br>');
}

function formatBotMarkdown(text) {
  if (!text) return '';
  let e = escapeHtml(text);
  return e.replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>');
}

function escapeAttr(text) {
  return (text||'').replace(/'/g,"\\'").replace(/"/g,'&quot;');
}

function showToast(type, msg) {
  document.querySelector('.toast')?.remove();
  const div = document.createElement('div');
  div.className = `toast toast-${type}`;
  div.innerText = msg || '✓ Done';
  document.body.appendChild(div);
  setTimeout(() => { div.remove(); }, 3500);
}
