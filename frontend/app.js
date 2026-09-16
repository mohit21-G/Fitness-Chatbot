/**
 * Fitness AI Chatbot — Main App (No Auth)
 */

const $ = (sel) => document.querySelector(sel);
const chatMessages = $('#chat-messages');
const chatInput = $('#chat-input');
const btnSend = $('#btn-send');
const btnVoice = $('#btn-voice');
const voiceIndicator = $('#voice-indicator');

let pendingAction = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// ============================================================
// INIT
// ============================================================

function init() {
  bindEvents();
  loadGreeting();
}

function bindEvents() {
  btnSend.addEventListener('click', handleSendText);
  chatInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendText(); } });

  // Voice - Toggle on click (click to start, click again to stop)
  btnVoice.addEventListener('click', toggleRecording);

  // Header buttons
  $('#btn-summary').addEventListener('click', showSummary);
  $('#btn-profile').addEventListener('click', showProfile);

  // Modal close
  $('#close-summary').addEventListener('click', () => $('#modal-summary').classList.add('hidden'));
  $('#close-profile').addEventListener('click', () => $('#modal-profile').classList.add('hidden'));
}

// ============================================================
// CHAT
// ============================================================

async function loadGreeting() {
  showTyping();
  const res = await api.sendMessage('hello');
  hideTyping();
  if (res.ok) {
    addBotMessage(res.data);
  } else {
    // Backend may be cold-starting on Render free tier — show a helpful message
    // instead of a generic error so the user knows to try again in a moment.
    addBotError(
      res.status === 0
        ? 'Backend is starting up — please send a message in a few seconds.'
        : (res.data.detail || 'Something went wrong')
    );
  }
}

async function handleSendText() {
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = '';
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
        : (res.data.detail || 'Something went wrong')
    );
  }
}

// ============================================================
// VOICE — Click to Start / Click to Stop Toggle
// ============================================================

function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

async function startRecording() {
  if (isRecording) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    let mimeType = 'audio/webm';
    if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
      mimeType = 'audio/ogg;codecs=opus';
    } else if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
      mimeType = 'audio/webm;codecs=opus';
    }
    mediaRecorder = new MediaRecorder(stream, { mimeType });
    audioChunks = [];
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };
    mediaRecorder.onstop = sendVoiceMessage;
    mediaRecorder.start(250); // collect chunk every 250ms
    isRecording = true;
    btnVoice.classList.add('recording');
    voiceIndicator.classList.remove('hidden');
  } catch (err) {
    addBotError('Microphone access denied. Please allow microphone.');
  }
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  isRecording = false;
  btnVoice.classList.remove('recording');
  voiceIndicator.classList.add('hidden');
  // Short delay to collect final chunk before stopping
  setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(t => t.stop());
    }
  }, 300);
}

async function sendVoiceMessage() {
  if (audioChunks.length === 0) {
    addBotError('No audio captured. Please try again.');
    return;
  }
  const mimeType = mediaRecorder ? (mediaRecorder.mimeType || 'audio/webm') : 'audio/webm';
  const blob = new Blob(audioChunks, { type: mimeType });
  console.log('Audio blob size:', blob.size, 'type:', mimeType);

  if (blob.size < 100) {
    addBotError('No audio captured. Please hold and speak.');
    return;
  }

  // Determine file extension from mime type
  let ext = 'webm';
  if (mimeType.includes('ogg')) ext = 'ogg';
  else if (mimeType.includes('mp4')) ext = 'mp4';

  addUserMessage('[Voice message]');
  showTyping();

  const res = await api.sendVoice(blob, ext);
  hideTyping();

  if (res.ok) {
    if (res.data.data && res.data.data.transcribed_text) {
      const lang = res.data.data.stt_language || '';
      const conf = res.data.data.stt_confidence;
      const confLabel = conf ? ` ${(conf * 100).toFixed(0)}%` : '';
      const langLabel = lang ? ` [${lang.toUpperCase()}${confLabel}]` : '';
      const normalised = res.data.data.normalised_text || '';
      let heardMsg = `🎤 Heard${langLabel}: "${res.data.data.transcribed_text}"`;
      if (normalised && normalised !== res.data.data.transcribed_text) {
        heardMsg += `\n→ Understood: "${normalised}"`;
      }
      addSystemMessage(heardMsg);
    }
    addBotMessage(res.data);
  } else {
    addBotError(res.data.detail || res.data.message || 'Voice processing failed');
  }
}

// ============================================================
// MESSAGE RENDERING
// ============================================================

function addUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg msg-user';
  div.innerHTML = `<div class="msg-text">${escapeHtml(text)}</div><div class="msg-time">${timeNow()}</div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function addBotMessage(data) {
  const div = document.createElement('div');
  div.className = 'msg msg-bot';

  let html = `<div class="msg-text">${formatBotMarkdown(data.message)}</div>`;

  // Nutrition card
  if (data.data && data.data.nutrition && data.data.nutrition.calories) {
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

  // Confirmation buttons
  if (data.needs_confirmation && data.pending_action) {
    pendingAction = data.pending_action;
    html += `<div class="confirm-card">
      <div class="confirm-btns">
        <button class="btn-yes" onclick="handleConfirm(true)">Haa, Save karo</button>
        <button class="btn-no" onclick="handleConfirm(false)">Na, Cancel</button>
      </div>
    </div>`;
  }

  // Option buttons (variant, quantity, meal type choices)
  if (data.options && data.options.length > 0) {
    html += `<div class="options-row">`;
    for (const opt of data.options) {
      html += `<button class="option-btn" onclick="handleOptionClick(this, '${escapeAttr(opt)}')">${escapeHtml(opt)}</button>`;
    }
    html += `</div>`;
  }

  html += `<div class="msg-time">${timeNow()}</div>`;
  div.innerHTML = html;
  chatMessages.appendChild(div);
  scrollToBottom();

  // Show toast for successful logs
  if (data.action_taken === 'food_logged' || data.action_taken === 'exercise_logged' || (data.data && (data.data.show_toast || data.data.action_taken === 'food_logged'))) {
    const toastMsg = (data.data && data.data.toast_message) ? data.data.toast_message : '✓ Log saved successfully';
    showToast('success', toastMsg);
  }
}

function addBotError(msg) {
  const div = document.createElement('div');
  div.className = 'msg msg-bot';
  div.innerHTML = `<div class="msg-text" style="color:var(--error)">${escapeHtml(msg)}</div><div class="msg-time">${timeNow()}</div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function addSystemMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg msg-bot';
  div.style.opacity = '0.7';
  div.style.fontSize = '0.82rem';
  div.innerHTML = `<div class="msg-text">${escapeHtml(text)}</div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'msg msg-typing';
  div.id = 'typing-indicator';
  div.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
  chatMessages.appendChild(div);
  scrollToBottom();
}

function hideTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
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
  if (res.ok) {
    addBotMessage(res.data);
  } else {
    addBotError('Failed to process confirmation');
    showToast('error', 'Failed to save log. Please try again.');
  }
}
window.handleConfirm = handleConfirm;

// Handle option button click — sends the option text as a message
async function handleOptionClick(btn, optionText) {
  // Mark selected, disable all in this group
  const row = btn.closest('.options-row');
  if (row) {
    row.querySelectorAll('.option-btn').forEach(b => { b.disabled = true; b.classList.remove('selected'); });
    btn.classList.add('selected');
    btn.disabled = true;
  }
  // Extract just the keyword (before parenthesis if any)
  let text = optionText.split('(')[0].trim();
  addUserMessage(text);
  showTyping();
  const res = await api.sendMessage(text, false);
  hideTyping();
  if (res.ok) addBotMessage(res.data);
  else addBotError(
    res.status === 0
      ? 'Could not reach the server — please try again in a moment.'
      : (res.data.detail || 'Something went wrong')
  );
}
window.handleOptionClick = handleOptionClick;

// Toast notification
function showToast(type, message) {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = (type === 'success' ? '\u2705 ' : '\u274C ') + message;
  document.body.appendChild(toast);
  setTimeout(() => { toast.remove(); }, 3000);
}

// ============================================================
// SUMMARY MODAL
// ============================================================

async function showSummary() {
  const modal = $('#modal-summary');
  const body = $('#summary-body');
  body.innerHTML = '<p>Loading...</p>';
  modal.classList.remove('hidden');

  const res = await api.getDailySummary();
  if (!res.ok) { body.innerHTML = '<p style="color:var(--error)">Failed to load summary</p>'; return; }

  const d = res.data;
  const pct = d.calorie_target > 0 ? Math.min((d.net_calories / d.calorie_target) * 100, 100) : 0;
  const isOver = d.net_calories > d.calorie_target;

  body.innerHTML = `
    <div class="summary-card">
      <h4>Today's Progress</h4>
      <div class="progress-bar"><div class="progress-fill ${isOver ? 'over' : ''}" style="width:${Math.min(pct, 100)}%"></div></div>
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

// Canonical chronological order for meal sections in the Daily Log.
// Matches the backend ordering: Breakfast → Morning → Lunch → Afternoon →
// Snack → Evening → Dinner. Any meal type not listed is appended after these
// (kept, never dropped) so nothing is lost.
const MEAL_DISPLAY_ORDER = ['breakfast', 'morning', 'lunch', 'afternoon', 'snack', 'evening', 'dinner'];
const MEAL_LABELS = {
  breakfast: '🍳 Breakfast', morning: '🌞 Morning', lunch: '🍱 Lunch', afternoon: '🍱 Afternoon',
  snack: '☕ Snack', evening: '🌇 Evening', dinner: '🍽️ Dinner',
};

function _mealLabel(meal) {
  return MEAL_LABELS[meal] || (meal ? `${getMealEmoji(meal)} ${meal.charAt(0).toUpperCase() + meal.slice(1)}` : '🍽️ Other');
}

function getMealEmoji(meal) {
  const m = (meal || '').toLowerCase().trim();
  if (m.includes('break') || m.includes('nashta') || m.includes('nasta') || m.includes('નાસ્તો') || m.includes('नाश्ता')) return '🍳';
  if (m.includes('morning') || m.includes('savar') || m.includes('subah') || m.includes('સવાર') || m.includes('सुबह')) return '🌞';
  if (m.includes('lunch') || m.includes('bapor') || m.includes('dopahar') || m.includes('બપોર') || m.includes('दोपहर')) return '🍱';
  if (m.includes('evening') || m.includes('sanj') || m.includes('shaam') || m.includes('sham') || m.includes('સાંજ') || m.includes('શામ') || m.includes('शाम')) return '🌇';
  if (m.includes('dinner') || m.includes('raat') || m.includes('ratre') || m.includes('supper') || m.includes('રાત') || m.includes('रात')) return '🍽️';
  if (m.includes('snack') || m.includes('tea') || m.includes('chai') || m.includes('coffee') || m.includes('nasto') || m.includes('ચા') || m.includes('चाय')) return '☕';
  return '🍽️';
}

function getExerciseEmoji(exercise) {
  const text = (exercise || '').toLowerCase();
  if (/push[- ]?ups?|bench press|chest|dip/i.test(text)) return '💪';
  if (/run|running|jog|jogging|sprint|treadmill|dodvu|dodhvu|dodyo|daud|દોડવું|दौड़/i.test(text)) return '🏃';
  if (/cycl|cycling|bike|biking|bicycle|સાયકલ|साइकिल/i.test(text)) return '🚴';
  if (/yoga|stretch|pilates|surya|pranayam|યોગ|योग|सूर्य/i.test(text)) return '🧘';
  if (/swim|swimming|taryo|tarvu|tarna|તરવું|तैरना/i.test(text)) return '🏊';
  if (/cricket|football|soccer|badminton|tennis|basketball|volleyball/i.test(text)) return '⚽';
  if (/walk|walking|brisk|stroll|hike|hiking|chalvu|chalyo|chalna|tehelna|sair|ચાલવું|घूमना|सैर/i.test(text)) return '🚶';
  if (/gym|workout|weight|lifting|squat|deadlift|dumbbell|barbell|pull[- ]?ups?|strength|kasrat|જિમ|जिम|કસરત/i.test(text)) return '🏋️';
  return '🏋️';
}

const FOOD_EMOJI_RULES = [
  { emoji: '🥪', keywords: ['sandwich', 'sandwitch', 'toast sandwich'] },
  { emoji: '🍜', keywords: ['noodles', 'noodle', 'noodels', 'nudles', 'maggi', 'pasta', 'spaghetti', 'macaroni', 'chowmein', 'ramen'] },
  { emoji: '🌯', keywords: ['roll', 'frankie', 'wrap', 'burrito', 'taco', 'shawarma'] },
  { emoji: '🍕', keywords: ['pizza', 'pizzza', 'calzone'] },
  { emoji: '🍔', keywords: ['burger', 'burgur', 'hamburger', 'cheeseburger', 'slider'] },
  { emoji: '🍟', keywords: ['fries', 'french fries'] },
  { emoji: '☕', keywords: ['chai', 'chaye', 'tea', 'green tea', 'black tea', 'milk tea', 'masala tea', 'coffee', 'coffe', 'filter coffee', 'espresso', 'cappuccino', 'latte', 'cold coffee'] },
  { emoji: '🥛', keywords: ['milk', 'doodh', 'lassi', 'lasssi', 'chaas', 'chhas', 'buttermilk', 'smoothie', 'shake', 'milkshake', 'protein shake', 'whey'] },
  { emoji: '🥤', keywords: ['juice', 'water', 'soda', 'drink', 'beverage', 'lemonade', 'nimbu pani', 'coconut water', 'nariyal pani', 'cola', 'pepsi', 'coke'] },
  { emoji: '🫓', keywords: ['roti', 'rotli', 'chapati', 'chapatti', 'phulka', 'naan', 'paratha', 'parotta', 'thepla', 'thepala', 'bhakri', 'bhakhari', 'poori', 'puri', 'bhatura', 'kulcha'] },
  { emoji: '🍞', keywords: ['bread', 'toast', 'pav', 'paav', 'bun', 'bagel', 'pita', 'croissant', 'tortilla'] },
  { emoji: '🍦', keywords: ['ice cream', 'icecream', 'gelato', 'kulfi', 'sundae', 'frozen yogurt', 'sorbet', 'popsicle'] },
  { emoji: '🥚', keywords: ['egg', 'eggs', 'anda', 'omelette', 'omlet', 'boiled egg', 'scrambled egg', 'bhurji', 'egg white'] },
  { emoji: '🍗', keywords: ['chicken', 'turkey', 'wings', 'drumstick', 'chicken breast', 'tandoori chicken', 'chicken tikka'] },
  { emoji: '🥩', keywords: ['mutton', 'lamb', 'beef', 'pork', 'keema', 'kebab', 'tikka', 'bacon', 'sausage', 'ham', 'steak', 'meat', 'ghost'] },
  { emoji: '🐟', keywords: ['fish', 'salmon', 'tuna', 'pomfret', 'rohu', 'surmai', 'bangda'] },
  { emoji: '🦐', keywords: ['prawn', 'prawns', 'shrimp', 'crab', 'lobster', 'seafood'] },
  { emoji: '🍚', keywords: ['rice', 'biryani', 'pulao', 'pulav', 'khichdi', 'khichri', 'keechdi', 'fried rice', 'jeera rice', 'curd rice', 'lemon rice', 'chawal', 'bhat', 'poha'] },
  { emoji: '🥣', keywords: ['upma', 'upmaa', 'uppma', 'poha', 'oats', 'oatmeal', 'quinoa', 'daliya', 'porridge', 'soup', 'stew', 'broth', 'shorba', 'cornflakes', 'muesli'] },
  { emoji: '🍲', keywords: ['dal', 'daal', 'sambar', 'sambhar', 'rasam', 'rajma', 'chhole', 'chole', 'chana', 'kadhi', 'curry', 'gravy'] },
  { emoji: '🧀', keywords: ['paneer', 'cheese', 'curd', 'dahi', 'yogurt', 'yoghurt', 'butter', 'ghee', 'makhan', 'cream', 'mayo', 'mayonnaise', 'tofu'] },
  { emoji: '🍎', keywords: ['apple', 'banana', 'mango', 'orange', 'berry', 'berries', 'strawberry', 'blueberry', 'watermelon', 'melon', 'papaya', 'guava', 'grapes', 'grape', 'pineapple', 'pomegranate', 'kiwi', 'pear', 'peach', 'plum', 'cherry', 'fruit', 'fruits', 'chiku', 'chikoo', 'avocado', 'lemon', 'lime', 'fig', 'dates', 'kela', 'safarchand', 'keri'] },
  { emoji: '🥗', keywords: ['salad', 'vegetable', 'vegetables', 'veggie', 'sabzi', 'subzi', 'sabji', 'bhindi', 'okra', 'aloo', 'potato', 'bataka', 'gobi', 'cauliflower', 'cabbage', 'palak', 'spinach', 'methi', 'tomato', 'cucumber', 'carrot', 'broccoli', 'capsicum', 'bell pepper', 'mushroom', 'peas', 'matar', 'corn', 'onion', 'brinjal', 'eggplant', 'lauki', 'doodhi', 'lettuce', 'sprouts'] },
  { emoji: '🥟', keywords: ['samosa', 'kachori', 'pakora', 'pakoda', 'bhajia', 'dosa', 'idli', 'vada', 'medu vada', 'uttapam', 'chaat', 'pani puri', 'sev puri', 'bhel', 'bhelpuri', 'momo', 'momos', 'dumpling', 'khaman', 'dhokla', 'handvo'] },
  { emoji: '🍬', keywords: ['gulab jamun', 'jalebi', 'barfi', 'burfi', 'halwa', 'laddu', 'ladoo', 'kheer', 'rasgulla', 'rasmalai', 'sandesh', 'peda', 'cake', 'pastry', 'chocolate', 'cookie', 'biscuit', 'dessert', 'mithai', 'brownie', 'pudding', 'sweet', 'donut', 'doughnut', 'waffle', 'pancake', 'sheera', 'mitha', 'chikki'] },
  { emoji: '🥜', keywords: ['almond', 'almonds', 'badam', 'walnut', 'walnuts', 'akhrot', 'cashew', 'cashews', 'kaju', 'peanut', 'peanuts', 'mungfali', 'sing', 'pista', 'pistachio', 'chia', 'flax', 'seed', 'seeds', 'dry fruit', 'dry fruits', 'raisin', 'raisins', 'kishmish'] },
];

function getFoodEmoji(foodName) {
  const text = (foodName || '').toLowerCase();
  if (!text) return '🍽️';
  for (const rule of FOOD_EMOJI_RULES) {
    for (const kw of rule.keywords) {
      const regex = new RegExp(`\\b${kw}\\b`, 'i');
      if (regex.test(text)) return rule.emoji;
    }
  }
  return '🍽️';
}

// Render the Daily Log grouped by meal. Shows ONLY meals that have logged foods,
// in chronological order, listing every food under its meal. The backend already
// returns meal_breakdown chronologically ordered; we additionally enforce the
// canonical order client-side so display order never depends on object key order.
function renderDailyLog(mealBreakdown) {
  if (!mealBreakdown || typeof mealBreakdown !== 'object') return '';

  // Order meals by the canonical rank; unknown meals keep their given order last.
  const meals = Object.keys(mealBreakdown);
  meals.sort((a, b) => {
    const ra = MEAL_DISPLAY_ORDER.indexOf(a);
    const rb = MEAL_DISPLAY_ORDER.indexOf(b);
    return (ra === -1 ? MEAL_DISPLAY_ORDER.length : ra) - (rb === -1 ? MEAL_DISPLAY_ORDER.length : rb);
  });

  const sections = [];
  for (const meal of meals) {
    const entry = mealBreakdown[meal] || {};
    const itemDetails = Array.isArray(entry.item_details) ? entry.item_details : null;
    const items = Array.isArray(entry.items) ? entry.items : [];
    // Skip meals with no logged foods — no empty sections.
    if (items.length === 0 && (!itemDetails || itemDetails.length === 0)) continue;

    const cals = typeof entry.calories === 'number' ? entry.calories : 0;
    let itemsHtml = '';
    if (itemDetails && itemDetails.length > 0) {
      itemsHtml = itemDetails.map(item => {
        const emoji = getFoodEmoji(item.food_name || item.category);
        const name = escapeHtml(item.food_name || '?');
        const qty = item.quantity ? ` — <strong>${escapeHtml(item.quantity)}</strong>` : '';
        const cal = typeof item.calories === 'number' ? ` — <strong>${item.calories.toFixed(0)} kcal</strong>` : '';
        return `<div class="log-item">• ${emoji} <strong>${name}</strong>${qty}${cal}</div>`;
      }).join('');
    } else {
      itemsHtml = items
        .map(name => {
          const emoji = getFoodEmoji(name);
          return `<div class="log-item">• ${emoji} <strong>${escapeHtml(name)}</strong></div>`;
        })
        .join('');
    }

    sections.push(`
      <div class="meal-section">
        <div class="summary-row"><span><strong>${escapeHtml(_mealLabel(meal))}</strong></span><span><strong>${cals.toFixed(0)} kcal</strong></span></div>
        ${itemsHtml}
      </div>`);
  }

  if (sections.length === 0) return '';  // nothing logged → no Daily Log block

  return `
    <div style="margin-top:1rem">
      <h4 style="margin-bottom:0.5rem">Daily Log</h4>
      ${sections.join('')}
    </div>`;
}

// ============================================================
// PROFILE MODAL
// ============================================================

// Allowed option values (mirror backend enums in schemas.py).
const PROFILE_OPTIONS = {
  gender: ['male', 'female'],
  activity_level: ['sedentary', 'light', 'moderate', 'active', 'very_active'],
  fitness_goal: ['lose_weight', 'maintain', 'gain_muscle', 'gain_weight'],
  diet_type: ['veg', 'non_veg', 'eggetarian', 'vegan'],
};

function _optionsHtml(field, selected) {
  return PROFILE_OPTIONS[field].map(v =>
    `<option value="${v}"${v === selected ? ' selected' : ''}>${v.replace(/_/g, ' ')}</option>`
  ).join('');
}

async function showProfile() {
  const modal = $('#modal-profile');
  const body = $('#profile-body');
  body.innerHTML = '<p>Loading...</p>';
  modal.classList.remove('hidden');

  const [profileRes, targetRes] = await Promise.all([api.getProfile(), api.getCalorieTarget()]);
  if (!profileRes.ok) { body.innerHTML = '<p style="color:var(--error)">Failed to load profile</p>'; return; }

  renderProfileForm(profileRes.data, targetRes.ok ? targetRes.data : {});
}

// Render the editable profile form. The user enters their ACTUAL Age, Weight,
// Height, Gender, Activity and Goal; saving persists them and recomputes
// BMR/TDEE/targets from the saved values.
function renderProfileForm(p, t) {
  const body = $('#profile-body');
  // Show the computed TDEE target as the placeholder suggestion for the custom goal
  const computedTarget = t && t.tdee ? Math.round(t.tdee + (
    p.fitness_goal === 'lose_weight' ? -500 :
    p.fitness_goal === 'gain_muscle' ? 300 :
    p.fitness_goal === 'gain_weight' ? 500 : 0
  )) : 2000;
  const customGoalVal = (p.custom_calorie_goal != null && p.custom_calorie_goal > 0)
    ? p.custom_calorie_goal : '';
  const isCustomActive = customGoalVal !== '';

  body.innerHTML = `
    <form id="profile-form" class="profile-grid">
      <div class="profile-item"><label>Name</label>
        <input type="text" name="name" value="${escapeAttr(p.name || '')}" required></div>
      <div class="profile-item"><label>Age</label>
        <input type="number" name="age" min="10" max="120" step="1" value="${p.age ?? ''}" required></div>
      <div class="profile-item"><label>Gender</label>
        <select name="gender">${_optionsHtml('gender', p.gender)}</select></div>
      <div class="profile-item"><label>Height (cm)</label>
        <input type="number" name="height_cm" min="50" max="300" step="0.1" value="${p.height_cm ?? ''}" required></div>
      <div class="profile-item"><label>Weight (kg)</label>
        <input type="number" name="weight_kg" min="20" max="500" step="0.1" value="${p.weight_kg ?? ''}" required></div>
      <div class="profile-item"><label>Activity</label>
        <select name="activity_level">${_optionsHtml('activity_level', p.activity_level)}</select></div>
      <div class="profile-item"><label>Goal</label>
        <select name="fitness_goal">${_optionsHtml('fitness_goal', p.fitness_goal)}</select></div>
      <div class="profile-item"><label>Diet</label>
        <select name="diet_type">${_optionsHtml('diet_type', p.diet_type)}</select></div>
    </form>

    <div style="margin-top:1rem;border-top:1px solid var(--border,#e0e0e0);padding-top:1rem">
      <h4 style="margin-bottom:0.4rem">Daily Calorie Goal</h4>
      <p style="font-size:0.83rem;color:var(--text-muted,#666);margin:0 0 0.6rem">
        Leave blank to use your auto-computed target (${computedTarget} kcal based on your profile).
        Enter your own value to override it everywhere.
      </p>
      <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap">
        <input id="custom-calorie-input" type="number" name="custom_calorie_goal"
          min="500" max="10000" step="50"
          value="${escapeAttr(String(customGoalVal))}"
          placeholder="e.g. 2000"
          style="width:120px;padding:0.35rem 0.5rem;border:1px solid var(--border,#ccc);border-radius:6px;font-size:0.95rem">
        <span style="font-size:0.9rem;color:var(--text-muted,#666)">kcal / day</span>
        ${isCustomActive ? `<button type="button" id="clear-custom-goal" style="font-size:0.8rem;padding:0.25rem 0.6rem;border:1px solid var(--border,#ccc);border-radius:5px;background:none;cursor:pointer;color:var(--text-muted,#666)">✕ Clear (use auto)</button>` : ''}
      </div>
      ${isCustomActive ? `<div style="margin-top:0.4rem;font-size:0.82rem;color:var(--success,#2e7d32)">✅ Custom goal active: <strong>${customGoalVal} kcal</strong></div>` : ''}
    </div>

    <div id="profile-targets">
      ${t && t.bmr ? `
      <div style="margin-top:1rem">
        <h4 style="margin-bottom:0.5rem">Calorie Targets</h4>
        <div class="summary-row"><span>BMR</span><span>${t.bmr.toFixed(0)} kcal</span></div>
        <div class="summary-row"><span>TDEE</span><span>${t.tdee.toFixed(0)} kcal</span></div>
        <div class="summary-row highlight"><span>Daily Target ${isCustomActive ? '(custom)' : '(auto)'}</span><span>${t.calorie_target.toFixed(0)} kcal</span></div>
      </div>` : ''}
    </div>
    <div style="margin-top:1rem;display:flex;gap:0.5rem;align-items:center">
      <button id="profile-save-btn" class="btn-primary" type="button">Save Profile</button>
      <span id="profile-save-status" style="font-size:0.85rem"></span>
    </div>`;

  const saveBtn = $('#profile-save-btn');
  if (saveBtn) saveBtn.addEventListener('click', saveProfile);

  // Clear custom goal button
  const clearBtn = $('#clear-custom-goal');
  if (clearBtn) clearBtn.addEventListener('click', () => {
    const inp = $('#custom-calorie-input');
    if (inp) inp.value = '';
  });
}

async function saveProfile() {
  const form = $('#profile-form');
  const status = $('#profile-save-status');
  if (!form) return;

  const fd = new FormData(form);
  const payload = {
    name: (fd.get('name') || '').trim(),
    age: parseInt(fd.get('age'), 10),
    gender: fd.get('gender'),
    height_cm: parseFloat(fd.get('height_cm')),
    weight_kg: parseFloat(fd.get('weight_kg')),
    activity_level: fd.get('activity_level'),
    fitness_goal: fd.get('fitness_goal'),
    diet_type: fd.get('diet_type'),
  };

  // Custom calorie goal — include explicitly so the server can clear it (null) or set it
  const rawCustom = (fd.get('custom_calorie_goal') || '').trim();
  if (rawCustom === '') {
    payload.custom_calorie_goal = null;   // explicitly clear any saved custom goal
  } else {
    const parsed = parseFloat(rawCustom);
    if (isNaN(parsed) || parsed < 500 || parsed > 10000) {
      status.textContent = 'Custom calorie goal must be between 500 and 10 000 kcal';
      status.style.color = 'var(--error)';
      return;
    }
    payload.custom_calorie_goal = parsed;
  }

  // Basic client-side validation against backend bounds.
  if (!payload.name) { status.textContent = 'Name is required'; status.style.color = 'var(--error)'; return; }
  if (!(payload.age >= 10 && payload.age <= 120)) { status.textContent = 'Age must be 10–120'; status.style.color = 'var(--error)'; return; }
  if (!(payload.height_cm >= 50 && payload.height_cm <= 300)) { status.textContent = 'Height must be 50–300 cm'; status.style.color = 'var(--error)'; return; }
  if (!(payload.weight_kg >= 20 && payload.weight_kg <= 500)) { status.textContent = 'Weight must be 20–500 kg'; status.style.color = 'var(--error)'; return; }

  status.style.color = 'var(--text-muted, #888)';
  status.textContent = 'Saving…';

  const res = await api.updateProfile(payload);
  if (!res.ok) {
    let detail = 'Save failed';
    if (res.data && res.data.detail) detail = typeof res.data.detail === 'string' ? res.data.detail : 'Invalid values';
    status.textContent = detail; status.style.color = 'var(--error)';
    return;
  }

  // Re-fetch the saved profile + freshly recomputed targets and re-render.
  const [profileRes, targetRes] = await Promise.all([api.getProfile(), api.getCalorieTarget()]);
  if (profileRes.ok) renderProfileForm(profileRes.data, targetRes.ok ? targetRes.data : {});
  const st = $('#profile-save-status');
  if (st) { st.textContent = '✓ Saved — targets updated'; st.style.color = 'var(--success, #2e7d32)'; }
  showToast('success', '✓ Profile saved');
}

// ============================================================
// UTILITIES
// ============================================================

function scrollToBottom() { chatMessages.scrollTop = chatMessages.scrollHeight; }
function timeNow() { return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML.replace(/\n/g, '<br>');
}
function formatBotMarkdown(text) {
  if (!text) return '';
  let escaped = escapeHtml(text);
  return escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}
function escapeAttr(text) {
  return (text || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function showToast(type, msg) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const div = document.createElement('div');
  div.className = `toast toast-${type}`;
  div.innerText = msg || '✓ Log saved successfully';
  document.body.appendChild(div);
  setTimeout(() => { div.remove(); }, 3500);
}

// ============================================================
// START
// ============================================================

document.addEventListener('DOMContentLoaded', init);
