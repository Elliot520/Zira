"use strict";

// Guards every top-level addEventListener call below: a missing element (e.g. a browser serving
// mismatched cached HTML/JS across a server restart) must never throw here - an uncaught error in
// a top-level <script> statement halts ALL remaining script execution, including unrelated wiring
// later in the file (this is a real bug that was found: the voice mic button's own listeners are
// registered near the bottom of this file, so a crash in an earlier, unrelated feature's setup
// code - e.g. a stale page missing the model-mode toggle - could silently break voice entirely).
function on(el, event, handler) {
  if (el) el.addEventListener(event, handler);
}

const STORAGE_KEY = "jarvis.conversation_id";
const HEALTH_INTERVAL_MS = 8000;
// Image/video generations: they keep running on the server when the connection drops.
const LONG_TOOLS = ["create_image", "edit_image", "create_video", "create_song", "create_ad"];
// What a long tool makes: its progress phrases and what the Stop button stops (STOP_URLS).
const GEN_KINDS = { create_video: "video", create_song: "song", create_ad: "ad" };
function genKind(tool) {
  return GEN_KINDS[tool] || "image";
}
const PENDING_POLL_MS = 3000;
const MODE_KEY = "jarvis.mode";
const VOICE_ENABLED_KEY = "jarvis.voice.enabled";
const MODE_LABELS = { chat: "Chat", plan: "Plan (read-only)", edit: "Edit (asks before changing)" };
const MODE_HINTS = {
  chat: "Chat mode. Zira can look at your project folders but changes nothing.",
  plan: "Plan mode: Zira investigates and proposes a plan. It changes nothing.",
  edit: "Edit mode: Zira may propose code changes. Nothing changes until you click Approve.",
};

const els = {
  messages: document.getElementById("messages"),
  empty: document.getElementById("empty"),
  form: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  newChat: document.getElementById("new-chat"),
  status: document.getElementById("status"),
  statusText: document.getElementById("status-text"),
  modelLabel: document.getElementById("model-label"),
  errorBanner: document.getElementById("error-banner"),
  errorText: document.getElementById("error-text"),
  errorClose: document.getElementById("error-close"),
  mode: document.getElementById("mode"),
  modeHint: document.getElementById("mode-hint"),
  voiceBar: document.getElementById("voice-bar"),
  voicePowerBtn: document.getElementById("voice-power-btn"),
  voiceBtn: document.getElementById("voice-btn"),
  voiceIcon: document.getElementById("voice-icon"),
  voiceStatus: document.getElementById("voice-status"),
  voiceTranscript: document.getElementById("voice-transcript"),
  voiceContinuous: document.getElementById("voice-continuous"),
  voiceLevel: document.getElementById("voice-level"),
  voiceLevelFill: document.getElementById("voice-level-fill"),
  voiceLevelThreshold: document.getElementById("voice-level-threshold"),
  musicPlayer: document.getElementById("music-player"),
  musicTitle: document.getElementById("music-title"),
  musicSource: document.getElementById("music-source"),
  musicPlay: document.getElementById("music-play"),
  musicPause: document.getElementById("music-pause"),
  musicStop: document.getElementById("music-stop"),
  modelModeToggle: document.getElementById("model-mode-toggle"),
  modelModeLight: document.getElementById("model-mode-light"),
  modelModeNewlight: document.getElementById("model-mode-newlight"),
  modelModeBalanced: document.getElementById("model-mode-balanced"),
  modelModeDeep: document.getElementById("model-mode-deep"),
  modelModeStatus: document.getElementById("model-mode-status"),
  imageUploadBtn: document.getElementById("image-upload-btn"),
  imageUploadInput: document.getElementById("image-upload-input"),
  imageAttached: document.getElementById("image-attached"),
  imageAttachedName: document.getElementById("image-attached-name"),
  imageAttachedRemove: document.getElementById("image-attached-remove"),
  imageStyleToggle: document.getElementById("image-style-toggle"),
  imageStyleNone: document.getElementById("image-style-none"),
  imageStyleRealistic: document.getElementById("image-style-realistic"),
  imageStyleLightning: document.getElementById("image-style-lightning"),
  imageStyleRealvis5: document.getElementById("image-style-realvis5"),
  imageResolutionToggle: document.getElementById("image-resolution-toggle"),
  imageResolution720: document.getElementById("image-resolution-720"),
  imageResolution1024: document.getElementById("image-resolution-1024"),
  videoModelToggle: document.getElementById("video-model-toggle"),
  ttsVoiceToggle: document.getElementById("tts-voice-toggle"),
  ttsVoiceQwen3: document.getElementById("tts-voice-qwen3"),
  ttsVoiceKokoro: document.getElementById("tts-voice-kokoro"),
  ttsVoiceIndicf5: document.getElementById("tts-voice-indicf5"),
  videoModelNone: document.getElementById("video-model-none"),
  videoModelLtx: document.getElementById("video-model-ltx"),
  videoModelFastmetal: document.getElementById("video-model-fastmetal"),
  videoModelHunyuan: document.getElementById("video-model-hunyuan"),
  themeToggle: document.getElementById("theme-toggle"),
  galleryBtn: document.getElementById("gallery-btn"),
  resetBtn: document.getElementById("reset-btn"),
  libraryBtn: document.getElementById("library-btn"),
  library: document.getElementById("library"),
  libraryTabs: document.getElementById("library-tabs"),
  libraryClose: document.getElementById("library-close"),
  libraryChats: document.getElementById("library-chats"),
  librarySearch: document.getElementById("library-search"),
  libraryChatList: document.getElementById("library-chat-list"),
  libraryChatsEmpty: document.getElementById("library-chats-empty"),
  libraryMore: document.getElementById("library-more"),
  libraryMemory: document.getElementById("library-memory"),
  libraryMemoryList: document.getElementById("library-memory-list"),
  libraryLearned: document.getElementById("library-learned"),
  libraryLearnedList: document.getElementById("library-learned-list"),
  libraryLearnedEmpty: document.getElementById("library-learned-empty"),
  learnedStudy: document.getElementById("learned-study"),
  learnedStatus: document.getElementById("learned-status"),
  libraryMemoryEmpty: document.getElementById("library-memory-empty"),
  memoryAdd: document.getElementById("memory-add"),
  notifyControl: document.getElementById("notify-control"),
  notifyBtn: document.getElementById("notify-btn"),
  notifyStatus: document.getElementById("notify-status"),
  loraBtn: document.getElementById("lora-btn"),
  gallery: document.getElementById("gallery"),
  galleryBack: document.getElementById("gallery-back"),
  galleryTabs: document.getElementById("gallery-tabs"),
  galleryClose: document.getElementById("gallery-close"),
  galleryGrid: document.getElementById("gallery-grid"),
  galleryEmpty: document.getElementById("gallery-empty"),
  galleryMore: document.getElementById("gallery-more"),
  galleryDetail: document.getElementById("gallery-detail"),
  galleryMedia: document.getElementById("gallery-media"),
  galleryInfo: document.getElementById("gallery-info"),
  galleryAgain: document.getElementById("gallery-again"),
  galleryEditRequest: document.getElementById("gallery-edit-request"),
  galleryEditImage: document.getElementById("gallery-edit-image"),
  galleryAnimate: document.getElementById("gallery-animate"),
  galleryLonger: document.getElementById("gallery-longer"),
  galleryDownload: document.getElementById("gallery-download"),
  galleryDelete: document.getElementById("gallery-delete"),
  videoResolutionToggle: document.getElementById("video-resolution-toggle"),
  videoResolution320p: document.getElementById("video-resolution-320p"),
  videoResolution480p: document.getElementById("video-resolution-480p"),
  videoOrientationToggle: document.getElementById("video-orientation-toggle"),
  videoOrientationLandscape: document.getElementById("video-orientation-landscape"),
  videoOrientationPortrait: document.getElementById("video-orientation-portrait"),
  lightbox: document.getElementById("lightbox"),
  lightboxImage: document.getElementById("lightbox-image"),
  lightboxDownload: document.getElementById("lightbox-download"),
  lightboxClose: document.getElementById("lightbox-close"),
};

let conversationId = localStorage.getItem(STORAGE_KEY);
let ws = null;
let busy = false;
let mode = localStorage.getItem(MODE_KEY) || "chat";
let capabilities = { modes: ["chat"], file_write: false };
let voiceStatus = { stt_available: false, tts_available: false };
// Independent of stt_available (whether STT is configured at all) - this is the user's own runtime
// on/off choice, so the STT model (a real ~1.6GB resident load once used) isn't forced into memory
// just because voice happens to be configured. Defaults on so existing setups see no behavior change.
let voiceEnabled = localStorage.getItem(VOICE_ENABLED_KEY) !== "0";
let modelMode = null; // "light" | "newlight" | "balanced" | "deep", from /api/health - authoritative, not user-set locally
let awaitingRestart = null; // the target mode while a switch-triggered restart is in flight
let fastHealthPoll = null; // tightened setInterval while awaitingRestart, so recovery isn't stuck at 8s
let imageStyle = null; // "none" | "realistic" | "lightning" | "realvis5", from /api/capabilities - no restart involved, unlike modelMode
let switchingImageStyle = false;
let imageResolution = null; // 720 | 1024, from /api/capabilities - independent of imageStyle
let switchingImageResolution = false;
let videoModel = null; // "none" | "ltx" | "fastmetal" | "hunyuan", from /api/capabilities - runtime-only, the server always starts on "none"
let switchingVideoModel = false;
let ttsVoice = "qwen3"; // runtime-only on the server; when available, Qwen3 is the preferred default
let switchingTtsVoice = false;

// ---------------------------------------------------------------- UI helpers
function showError(message) {
  els.errorText.textContent = message;
  els.errorBanner.classList.remove("notice");
  els.errorBanner.hidden = false;
}

// Good news in the same banner (not red), gone by itself after a few seconds.
let noticeTimer = null;
function showNotice(message) {
  showError(message);
  els.errorBanner.classList.add("notice");
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => {
    if (els.errorBanner.classList.contains("notice")) clearError();
  }, 6000);
}

function clearError() {
  els.errorBanner.hidden = true;
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

// Builds DOM nodes (never innerHTML) so model output cannot inject markup. http(s) URLs become
// links: markdown [label](url) and bare URLs. /api/exports/... is also matched bare (no markdown
// form) - app/main.py hands out root-relative download links, not absolute ones, since a single
// host baked in at startup can never be right for every client (127.0.0.1, the LAN IP, ...); the
// browser resolves a relative href/src against whatever origin actually loaded the page, so this
// needs no special "which host" logic here either - just recognizing the bare path as a link.
const LINK_PATTERN = /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s<>()\[\]"']+)|(\/api\/exports\/[^\s<>()\[\]"']+)/g;
// A create_image/edit_image download link (see app/tools/image.py) - shown as an inline preview
// instead of a text link. /api/exports/{filename} serves these with Content-Disposition: attachment
// (so a direct click/navigation downloads), but that header doesn't stop <img src> from rendering
// them inline - confirmed for real via headless Chrome, not assumed.
const IMAGE_URL_PATTERN = /\.(png|jpe?g|webp|gif)(\?|#|$)/i;
// A create_video result (see app/tools/video.py): played inline with controls instead of a text link.
const VIDEO_URL_PATTERN = /\.(mp4|webm|mov)(\?|#|$)/i;
// A create_song result (see app/tools/song.py): an inline player.
const AUDIO_URL_PATTERN = /\.(mp3|wav|flac|m4a|ogg)(\?|#|$)/i;

function makeLink(href, label) {
  const a = document.createElement("a");
  a.href = href;
  a.textContent = label;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  return a;
}

function openLightbox(url, title) {
  els.lightboxImage.src = url;
  els.lightboxImage.alt = title;
  els.lightboxDownload.href = url;
  els.lightbox.hidden = false;
}

function closeLightbox() {
  els.lightbox.hidden = true;
  els.lightboxImage.src = "";
}

function makeVideoPreview(url) {
  const video = document.createElement("video");
  video.className = "video-preview";
  video.src = url;
  video.controls = true;
  video.playsInline = true; // iPhone Safari would otherwise force fullscreen
  video.preload = "metadata";
  return video;
}

function makeAudioPreview(url) {
  const audio = document.createElement("audio");
  audio.className = "audio-preview";
  audio.src = url;
  audio.controls = true;
  audio.preload = "metadata";
  return audio;
}

function makeMediaOrLink(url, label) {
  if (VIDEO_URL_PATTERN.test(url)) return makeVideoPreview(url);
  if (AUDIO_URL_PATTERN.test(url)) return makeAudioPreview(url);
  return IMAGE_URL_PATTERN.test(url) ? makeImagePreview(url, label) : makeLink(url, label);
}

function makeImagePreview(url, title) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "image-preview-btn";
  btn.title = "Click to view full size";
  const img = document.createElement("img");
  img.src = url;
  img.alt = title || "Generated image";
  img.loading = "lazy";
  btn.appendChild(img);
  btn.addEventListener("click", () => openLightbox(url, title || "Generated image"));
  return btn;
}

function renderText(el, text) {
  const nodes = [];
  let last = 0;
  for (const match of text.matchAll(LINK_PATTERN)) {
    if (match.index > last) nodes.push(document.createTextNode(text.slice(last, match.index)));
    if (match[2]) {
      nodes.push(makeMediaOrLink(match[2], match[1]));
    } else if (match[4]) {
      const url = match[4];
      nodes.push(makeMediaOrLink(url, url));
    } else {
      const url = match[3].replace(/[.,;:!?]+$/, "");
      nodes.push(makeMediaOrLink(url, url));
      if (url.length < match[3].length) nodes.push(document.createTextNode(match[3].slice(url.length)));
    }
    last = match.index + match[0].length;
  }
  if (last < text.length) nodes.push(document.createTextNode(text.slice(last)));
  el.replaceChildren(...nodes);
}

function addMessage(role, text) {
  els.empty.hidden = true;
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.dataset.raw = text;
  renderText(div, text);
  els.messages.appendChild(div);
  scrollToBottom();
  return div;
}

function setMessageText(div, text) {
  div.dataset.raw = text;
  renderText(div, text);
}

function typingIndicator() {
  const wrap = document.createElement("span");
  wrap.className = "typing";
  wrap.setAttribute("aria-label", "Zira is thinking");
  for (let i = 0; i < 3; i++) wrap.appendChild(document.createElement("span"));
  return wrap;
}

function setBusy(value) {
  busy = value;
  updateSendButton();
}

// Send, "…" while a reply is being written, or Stop while an image or video is being made and nothing (or only a
// stop word) is typed - with some other text typed it is Send again: the message is answered while an image is
// still being made (see detachImageTurn).
function updateSendButton() {
  const text = els.input.value.trim();
  const stopMode = !!stoppable && (!text || STOP_WORDS.test(text));
  els.send.classList.toggle("stop-btn", stopMode);
  els.send.title = stopMode ? `Stop the ${stoppable} being made (or type or say stop)` : "";
  if (stopMode) {
    els.send.disabled = stopping;
    els.send.textContent = stopping ? "Stopping…" : "Stop";
  } else {
    els.send.disabled = busy;
    els.send.textContent = busy ? "…" : "Send";
  }
}

// ------------------------------------------------------------------ stopping an image or video
// While an image or video is being made the Send button becomes Stop, and typing "stop" (or "cancel", "ruk jao")
// + Enter, or pressing the mic and saying it, does the same: POST /api/images/cancel or /api/videos/cancel ends
// it on the server (the image model stays loaded; a video's LLM is loaded again) and the reply says it was stopped.
const STOP_WORDS = /^\s*(please\s+)?(stop|cancel|ruko|ruk\s+jao|rok\s+do|bas|band\s+karo|stop\s+(it|karo|kar\s+do)|cancel\s+(it|karo|kar\s+do)|रुको|रुक\s+जाओ|रोक\s+दो|बस|बंद\s+करो|स्टॉप|कैंसल)[\s.!।,]*$/i;
const STOP_URLS = { image: "/api/images/cancel", video: "/api/videos/cancel", song: "/api/songs/cancel", ad: "/api/ads/cancel" };
let stoppable = null; // null | "image" | "video": what the Stop button stops right now
let stopListen = false; // the mic is open only to hear "stop" (listenForStop)
let stopping = false; // a Stop request is on its way

// The image/video model can now change without a button press: switched on by itself for a request, or back to
// None after a failure (app/api/images.py select_image_style). While something is being made, and once when it
// ends, the selectors ask the server so they always show what is really loaded.
let modelSyncTimer = null;
async function syncModelSelectors() {
  try {
    if (imageStyle !== null && !switchingImageStyle) {
      const state = await (await fetch("/api/images/style", { cache: "no-store" })).json();
      if (state && state.style && !switchingImageStyle && state.style !== imageStyle) applyImageStyle(state.style);
    }
  } catch { /* not enabled or unreachable: leave it */ }
  try {
    if (videoModel !== null && !switchingVideoModel) {
      const state = await (await fetch("/api/videos/model", { cache: "no-store" })).json();
      if (state && state.model && !switchingVideoModel && state.model !== videoModel) applyVideoModel(state.model);
    }
  } catch { /* not enabled or unreachable: leave it */ }
}

function setStoppable(kind) {
  if (kind && !modelSyncTimer) modelSyncTimer = setInterval(syncModelSelectors, 5000);
  if (!kind && modelSyncTimer) {
    clearInterval(modelSyncTimer);
    modelSyncTimer = null;
    syncModelSelectors();
  }
  stoppable = kind;
  if (!kind) stopping = false;
  if (!kind && stopListen) abortStopListen();
  updateSendButton();
}

async function stopGeneration() {
  if (!stoppable || stopping) return;
  const kind = stoppable;
  stopping = true;
  updateSendButton();
  const statuses = [...document.querySelectorAll(".gen-status")].filter((s) => s.gen);
  statuses.forEach((s) => { s.gen.stopping = true; paintGenStatus(s); });
  try {
    const res = await fetch(STOP_URLS[kind], { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok || !body.stopped) throw new Error(body.detail || `Could not stop the ${kind}.`);
  } catch (err) {
    showError(err.message);
    statuses.forEach((s) => { s.gen.stopping = false; paintGenStatus(s); });
    stopping = false;
    updateSendButton();
  }
}

function setConversation(id) {
  conversationId = id;
  if (id) localStorage.setItem(STORAGE_KEY, id);
  else localStorage.removeItem(STORAGE_KEY);
}

// -------------------------------------------------------------------- health
async function refreshHealth() {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    const data = await res.json();
    els.modelLabel.textContent = data.model_label;
    els.modelLabel.title = data.model;
    if (data.status === "ok") {
      setStatus("online", "AI ONLINE", "Ollama is running and the model is ready");
    } else if (!data.ollama_online) {
      setStatus("offline", "AI OFFLINE", data.detail);
    } else {
      setStatus("degraded", "MODEL MISSING", data.detail);
    }
    applyModelMode(data);
    if (awaitingRestart && data.model_mode === awaitingRestart) {
      awaitingRestart = null;
      if (fastHealthPoll) { clearInterval(fastHealthPoll); fastHealthPoll = null; }
      location.reload();
    }
  } catch (err) {
    setStatus("offline", "SERVER OFFLINE", "Cannot reach the Zira server");
  }
}

// ------------------------------------------------------------------ model mode
// Data-driven over MODEL_MODE_BUTTONS rather than hardcoding, so a future mode only needs an entry
// here plus the matching button in index.html - not a rewrite of this logic every time.
const MODEL_MODE_BUTTONS = [
  { mode: "light", el: () => els.modelModeLight, installedKey: "light_model_installed", modelKey: "light_model",
    title: "Fast, low-latency - for realtime/voice/casual chat" },
  { mode: "newlight", el: () => els.modelModeNewlight, installedKey: "newlight_model_installed", modelKey: "newlight_model",
    title: "The newer fast model (Qwen3.5 4B) - try it against LIGHT" },
  { mode: "balanced", el: () => els.modelModeBalanced, installedKey: "balanced_model_installed", modelKey: "balanced_model",
    title: "A balance of speed and quality - for everyday chat and tasks" },
  { mode: "deep", el: () => els.modelModeDeep, installedKey: "deep_model_installed", modelKey: "deep_model",
    title: "For coding, reasoning, tools, complex tasks" },
];

function applyModelMode(data) {
  modelMode = data.model_mode;
  els.modelModeToggle.hidden = false;
  for (const { mode, el, installedKey, modelKey, title } of MODEL_MODE_BUTTONS) {
    const btn = el();
    if (!btn) continue;
    const installed = data[installedKey];
    btn.classList.toggle("active", modelMode === mode);
    btn.disabled = !!awaitingRestart || modelMode === mode || !installed;
    btn.title = installed ? title : `Not installed - run: ollama pull ${data[modelKey]}`;
  }
}

async function switchModelMode(target) {
  if (awaitingRestart || target === modelMode) return;
  if (currentAudio) currentAudio.pause(); // same manual-interrupt path the mic button already uses
  for (const { el } of MODEL_MODE_BUTTONS) {
    if (el()) el().disabled = true;
  }
  els.modelModeStatus.hidden = false;
  els.modelModeStatus.textContent = `Switching to ${target.toUpperCase()}… Stopping current model…`;
  try {
    const res = await fetch("/api/model/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: target }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Could not switch model.");
    if (!body.restart_required) {
      els.modelModeStatus.hidden = true;
      return;
    }
    els.modelModeStatus.textContent = `Restarting Zira with ${body.model_label}…`;
    awaitingRestart = target;
    fastHealthPoll = setInterval(refreshHealth, 1000);
  } catch (err) {
    showError(err.message);
    els.modelModeStatus.hidden = true;
    await refreshHealth(); // restores real button state (enabled/disabled/installed) from the server
  }
}

for (const { mode, el } of MODEL_MODE_BUTTONS) {
  on(el(), "click", () => switchModelMode(mode));
}

// ------------------------------------------------------------------ image style
// No restart involved (see ImagePipelines.switch_style's docstring - the pipeline lives in this
// same process, unlike the Ollama-backed LLM), so this is a plain disable-post-reenable, not the
// restart/recovery-polling dance switchModelMode above needs.
const IMAGE_STYLE_BUTTONS = [
  { style: "none", el: () => els.imageStyleNone },
  { style: "realistic", el: () => els.imageStyleRealistic },
  { style: "lightning", el: () => els.imageStyleLightning },
  { style: "realvis5", el: () => els.imageStyleRealvis5 },
];

function applyImageStyle(style) {
  imageStyle = style;
  for (const { style: s, el } of IMAGE_STYLE_BUTTONS) {
    const btn = el();
    if (!btn) continue;
    btn.classList.toggle("active", imageStyle === s);
    btn.disabled = switchingImageStyle || imageStyle === s;
  }
}

async function switchImageStyle(target) {
  if (switchingImageStyle || target === imageStyle) return;
  switchingImageStyle = true;
  for (const { el, style } of IMAGE_STYLE_BUTTONS) {
    if (el()) {
      el().disabled = true;
      el().classList.toggle("loading", style === target && target !== "none"); // selecting a model loads it (can take a while)
    }
  }
  try {
    const res = await fetch("/api/images/style", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ style: target }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Could not switch image style.");
    switchingImageStyle = false;
    clearImageStyleLoading();
    applyImageStyle(body.style);
  } catch (err) {
    switchingImageStyle = false;
    clearImageStyleLoading();
    showError(err.message);
    // A failed load leaves the server on "none" (the previous model was already unloaded), so ask it
    // rather than assuming the old style is still active.
    let actual = imageStyle;
    try {
      const state = await (await fetch("/api/images/style", { cache: "no-store" })).json();
      if (state && state.style) actual = state.style;
    } catch { /* keep the last known style */ }
    applyImageStyle(actual); // re-enables buttons at the style that is really active
  }
}

function clearImageStyleLoading() {
  for (const { el } of IMAGE_STYLE_BUTTONS) {
    if (el()) el().classList.remove("loading");
  }
}

for (const { style, el } of IMAGE_STYLE_BUTTONS) {
  on(el(), "click", () => switchImageStyle(style));
}

// ------------------------------------------------------------------ video model
// Same pattern as the image style above: None unloads, picking a model loads it (and keeps it loaded).
// Runtime-only on the server, so a restart always comes back on None.
const VIDEO_MODEL_BUTTONS = [
  { model: "none", el: () => els.videoModelNone },
  { model: "ltx", el: () => els.videoModelLtx },
  { model: "fastmetal", el: () => els.videoModelFastmetal },
  { model: "hunyuan", el: () => els.videoModelHunyuan },
];

function applyVideoModel(model) {
  videoModel = model;
  for (const { model: m, el } of VIDEO_MODEL_BUTTONS) {
    const btn = el();
    if (!btn) continue;
    btn.classList.toggle("active", videoModel === m);
    btn.disabled = switchingVideoModel || videoModel === m;
  }
}

function clearVideoModelLoading() {
  for (const { el } of VIDEO_MODEL_BUTTONS) {
    if (el()) el().classList.remove("loading");
  }
}

async function switchVideoModel(target) {
  if (switchingVideoModel || target === videoModel) return;
  switchingVideoModel = true;
  for (const { el, model: m } of VIDEO_MODEL_BUTTONS) {
    if (el()) {
      el().disabled = true;
      el().classList.toggle("loading", m === target && target !== "none"); // loading a model takes a while
    }
  }
  try {
    const res = await fetch("/api/videos/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: target }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Could not switch the video model.");
    switchingVideoModel = false;
    clearVideoModelLoading();
    applyVideoModel(body.model);
  } catch (err) {
    switchingVideoModel = false;
    clearVideoModelLoading();
    showError(err.message);
    let actual = videoModel; // a failed load leaves the server on "none"; ask rather than assume
    try {
      const state = await (await fetch("/api/videos/model", { cache: "no-store" })).json();
      if (state && state.model) actual = state.model;
    } catch { /* keep the last known model */ }
    applyVideoModel(actual);
  }
}

for (const { model, el } of VIDEO_MODEL_BUTTONS) {
  on(el(), "click", () => switchVideoModel(model));
}

// ------------------------------------------------------------------ reply voice (Qwen3 / Kokoro / IndicF5)
// Same pattern as the image/video selectors: picking a worker voice starts it (the request returns once the
// model is loaded). Only the voices the server has set up (voiceStatus.tts_choices) are shown.
const TTS_VOICE_BUTTONS = [
  { voice: "qwen3", el: () => els.ttsVoiceQwen3 },
  { voice: "kokoro", el: () => els.ttsVoiceKokoro },
  { voice: "indicf5", el: () => els.ttsVoiceIndicf5 },
];

function applyTtsVoice(voice) {
  ttsVoice = voice;
  for (const { voice: v, el } of TTS_VOICE_BUTTONS) {
    const btn = el();
    if (!btn) continue;
    btn.classList.toggle("active", ttsVoice === v);
    btn.disabled = switchingTtsVoice || ttsVoice === v;
    btn.hidden = !(voiceStatus.tts_choices || []).includes(v);
  }
}

async function switchTtsVoice(target) {
  if (switchingTtsVoice || target === ttsVoice) return;
  switchingTtsVoice = true;
  for (const { el, voice } of TTS_VOICE_BUTTONS) {
    if (el()) {
      el().disabled = true;
      el().classList.toggle("loading", voice === target && target !== "qwen3");
    }
  }
  let actual = ttsVoice;
  try {
    const res = await fetch("/api/voice/tts-voice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice: target }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Could not switch the voice.");
    actual = body.voice;
  } catch (err) {
    showError(err.message);
  } finally {
    switchingTtsVoice = false;
    for (const { el } of TTS_VOICE_BUTTONS) if (el()) el().classList.remove("loading");
    applyTtsVoice(actual);
  }
}

for (const { voice, el } of TTS_VOICE_BUTTONS) {
  on(el(), "click", () => switchTtsVoice(voice));
}

// ------------------------------------------------------------------ image resolution
// Independent of image style (unlike steps/guidance, which stay per-style server-side) - a plain
// runtime-switchable value, so this mirrors the style toggle above but with no model/checkpoint
// concept at all: switching never touches the loaded pipeline, just a number read at generation time.
const IMAGE_RESOLUTION_BUTTONS = [
  { resolution: 720, el: () => els.imageResolution720 },
  { resolution: 1024, el: () => els.imageResolution1024 },
];

function applyImageResolution(resolution) {
  imageResolution = resolution;
  for (const { resolution: r, el } of IMAGE_RESOLUTION_BUTTONS) {
    const btn = el();
    if (!btn) continue;
    btn.classList.toggle("active", imageResolution === r);
    btn.disabled = switchingImageResolution || imageResolution === r;
  }
}

async function switchImageResolution(target) {
  if (switchingImageResolution || target === imageResolution) return;
  switchingImageResolution = true;
  for (const { el } of IMAGE_RESOLUTION_BUTTONS) {
    if (el()) el().disabled = true;
  }
  try {
    const res = await fetch("/api/images/resolution", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resolution: target }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Could not switch image resolution.");
    switchingImageResolution = false;
    applyImageResolution(body.resolution);
  } catch (err) {
    switchingImageResolution = false;
    showError(err.message);
    applyImageResolution(imageResolution); // re-enables buttons at the still-actually-active value
  }
}

for (const { resolution, el } of IMAGE_RESOLUTION_BUTTONS) {
  on(el(), "click", () => switchImageResolution(resolution));
}

// ------------------------------------------------------------------ video size
// Resolution (320p/480p) and orientation (landscape/portrait) for every video model - video only, the
// image resolution above is separate. Like it: saved on the server (.env), applied to the next video,
// no model reload.
let videoFormat = { resolution: null, orientation: null }; // from /api/capabilities
let switchingVideoFormat = false;
const VIDEO_FORMAT_BUTTONS = [
  { field: "resolution", value: "320p", el: () => els.videoResolution320p },
  { field: "resolution", value: "480p", el: () => els.videoResolution480p },
  { field: "orientation", value: "landscape", el: () => els.videoOrientationLandscape },
  { field: "orientation", value: "portrait", el: () => els.videoOrientationPortrait },
];

function applyVideoFormat(format) {
  videoFormat = { resolution: format.resolution, orientation: format.orientation };
  for (const { field, value, el } of VIDEO_FORMAT_BUTTONS) {
    const btn = el();
    if (!btn) continue;
    btn.classList.toggle("active", videoFormat[field] === value);
    btn.disabled = switchingVideoFormat || videoFormat[field] === value;
  }
}

async function switchVideoFormat(field, value) {
  if (switchingVideoFormat || videoFormat[field] === value) return;
  switchingVideoFormat = true;
  for (const { el } of VIDEO_FORMAT_BUTTONS) {
    if (el()) el().disabled = true;
  }
  try {
    const res = await fetch("/api/videos/format", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ [field]: value }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Could not change the video size.");
    switchingVideoFormat = false;
    applyVideoFormat(body);
  } catch (err) {
    switchingVideoFormat = false;
    showError(err.message);
    applyVideoFormat(videoFormat); // re-enables the buttons at the still-actually-active values
  }
}

for (const { field, value, el } of VIDEO_FORMAT_BUTTONS) {
  on(el(), "click", () => switchVideoFormat(field, value));
}

function setStatus(kind, text, title) {
  els.status.className = `status ${kind}`;
  els.statusText.textContent = text;
  els.status.title = title || text;
}

// ----------------------------------------------------------------- websocket
// One handler for the whole connection. `currentTurn` is the reply being streamed;
// `lastCompleted` is the previous finished reply, which a trailing "memory" event
// (sent by the server after "done") belongs to.
let currentTurn = null;
let lastCompleted = null;

function finishTurn(turn) {
  if (turn.finished) return;
  turn.finished = true;
  lastCompleted = turn;
  if (currentTurn === turn) currentTurn = null;
  setStoppable(null);
  setBusy(false);
  els.input.focus();
  if (turn.gotToken && turn.startedAt) showTiming(turn);
  if (turn.mode === "plan" && turn.gotToken) addPlanBar(turn);
  if (turn.onDone) turn.onDone(turn);
  if (turn.songTool) autoPlaySong(turn.bubble.dataset.raw || "");
}

// A finished song starts playing by itself (user request), on the music player: its <audio> element is the one a
// tap has already unlocked on an iPhone, and the player panel gives pause/stop. The inline player stays in the chat.
const SONG_LINK = /\/api\/exports\/([^\s)\]>"']+\.(?:mp3|wav|flac|m4a|ogg))/i;
function autoPlaySong(text) {
  const match = SONG_LINK.exec(text || "");
  if (!match) return;
  const title = decodeURIComponent(match[1]).replace(/\.[^.]+$/, "").replace(/[-_]+/g, " ");
  handleMusicCommand({ action: "play", url: match[0], title: `Zira sings: ${title}`, source: "local" });
}

// A small "2.3s - Qwen3 8B" label under each reply, so LIGHT vs DEEP speed is directly visible
// and comparable turn to turn, not just guessed at.
function showTiming(turn) {
  const seconds = ((performance.now() - turn.startedAt) / 1000).toFixed(1);
  const label = document.createElement("div");
  label.className = "turn-timing";
  label.textContent = `${seconds}s · ${turn.modelLabel || modelMode || ""}`;
  turn.bubble.after(label);
}

function showMemoryNote(memories) {
  const anchor = lastCompleted ? lastCompleted.bubble : null;
  if (!anchor || !memories || memories.length === 0) return;
  const note = document.createElement("div");
  note.className = "memory-note";
  note.textContent = "Saved to memory: " + memories.map((m) => m.text).join(" ");
  anchor.after(note);
  scrollToBottom();
}

function handleSocketMessage(event) {
  let data;
  try {
    data = JSON.parse(event.data);
  } catch {
    return;
  }
  if (data.turn_id && backgroundTurns.has(data.turn_id)) return; // an image finishing in the background: polled
  const turn = currentTurn;

  // Streamed speech keeps arriving after the text's "done", so it is routed by its own player, not the turn.
  if (data.type === "audio" || data.type === "speech_end" || data.type === "speech_error") {
    const speech = activeSpeech;
    if (!speech) return;
    if (data.type === "audio") speech.enqueue(data);
    else if (data.type === "speech_end") speech.end();
    else speech.errors.push(data.detail || "speech failed");
    return;
  }
  if (data.type === "memory") {
    showMemoryNote(data.memories);
    return;
  }
  if (data.type === "change") {
    addChangeCard(data.change);
    return;
  }
  if (data.type === "music") {
    handleMusicCommand(data.music);
    return;
  }
  if (!turn) return;

  if (data.type === "start") {
    setConversation(data.conversation_id);
  } else if (data.type === "tool") {
    turn.longTool = LONG_TOOLS.includes(data.tool);
    if (data.tool === "create_song") turn.songTool = true;
    if (!turn.gotToken && turn.longTool) {
      startGenStatus(turn.bubble, data.tool, `${data.detail || data.tool}…`);
      setStoppable(genKind(data.tool));
      scrollToBottom();
      if (genKind(data.tool) === "image" && capabilities.chat_while_generating && turn.id) detachImageTurn(turn);
    } else if (!turn.gotToken) {
      const status = document.createElement("span");
      status.className = "tool-status";
      const base = data.tool === "web_search" ? `Searching the web for "${data.detail}"…` : `${data.detail || data.tool}…`;
      status.dataset.base = base;
      status.textContent = base;
      turn.bubble.replaceChildren(status);
      scrollToBottom();
    }
  } else if (data.type === "image_progress") {
    if (!turn.gotToken) {
      const status = turn.bubble.querySelector(".tool-status");
      if (status && status.gen) {
        setGenProgress(status, data.progress || null);
      } else if (status) {
        const p = data.progress || {};
        const base = status.dataset.base || status.textContent;
        const eta = p.eta_seconds != null ? ` (~${Math.max(1, Math.round(p.eta_seconds))}s left)` : "";
        status.textContent = p.total_steps ? `${base} step ${p.step}/${p.total_steps}${eta}` : base;
        scrollToBottom();
      }
    }
  } else if (data.type === "token") {
    if (!turn.gotToken) {
      setMessageText(turn.bubble, "");
      turn.gotToken = true;
    }
    setMessageText(turn.bubble, (turn.bubble.dataset.raw || "") + data.content);
    scrollToBottom();
  } else if (data.type === "done") {
    finishTurn(turn);
  } else if (data.type === "error") {
    failTurn(turn, data.detail || data.error || "Something went wrong.");
  }
}

function failTurn(turn, message) {
  if (turn.speech) turn.speech.stop();
  if (!turn.gotToken) turn.bubble.remove();
  else setMessageText(turn.bubble, (turn.bubble.dataset.raw || "") + "\n\n[response interrupted]");
  showError(message);
  finishTurn(turn);
}

function handleSocketClose() {
  ws = null;
  if (activeSpeech) activeSpeech.stop(); // its remaining sentences can no longer arrive
  const turn = currentTurn;
  if (!turn || turn.finished) return;
  if (turn.longTool && !turn.gotToken) {
    // An image/video keeps generating on the server without this socket (a phone switching apps
    // closes it): keep the message and the progress on screen and follow the job by polling instead.
    turn.finished = true;
    if (currentTurn === turn) currentTurn = null;
    if (turn.onDone) turn.onDone({ ...turn, bubble: document.createElement("div") }); // voice: nothing to speak
    watchPending(turn.bubble);
    return;
  }
  failTurn(turn, "Connection to the server was lost. Try again.");
}

// ------------------------------------------------------- generations that outlive the socket
// The server keeps a running image/video job per conversation (GET .../pending) until its reply is
// saved. A page that lost its socket, or was reloaded (iOS reloads tabs it put away), shows the job's
// message and progress from there and reloads the history once the job is gone.
let pendingWatch = false;

async function fetchPending() {
  try {
    const res = await fetch(`/api/conversations/${encodeURIComponent(conversationId)}/pending`);
    if (res.status === 404) return null; // a server without this endpoint: nothing to follow
    if (!res.ok) return undefined;
    return (await res.json()).pending || null;
  } catch {
    return undefined; // unreachable for a moment (the phone just came back): try again next tick
  }
}

function renderPendingStatus(bubble, job) {
  const status = bubble.querySelector(".gen-status") || startGenStatus(bubble, job.tool, `${job.label || job.tool}…`);
  setGenProgress(status, job.progress, job.progress_age_seconds || 0, job.finished);
}

// ------------------------------------------------------- image/video status text
// ChatGPT-style: a phrase for the stage the generation is really in (from its step progress), changing
// every few seconds within that stage so a long step (minutes, for video) never looks frozen, with the
// step and a live countdown underneath. What is being made (the tool's label) is the tooltip.
const GEN_PHRASES = {
  video: [
    ["Reading your prompt…", "Planning the scene…"],
    ["Sketching the scene…", "Setting up the shot…", "Blocking out the motion…"],
    ["Filling in details…", "Shaping the movement…", "Adding color and light…"],
    ["Polishing…", "A little more polishing…", "Refining every frame…"],
    ["Adding final touches…", "Putting the frames together…"],
  ],
  ad: [
    ["Writing the script…", "Planning the scenes…"],
    ["Filming the scenes…", "Setting up the shots…"],
    ["Filming the scenes…", "Adding motion and light…"],
    ["Making the jingle…", "Mixing voice and music…"],
    ["Putting the ad together…", "Almost ready…"],
  ],
  song: [
    ["Reading the lyrics…", "Warming up the voice…"],
    ["Finding the melody…", "Setting the rhythm…"],
    ["Singing…", "Laying down the vocals…", "Adding the music…"],
    ["Mixing…", "Balancing voice and music…"],
    ["Adding final touches…", "Almost ready to play…"],
  ],
  image: [
    ["Reading your prompt…", "Getting ready…"],
    ["Creating…", "Sketching the composition…"],
    ["Filling in pixels…", "Adding detail…", "Working on the lighting…"],
    ["Polishing…", "A little more polishing…", "Sharpening details…"],
    ["Adding final touches…", "Almost there…"],
  ],
};
const GEN_PHRASE_MS = 6000;
let genTimer = null;

function startGenStatus(bubble, tool, label) {
  const status = document.createElement("span");
  status.className = "tool-status gen-status";
  status.title = label || "";
  const phrase = document.createElement("span");
  phrase.className = "gen-phrase";
  const detail = document.createElement("span");
  detail.className = "gen-detail";
  status.append(phrase, detail);
  const now = Date.now();
  status.gen = { kind: genKind(tool), progress: null, at: now, started: now, finished: false };
  bubble.replaceChildren(status);
  paintGenStatus(status);
  if (!genTimer) genTimer = setInterval(tickGenStatuses, 1000);
  return status;
}

function setGenProgress(status, progress, ageSeconds = 0, finished = false) {
  status.gen.progress = progress;
  status.gen.at = Date.now() - ageSeconds * 1000; // when that progress was measured, for the countdown
  status.gen.finished = finished;
  paintGenStatus(status);
}

function genStage(g) {
  const p = g.progress;
  if (g.finished) return 4;
  if (!p || !p.total_steps) return 0; // prompt still being encoded: no step yet
  if (p.step >= p.total_steps) return 4; // all steps done: decoding and saving
  const f = p.step / p.total_steps;
  return f < 1 / 3 ? 1 : f < 2 / 3 ? 2 : 3;
}

function formatTimeLeft(seconds) {
  if (seconds <= 0) return "almost done";
  const s = Math.round(seconds);
  return s < 60 ? `~${s}s left` : `~${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s left`;
}

function paintGenStatus(status) {
  const g = status.gen;
  const stage = genStage(g);
  const phrases = GEN_PHRASES[g.kind][stage];
  status.querySelector(".gen-phrase").textContent = phrases[Math.floor((Date.now() - g.started) / GEN_PHRASE_MS) % phrases.length];
  const p = g.progress;
  let detail = "starting";
  if (g.stopping) {
    detail = "stopping…";
  } else if (p && p.stage === "loading") {
    detail = p.label || "setting up the model…"; // a model being switched on for this request
  } else if (p && p.label && p.total_steps) {
    detail = `${p.label} · ${Math.round((100 * p.step) / p.total_steps)}%`; // an ad: which part it is on
  } else if (stage === 4) {
    detail = "finishing up";
  } else if (p && p.total_steps) {
    const left = p.eta_seconds != null ? ` · ${formatTimeLeft(p.eta_seconds - (Date.now() - g.at) / 1000)}` : "";
    detail = `step ${p.step}/${p.total_steps}${left}`;
  }
  status.querySelector(".gen-detail").textContent = detail;
}

function tickGenStatuses() {
  const live = document.querySelectorAll(".gen-status");
  if (!live.length) {
    clearInterval(genTimer);
    genTimer = null;
    return;
  }
  live.forEach((status) => status.gen && paintGenStatus(status));
}

async function reloadHistory() {
  els.messages.querySelectorAll(".msg, .memory-note, .turn-timing, .plan-bar").forEach((node) => node.remove());
  await loadHistory();
}

// background: an image turn moved out of the way so the chat stays free (detachImageTurn) - nothing is blocked,
// and the history is reloaded only once no reply is being written.
async function watchPending(bubble, { background = false } = {}) {
  if (pendingWatch) return;
  pendingWatch = true;
  if (!background) setBusy(true);
  const since = Date.now();
  let seen = false;
  let lastTool = null;
  try {
    for (;;) {
      const job = await fetchPending();
      // Just moved to the background, the job may not be registered yet: give it a moment before calling it done.
      if (job === null && (seen || !background || Date.now() - since > PENDING_GRACE_MS)) break;
      if (job) {
        seen = true;
        lastTool = job.tool;
        renderPendingStatus(bubble, job);
        if (!job.finished && !stoppable) setStoppable(genKind(job.tool));
      }
      await new Promise((resolve) => setTimeout(resolve, background && !seen ? 1000 : PENDING_POLL_MS));
    }
    setStoppable(null);
    while (currentTurn) await new Promise((resolve) => setTimeout(resolve, 500)); // never wipe a reply being written
    await reloadHistory(); // the finished image/video, or what went wrong, is in the history now
    if (lastTool === "create_song") {
      const players = document.querySelectorAll(".audio-preview");
      if (players.length) autoPlaySong(new URL(players[players.length - 1].src).pathname);
    }
  } finally {
    pendingWatch = false;
    setStoppable(null);
    if (!background) setBusy(false);
  }
}

// ------------------------------------------------------------------ chatting while an image is made
// Once a turn starts making an image the chat is free again: the turn is finished here (a voice turn goes back to
// listening), its progress is followed by polling /pending as after a reconnect, and the server answers the next
// message alongside it (chat_ws). Its own events are ignored from now on. Videos keep the chat busy - the chat model
// is unloaded while one is made.
const PENDING_GRACE_MS = 15000;
const backgroundTurns = new Set();

function newTurnId() {
  try {
    return crypto.randomUUID();
  } catch {
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }
}

function detachImageTurn(turn) {
  if (turn.finished || pendingWatch) return;
  backgroundTurns.add(turn.id);
  turn.finished = true;
  if (currentTurn === turn) currentTurn = null;
  if (turn.speech) {
    turn.speech.stop();
    if (activeSpeech === turn.speech) activeSpeech = null;
  }
  setBusy(false);
  // Voice: nothing to speak for an image - back to listening (the same as after a reconnect, handleSocketClose).
  if (turn.onDone) turn.onDone({ ...turn, bubble: document.createElement("div"), speech: null });
  watchPending(turn.bubble, { background: true });
}

async function resumePending() {
  if (!conversationId || currentTurn || pendingWatch) return;
  const job = await fetchPending();
  if (!job || currentTurn || pendingWatch) return;
  addMessage("user", job.user_message);
  const bubble = addMessage("assistant", "");
  renderPendingStatus(bubble, job);
  watchPending(bubble);
}

function connect() {
  return new Promise((resolve, reject) => {
    if (ws && ws.readyState === WebSocket.OPEN) return resolve(ws);
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/ws/chat`);
    socket.onmessage = handleSocketMessage;
    socket.onclose = handleSocketClose;
    socket.onopen = () => {
      ws = socket;
      resolve(socket);
    };
    socket.onerror = () => reject(new Error("Could not connect to the Zira server."));
  });
}

async function sendMessage(text, { onDone, source = "text", speak = false } = {}) {
  noteUserActivity();
  clearError();
  document.querySelectorAll(".plan-bar").forEach((node) => node.remove());
  addMessage("user", text);
  const bubble = addMessage("assistant", "");
  bubble.appendChild(typingIndicator());
  setBusy(true);

  const turn = { id: newTurnId(), bubble, gotToken: false, finished: false, mode, onDone, startedAt: performance.now(), modelLabel: els.modelLabel.textContent };
  currentTurn = turn;
  if (activeSpeech) activeSpeech.stop(); // a new request ends whatever was still being said
  if (speak) {
    activeSpeech = new SpeechPlayer(turn.startedAt);
    turn.speech = activeSpeech;
  }

  let socket;
  try {
    socket = await connect();
  } catch (err) {
    bubble.remove();
    showError(err.message);
    finishTurn(turn);
    return;
  }
  socket.send(JSON.stringify({ conversation_id: conversationId, message: text, mode, source, turn_id: turn.id, ...(speak ? { speak: true } : {}) }));
}

// ------------------------------------------------------- modes and plan approval
function applyMode(value) {
  mode = capabilities.modes.includes(value) ? value : "chat";
  localStorage.setItem(MODE_KEY, mode);
  els.mode.value = mode;
  els.modeHint.textContent = MODE_HINTS[mode];
}

async function loadCapabilities() {
  try {
    const res = await fetch("/api/capabilities", { cache: "no-store" });
    capabilities = await res.json();
  } catch {
    return;
  }
  els.mode.replaceChildren(
    ...capabilities.modes.map((m) => {
      const option = document.createElement("option");
      option.value = m;
      option.textContent = MODE_LABELS[m];
      return option;
    })
  );
  els.mode.hidden = capabilities.modes.length < 2;
  applyMode(mode);
  if (!capabilities.music) els.musicPlayer.hidden = true;
  if (els.imageUploadBtn) els.imageUploadBtn.hidden = false; // documents can always be attached; photos need image generation
  if (els.imageStyleToggle) {
    els.imageStyleToggle.hidden = !capabilities.image_generation;
    if (capabilities.image_generation) applyImageStyle(capabilities.image_style);
  }
  if (els.imageResolutionToggle) {
    els.imageResolutionToggle.hidden = !capabilities.image_generation;
    if (capabilities.image_generation) applyImageResolution(capabilities.image_resolution);
  }
  if (els.videoModelToggle) {
    els.videoModelToggle.hidden = !capabilities.video_generation;
    if (capabilities.video_generation) applyVideoModel(capabilities.video_model);
  }
  for (const toggle of [els.videoResolutionToggle, els.videoOrientationToggle]) {
    if (toggle) toggle.hidden = !capabilities.video_generation;
  }
  if (capabilities.video_generation && capabilities.video_resolution) {
    applyVideoFormat({ resolution: capabilities.video_resolution, orientation: capabilities.video_orientation });
  }
}

function addPlanBar(turn) {
  const bar = document.createElement("div");
  bar.className = "plan-bar";
  const label = document.createElement("span");
  label.textContent = "Happy with this plan?";
  bar.appendChild(label);
  if (capabilities.modes.includes("edit")) {
    const approve = document.createElement("button");
    approve.type = "button";
    approve.className = "approve";
    approve.textContent = "Approve plan → Edit mode";
    approve.addEventListener("click", () => {
      if (busy) return;
      applyMode("edit");
      sendMessage("The plan is approved. Implement it step by step. Propose each change for my review.");
    });
    bar.appendChild(approve);
  } else {
    label.textContent = "To have Zira carry it out, enable FILE_WRITE_ROOTS in .env and restart.";
  }
  turn.bubble.after(bar);
  scrollToBottom();
}

// ---------------------------------------------------------- change approval cards
const changeCards = new Map();

function renderDiff(pre, diff) {
  const nodes = [];
  for (const line of diff.split("\n")) {
    const span = document.createElement("span");
    if (line.startsWith("+") && !line.startsWith("+++")) span.className = "diff-add";
    else if (line.startsWith("-") && !line.startsWith("---")) span.className = "diff-del";
    else if (line.startsWith("@@")) span.className = "diff-hunk";
    span.textContent = line;
    nodes.push(span, document.createTextNode("\n"));
  }
  pre.replaceChildren(...nodes);
}

async function decide(change, action, card) {
  const buttons = card.querySelectorAll("button");
  buttons.forEach((b) => (b.disabled = true));
  try {
    const res = await fetch(`/api/changes/${encodeURIComponent(change.id)}/${action}`, { method: "POST" });
    const body = await res.json();
    if (!res.ok) {
      card.querySelector(".change-note").textContent = body.detail || "Could not do that.";
      const fresh = await fetch(`/api/changes/${encodeURIComponent(change.id)}`);
      if (fresh.ok) updateChangeCard(card, await fresh.json(), true);
      return;
    }
    updateChangeCard(card, body);
  } catch {
    card.querySelector(".change-note").textContent = "Cannot reach the Zira server.";
    buttons.forEach((b) => (b.disabled = false));
  }
}

function updateChangeCard(card, change, keepNote = false) {
  const status = card.querySelector(".change-status");
  const labels = { pending: "Waiting for your approval", applied: "Applied", rejected: "Rejected", undone: "Undone", failed: "Not applied", expired: "Expired" };
  status.textContent = labels[change.status] || change.status;
  status.className = `change-status ${change.status}`;
  if (!keepNote) card.querySelector(".change-note").textContent = change.note || "";
  const actions = card.querySelector(".change-buttons");
  actions.replaceChildren();
  const make = (text, cls, action) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = text;
    b.addEventListener("click", () => decide(change, action, card));
    actions.appendChild(b);
  };
  if (change.status === "pending") {
    make("Approve", "approve", "approve");
    make("Reject", "reject", "reject");
  } else if (change.status === "applied") {
    make("Undo", "", "undo");
  }
}

function addChangeCard(change) {
  if (changeCards.has(change.id)) return;
  els.empty.hidden = true;
  const card = document.createElement("div");
  card.className = "change-card" + (change.risky ? " risky" : "");
  const head = document.createElement("div");
  head.className = "change-head";
  const kind = document.createElement("span");
  kind.textContent = change.kind === "create" ? "New file" : "Edit";
  const path = document.createElement("span");
  path.className = "change-path";
  path.textContent = change.path;
  head.append(kind, path);
  if (change.risky) {
    const badge = document.createElement("span");
    badge.className = "badge risky";
    badge.textContent = "config / build / script: review carefully";
    head.appendChild(badge);
  }
  card.appendChild(head);
  if (change.explanation) {
    const why = document.createElement("div");
    why.className = "change-why";
    why.textContent = change.explanation;
    card.appendChild(why);
  }
  const pre = document.createElement("pre");
  pre.className = "diff";
  renderDiff(pre, change.diff);
  card.appendChild(pre);
  const actions = document.createElement("div");
  actions.className = "change-actions";
  const status = document.createElement("span");
  status.className = "change-status";
  const buttons = document.createElement("span");
  buttons.className = "change-buttons";
  buttons.style.display = "inline-flex";
  buttons.style.gap = "8px";
  actions.append(status, buttons);
  card.appendChild(actions);
  const note = document.createElement("div");
  note.className = "change-note";
  card.appendChild(note);
  updateChangeCard(card, change);
  changeCards.set(change.id, card);
  els.messages.appendChild(card);
  scrollToBottom();
}

// ------------------------------------------------------------------- music
// A single plain Audio() object, the same "what's currently playing" pattern already used for TTS
// (currentAudio) - a track is just a direct audio URL (either JARVIS's own /api/music/local
// endpoint, or the user's own server), so no player library/embed is needed. No persistence across
// reload (same as TTS today) - an accepted v1 limitation, not a silent gap.
let currentMusicAudio = null;

// --- audio that works on an iPhone -----------------------------------------------------------
// Music and spoken replies are started from async callbacks (a WebSocket "music" event, a fetched
// TTS clip), never directly inside a tap. Desktop browsers mostly allow that once you've interacted
// with the page; iOS Safari - the engine behind every iPhone browser - does not: audio.play() only
// works inside a user gesture, so a fresh `new Audio(url).play()` from a callback is silently
// refused. What iOS does allow: an <audio> element that a real tap has started once stays
// allowed, even after its src is swapped and play() is called later from async code. So there is
// one long-lived element per role, both "unlocked" (a 0.1s silent clip played) on the first tap.
// The audio itself plays on whichever device is running the page - from an iPhone, that's the
// iPhone's speaker - since the server only ever sends a URL and never plays anything on the Mac.
function makeSilentWavUrl() {
  const n = 800; // 0.1s of 8 kHz 8-bit mono silence
  const buf = new ArrayBuffer(44 + n);
  const v = new DataView(buf);
  const tag = (offset, s) => { for (let i = 0; i < s.length; i++) v.setUint8(offset + i, s.charCodeAt(i)); };
  tag(0, "RIFF"); v.setUint32(4, 36 + n, true); tag(8, "WAVE"); tag(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 8000, true); v.setUint32(28, 8000, true); v.setUint16(32, 1, true); v.setUint16(34, 8, true);
  tag(36, "data"); v.setUint32(40, n, true);
  for (let i = 0; i < n; i++) v.setUint8(44 + i, 128); // 128 = zero level for unsigned 8-bit PCM
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}

const musicEl = new Audio();
const speechEl = new Audio();
let audioUnlocked = false;

// One long-lived AudioContext for the hands-free silence detector. It used to create (and close) a
// new one on every listen; on iOS a context made outside a tap starts "suspended" and stays that
// way, and hands-free re-listens after each reply from an async callback, not a tap - so from the
// second turn the analyser read silence and the end of your speech was never detected. Created and
// resumed inside the first tap instead (see onUserGesture), then reused for every listen.
let sharedAudioCtx = null;
function getAudioCtx() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return null;
  if (!sharedAudioCtx || sharedAudioCtx.state === "closed") sharedAudioCtx = new AudioCtx();
  return sharedAudioCtx;
}
function resumeSharedAudioCtx() {
  const ctx = getAudioCtx();
  if (ctx && ctx.state === "suspended") ctx.resume().catch(() => {});
}

function unlockAudio() {
  if (audioUnlocked) return;
  audioUnlocked = true;
  const silent = makeSilentWavUrl();
  for (const el of [musicEl, speechEl]) {
    try {
      el.src = silent;
      const p = el.play();
      // If this attempt was refused, allow the next tap to try again instead of staying "unlocked".
      if (p && p.then) p.then(() => el.pause()).catch(() => { audioUnlocked = false; });
    } catch {
      audioUnlocked = false;
    }
  }
}
// click + touchend are the two events iOS counts as user activation for media (pointerdown is not).
function onUserGesture() {
  unlockAudio();
  resumeSharedAudioCtx(); // every tap, not just the first: iOS can re-suspend the context (interruptions)
}
for (const evt of ["click", "touchend"]) document.addEventListener(evt, onUserGesture, { capture: true, passive: true });

function handleMusicCommand(cmd) {
  if (!cmd || !cmd.action) return;
  if (cmd.action === "play") {
    if (currentAudio) currentAudio.pause(); // music and spoken replies are mutually exclusive
    if (currentMusicAudio) currentMusicAudio.pause();
    releaseMicForPlayback(); // loudspeaker, not the earpiece (see releaseMicForPlayback)
    currentMusicAudio = musicEl;
    musicEl.src = cmd.url;
    musicEl.play().catch((err) => {
      // Browsers block audio.play() unless it's tied to a direct, recent user gesture - this call
      // comes from an async WS message handler (after however long the reply took to generate), so
      // that block can trigger silently: the player panel still shows "Now playing" with nothing
      // audible. The panel's own Play button (a real click) always bypasses this, so surface it
      // instead of swallowing the rejection into the console where only a dev would ever see it.
      console.debug("[music] play failed", err);
      showError('Playback was blocked by your browser. Press ▶ on the music player below to start it.');
    });
    els.musicTitle.textContent = cmd.title || "";
    els.musicSource.textContent = cmd.source === "remote" ? "Your server" : "Local library";
    els.musicPlayer.hidden = false;
    // Music now playing through the speakers could otherwise bleed into the mic and be mistaken
    // for a follow-up command under an already-open no-wake-word grace window.
    requireWakeWordAgain();
    return;
  }
  if (!currentMusicAudio) return; // pause/resume/stop with nothing loaded: safe no-op
  if (cmd.action === "pause") {
    currentMusicAudio.pause();
  } else if (cmd.action === "resume") {
    currentMusicAudio.play().catch((err) => {
      console.debug("[music] resume failed", err);
      showError('Playback was blocked by your browser. Press ▶ on the music player below to start it.');
    });
  } else if (cmd.action === "stop") {
    currentMusicAudio.pause();
    currentMusicAudio.currentTime = 0;
    els.musicPlayer.hidden = true;
  }
}

// Manual buttons call the exact same paths the AI-driven commands use, so manual and AI control
// can never drift out of sync.
on(els.musicPlay, "click", () => handleMusicCommand({ action: "resume" }));
on(els.musicPause, "click", () => handleMusicCommand({ action: "pause" }));
on(els.musicStop, "click", () => handleMusicCommand({ action: "stop" }));

// ------------------------------------------------------------ image upload/edit
// Uploads immediately on file selection (so failures surface right away, not at send-time), then
// holds the returned id until the next message is sent - handleSubmit prefixes that message with
// "[Uploaded image: <id>]" (see app/tools/image.py::EditImageTool's description, which tells the
// model to extract the id from exactly that literal text) so edit_image can find the file.
let attachedImage = null; // {id, name} | null

function clearAttachedImage() {
  attachedImage = null;
  els.imageAttached.hidden = true;
  els.imageUploadInput.value = "";
}

on(els.imageUploadBtn, "click", () => els.imageUploadInput.click());

on(els.imageUploadInput, "change", async () => {
  const file = els.imageUploadInput.files[0];
  if (!file) return;
  const previousName = els.imageAttachedName.textContent;
  els.imageAttachedName.textContent = `Uploading ${file.name}…`;
  els.imageAttached.hidden = false;
  try {
    const form = new FormData();
    form.append("file", file);
    // A photo goes to the image uploads (ask about it, edit or animate it); anything else is a document to ask about.
    const isImage = /^image\//.test(file.type);
    if (isImage && !capabilities.image_generation) throw new Error("Photos can be attached when image generation is on.");
    const res = await fetch(isImage ? "/api/images/upload" : "/api/documents/upload", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || `Could not upload that ${isImage ? "image" : "document"}.`);
    attachedImage = isImage ? { id: body.image_id, name: file.name } : { id: body.doc_id, name: file.name, kind: "document" };
    els.imageAttachedName.textContent = `${isImage ? "📎" : "📄"} ${file.name}`;
  } catch (err) {
    showError(err.message);
    if (attachedImage) {
      els.imageAttachedName.textContent = previousName;
    } else {
      clearAttachedImage();
    }
  }
});

on(els.imageAttachedRemove, "click", clearAttachedImage);

async function loadPendingChanges() {
  if (!conversationId || !capabilities.file_write) return;
  try {
    const res = await fetch(`/api/changes?conversation_id=${encodeURIComponent(conversationId)}&status=pending`);
    if (!res.ok) return;
    for (const change of (await res.json()).reverse()) addChangeCard(change);
  } catch {
    /* pending changes are a convenience on reload */
  }
}

// ------------------------------------------------------------------- voice
// Two interaction modes, both feeding transcribed text through the SAME sendMessage()/chat
// pipeline used for typed text, so a voice turn is stored in conversation history identically to
// a typed one:
//   - Push-to-talk (default): press and hold the mic button, speak, release.
//   - Continuous/hands-free ("Continuous" checkbox): the mic stays on in a loop. Each utterance is
//     transcribed locally and must start with the wake word "Hello JARVIS" (fuzzy-matched - any
//     of the first few words being "jarvis" counts) or it is silently discarded and JARVIS keeps
//     listening. This runs entirely through the existing faster-whisper /api/voice/transcribe
//     endpoint - there is no separate always-on wake-word engine - so it is CPU/battery-heavier
//     than push-to-talk while active. That trade-off (plus a real, local, low-power wake-word
//     engine as a future upgrade) is documented in the README "Voice" section.
// After the reply finishes, its text (markdown stripped server-side) is sent to /api/voice/speak
// and played back; in continuous mode JARVIS then automatically starts listening again.
const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
const MIN_CLIP_BYTES = 800; // below this, a real spoken clip is essentially impossible (e.g. an accidental tap)
const CONTINUOUS_KEY = "jarvis.voice.continuous";
// Wake words. "Zira" is the assistant's name and its wake word; "jarvis" (the old name) stays as a
// hidden backup, because the speech model spells it reliably - a fallback while "Zira" is tuned.
// Whether hands-free wakes at all depends on how Whisper SPELLS the word, and a rare name gets many
// spellings ("Zira", "Zyra", "Zeera"...), so matching has three layers: exact known spellings, a
// phonetic key for other spellings of the same sound, and Devanagari spellings. The spellings come
// from real transcripts - see scripts/wake_word_benchmark.py - not from guessing.
const WAKE_WORD = "zira";
// The backup's spellings as measured: "jarvis" was written exactly in only 6 of 12 clips - the rest
// came out as Jivus/Jovers/Java or in Devanagari. ("java" is a real word, so it is deliberately left out.)
const WAKE_WORD_BACKUP_ALIASES = new Set(["jarvis", "jarvus", "jervis", "jivus", "jovers"]);
// Whisper doesn't reliably keep a Latin name as Latin when the rest of the utterance is Hindi -
// measured for real (scripts/voice_benchmark.py): "Haan JARVIS, kya scene hai?" came back as
// "हाँ जारवेस, क्या सीन है?", which a plain Latin match never wakes on. Best-effort transliterations,
// not an exhaustive set. (In Hindi, ज़ीरा/जीरा is also the word for cumin, so a Hindi clip may well
// contain it by accident - the reason this only matters in the first few words of a clip.)
// Non-Latin spellings. Seen without the prompt: जीरा, जिरा, जिर (Devanagari) and زیرا (Urdu script).
const WAKE_WORD_DEVANAGARI = new Set([
  "ज़ीरा", "जीरा", "ज़िरा", "जिरा", "ज़ेरा", "जेरा", "ज़ायरा", "जायरा", "जिर", "जीर", // zira
  "زیرا", "زيرا", // zira, Urdu/Arabic script
  "जारवेस", "जार्विस", "जारविस", "जर्विस", "जॉर्विस", "जोबर्स", // jarvis (backup)
]);
// An ordinary English word that also happens to sound like "zira" (Whisper wrote "Zero" 3 times for
// it): it only counts right after a greeting, never as the first word of a clip.
const WAKE_WORD_NEEDS_GREETING = new Set(["zero"]);
// The words that may directly precede the wake word ("Hello Zira", "Hey Zira").
const GREETING_WORD = /^(h[ae]l+o+|hey+|hi+|hai|ok|okay|namaste)$/;
const WAKE_WORD_FILLER = /^[,.!]*(please|ok|okay)?[,.!]*$/i;
// A farewell in the command ends the hands-free session right away instead of waiting out
// AWAKE_GRACE_MS - saying "goodbye"/"good night" is itself the signal that the conversation is
// over, so the next utterance should need "Hello JARVIS" again immediately, not 10s later.
const FAREWELL_PATTERN = /\b(good\s*bye|bye|good\s*night|goodnight|see\s*you|take\s*care)\b/i;
// Silence-based auto-stop for continuous mode. The speech-detection threshold is NOT a fixed
// guessed number (a fixed 0.02 turned out to be too high for at least one real microphone and
// made continuous mode never detect speech at all) - instead it is calibrated against your own
// mic's ambient noise level for the first CALIBRATION_MS of every listen, then set to a multiple
// of that. On top of that, if no speech is detected for a while (still possible on hardware with
// aggressive automatic gain control, which actively works against amplitude-based detection by
// keeping levels roughly constant whether you're talking or not - common on laptop mics), the
// threshold RELAXES (lowers) every RELAX_INTERVAL_MS so it gets progressively more sensitive
// rather than staying stuck too strict for the whole listen. MAX_LISTEN_MS is also a firm, fairly
// short cap so a bad calibration never looks like "nothing is happening" for too long - if
// automatic detection never works for you, clicking the mic while it says LISTENING always forces
// it to stop and process immediately, regardless of any of this.
const CALIBRATION_MS = 400;
const SPEECH_LEVEL_MULTIPLIER = 2.5;
const MIN_SPEECH_THRESHOLD = 0.004; // floor for a near-silent room, so it isn't hypersensitive
const RELAX_INTERVAL_MS = 1500; // how often the threshold lowers if still no speech detected
const RELAX_FACTOR = 0.7;
// Was 1200 - pure dead time between "you stopped talking" and STT even starting, on top of STT's
// own real latency (~4-5s with mlx-whisper/large-v3-turbo). Lowered as part of addressing a real
// "recognition is good but late" report; still enough headroom for a natural brief mid-sentence
// pause without cutting you off (SILENCE_MS is time-since-last-loud-sample, not a hard per-word cap).
const SILENCE_MS = 700;
const MAX_LISTEN_MS = 8000;
// Once woken with "Hello JARVIS", you don't have to repeat it for follow-ups - each one just
// needs to come within AWAKE_GRACE_MS of real listening time since your last utterance (JARVIS
// thinking/speaking doesn't count against this; only time spent actually listening with nothing
// heard does). Go quiet for that long and the next utterance needs the wake word again.
const AWAKE_GRACE_MS = 10000;

let micStream = null;
let recorder = null;
let recordedChunks = [];
let voiceState = "ready"; // ready | listening | transcribing | thinking | speaking
let currentAudio = null;
let turnStartedAt = 0;
let continuousMode = localStorage.getItem(CONTINUOUS_KEY) === "1";
let silenceWatch = null; // mutable state object while a silence watch is running, else null
let continuousSafetyTimer = null;
let sessionAwake = false; // true from a heard wake word until AWAKE_GRACE_MS of silence passes
let silentListenMs = 0; // accumulated listening time (not counting JARVIS's own turn) with nothing heard

// Ends the current hands-free session immediately, instead of waiting out AWAKE_GRACE_MS - the
// next voice command will need "Hello JARVIS" again. Used both when a farewell phrase is heard
// (see stopListeningAndProcess) and whenever music starts playing (see handleMusicCommand): music
// coming out of the speakers can otherwise bleed into the mic and be mistaken for a follow-up
// command under an already-open grace window.
function requireWakeWordAgain() {
  sessionAwake = false;
  silentListenMs = 0;
}

function pickMimeType() {
  if (!window.MediaRecorder) return "";
  return MIME_CANDIDATES.find((t) => MediaRecorder.isTypeSupported(t)) || "";
}

function extensionForMime(mimeType) {
  if (mimeType.includes("mp4")) return "mp4";
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("wav")) return "wav";
  return "webm";
}

function setVoiceState(next) {
  voiceState = next;
  const labels = { ready: "READY", listening: "LISTENING…", transcribing: "TRANSCRIBING…", thinking: "THINKING…", speaking: "SPEAKING…" };
  els.voiceStatus.textContent = labels[next] || next;
  els.voiceBar.classList.toggle("listening", next === "listening");
  els.voiceBar.classList.toggle("active", next !== "ready");
}

async function loadVoiceStatus() {
  try {
    const res = await fetch("/api/voice/status", { cache: "no-store" });
    voiceStatus = await res.json();
  } catch {
    voiceStatus = { stt_available: false, tts_available: false };
  }
  els.voiceBar.hidden = !voiceStatus.stt_available;
  if (els.ttsVoiceToggle) {
    const switchable = (voiceStatus.tts_choices || []).length > 1;
    els.ttsVoiceToggle.hidden = !switchable;
    if (switchable) applyTtsVoice(voiceStatus.tts_choice || "qwen3");
  }
  applyVoicePowerState();
  updateVoiceBtnTitle();
  els.voiceContinuous.checked = continuousMode;
  if (continuousMode && voiceStatus.stt_available && voiceEnabled) startListening();
}

// The interactive parts of the voice bar (mic button, level meter, transcript, continuous checkbox)
// - everything except the power toggle itself, which must stay reachable so voice can be turned
// back on. Hiding these rather than the whole bar keeps the power toggle visible/accessible even
// while voice is off.
const VOICE_INTERACTIVE_ELS = () => [els.voiceBtn, els.voiceStatus, els.voiceLevel, els.voiceTranscript, els.voiceContinuous?.closest("label")];

function applyVoicePowerState() {
  if (!els.voicePowerBtn) return;
  els.voicePowerBtn.textContent = voiceEnabled ? "🎤 ON" : "🎤 OFF";
  els.voicePowerBtn.classList.toggle("off", !voiceEnabled);
  els.voicePowerBtn.title = voiceEnabled
    ? "Voice is on - the STT model loads into RAM on first use. Click to turn off and free that memory."
    : "Voice is off (no model in memory). Click to turn back on.";
  for (const el of VOICE_INTERACTIVE_ELS()) {
    if (el) el.hidden = !voiceEnabled;
  }
}

async function toggleVoicePower() {
  const next = !voiceEnabled;
  if (!next) {
    teardownListening();
    try {
      await fetch("/api/voice/unload", { method: "POST" });
    } catch {
      // Best-effort - the page reload right after this still leaves the UI in the right state even
      // if the unload request itself failed (e.g. server briefly unreachable).
    }
  }
  localStorage.setItem(VOICE_ENABLED_KEY, next ? "1" : "0");
  location.reload();
}

on(els.voicePowerBtn, "click", toggleVoicePower);

function updateVoiceBtnTitle() {
  if (continuousMode) {
    els.voiceBtn.title = 'Click to stop listening (say "Hello Zira" to talk while this is on)';
  } else {
    els.voiceBtn.title = voiceStatus.tts_available
      ? "Press and hold, then speak"
      : "Press and hold, then speak (spoken replies are off: no TTS_PROVIDER configured)";
  }
}

async function getMicStream() {
  // A cached stream is only reusable while a track is live: iOS can end the track (a phone call, the
  // page being backgrounded), and recording from a dead stream silently produces empty clips.
  if (micStream && micStream.getAudioTracks().some((t) => t.readyState === "live")) return micStream;
  micStream = null;
  // Browsers expose the microphone only on a secure context (HTTPS or localhost). On plain
  // http://<ip>:8000 - how a phone reaches this server over Wi-Fi or Tailscale - navigator.mediaDevices
  // is undefined, which used to surface as "undefined is not an object (evaluating ...)".
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const err = new Error("The microphone needs a secure (HTTPS) page.");
    err.name = "InsecureContextError";
    throw err;
  }
  setAudioSession("auto"); // back to recording mode (see releaseMicForPlayback)
  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  return micStream;
}

// iPhone: while a page holds the microphone, iOS treats it like a phone call and plays everything through the
// small earpiece at the top, quietly (the user's report: "voice is very low ... like a phone call, not from the
// speaker below"). So before Zira speaks, the mic is released and the page's audio session is set to "playback"
// (Safari 16.4+'s navigator.audioSession), which sends sound to the loudspeaker. The next listen takes the mic
// again (getMicStream). Never while recording: only outside the "listening" state.
function setAudioSession(type) {
  try {
    if (navigator.audioSession && navigator.audioSession.type !== type) navigator.audioSession.type = type;
  } catch {
    // older browsers: nothing to set
  }
}

function releaseMicForPlayback() {
  if (voiceState === "listening") return;
  if (micStream) {
    for (const track of micStream.getTracks()) track.stop();
    micStream = null;
  }
  setAudioSession("playback");
}

function micErrorMessage(err) {
  if (err.name === "InsecureContextError") {
    return "The microphone only works on an HTTPS page - browsers block it on plain http:// addresses. Open Zira through its HTTPS address (see the README, \"Voice input from the phone\").";
  }
  if (err.name === "NotAllowedError" || err.name === "SecurityError") {
    return "Microphone access was denied. Allow it for this page in your browser's site settings.";
  }
  if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
    return "No microphone was found on this device.";
  }
  return `Could not access the microphone (${err.message || err.name}).`;
}

// --------------------------------------------------- silence-based auto-stop (continuous mode)
// `onLevel(level, threshold)` is called every frame if given, purely to drive the UI meter - it
// never affects the detection logic itself.
function startSilenceWatch(stream, onTrigger, onLevel) {
  stopSilenceWatch();
  const audioCtx = getAudioCtx();
  if (!audioCtx) return false;
  audioCtx.resume().catch(() => {}); // some browsers create it suspended until explicitly resumed
  const source = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);
  const buf = new Float32Array(analyser.fftSize);
  const state = {
    audioCtx,
    source,
    analyser,
    rafId: 0,
    stopped: false,
    startedAt: performance.now(),
    heardSpeech: false,
    lastLoudAt: performance.now(),
    calibrating: true,
    ambientSamples: [],
    threshold: null,
    lastRelaxAt: 0,
    lastLogAt: 0,
  };
  silenceWatch = state;

  function rms() {
    analyser.getFloatTimeDomainData(buf);
    let sumSquares = 0;
    for (let i = 0; i < buf.length; i++) sumSquares += buf[i] * buf[i];
    return Math.sqrt(sumSquares / buf.length);
  }

  function tick() {
    if (state.stopped) return;
    const level = rms();
    const now = performance.now();

    if (state.calibrating) {
      state.ambientSamples.push(level);
      if (now - state.startedAt >= CALIBRATION_MS) {
        const ambient = state.ambientSamples.reduce((a, b) => a + b, 0) / state.ambientSamples.length;
        state.threshold = Math.max(MIN_SPEECH_THRESHOLD, ambient * SPEECH_LEVEL_MULTIPLIER);
        state.calibrating = false;
        state.lastLoudAt = now; // calibration time must not count against the silence timer
        state.lastRelaxAt = now;
        console.debug(`[voice] mic calibrated: ambient=${ambient.toFixed(4)} threshold=${state.threshold.toFixed(4)}`);
      }
      if (onLevel) onLevel(level, null);
      state.rafId = requestAnimationFrame(tick);
      return;
    }

    if (onLevel) onLevel(level, state.threshold);
    if (level > state.threshold) {
      state.heardSpeech = true;
      state.lastLoudAt = now;
    } else if (!state.heardSpeech && now - state.lastRelaxAt > RELAX_INTERVAL_MS) {
      // Still nothing detected after a while - lower the bar rather than staying stuck too
      // strict, e.g. for a mic with automatic gain control that keeps levels nearly constant.
      state.threshold = Math.max(MIN_SPEECH_THRESHOLD, state.threshold * RELAX_FACTOR);
      state.lastRelaxAt = now;
      console.debug(`[voice] no speech detected yet, relaxing threshold to ${state.threshold.toFixed(4)}`);
    }
    if (now - state.lastLogAt > 1000) {
      state.lastLogAt = now;
      console.debug(`[voice] level=${level.toFixed(4)} threshold=${state.threshold.toFixed(4)} heardSpeech=${state.heardSpeech}`);
    }
    const timedOut = now - state.startedAt > MAX_LISTEN_MS;
    const wentSilentAfterSpeech = state.heardSpeech && now - state.lastLoudAt > SILENCE_MS;
    if (timedOut || wentSilentAfterSpeech) {
      console.debug(`[voice] stopping: ${timedOut ? "hit MAX_LISTEN_MS" : "silence after speech"}`);
      onTrigger();
      return;
    }
    state.rafId = requestAnimationFrame(tick);
  }
  state.rafId = requestAnimationFrame(tick);
  return true;
}

function stopSilenceWatch() {
  if (!silenceWatch) return;
  silenceWatch.stopped = true;
  cancelAnimationFrame(silenceWatch.rafId);
  // Disconnect this listen's nodes but keep the shared context alive (see getAudioCtx).
  try {
    silenceWatch.source.disconnect();
    silenceWatch.analyser.disconnect();
  } catch {
    // already disconnected
  }
  silenceWatch = null;
}

// A phonetic key for wake-word spellings: Whisper writes the same sound many ways, so "zyra", "zeera",
// "zera", "xera", "xero", "sira" and "zirah" all reduce to "zira" (a leading s/z/x merged, runs of
// y/i/e collapsed to one i, a trailing h dropped, a final e/o turned into a). Other names such as
// "sarah", "kira" and "zara" keep a different key and do not match.
function wakePhoneticKey(word) {
  return word.replace(/[^a-z]/g, "").replace(/^[szx]/, "z").replace(/[yie]+/g, "i").replace(/h$/, "").replace(/[eo]$/, "a");
}

// `prev` is the normalized word before this one, or undefined at the start of the clip. An exact
// spelling counts anywhere in the first few words; a merely phonetically-similar spelling counts only
// at the very start or right after a greeting, which keeps false wake-ups low.
function isWakeToken(word, prev) {
  if (word === WAKE_WORD || WAKE_WORD_BACKUP_ALIASES.has(word) || WAKE_WORD_DEVANAGARI.has(word)) return true;
  if (prev !== undefined && !GREETING_WORD.test(prev)) return false;
  if (prev === undefined && WAKE_WORD_NEEDS_GREETING.has(word)) return false;
  return wakePhoneticKey(word) === WAKE_WORD;
}

// ------------------------------------------------------------- wake word (continuous mode only)
// Returns the command text after the wake word, "" if only the wake word was heard (keep
// listening for the actual command), or null if no wake word was heard at all (ignore this clip).
function extractCommand(rawText) {
  const words = rawText.trim().split(/\s+/).filter(Boolean);
  if (!words.length) return null;
  // \p{M} (combining marks) must be kept alongside \p{L} - found for real via
  // scripts/voice_benchmark.py: stripping only \p{L} silently ate Devanagari matras (vowel signs
  // are Marks, not Letters), corrupting "जारवेस" into "जरवस" and breaking the Devanagari wake-word
  // match below before it could ever run.
  const normalized = words.map((w) => w.toLowerCase().replace(/[^\p{L}\p{M}\p{N}']/gu, ""));
  const idx = normalized.slice(0, 5).findIndex((w, i) => isWakeToken(w, i === 0 ? undefined : normalized[i - 1]));
  if (idx === -1) return null;
  const rest = words.slice(idx + 1);
  while (rest.length && WAKE_WORD_FILLER.test(rest[0])) rest.shift();
  return rest.join(" ").trim();
}

// Drives the live mic-level meter. `threshold` is null during the brief calibration phase (the
// bar is shown but the marker/color aren't meaningful yet). Level/threshold are RMS amplitudes
// roughly in [0, 0.3] for normal mic input, scaled here against a fixed visual ceiling.
const METER_VISUAL_CEILING = 0.15;

function updateLevelMeter(level, threshold) {
  const pct = Math.max(0, Math.min(100, (level / METER_VISUAL_CEILING) * 100));
  els.voiceLevelFill.style.width = `${pct}%`;
  els.voiceLevelFill.classList.toggle("above-threshold", threshold !== null && level > threshold);
  if (threshold !== null) {
    els.voiceLevelThreshold.style.left = `${Math.max(0, Math.min(100, (threshold / METER_VISUAL_CEILING) * 100))}%`;
  }
}

// ---------------------------------------------------------------------- recording lifecycle
async function startListening() {
  if (!stopListen) stopSpeech("interrupted"); // a streamed reply: drop its queued sentences and stop Qwen writing it
  if (currentAudio) {
    currentAudio.pause(); // interrupt: starting to listen again stops any reply currently playing
    currentAudio = null;
  }
  clearError();
  els.voiceTranscript.textContent = continuousMode
    ? sessionAwake
      ? "Listening… (or click mic when done talking)"
      : 'Say "Hello Zira"… (or click mic when done talking)'
    : "";
  let stream;
  try {
    stream = await getMicStream();
  } catch (err) {
    showError(micErrorMessage(err));
    setVoiceState("ready");
    return;
  }
  recordedChunks = [];
  const mimeType = pickMimeType();
  try {
    recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  } catch (err) {
    showError(`Could not start recording (${err.message || err.name}).`);
    setVoiceState("ready");
    return;
  }
  recorder.addEventListener("dataavailable", (e) => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  });
  turnStartedAt = performance.now();
  recorder.start();
  setVoiceState("listening");

  if (continuousMode) {
    els.voiceLevel.hidden = false;
    const watching = startSilenceWatch(stream, () => stopListeningAndProcess(), updateLevelMeter);
    if (!watching) {
      // No Web Audio API available: fall back to a fixed listening window instead of silence
      // detection, so continuous mode still terminates a turn rather than listening forever.
      continuousSafetyTimer = setTimeout(() => stopListeningAndProcess(), MAX_LISTEN_MS);
    }
  }
}

function teardownListening() {
  stopSilenceWatch();
  els.voiceLevel.hidden = true;
  if (continuousSafetyTimer) {
    clearTimeout(continuousSafetyTimer);
    continuousSafetyTimer = null;
  }
}

function stopRecorder() {
  return new Promise((resolve) => {
    if (!recorder || recorder.state === "inactive") return resolve(null);
    recorder.addEventListener("stop", () => resolve(new Blob(recordedChunks, { type: recorder.mimeType || "audio/webm" })), { once: true });
    recorder.stop();
  });
}

function relistenIfContinuous() {
  if (continuousMode && voiceState === "ready") startListening();
}

// Call after a listen cycle that captured no usable speech, with how long that cycle actually
// listened for. Repeated calls accumulate; once the total reaches AWAKE_GRACE_MS, the hands-free
// conversation is considered over and the next utterance will need "Hello JARVIS" again.
function trackIdleListening(ms) {
  silentListenMs += ms;
  if (sessionAwake && silentListenMs >= AWAKE_GRACE_MS) {
    sessionAwake = false;
    console.debug(`[voice] no speech for ~${(silentListenMs / 1000).toFixed(1)}s - wake word required again`);
  }
}

// Saying "stop" while an image or video is being made: the voice state is "thinking" then (the reply is waiting
// on the tool), so the mic is normally idle. Pressing it now records one clip that can only stop the
// generation - anything else is shown and dropped, and the voice state goes back to waiting for the reply.
function canListenForStop() {
  return !!stoppable && voiceState === "thinking";
}

function listenForStop() {
  stopListen = true;
  startListening();
  els.voiceTranscript.textContent = `Say "stop" to stop the ${stoppable}… (release or click mic when done)`;
}

function endStopListen(text) {
  stopListen = false;
  const heard = text.trim();
  if (heard && STOP_WORDS.test(heard) && stoppable) {
    els.voiceTranscript.textContent = `Heard "${heard}" - stopping the ${stoppable}.`;
    stopGeneration();
  } else if (heard) {
    els.voiceTranscript.textContent = `Heard: "${heard.slice(0, 80)}" - say "stop" to stop it.`;
  } else {
    els.voiceTranscript.textContent = "";
  }
  if (voiceState !== "transcribing") return; // the reply finished meanwhile and its own handler took over
  setVoiceState(currentTurn ? "thinking" : "ready");
  if (!currentTurn) relistenIfContinuous();
}

// The image/video ended (finished or stopped) while the mic was open for "stop": drop the clip.
function abortStopListen() {
  stopListen = false;
  if (voiceState !== "listening") return; // already transcribing: endStopListen settles it
  teardownListening();
  stopRecorder();
  els.voiceTranscript.textContent = "";
  setVoiceState(currentTurn ? "thinking" : "ready");
}

async function stopListeningAndProcess() {
  if (voiceState !== "listening") return;
  const forStop = stopListen;
  // Captured before teardown clears silenceWatch - distinguishes "the amplitude watch genuinely
  // detected your voice, but the clip still failed to transcribe" (worth telling you about) from
  // ordinary idle relistening that only ever heard room silence (not worth mentioning every cycle).
  const heardSpeech = silenceWatch ? silenceWatch.heardSpeech : false;
  const wakeListening = continuousMode && !sessionAwake;
  teardownListening();
  const recordEndedAt = performance.now();
  const blob = await stopRecorder();
  setVoiceState("transcribing");
  if (forStop && (!blob || blob.size < MIN_CLIP_BYTES)) {
    endStopListen("");
    return;
  }
  if (!blob || blob.size < MIN_CLIP_BYTES) {
    if (continuousMode) trackIdleListening(recordEndedAt - turnStartedAt);
    setVoiceState("ready");
    relistenIfContinuous();
    return;
  }

  let text = "";
  try {
    const form = new FormData();
    form.append("audio", blob, `clip.${extensionForMime(recorder.mimeType || "")}`);
    // Only a clip recorded while waiting for the wake word asks the server to prime the speech model
    // with it; push-to-talk and follow-ups inside an active conversation are transcribed plainly, so
    // a spoken name like "Sarah" in a real command is never bent toward the wake word.
    const res = await fetch(wakeListening ? "/api/voice/transcribe?wake=true" : "/api/voice/transcribe", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Could not transcribe that.");
    text = body.text;
    console.debug(`[voice] mic->transcript: ${(performance.now() - recordEndedAt).toFixed(0)}ms (server STT: ${body.duration_ms}ms)`);
  } catch (err) {
    // A real failure (network/server): stop and let the user retry manually rather than
    // auto-relistening into a possibly-repeating error.
    showError(err.message);
    if (forStop) endStopListen("");
    else setVoiceState("ready");
    return;
  }

  if (forStop && !text.trim()) {
    endStopListen("");
    return;
  }
  if (!text.trim()) {
    if (continuousMode) trackIdleListening(recordEndedAt - turnStartedAt);
    setVoiceState("ready");
    if (continuousMode && heardSpeech) {
      // Real speech was detected (not just ambient noise during idle relistening) but transcription
      // still came back empty - a genuine miss, not silence indistinguishable from "nothing
      // happened". Relisten is delayed briefly so this message is actually readable instead of
      // being overwritten instantly by the next "Listening…" from relistenIfContinuous().
      els.voiceTranscript.textContent = "Didn't catch that — try again.";
      setTimeout(relistenIfContinuous, 1500);
    } else {
      els.voiceTranscript.textContent = continuousMode ? "" : "Didn't catch that — try again.";
      relistenIfContinuous();
    }
    return;
  }

  if (forStop) {
    endStopListen(text);
    return;
  }

  let command = text;
  lastVoiceActivityAt = Date.now(); // someone is talking - never interrupt with an opener right now
  if (continuousMode) {
    silentListenMs = 0; // real speech was transcribed this cycle, whatever it turns out to say
    const extracted = extractCommand(text);
    if (extracted !== null) {
      sessionAwake = true; // wake word heard ("Hello Zira") - (re)start the no-wake-word grace window
      command = extracted;
    } else if (sessionAwake) {
      // No wake word, but we're still within AWAKE_GRACE_MS of the last exchange - no need to
      // repeat "Hello JARVIS" for a follow-up question.
      command = text.trim();
    } else {
      console.debug(`[voice] no wake word heard in "${text}" and not in an active conversation - ignoring`);
      // Shown (not only logged) so a wake word that Whisper spelled unexpectedly is visible and can
      // be reported; the delay keeps it readable before "Listening…" replaces it.
      els.voiceTranscript.textContent = `Heard: "${text.trim().slice(0, 80)}" - say "Hello Zira"`;
      setVoiceState("ready");
      setTimeout(relistenIfContinuous, 1500);
      return;
    }
    if (!command) {
      // Only the wake word was heard ("Hello JARVIS" with nothing else yet, no actual request) -
      // greet back like a person would instead of silently sitting there waiting. Sends the heard
      // greeting itself (e.g. "Hello JARVIS") through the normal chat pipeline, same as any other
      // command - the system prompt has specific guidance for replying to a bare greeting.
      command = text.trim();
    }
  }

  // Detected before sending, applied after the reply so JARVIS still gets to say goodbye back -
  // but the wake word is required again right after, not after the usual 10s grace window.
  const isFarewell = continuousMode && FAREWELL_PATTERN.test(command);

  if (stoppable && STOP_WORDS.test(command)) {
    els.voiceTranscript.textContent = `Heard "${command.trim()}" - stopping the ${stoppable}.`;
    stopGeneration();
    setVoiceState("ready");
    return;
  }

  els.voiceTranscript.textContent = "";
  setVoiceState("thinking");
  const chatStartedAt = performance.now();
  sendMessage(command, {
    source: "voice",
    speak: !!(voiceStatus.tts_available && voiceStatus.tts_streaming), // spoken while it is being written
    onDone: async (turn) => {
      console.debug(`[voice] chat response: ${(performance.now() - chatStartedAt).toFixed(0)}ms`);
      if (isFarewell) {
        requireWakeWordAgain();
        console.debug('[voice] farewell heard - wake word required again immediately');
      }
      if (turn.speech) {
        // Streamed: it has been speaking since its first sentence; wait for the last one (or an interruption).
        try {
          await turn.speech.finished;
        } finally {
          console.debug(`[voice] total turn: ${(performance.now() - turnStartedAt).toFixed(0)}ms`);
          if (!turn.speech.stopped) {
            setVoiceState("ready");
            relistenIfContinuous();
          }
        }
        return;
      }
      const reply = (turn.bubble.dataset.raw || "").trim();
      if (!reply || !voiceStatus.tts_available) {
        setVoiceState("ready");
        relistenIfContinuous();
        return;
      }
      setVoiceState("speaking");
      try {
        await speakText(reply);
      } catch (err) {
        showError(err.message);
      } finally {
        console.debug(`[voice] total turn: ${(performance.now() - turnStartedAt).toFixed(0)}ms`);
        setVoiceState("ready");
        relistenIfContinuous();
      }
    },
  });
}

// ------------------------------------------------------------------ streamed speech (Kokoro)
// A voice reply is spoken sentence by sentence while Qwen is still writing it: the server sends each sentence's
// audio on the chat WebSocket ("audio" events, then "speech_end"), and each clip plays right after the previous
// one on speechEl - the element a tap already unlocked (see "audio that works on an iPhone"). Starting the mic,
// or a new request, stops it: queued sentences are dropped and the server is told to stop generating.
let activeSpeech = null;

class SpeechPlayer {
  constructor(startedAt) {
    this.queue = [];
    this.playing = false;
    this.ended = false;
    this.stopped = false;
    this.done = false;
    this.startedAt = startedAt;
    this.played = 0;
    this.errors = [];
    this.firstPlaybackMs = null;
    this.finished = new Promise((resolve) => { this._resolve = resolve; });
  }

  enqueue(data) {
    if (this.stopped || this.done) return;
    const bytes = Uint8Array.from(atob(data.audio), (c) => c.charCodeAt(0));
    this.queue.push({ seq: data.seq, url: URL.createObjectURL(new Blob([bytes], { type: data.mime || "audio/wav" })) });
    this.queue.sort((a, b) => a.seq - b.seq);
    if (!this.playing) this._next();
  }

  end() {
    this.ended = true;
    if (!this.playing && this.queue.length === 0) this._finish();
  }

  stop() {
    if (this.stopped || this.done) return;
    this.stopped = true;
    for (const item of this.queue) URL.revokeObjectURL(item.url);
    this.queue = [];
    if (currentAudio === speechEl) {
      speechEl.pause();
      currentAudio = null;
    }
    this._finish();
  }

  _next() {
    if (this.stopped) return;
    const item = this.queue.shift();
    if (!item) {
      this.playing = false;
      if (this.ended) this._finish();
      return;
    }
    this.playing = true;
    const audio = speechEl;
    let moved = false;
    const next = () => {
      if (moved) return;
      moved = true;
      audio.onended = null;
      audio.onerror = null;
      URL.revokeObjectURL(item.url);
      this._next();
    };
    audio.onended = next;
    audio.onerror = next;
    releaseMicForPlayback();
    audio.src = item.url;
    currentAudio = audio;
    setVoiceState("speaking");
    audio.play().then(() => {
      this.played += 1;
      if (this.firstPlaybackMs === null) {
        this.firstPlaybackMs = Math.round(performance.now() - this.startedAt);
        console.debug(`[AUDIO] first playback: ${this.firstPlaybackMs}ms after the request`);
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "speech_metrics", first_playback_ms: this.firstPlaybackMs, chunks_played: 1 }));
        }
      }
    }).catch(next);
  }

  _finish() {
    if (this.done) return;
    this.done = true;
    if (!this.stopped && currentAudio === speechEl) currentAudio = null;
    if (!this.stopped && this.played === 0 && this.errors.length) showError(`Could not speak the reply: ${this.errors[0]}`);
    if (activeSpeech === this) activeSpeech = null;
    this._resolve();
  }
}

// Stops the reply being spoken (and, if Qwen is still writing it, its generation) - the user interrupted.
function stopSpeech(reason) {
  const speech = activeSpeech;
  if (!speech) return;
  speech.stop();
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop", reason }));
}

// Speaks `text` with the current reply voice and resolves when playback ends (or fails to start).
async function speakText(text) {
  const speakStartedAt = performance.now();
  const res = await fetch("/api/voice/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Could not speak the reply.");
  }
  const url = URL.createObjectURL(await res.blob());
  // speechEl, not `new Audio(url)`: see "audio that works on an iPhone" above. Property handlers
  // (onended/onerror), not addEventListener, since the element is reused every turn and listeners
  // would otherwise pile up on it.
  const audio = speechEl;
  releaseMicForPlayback();
  audio.src = url;
  currentAudio = audio;
  console.debug(`[voice] tts: ${(performance.now() - speakStartedAt).toFixed(0)}ms`);
  try {
    await new Promise((resolve) => {
      audio.onended = resolve;
      audio.onerror = resolve;
      audio.play().catch(resolve);
    });
  } finally {
    audio.onended = null;
    audio.onerror = null;
    URL.revokeObjectURL(url);
    if (currentAudio === audio) currentAudio = null;
  }
}

// ------------------------------------------------------------------ Zira talks first
// In hands-free mode, after a random quiet spell (capabilities.proactive: 4-10 minutes by default)
// Zira starts the conversation herself: a question to learn about you, a compliment, or a fun point
// about one of your interests (written by the server, POST /api/chat/proactive). Your spoken answer
// is an ordinary turn, so it is remembered like anything else you tell her. She only does this while
// the mic is idly listening (never mid-sentence, while busy, or while music plays), and stops after
// `max_unanswered` openers in a row get no answer, until you speak again.
let lastVoiceActivityAt = Date.now();
let proactiveUnanswered = 0;
let nextProactiveMs = null;
let proactiveRunning = false;

function noteUserActivity() {
  lastVoiceActivityAt = Date.now();
  proactiveUnanswered = 0;
  nextProactiveMs = null;
}

function pickProactiveDelay() {
  const p = capabilities.proactive;
  const minutes = p.min_minutes + Math.random() * Math.max(0, p.max_minutes - p.min_minutes);
  return minutes * 60000 * (proactiveUnanswered > 0 ? 2 : 1); // wait longer after an unanswered one
}

function proactiveAllowed() {
  const p = capabilities && capabilities.proactive;
  return (
    p && continuousMode && voiceEnabled && voiceStatus.stt_available && voiceStatus.tts_available &&
    document.visibilityState === "visible" && !busy && !proactiveRunning &&
    voiceState === "listening" && !(silenceWatch && silenceWatch.heardSpeech) &&
    !(currentMusicAudio && !currentMusicAudio.paused) && proactiveUnanswered < p.max_unanswered
  );
}

async function maybeStartProactive() {
  if (!proactiveAllowed()) return;
  if (nextProactiveMs === null) nextProactiveMs = pickProactiveDelay();
  if (Date.now() - lastVoiceActivityAt < nextProactiveMs) return;

  proactiveRunning = true;
  teardownListening();
  await stopRecorder(); // this idle clip is discarded, nothing was said in it
  setVoiceState("thinking");
  try {
    const res = await fetch("/api/chat/proactive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversationId }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Could not start a conversation.");
    setConversation(body.conversation_id);
    addMessage("assistant", body.text);
    setVoiceState("speaking");
    await speakText(body.text);
    proactiveUnanswered += 1;
    // Answer without "Hello Zira": the grace window starts extra wide, since you may need a moment.
    sessionAwake = true;
    silentListenMs = -AWAKE_GRACE_MS;
  } catch (err) {
    console.debug(`[voice] proactive opener skipped: ${err.message}`);
  } finally {
    lastVoiceActivityAt = Date.now();
    nextProactiveMs = null;
    proactiveRunning = false;
    setVoiceState("ready");
    relistenIfContinuous();
  }
}

setInterval(maybeStartProactive, 15000);

// Push-to-talk: press and hold (pointerdown/up). Continuous mode: a single click toggles
// listening on/off, since nothing is being held down in hands-free use.
els.voiceBtn.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  if (continuousMode) return;
  if (voiceState === "ready" || voiceState === "speaking") startListening();
  else if (canListenForStop()) listenForStop();
});
["pointerup", "pointerleave", "pointercancel"].forEach((type) => {
  els.voiceBtn.addEventListener(type, () => {
    if (continuousMode) return;
    if (voiceState === "listening") stopListeningAndProcess();
  });
});
els.voiceBtn.addEventListener("click", (event) => {
  if (!continuousMode) return;
  event.preventDefault();
  if (voiceState === "listening") stopListeningAndProcess();
  else if (voiceState === "ready" || voiceState === "speaking") startListening();
  else if (canListenForStop()) listenForStop();
});

// Keep the screen on while hands-free is on: an iPhone that auto-locks suspends the page and the mic,
// which silently ends the conversation. Feature-detected (Safari 16.4+); failure only means the
// screen may still sleep.
let wakeLock = null;
async function acquireWakeLock() {
  if (wakeLock || !("wakeLock" in navigator)) return;
  try {
    wakeLock = await navigator.wakeLock.request("screen");
    wakeLock.addEventListener("release", () => { wakeLock = null; });
  } catch (err) {
    console.debug("[voice] screen wake lock unavailable", err);
  }
}
function releaseWakeLock() {
  if (!wakeLock) return;
  wakeLock.release().catch(() => {});
  wakeLock = null;
}

els.voiceContinuous.addEventListener("change", () => {
  continuousMode = els.voiceContinuous.checked;
  if (continuousMode) noteUserActivity();
  if (continuousMode) acquireWakeLock();
  else releaseWakeLock();
  localStorage.setItem(CONTINUOUS_KEY, continuousMode ? "1" : "0");
  updateVoiceBtnTitle();
  if (!continuousMode) {
    // Leave hands-free mode: abort whatever is in flight rather than submitting a partial clip.
    teardownListening();
    if (recorder && recorder.state !== "inactive") recorder.stop();
    if (voiceState === "listening") setVoiceState("ready");
    sessionAwake = false;
    silentListenMs = 0;
  } else if (voiceState === "ready") {
    startListening(); // the checkbox click is itself a user gesture, so this may start immediately
  }
});

// The OS drops a wake lock whenever the page is hidden, and may have suspended the audio context and
// ended the mic track meanwhile: on return, take the lock again, and if we still think we are
// listening but the recorder is not actually recording, start that listen over cleanly.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible" || !continuousMode) return;
  acquireWakeLock();
  resumeSharedAudioCtx();
  if (voiceState === "listening" && (!recorder || recorder.state !== "recording")) {
    teardownListening();
    setVoiceState("ready");
    startListening();
  }
});

// ------------------------------------------------------------------- history
async function loadHistory() {
  if (!conversationId) return;
  try {
    const res = await fetch(`/api/conversations/${encodeURIComponent(conversationId)}/messages`);
    if (!res.ok) return;
    const messages = await res.json();
    for (const m of messages) addMessage(m.role, m.content);
  } catch {
    /* history is a convenience; the status pill already reports server problems */
  }
}

// -------------------------------------------------------------------- events
els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = els.input.value.trim();
  if (stoppable && (!text || STOP_WORDS.test(text))) {
    els.input.value = "";
    autoGrow();
    stopGeneration();
    return;
  }
  if (!text || busy) return;
  els.input.value = "";
  autoGrow();
  const tag = attachedImage && (attachedImage.kind === "document" ? `[Document: ${attachedImage.id}]` : `[Uploaded image: ${attachedImage.id}]`);
  const message = tag ? `${tag} ${text}` : text;
  clearAttachedImage();
  sendMessage(message);
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});

function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 160)}px`;
}
els.input.addEventListener("input", () => { autoGrow(); updateSendButton(); });

function clearChatView() {
  clearError();
  els.messages.querySelectorAll(".msg, .memory-note, .change-card, .plan-bar, .turn-timing").forEach((node) => node.remove());
  changeCards.clear();
  lastCompleted = null;
  els.empty.hidden = false;
}

els.newChat.addEventListener("click", () => {
  if (busy) return;
  setConversation(null);
  clearChatView();
  els.input.focus();
});

els.errorClose.addEventListener("click", clearError);
els.mode.addEventListener("change", () => applyMode(els.mode.value));

on(els.lightboxClose, "click", closeLightbox);
on(els.lightbox, "click", (event) => {
  if (event.target === els.lightbox) closeLightbox(); // click on the backdrop, not the image/buttons
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!els.lightbox.hidden) closeLightbox();
  else if (els.gallery && !els.gallery.hidden) (els.galleryDetail.hidden ? closeGallery() : showGalleryGrid());
  else if (els.library && !els.library.hidden) closeLibrary();
});

// ------------------------------------------------------------------- gallery
// Every image and video Zira made (GET /api/media), newest first, as thumbnails; tap one for the full video or
// image, what was asked, the prompt the model wrote, and Make again / Edit request / Edit this image / Delete.
const GALLERY_PAGE = 48;
const MODEL_NAMES = {
  fastmetal: "FastMetal 1.3B", fastmetal5b: "FastMetal 5B", hunyuan: "HunyuanVideo 1.5", ltx: "LTX 2B", wan: "Wan 2.1 1.3B",
  lightning: "Lightning", realistic: "Realistic", realvis5: "RealVisXL 5.0", flux: "FLUX",
};
const SOURCE_NAMES = { text: "from a request", edit: "an edited image", photo: "an animated photo", extend: "a longer version", face: "from a face photo", earlier: "made before the gallery existed" };
const gallery = { kind: "", items: [], before: null, current: null, loading: false };

function openGallery() {
  els.gallery.hidden = false;
  document.body.classList.add("gallery-open");
  showGalleryGrid();
  loadGallery(true);
}

function closeGallery() {
  els.galleryMedia.replaceChildren(); // stops a playing video
  els.gallery.hidden = true;
  document.body.classList.remove("gallery-open");
}

async function loadGallery(reset) {
  if (gallery.loading) return;
  gallery.loading = true;
  if (reset) {
    gallery.items = [];
    gallery.before = null;
    els.galleryGrid.replaceChildren();
  }
  const params = new URLSearchParams({ limit: String(GALLERY_PAGE) });
  if (gallery.kind) params.set("kind", gallery.kind);
  if (gallery.before != null) params.set("before", String(gallery.before));
  try {
    const res = await fetch(`/api/media?${params}`);
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Could not load the gallery.");
    for (const item of body.items) {
      gallery.items.push(item);
      els.galleryGrid.appendChild(galleryTile(item));
    }
    if (body.items.length) gallery.before = body.items[body.items.length - 1].id;
    els.galleryMore.hidden = body.items.length < GALLERY_PAGE;
    els.galleryEmpty.hidden = gallery.items.length > 0;
  } catch (err) {
    showError(err.message);
  } finally {
    gallery.loading = false;
  }
}

function galleryTile(item) {
  const tile = document.createElement("button");
  tile.type = "button";
  tile.className = "gallery-tile";
  tile.dataset.filename = item.filename;
  tile.title = item.request || item.prompt || item.filename;
  if (item.kind === "audio") {
    // A song has no picture: a note and its title instead of a thumbnail.
    tile.classList.add("gallery-audio");
    const note = document.createElement("span");
    note.className = "gallery-noimg";
    note.textContent = "🎵";
    const name = document.createElement("span");
    name.className = "gallery-audio-title";
    name.textContent = item.filename.replace(/\.[^.]+$/, "").replace(/[-_]+/g, " ");
    tile.append(note, name);
    if (item.seconds) {
      const badge = document.createElement("span");
      badge.className = "gallery-badge";
      badge.textContent = `♪ ${Math.round(item.seconds)}s`;
      tile.appendChild(badge);
    }
    tile.addEventListener("click", () => showGalleryItem(item));
    return tile;
  }
  const img = document.createElement("img");
  img.loading = "lazy";
  img.alt = tile.title;
  img.src = item.thumb;
  img.addEventListener("error", () => {
    const placeholder = document.createElement("span");
    placeholder.className = "gallery-noimg";
    placeholder.textContent = item.kind === "video" ? "▶" : "🖼";
    img.replaceWith(placeholder);
  });
  tile.appendChild(img);
  if (item.kind === "video") {
    const badge = document.createElement("span");
    badge.className = "gallery-badge";
    badge.textContent = item.seconds ? `▶ ${Math.round(item.seconds)}s` : "▶";
    tile.appendChild(badge);
  }
  tile.addEventListener("click", () => showGalleryItem(item));
  return tile;
}

function galleryInfoRow(label, value) {
  if (!value) return;
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  els.galleryInfo.append(dt, dd);
}

function showGalleryItem(item) {
  gallery.current = item;
  for (const el of [els.galleryGrid, els.galleryMore, els.galleryEmpty, els.galleryTabs]) el.hidden = true;
  els.galleryBack.hidden = false;
  els.galleryDetail.hidden = false;
  let media;
  if (item.kind === "video") {
    media = document.createElement("video");
    Object.assign(media, { src: item.url, controls: true, playsInline: true, loop: true, preload: "metadata" });
    media.poster = item.thumb;
  } else if (item.kind === "audio") {
    media = document.createElement("audio");
    Object.assign(media, { src: item.url, controls: true, preload: "metadata" });
    media.className = "audio-preview";
  } else {
    media = document.createElement("img");
    media.src = item.url;
    media.alt = item.request || item.prompt || item.filename;
  }
  els.galleryMedia.replaceChildren(media);
  els.galleryInfo.replaceChildren();
  galleryInfoRow("You asked", (item.request || "").replace(/^\[Uploaded image: [^\]]+\]\s*/, ""));
  galleryInfoRow(item.kind === "audio" ? "Style" : "Prompt", item.prompt !== item.request ? item.prompt : "");
  galleryInfoRow("Model", MODEL_NAMES[item.model] || item.model);
  galleryInfoRow("Size", item.width && item.height ? `${item.width}×${item.height}` : "");
  galleryInfoRow("Length", item.seconds ? `${item.seconds.toFixed(1)} s` : "");
  galleryInfoRow("Made", `${new Date(item.created_at).toLocaleString()} - ${SOURCE_NAMES[item.source] || item.source}`);
  els.galleryAgain.disabled = els.galleryEditRequest.disabled = !(item.request || item.prompt);
  els.galleryEditImage.hidden = els.galleryAnimate.hidden = item.kind !== "image";
  els.galleryLonger.hidden = item.kind !== "video";
  els.galleryDownload.href = item.url;
  els.galleryDownload.setAttribute("download", item.filename);
  els.gallery.scrollTop = 0;
}

function showGalleryGrid() {
  els.galleryMedia.replaceChildren();
  els.galleryDetail.hidden = true;
  els.galleryBack.hidden = true;
  els.galleryGrid.hidden = false;
  els.galleryTabs.hidden = false;
  els.galleryEmpty.hidden = gallery.items.length > 0 || gallery.loading;
  els.galleryMore.hidden = gallery.items.length === 0 || gallery.items.length % GALLERY_PAGE !== 0;
}

function useGalleryRequest(send) {
  const item = gallery.current;
  const text = item && (item.request || item.prompt);
  if (!text) return;
  closeGallery();
  els.input.value = text;
  autoGrow();
  els.input.focus();
  if (send) els.form.requestSubmit(); // while Zira is busy this just leaves the request in the box
}

on(els.galleryBtn, "click", openGallery);

// ⟲ Reset (app/api/system.py): stop what is being made, image/video models to None (memory freed), chat model back.
on(els.resetBtn, "click", async () => {
  if (!window.confirm("Reset Zira?\n\nStops anything being made (image, video, song), unloads the image and video models to free memory, and loads the chat model again. Nothing is deleted.")) return;
  const label = els.resetBtn.textContent;
  els.resetBtn.disabled = true;
  els.resetBtn.textContent = "…";
  try {
    const res = await fetch("/api/system/reset", { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || "Reset failed.");
    if (body.image_style && imageStyle !== null) applyImageStyle(body.image_style);
    if (body.video_model && videoModel !== null) applyVideoModel(body.video_model);
    if (body.ok) showNotice(body.detail);
    else showError(body.detail);
  } catch (err) {
    showError(err.message);
  } finally {
    els.resetBtn.disabled = false;
    els.resetBtn.textContent = label;
  }
});

// ---------------------------------------------------------------- LoRA Studio
// Starts the separate LoRA Studio app on the Mac if it isn't running (app/api/lora_studio.py) and
// opens it in a new tab through Zira, so it works from the phone too.
async function openLoraStudio() {
  const tab = window.open("", "_blank"); // opened inside the tap itself, or Safari blocks it as a popup
  if (tab) tab.document.body.textContent = "Starting LoRA Studio…";
  els.loraBtn.disabled = true;
  try {
    const res = await fetch("/api/lora-studio/start", { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `LoRA Studio did not start (HTTP ${res.status}).`);
    if (tab) tab.location.href = body.url;
    else window.location.href = body.url;
  } catch (err) {
    if (tab) tab.close();
    showError(err.message);
  } finally {
    els.loraBtn.disabled = false;
  }
}
on(els.loraBtn, "click", openLoraStudio);
on(els.galleryClose, "click", closeGallery);
on(els.galleryBack, "click", showGalleryGrid);
on(els.galleryMore, "click", () => loadGallery(false));
on(els.galleryAgain, "click", () => useGalleryRequest(true));
on(els.galleryEditRequest, "click", () => useGalleryRequest(false));
if (els.galleryTabs) {
  for (const tab of els.galleryTabs.querySelectorAll("button[data-kind]")) {
    tab.addEventListener("click", () => {
      gallery.kind = tab.dataset.kind;
      for (const other of els.galleryTabs.querySelectorAll("button")) other.classList.toggle("active", other === tab);
      loadGallery(true);
    });
  }
}
// The gallery image as an attachment ("[Uploaded image: <id>]" on the next message), then `text` in the box.
async function attachGalleryImage(text) {
  const item = gallery.current;
  try {
    const res = await fetch(`/api/media/${encodeURIComponent(item.filename)}/to-upload`, { method: "POST" });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Could not use that image.");
    attachedImage = { id: body.image_id, name: item.filename };
    els.imageAttachedName.textContent = `📎 ${item.filename}`;
    els.imageAttached.hidden = false;
    closeGallery();
    els.input.value = text;
    autoGrow();
    els.input.focus();
  } catch (err) {
    showError(err.message);
  }
}

on(els.galleryEditImage, "click", () => attachGalleryImage(""));
on(els.galleryAnimate, "click", () => attachGalleryImage("Animate this photo: "));
on(els.galleryLonger, "click", () => {
  const item = gallery.current;
  closeGallery();
  els.input.value = `[Video: ${item.filename}] Make this video 5 seconds longer`;
  autoGrow();
  els.input.focus();
});
on(els.galleryDelete, "click", async () => {
  const item = gallery.current;
  if (!item || !window.confirm(`Delete this ${item.kind}? This cannot be undone.`)) return;
  try {
    const res = await fetch(`/api/media/${encodeURIComponent(item.filename)}`, { method: "DELETE" });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Could not delete it.");
    gallery.items = gallery.items.filter((i) => i !== item);
    els.galleryGrid.querySelector(`[data-filename="${CSS.escape(item.filename)}"]`)?.remove();
    showGalleryGrid();
  } catch (err) {
    showError(err.message);
  }
});

// ------------------------------------------------------------------- chats and memory
// Every conversation (GET /api/conversations: most recent first, searchable; tap one to reopen it) and what
// Zira remembers (the memory page: add, change, delete, importance) - one panel, two tabs.
const CHATS_PAGE = 40;
const MEMORY_CATEGORIES = { preference: "preference", personal: "personal", work: "work", project: "project", instruction: "instruction", fact: "fact" };
const library = { tab: "chats", query: "", before: null, loading: false, searchTimer: null };

function openLibrary(tab = "chats") {
  els.library.hidden = false;
  document.body.classList.add("gallery-open");
  showLibraryTab(tab);
}

function closeLibrary() {
  els.library.hidden = true;
  if (els.gallery.hidden) document.body.classList.remove("gallery-open");
}

function showLibraryTab(tab) {
  library.tab = tab;
  for (const button of els.libraryTabs.querySelectorAll("button[data-tab]")) button.classList.toggle("active", button.dataset.tab === tab);
  els.libraryChats.hidden = tab !== "chats";
  els.libraryMemory.hidden = tab !== "memory";
  if (els.libraryLearned) els.libraryLearned.hidden = tab !== "learned";
  if (tab === "chats") loadChats(true);
  else if (tab === "learned") loadLearned();
  else loadMemories();
}

function when(iso) {
  const date = new Date(iso);
  return date.toDateString() === new Date().toDateString()
    ? date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString([], { day: "numeric", month: "short" });
}

async function loadChats(reset) {
  if (library.loading) return;
  library.loading = true;
  if (reset) {
    library.before = null;
    els.libraryChatList.replaceChildren();
  }
  const params = new URLSearchParams({ limit: String(CHATS_PAGE) });
  if (library.query) params.set("q", library.query);
  if (library.before != null) params.set("before", String(library.before));
  try {
    const res = await fetch(`/api/conversations?${params}`);
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Could not load your chats.");
    for (const item of body.items) els.libraryChatList.appendChild(chatRow(item));
    if (body.items.length) library.before = body.items[body.items.length - 1].last_id;
    els.libraryMore.hidden = body.items.length < CHATS_PAGE;
    els.libraryChatsEmpty.hidden = els.libraryChatList.children.length > 0;
  } catch (err) {
    showError(err.message);
  } finally {
    library.loading = false;
  }
}

function chatRow(item) {
  const row = document.createElement("li");
  row.className = "library-row";
  const open = document.createElement("button");
  open.type = "button";
  open.className = "library-item" + (item.id === conversationId ? " current" : "");
  const title = document.createElement("span");
  title.className = "library-item-title";
  title.textContent = item.title;
  const meta = document.createElement("span");
  meta.className = "library-item-meta";
  meta.textContent = `${when(item.updated_at)} · ${item.messages} messages${item.id === conversationId ? " · open now" : ""}`;
  open.append(title, meta);
  if (item.snippet) {
    const snippet = document.createElement("span");
    snippet.className = "library-item-snippet";
    snippet.textContent = item.snippet;
    open.appendChild(snippet);
  }
  open.addEventListener("click", () => openConversation(item.id));
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "library-delete";
  remove.title = "Delete this chat";
  remove.setAttribute("aria-label", `Delete the chat "${item.title}"`);
  remove.textContent = "✕";
  remove.addEventListener("click", async () => {
    if (!window.confirm(`Delete the chat "${item.title}"? Its images and videos stay in the gallery.`)) return;
    try {
      const res = await fetch(`/api/conversations/${encodeURIComponent(item.id)}`, { method: "DELETE" });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Could not delete it.");
      row.remove();
      if (item.id === conversationId && !busy) {
        setConversation(null);
        clearChatView();
      }
    } catch (err) {
      showError(err.message);
    }
  });
  row.append(open, remove);
  return row;
}

async function openConversation(id) {
  if (busy || pendingWatch) {
    showError("Wait until Zira has finished, then open another chat.");
    return;
  }
  closeLibrary();
  if (id === conversationId) return;
  setConversation(id);
  clearChatView();
  await loadHistory();
  await resumePending();
  await loadPendingChanges();
  scrollToBottom();
}

async function loadMemories() {
  try {
    const res = await fetch("/api/memories");
    const memories = await res.json();
    if (!res.ok) throw new Error(memories.detail || "Could not load what Zira remembers.");
    els.libraryMemoryList.replaceChildren(...memories.map(memoryRow));
    els.libraryMemoryEmpty.hidden = memories.length > 0;
  } catch (err) {
    showError(err.message);
  }
}

async function saveMemory(id, change) {
  const res = await fetch(id == null ? "/api/memories" : `/api/memories/${id}`, {
    method: id == null ? "POST" : "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(change),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Could not save that memory.");
  return body;
}

function memoryRow(memory) {
  const row = document.createElement("li");
  row.className = "library-row";
  const edit = document.createElement("button");
  edit.type = "button";
  edit.className = "library-item";
  edit.title = "Change this memory";
  const text = document.createElement("span");
  text.className = "library-item-text";
  text.textContent = memory.text;
  const meta = document.createElement("span");
  meta.className = "library-item-meta";
  meta.textContent = `${MEMORY_CATEGORIES[memory.category] || memory.category} · updated ${when(memory.updated_at)}`;
  edit.append(text, meta);
  edit.addEventListener("click", async () => {
    const changed = window.prompt("Change this memory:", memory.text);
    if (changed == null || !changed.trim() || changed.trim() === memory.text) return;
    try {
      await saveMemory(memory.id, { text: changed.trim() });
      loadMemories();
    } catch (err) {
      showError(err.message);
    }
  });
  const importance = document.createElement("select");
  importance.className = "library-importance";
  importance.title = "How important (5 = always keep in mind)";
  importance.setAttribute("aria-label", "Importance");
  for (let level = 1; level <= 5; level++) importance.add(new Option(`★${level}`, String(level), false, level === memory.importance));
  importance.addEventListener("change", async () => {
    try {
      await saveMemory(memory.id, { importance: Number(importance.value) });
    } catch (err) {
      showError(err.message);
    }
  });
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "library-delete";
  remove.title = "Forget this";
  remove.setAttribute("aria-label", "Forget this memory");
  remove.textContent = "✕";
  remove.addEventListener("click", async () => {
    if (!window.confirm(`Forget "${memory.text}"?`)) return;
    try {
      const res = await fetch(`/api/memories/${memory.id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Could not forget it.");
      row.remove();
      els.libraryMemoryEmpty.hidden = els.libraryMemoryList.children.length > 0;
    } catch (err) {
      showError(err.message);
    }
  });
  row.append(edit, importance, remove);
  return row;
}

// ------------------------------------------------------------------ what Zira learned by itself
// The nightly study (GET /api/learned, /api/learning/status): each question Zira worked out again in its own
// time, with the better answer it now uses when a question like it comes up. Tap one to read the answer;
// ✕ removes it (Zira then stops using it). "Study now" runs a study at once (it pauses if you start chatting).
async function loadLearned() {
  try {
    const [res, statusRes] = await Promise.all([fetch("/api/learned"), fetch("/api/learning/status")]);
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Could not load what Zira learned.");
    const status = statusRes.ok ? await statusRes.json() : null;
    els.libraryLearnedList.replaceChildren(...body.items.map(learnedRow));
    els.libraryLearnedEmpty.hidden = body.items.length > 0;
    els.learnedStatus.textContent = learnedStatusText(status);
    els.learnedStudy.disabled = !!(status && status.running);
  } catch (err) {
    showError(err.message);
  }
}

function learnedStatusText(status) {
  if (!status) return "";
  if (status.running) return "Studying now…";
  const last = status.last_run;
  const lastText = last ? `Last study ${when(last.finished_at || last.started_at)}: ${last.studied} learned` +
    (last.finished_at ? "" : " (paused)") : "Not studied yet";
  return `${lastText} · studies nightly ${status.window} when you're idle`;
}

function learnedRow(item) {
  const row = document.createElement("li");
  row.className = "library-row";
  const open = document.createElement("button");
  open.type = "button";
  open.className = "library-item";
  open.title = "Read the answer Zira worked out";
  const title = document.createElement("span");
  title.className = "library-item-title";
  title.textContent = item.question;
  const answer = document.createElement("span");
  answer.className = "library-item-snippet";
  answer.textContent = item.answer.length > 140 ? item.answer.slice(0, 137) + "..." : item.answer;
  const meta = document.createElement("span");
  meta.className = "library-item-meta";
  meta.textContent = `learned ${when(item.created_at)}` + (item.lesson ? " · you corrected it" : "") +
    (item.used_count ? ` · used ${item.used_count}×` : "");
  open.append(title, answer, meta);
  open.addEventListener("click", () => {
    window.alert(`${item.question}\n\n${item.answer}` + (item.lesson ? `\n\n(${item.lesson})` : ""));
  });
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "library-delete";
  remove.title = "Remove this - Zira stops using it";
  remove.setAttribute("aria-label", "Remove this learned answer");
  remove.textContent = "✕";
  remove.addEventListener("click", async () => {
    if (!window.confirm(`Remove what Zira learned about "${item.question}"?`)) return;
    try {
      const res = await fetch(`/api/learned/${item.id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Could not remove it.");
      row.remove();
      els.libraryLearnedEmpty.hidden = els.libraryLearnedList.children.length > 0;
    } catch (err) {
      showError(err.message);
    }
  });
  row.append(open, remove);
  return row;
}

on(els.learnedStudy, "click", async () => {
  els.learnedStudy.disabled = true;
  try {
    const res = await fetch("/api/learning/study", { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || "Could not start a study.");
    els.learnedStatus.textContent = "Studying now… (it pauses if you start chatting)";
    setTimeout(() => { if (library.tab === "learned") loadLearned(); }, 60000);
  } catch (err) {
    showError(err.message);
    els.learnedStudy.disabled = false;
  }
});

on(els.libraryBtn, "click", () => openLibrary("chats"));
on(els.libraryClose, "click", closeLibrary);
on(els.libraryMore, "click", () => loadChats(false));
on(els.memoryAdd, "click", async () => {
  const text = window.prompt("Something Zira should remember about you:");
  if (!text || !text.trim()) return;
  try {
    await saveMemory(null, { text: text.trim() });
    loadMemories();
  } catch (err) {
    showError(err.message);
  }
});
if (els.libraryTabs) {
  for (const button of els.libraryTabs.querySelectorAll("button[data-tab]")) {
    button.addEventListener("click", () => showLibraryTab(button.dataset.tab));
  }
}
on(els.librarySearch, "input", () => {
  clearTimeout(library.searchTimer);
  library.searchTimer = setTimeout(() => {
    library.query = els.librarySearch.value.trim();
    loadChats(true);
  }, 300);
});

// ------------------------------------------------------------------- phone notifications
// Web Push (app/push.py): the Mac tells this phone when a video is ready or was not made, even with Zira closed.
// On an iPhone it only works once Zira is on the Home Screen (Safari: Share -> Add to Home Screen) and opened
// from that icon; the permission has to be asked from a tap, hence the button.
let notifyState = "off";

function setNotifyState(state, text) {
  notifyState = state;
  if (!els.notifyControl) return;
  els.notifyControl.hidden = false;
  const labels = { off: "🔔 Enable notifications", on: "🔔 Send a test", busy: "🔔 …" };
  els.notifyBtn.hidden = !(state in labels);
  els.notifyBtn.disabled = state === "busy";
  if (state in labels) els.notifyBtn.textContent = labels[state];
  els.notifyStatus.textContent = text || "";
}

function base64UrlToBytes(value) {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (value.length % 4)) % 4);
  return Uint8Array.from(atob(padded), (c) => c.charCodeAt(0));
}

async function saveSubscription(subscription) {
  const res = await fetch("/api/push/subscribe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(subscription.toJSON ? subscription.toJSON() : subscription),
  });
  if (!res.ok) throw new Error("Zira could not save this phone for notifications.");
}

async function setupNotifications() {
  if (!els.notifyControl || !window.isSecureContext || !("serviceWorker" in navigator)) return;
  let registration;
  try {
    registration = await navigator.serviceWorker.register("/sw.js");
  } catch {
    return; // no service worker here: no notifications, nothing to show
  }
  const iPhone = /iPhone|iPad|iPod/.test(navigator.userAgent);
  const installed = navigator.standalone === true || window.matchMedia("(display-mode: standalone)").matches;
  if (!("PushManager" in window) || !("Notification" in window)) {
    if (iPhone && !installed) setNotifyState("install", "For notifications: Share → Add to Home Screen, then open Zira from its icon.");
    return;
  }
  if (Notification.permission === "denied") {
    setNotifyState("denied", "Notifications are blocked for Zira in the phone's settings.");
    return;
  }
  const subscription = await registration.pushManager.getSubscription();
  if (subscription && Notification.permission === "granted") {
    try {
      await saveSubscription(subscription); // again on every visit, in case the Mac lost it
      setNotifyState("on", "On - you'll hear when a video is ready.");
    } catch (err) {
      setNotifyState("off", err.message);
    }
  } else {
    setNotifyState("off", "");
  }
}

async function enableNotifications() {
  setNotifyState("busy");
  try {
    const permission = await Notification.requestPermission();
    if (permission !== "granted") {
      setNotifyState(permission === "denied" ? "denied" : "off", "Notifications were not allowed.");
      return;
    }
    const keyRes = await fetch("/api/push/key");
    const { public_key: publicKey } = await keyRes.json();
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: base64UrlToBytes(publicKey),
    });
    await saveSubscription(subscription);
    setNotifyState("on", "On - you'll hear when a video is ready.");
  } catch (err) {
    setNotifyState("off", err.message || "Could not turn notifications on.");
  }
}

on(els.notifyBtn, "click", async () => {
  if (notifyState === "off") return enableNotifications();
  if (notifyState !== "on") return;
  setNotifyState("busy");
  try {
    const res = await fetch("/api/push/test", { method: "POST" });
    const body = await res.json();
    if (body.sent) setNotifyState("on", "Test sent - check your notifications.");
    else setNotifyState("off", "No phone got it - tap to enable again.");
  } catch {
    setNotifyState("on", "Could not send the test.");
  }
});
setupNotifications();

// ------------------------------------------------------------------- theme
// Light by default (the user's preference), dark on request, remembered in this browser; index.html's inline
// script applies it before the first paint.
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  if (!els.themeToggle) return;
  const dark = theme === "dark";
  els.themeToggle.textContent = dark ? "☀" : "☾";
  els.themeToggle.title = dark ? "Switch to light mode" : "Switch to dark mode";
  els.themeToggle.setAttribute("aria-label", els.themeToggle.title);
}
applyTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
on(els.themeToggle, "click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  try {
    localStorage.setItem("zira-theme", next);
  } catch {
    /* private browsing: the choice lasts for this page only */
  }
  applyTheme(next);
});

refreshHealth();
setInterval(refreshHealth, HEALTH_INTERVAL_MS);
els.modeHint.textContent = MODE_HINTS[mode] || "";
loadCapabilities().then(async () => {
  await loadHistory();
  await resumePending();
  await loadPendingChanges();
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") resumePending();
});
loadVoiceStatus();
