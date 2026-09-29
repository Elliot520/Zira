"use strict";

// Phone-only settings drawer. On a narrow viewport the configuration controls (mode, Light/Deep,
// image style + resolution, voice power) move out of the cramped header/composer into a slide-out
// menu; the mic and "Continuous" stay on top of the chat. The controls are MOVED, not copied, so
// every id, event listener and piece of state in app.js keeps working untouched; widening the window
// puts them back exactly where they were. Everything here is optional progressive enhancement: if
// anything it needs is missing (e.g. cached HTML from before this file existed) it does nothing and
// the regular layout stays as it was.
(function () {
  const $ = (id) => document.getElementById(id);
  const drawer = $("settings-drawer");
  const backdrop = $("drawer-backdrop");
  const openBtn = $("menu-btn");
  const closeBtn = $("drawer-close");
  if (!drawer || !backdrop || !openBtn || !closeBtn || !window.matchMedia) return;

  const mq = window.matchMedia("(max-width: 640px)");
  const MOVES = [
    ["mode-hint", "slot-mode-hint"],
    ["model-mode-toggle", "slot-model-toggle"],
    ["model-mode-status", "slot-model-label"],
    ["model-label", "slot-model-label"],
    ["image-style-toggle", "slot-style"],
    ["image-resolution-toggle", "slot-resolution"],
    ["video-model-toggle", "slot-video"],
    ["video-resolution-toggle", "slot-video-resolution"],
    ["video-orientation-toggle", "slot-video-orientation"],
    ["notify-control", "slot-notify"],
    ["theme-toggle", "slot-theme"],
    ["voice-power-btn", "slot-voice-power"],
    ["tts-voice-toggle", "slot-tts-voice"],
    ["lora-btn", "slot-lora"],
  ];
  const moved = [];
  const input = $("input");
  const fullPlaceholder = input ? input.placeholder : "";
  let active = false;
  let open = false;

  // ------------------------------------------------------------ moving controls in and out
  function enter() {
    if (active) return;
    for (const [id, slotId] of MOVES) {
      const el = $(id);
      const slot = $(slotId);
      if (!el || !slot) continue;
      const marker = document.createComment(`mobile-menu:${id}`); // remembers the original spot
      el.replaceWith(marker);
      slot.appendChild(el);
      moved.push({ el, marker });
    }
    document.body.classList.add("menu-on");
    if (input) input.placeholder = "Message Zira…"; // the keyboard hints don't apply on a phone and wrap to a cut-off second line
    active = true;
    syncGroups();
    syncCaption();
    syncVideoCaption();
    syncTtsCaption();
    renderModeSeg();
  }

  function leave() {
    if (!active) return;
    closeDrawer(true);
    while (moved.length) {
      const { el, marker } = moved.pop();
      marker.replaceWith(el);
    }
    document.body.classList.remove("menu-on");
    if (input) input.placeholder = fullPlaceholder;
    active = false;
  }

  function apply() {
    if (mq.matches) enter();
    else leave();
  }

  // ------------------------------------------------------------ open / close
  function openDrawer() {
    if (!active || open) return;
    open = true;
    renderModeSeg();
    syncCaption();
    syncVideoCaption();
    document.body.classList.add("drawer-open");
    drawer.setAttribute("aria-hidden", "false");
    openBtn.setAttribute("aria-expanded", "true");
    closeBtn.focus({ preventScroll: true });
  }

  function closeDrawer(silent) {
    if (!open) return;
    open = false;
    document.body.classList.remove("drawer-open");
    drawer.setAttribute("aria-hidden", "true");
    openBtn.setAttribute("aria-expanded", "false");
    if (!silent) openBtn.focus({ preventScroll: true });
  }

  openBtn.addEventListener("click", () => (open ? closeDrawer() : openDrawer()));
  closeBtn.addEventListener("click", () => closeDrawer());
  backdrop.addEventListener("click", () => closeDrawer());
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && open) closeDrawer();
  });

  // Swipe left inside the drawer to close it; the panel follows the finger, then either slides
  // shut or snaps back. Vertical drags are left alone so the menu itself can still scroll.
  let startX = 0;
  let startY = 0;
  let dx = 0;
  let tracking = false;
  let decided = false;

  drawer.addEventListener(
    "touchstart",
    (event) => {
      if (!open || event.touches.length !== 1) return;
      startX = event.touches[0].clientX;
      startY = event.touches[0].clientY;
      dx = 0;
      tracking = true;
      decided = false;
    },
    { passive: true }
  );

  drawer.addEventListener(
    "touchmove",
    (event) => {
      if (!tracking) return;
      const touch = event.touches[0];
      const moveX = touch.clientX - startX;
      const moveY = touch.clientY - startY;
      if (!decided) {
        if (Math.abs(moveX) < 8 && Math.abs(moveY) < 8) return;
        decided = true;
        if (Math.abs(moveY) > Math.abs(moveX) || moveX > 0) {
          tracking = false;
          return;
        }
        drawer.classList.add("dragging");
      }
      dx = Math.min(0, moveX);
      drawer.style.transform = `translateX(${dx}px)`;
      backdrop.style.opacity = String(Math.max(0, 1 + dx / Math.max(drawer.offsetWidth, 1)));
    },
    { passive: true }
  );

  function endDrag() {
    if (!tracking) return;
    tracking = false;
    drawer.classList.remove("dragging");
    drawer.style.transform = "";
    backdrop.style.opacity = "";
    if (dx < -Math.min(80, drawer.offsetWidth * 0.25)) closeDrawer();
  }
  drawer.addEventListener("touchend", endDrag);
  drawer.addEventListener("touchcancel", endDrag);

  // ------------------------------------------------------------ mode: segmented control
  // The real <select id="mode"> stays the source of truth (app.js populates it from /api/capabilities
  // and reads it); this just mirrors it as three buttons and writes back through it.
  const modeSelect = $("mode");
  const modeSeg = $("mode-seg");
  const MODE_SHORT = { chat: "Chat", plan: "Plan", edit: "Edit" };

  function renderModeSeg() {
    if (!modeSelect || !modeSeg) return;
    modeSeg.replaceChildren(
      ...Array.from(modeSelect.options).map((option) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = MODE_SHORT[option.value] || option.textContent;
        button.classList.toggle("active", modeSelect.value === option.value);
        button.addEventListener("click", () => {
          modeSelect.value = option.value;
          modeSelect.dispatchEvent(new Event("change"));
          renderModeSeg();
        });
        return button;
      })
    );
  }
  if (modeSelect) new MutationObserver(renderModeSeg).observe(modeSelect, { childList: true });

  // ------------------------------------------------------------ groups + captions
  // A group with nothing to show (image generation off, no speech-to-text) disappears instead of
  // leaving an empty card. data-watch lists the element ids whose `hidden` attribute decides it.
  const groups = Array.from(drawer.querySelectorAll(".drawer-group[data-watch]"));

  function syncGroups() {
    for (const group of groups) {
      const ids = group.dataset.watch.split(/\s+/);
      group.hidden = ids.every((id) => {
        const el = $(id);
        return !el || el.hidden;
      });
    }
  }

  const watchedIds = new Set(groups.flatMap((group) => group.dataset.watch.split(/\s+/)));
  const hiddenObserver = new MutationObserver(syncGroups);
  for (const id of watchedIds) {
    const el = $(id);
    if (el) hiddenObserver.observe(el, { attributes: true, attributeFilter: ["hidden"] });
  }

  const caption = $("image-style-caption");
  const styleButtons = Array.from(document.querySelectorAll(".image-style-btn[data-style]"));

  function syncCaption() {
    if (!caption) return;
    const activeButton = styleButtons.find((button) => button.classList.contains("active"));
    caption.textContent = activeButton ? activeButton.dataset.name || activeButton.textContent : "";
  }
  const ttsCaption = $("tts-voice-caption");
  const ttsButtons = Array.from(document.querySelectorAll(".image-style-btn[data-tts-voice]"));

  function syncTtsCaption() {
    if (!ttsCaption) return;
    const activeButton = ttsButtons.find((button) => button.classList.contains("active"));
    ttsCaption.textContent = activeButton ? activeButton.dataset.name || activeButton.textContent : "";
    const field = $("drawer-tts-field");
    const toggle = $("tts-voice-toggle");
    if (field) field.hidden = !toggle || toggle.hidden;
  }

  const videoCaption = $("video-model-caption");
  const videoButtons = Array.from(document.querySelectorAll(".image-style-btn[data-video-model]"));

  function syncVideoCaption() {
    if (!videoCaption) return;
    const activeButton = videoButtons.find((button) => button.classList.contains("active"));
    videoCaption.textContent = activeButton ? activeButton.dataset.name || activeButton.textContent : "";
  }

  const captionObserver = new MutationObserver(() => {
    syncCaption();
    syncVideoCaption();
    syncTtsCaption();
  });
  const ttsToggle = $("tts-voice-toggle");
  if (ttsToggle) captionObserver.observe(ttsToggle, { attributes: true, attributeFilter: ["hidden"] });
  for (const button of [...styleButtons, ...videoButtons, ...ttsButtons]) {
    captionObserver.observe(button, { attributes: true, attributeFilter: ["class"] });
  }

  // ------------------------------------------------------------ start
  if (mq.addEventListener) mq.addEventListener("change", apply);
  else if (mq.addListener) mq.addListener(apply);
  apply();
})();
