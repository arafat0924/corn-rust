() => {
  const initialize = () => {
    // Gradio may mount its interface inside a web-component shadow root.
    const host = document.querySelector("gradio-app");
    const scope = host?.shadowRoot || host || document;
    const mascot = scope.querySelector("#cc-kernel");
    const root = scope.querySelector(".gradio-container");
    if (!mascot || !root) return false;
    if (root.dataset.cornCareReady) return true;
    root.dataset.cornCareReady = "true";
    root.dataset.checkState = "ready";
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const finePointer = window.matchMedia("(pointer: fine)");
    const toggle = scope.querySelector("#cc-motion-toggle");
    const line = scope.querySelector("#cc-kernel-line");
    const pupils = scope.querySelector("#cc-pupils");
    const glow = scope.querySelector("#cc-cursor-glow");
    const layer = scope.querySelector("#cc-sprout-layer");
    let saved;
    try { saved = localStorage.getItem("corncare-motion"); } catch (_) {}
    let enabled = saved !== "off" && !reduceMotion.matches;
    let targetX = 0, targetY = 0, currentX = 0, currentY = 0, frame = 0;
    let captionIndex = 0, greetingTimer;
    const jokes = ["Leaf it to me. ✦", "I'm all ears. Technically, one ear.", "My favorite subject? Crop science.", "Corny jokes. Careful checks.", "Leaf selfies only, please."];
    const cancelFrame = () => { if (frame) cancelAnimationFrame(frame); frame = 0; };
    const motionActive = () => enabled && !reduceMotion.matches && !document.hidden;
    const applyMotion = () => {
      const active = motionActive();
      root.dataset.motion = active ? "on" : "off";
      toggle.setAttribute("aria-pressed", String(active));
      toggle.disabled = reduceMotion.matches;
      toggle.setAttribute("aria-label", reduceMotion.matches ? "Decorative motion follows your device's reduced-motion setting" : active ? "Pause decorative animations" : "Enable decorative animations");
      toggle.querySelector(".cc-motion-label").textContent = reduceMotion.matches ? "Reduced motion" : active ? "Motion on" : "Motion off";
      if (!active) {
        cancelFrame(); glow.style.opacity = "0"; pupils.style.transform = "";
        root.style.setProperty("--gaze-x", "0px"); root.style.setProperty("--gaze-y", "0px");
        layer.replaceChildren();
      }
    };
    toggle.addEventListener("click", () => {
      // A system reduced-motion preference always takes priority.
      if (reduceMotion.matches) { enabled = false; line.textContent = "Quiet mode. Same curious Kernel."; }
      else enabled = !enabled;
      try { localStorage.setItem("corncare-motion", enabled ? "on" : "off"); } catch (_) {}
      applyMotion();
    });
    reduceMotion.addEventListener("change", () => { if (reduceMotion.matches) enabled = false; applyMotion(); });
    document.addEventListener("visibilitychange", applyMotion);
    applyMotion();
    const draw = () => {
      frame = 0;
      if (!motionActive() || !finePointer.matches) return;
      currentX += (targetX - currentX) * .18; currentY += (targetY - currentY) * .18;
      glow.style.transform = `translate3d(${currentX}px, ${currentY}px, 0)`;
      const rect = mascot.getBoundingClientRect();
      const dx = Math.max(-1, Math.min(1, (targetX - rect.left - rect.width / 2) / 420));
      const dy = Math.max(-1, Math.min(1, (targetY - rect.top - rect.height / 2) / 320));
      pupils.style.transform = `translate(${dx * 3}px, ${dy * 2}px)`;
      root.style.setProperty("--gaze-x", `${dx * 3}px`); root.style.setProperty("--gaze-y", `${dy * 2}px`);
      if (Math.abs(targetX - currentX) + Math.abs(targetY - currentY) > .3) frame = requestAnimationFrame(draw);
    };
    document.addEventListener("pointermove", (event) => {
      if (!motionActive() || !finePointer.matches || event.pointerType === "touch") return;
      targetX = event.clientX; targetY = event.clientY;
      glow.style.opacity = ".65";
      if (!frame) frame = requestAnimationFrame(draw);
      const target = event.composedPath()[0];
      const panel = target.closest?.("#upload-panel, #results-panel");
      if (panel) {
        const bounds = panel.getBoundingClientRect();
        panel.style.setProperty("--spot-x", `${event.clientX - bounds.left}px`);
        panel.style.setProperty("--spot-y", `${event.clientY - bounds.top}px`);
      }
    }, {passive: true});
    document.documentElement.addEventListener("pointerleave", () => { glow.style.opacity = "0"; cancelFrame(); });
    mascot.addEventListener("click", () => {
      captionIndex = (captionIndex + 1) % jokes.length;
      line.textContent = jokes[captionIndex];
      root.classList.remove("cc-say-hello");
      if (motionActive()) { requestAnimationFrame(() => root.classList.add("cc-say-hello")); }
      clearTimeout(greetingTimer);
      greetingTimer = setTimeout(() => root.classList.remove("cc-say-hello"), 1000);
    });
    document.addEventListener("click", (event) => {
      if (!motionActive() || !finePointer.matches || !event.isTrusted) return;
      const target = event.composedPath()[0];
      if (target.closest?.("button,a,input,label,textarea,select,[role=button],[role=radio],#workspace")) return;
      if (layer.childElementCount >= 6) layer.firstElementChild?.remove();
      const sprout = document.createElement("span"); sprout.className = "cc-click-sprout";
      sprout.style.left = `${event.clientX}px`; sprout.style.top = `${event.clientY}px`;
      const stem = document.createElement("i"); const leafA = document.createElement("i"); const leafB = document.createElement("i");
      sprout.append(stem, leafA, leafB); layer.append(sprout);
      setTimeout(() => sprout.remove(), 1300);
    });
    const finish = (note) => {
      const text = String(note || "");
      let state = "ready", message = "Leaf it to me. ✦";
      if (text.includes("doesn't look like a corn leaf")) { state = "rejected"; message = "Leaf selfies only, please."; }
      else if (text.includes("Uncertain") || text.includes("Possible")) { state = "uncertain"; message = "A clearer close-up would help."; }
      else if (text.includes("Common rust detected")) { state = "rust"; message = "Check the result and highlighted regions."; }
      else if (text.includes("Leaf looks healthy")) { state = "healthy"; message = "One leaf check, ready to explore."; }
      root.dataset.checkState = state; line.textContent = message;
    };
    window.cornCareMotion = {
      begin() { root.dataset.checkState = "scanning"; line.textContent = "Putting on my science face…"; },
      finish,
      reset() { finish(""); },
      fail() { root.dataset.checkState = "uncertain"; line.textContent = "That check didn't finish. Please try again."; }
    };
    return true;
  };
  if (!initialize()) {
    let attempts = 0;
    const timer = setInterval(() => {
      if (initialize() || ++attempts >= 100) clearInterval(timer);
    }, 150);
  }
}
