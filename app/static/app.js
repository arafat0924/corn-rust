const $ = (s) => document.querySelector(s);
const el = {
  models: $("#models"), modelDetail: $("#modelDetail"),
  upload: $("#upload"), drop: $("#drop"), file: $("#file"), btnText: $("#btnText"), status: $("#status"),
  result: $("#result"), frameBg: $("#frameBg"), photo: $("#photo"), heat: $("#heat"), toggle: $("#viewToggle"),
  report: $("#report"), used: $("#usedModel"), title: $("#verdictTitle"), pct: $("#pct"),
  barWrap: $("#barWrap"), bar: $("#bar"), detail: $("#verdictDetail"), scores: $("#scores"),
  hint: $("#heatHint"), again: $("#again"),
};
// Uploads are switched off while the project is still in development (tested locally only).
// Set to true to let visitors upload their own photos again.
const UPLOAD_ENABLED = false;
const TOUCH = window.matchMedia("(hover: none) and (pointer: coarse)").matches;
const IDLE = !UPLOAD_ENABLED ? ""
  : TOUCH ? "Tap to choose or take a photo"
  : "Drop, paste or click · JPG, PNG or WebP";
const BTN = "Upload a leaf";
const MAX = 15 * 1024 * 1024;
let catalog = [], current = null, lastFile = null, lastSample = null, busy = false, localUrl = null;
// Demo mode: no Python server behind the page (static hosting). The model list and the sample
// results come from pre-computed files in /demo/, built by scripts/build_static_site.py.
let demo = false;

function setStatus(text, err = false) {
  el.status.textContent = text;
  el.status.classList.toggle("err", err);
}

/* ---------- models ---------- */
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

async function loadModels() {
  let d;
  try {
    d = await getJSON("/api/models");
  } catch {
    try {
      d = await getJSON("/demo/models.json");
      demo = true;
    } catch {
      setStatus("Couldn't reach the detector. Please try again later.", true);
      return;
    }
  }
  catalog = d.models;
  current = d.default;
  renderModels();
}

function renderModels() {
  el.models.replaceChildren(...catalog.map((m) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "model";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(m.id === current));
    b.tabIndex = m.id === current ? 0 : -1;
    b.textContent = m.label;
    b.disabled = busy;
    b.addEventListener("click", () => selectModel(m.id));
    return b;
  }));
  const m = catalog.find((x) => x.id === current);
  el.modelDetail.textContent = m ? m.detail : "";
}

function selectModel(id) {
  if (busy || id === current) return;
  current = id;
  renderModels();
  el.models.querySelector('[aria-checked="true"]')?.focus();
  if (el.result.hidden) return;                       // re-check the same photo with the new model
  if (lastSample) runSample(lastSample);
  else if (lastFile) analyze(lastFile);
}

el.models.addEventListener("keydown", (e) => {
  const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
  if (!step || !catalog.length) return;
  e.preventDefault();
  const i = catalog.findIndex((m) => m.id === current);
  selectModel(catalog[(i + step + catalog.length) % catalog.length].id);
});

/* ---------- analysis ---------- */
function setBusy(on) {
  busy = on;
  el.drop.classList.toggle("busy", on);
  el.result.classList.toggle("busy", on);
  el.btnText.textContent = on ? "Looking closely…" : BTN;
  el.models.querySelectorAll("button").forEach((b) => { b.disabled = on; });
}

async function analyze(file, sample = null) {
  if (busy || !file) return;
  if (demo) return fail("Uploads aren't available on this page yet.");
  if (file.type && !file.type.startsWith("image/")) return fail("That file isn't an image. Try a JPG, PNG or WebP.");
  if (file.size > MAX) return fail("That file is over 15 MB. Try a smaller photo.");
  lastFile = file;
  lastSample = sample;
  setBusy(true);
  if (el.result.hidden) setStatus("Checking the leaf…");
  const body = new FormData();
  body.append("image", file);
  body.append("model", current || "");
  try {
    const r = await fetch("/api/analyze", { method: "POST", body });
    const d = await r.json().catch(() => ({ status: "error", detail: "The detector sent an unexpected reply." }));
    if (!r.ok || d.status === "error") throw new Error(d.detail || "Something went wrong. Please try again.");
    show(d, file);
  } catch (e) {
    fail(e.message && e.message !== "Failed to fetch" ? e.message : "Couldn't reach the detector. Please try again.");
  } finally {
    setBusy(false);
  }
}

function fail(message) {
  el.result.hidden = true;
  el.upload.hidden = false;
  setStatus(message, true);
}

function show(d, file) {
  const ok = d.status === "ok";
  if (localUrl) { URL.revokeObjectURL(localUrl); localUrl = null; }
  let src = d.photo;
  if (!ok && file) { localUrl = URL.createObjectURL(file); src = localUrl; }   // rejected: show the photo as uploaded
  el.photo.src = src;
  el.frameBg.style.backgroundImage = `url("${src}")`;
  el.heat.src = d.heatmap || "";
  el.toggle.hidden = el.hint.hidden = !d.heatmap;
  setView("photo");

  const m = catalog.find((x) => x.id === current);
  el.report.dataset.verdict = d.verdict;
  el.used.textContent = ok ? (m ? m.label : "") : "Input check";
  el.title.textContent = d.title;
  el.detail.textContent = d.detail;
  el.pct.hidden = el.barWrap.hidden = el.scores.hidden = !ok;
  if (ok) {
    const p = Math.round(d.confidence * 100);
    el.pct.textContent = d.verdict === "uncertain" ? `${p}% ${d.label.toLowerCase()}` : `${p}%`;
    el.bar.style.width = "0";
    requestAnimationFrame(() => requestAnimationFrame(() => { el.bar.style.width = `${p}%`; }));
    el.scores.replaceChildren(...d.scores.map((s) => {
      const li = document.createElement("li");
      const a = document.createElement("span");
      const b = document.createElement("span");
      a.textContent = s.label;
      b.textContent = `${(s.value * 100).toFixed(1)}%`;
      li.append(a, b);
      return li;
    }));
  }
  el.upload.hidden = true;
  el.result.hidden = false;
  setStatus(IDLE);
}

function setView(view) {
  el.photo.hidden = view !== "photo";
  el.heat.hidden = view !== "heat";
  el.toggle.querySelectorAll("button").forEach((b) => {
    const on = b.dataset.view === view;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", String(on));
  });
}
el.toggle.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-view]");
  if (b) setView(b.dataset.view);
});

el.again.addEventListener("click", () => {
  el.result.hidden = true;
  el.upload.hidden = false;
  lastFile = lastSample = null;
  setStatus(IDLE);
  el.drop.focus();
});

/* ---------- inputs: click, keyboard, drag and drop, paste, samples ---------- */
if (UPLOAD_ENABLED) {
  el.file.addEventListener("change", () => {
    const f = el.file.files[0];
    el.file.value = "";
    analyze(f);
  });
  el.drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.file.click(); }
  });
  document.addEventListener("dragover", (e) => { e.preventDefault(); el.drop.classList.add("over"); });
  document.addEventListener("dragleave", (e) => { if (!e.relatedTarget) el.drop.classList.remove("over"); });
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    el.drop.classList.remove("over");
    analyze(e.dataTransfer?.files?.[0]);
  });
  document.addEventListener("paste", (e) => {
    const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
    if (f) analyze(f);
  });
} else {
  el.file.disabled = true;                       // a click on the pill no longer opens the file picker
  el.drop.classList.add("locked");
  el.drop.removeAttribute("tabindex");
  el.drop.setAttribute("aria-disabled", "true");
  $("#btnNote").hidden = false;
  $("#samplesLead").textContent = "Meanwhile, try a";
  document.addEventListener("dragover", (e) => e.preventDefault());   // a dropped file must not open in the tab
  document.addEventListener("drop", (e) => e.preventDefault());
}
async function runSample(name) {
  if (busy || !catalog.length) return;
  if (!demo) {                                    // live server: analyse the sample photo for real
    try {
      const r = await fetch(`/assets/sample-${name}.png`);
      if (!r.ok) throw new Error();
      await analyze(new File([await r.blob()], `sample-${name}.png`, { type: "image/png" }), name);
    } catch {
      fail("Couldn't load the sample photo.");
    }
    return;
  }
  const m = catalog.find((x) => x.id === current);   // static hosting: pre-computed result
  lastSample = name;
  lastFile = null;
  setBusy(true);
  try {
    const [d] = await Promise.all([getJSON(`/demo/${name}-${m.slug}.json`),
                                   new Promise((ok) => setTimeout(ok, 350))]);   // let the "busy" state show
    show(d, null);
  } catch {
    fail("Couldn't load the sample result.");
  } finally {
    setBusy(false);
  }
}
document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => runSample(b.dataset.sample)));

/* ---------- Method / Limits ---------- */
document.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => {
  document.getElementById(b.dataset.open).showModal();
}));
document.querySelectorAll("dialog.pop").forEach((d) => {
  d.querySelector(".x").addEventListener("click", () => d.close());
  d.addEventListener("click", (e) => { if (e.target === d) d.close(); });   // click outside closes
});

setStatus(IDLE);
loadModels();
