/* Receipt Classification System — demo frontend.
   Talks to app.py: POST /api/scan -> job id, then polls /api/jobs/<id>
   for per-stage progress until the result is ready. */

(() => {
  "use strict";

  // ─── Category metadata ──────────────────────────────────────────────────
  const ICON_PATHS = {
    "Food & Beverage": '<path d="M4 9h13v4a6 6 0 0 1-6 6h-1a6 6 0 0 1-6-6z"/><path d="M17 10.5h1.5a2.5 2.5 0 0 1 0 5H16.5"/><path d="M8 3v3M12 3v3"/>',
    "Transportation": '<path d="M3 16v-3l2.2-5.2A2 2 0 0 1 7 6.5h10a2 2 0 0 1 1.8 1.3L21 13v3"/><path d="M3 13h18"/><circle cx="7.5" cy="16.5" r="2"/><circle cx="16.5" cy="16.5" r="2"/>',
    "Household & Apparel": '<path d="M5 8h14l-1.2 12.2a1 1 0 0 1-1 .8H7.2a1 1 0 0 1-1-.8z"/><path d="M9 10V6.5a3 3 0 0 1 6 0V10"/>',
    "Home & Hardware": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v10h13V10"/><path d="M10 20v-5.5h4V20"/>',
    "Electronics & Electrical": '<path d="M13 2.5 4.5 14H11l-1 7.5L19.5 10H13z"/>',
    "Recreation & Lifestyle": '<path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/>',
    "Others": '<path d="M6 3h12v18l-3-2-3 2-3-2-3 2z"/><path d="M9 8h6M9 12h6M9 16h3"/>',
    "Blank / Not a Receipt": '<circle cx="12" cy="12" r="9"/><path d="m8.5 8.5 7 7M15.5 8.5l-7 7"/>',
  };
  const CATEGORY_COLORS = {
    "Food & Beverage": "#c0562f",
    "Transportation": "#2f5d8a",
    "Household & Apparel": "#8a4f7d",
    "Home & Hardware": "#8a6a2f",
    "Electronics & Electrical": "#2f7a74",
    "Recreation & Lifestyle": "#5f7a2a",
    "Others": "#6b6155",
    "Blank / Not a Receipt": "#9b2c2c",
  };
  const icon = (cat, size = 24) =>
    `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON_PATHS[cat] || ICON_PATHS.Others}</svg>`;

  const STAGE_META = {
    preprocessing:    { name: "Preprocess",       desc: "Grayscale, resize to 1800px, scan/photo filter" },
    ocr:              { name: "OCR",              desc: "PaddleOCR 3.x text recognition" },
    field_extraction: { name: "Field extraction", desc: "Regex first pass + LLM validator" },
    classification:   { name: "Classification",   desc: "DistilBERT, EfficientNet-B0, late fusion" },
    safety_net:       { name: "Safety net",       desc: "LLM check when confidence < 95%" },
  };
  const FIELD_LABELS = {
    transaction_date: "Date",
    transaction_time: "Time",
    tax_amount: "Tax / Service",
    total_amount: "Total",
  };
  const HISTORY_KEY = "receipt-demo-history-v1";
  const HISTORY_MAX = 12;

  // ─── State ──────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  let info = null;
  let selectedFile = null;
  let originalUrl = null;     // object URL / dataURL for the selected file
  let processedUrl = null;
  let currentView = "original";
  let current = null;         // { result, name, thumb, ts }
  let polling = null;
  let ocrView = "lines";

  // ─── Helpers ────────────────────────────────────────────────────────────
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x) => (x * 100).toFixed(1) + "%";
  const secs = (x) => (x == null ? "" : x < 0.1 ? `${Math.round(x * 1000)}ms` : `${x.toFixed(2)}s`);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const llmName = () => (info?.llm?.model || "LLM").split("/").pop();

  function toast(msg) {
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2200);
  }

  async function copyText(text, msg) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text; document.body.appendChild(ta); ta.select();
      document.execCommand("copy"); ta.remove();
    }
    toast(msg);
  }

  function showError(title, msg) {
    $("errorTitle").textContent = title;
    $("errorMsg").textContent = msg ? " " + msg : "";
    $("errorBox").hidden = false;
  }

  // ─── Info / samples ─────────────────────────────────────────────────────
  async function loadInfo() {
    try {
      const res = await fetch("/api/info");
      info = await res.json();
    } catch {
      setPill(false, "Server unreachable");
      return;
    }
    setPill(info.llm.enabled, info.llm.enabled ? `Safety net: ${llmName()}` : "Safety net off — no API key");
    $("acceptHint").textContent = `${info.allowed_extensions.map((e) => e.slice(1).toUpperCase()).join(", ")} · max ${info.max_file_mb} MB`;
    $("fusionBadge").textContent = `${info.fusion.cnn} CNN / ${info.fusion.bert} BERT`;
    $("meterThreshold").style.left = info.threshold * 100 + "%";
    $("meterThreshold").dataset.label = `${Math.round(info.threshold * 100)}% threshold`;
    renderSamples(info.samples);
    renderAbout();
  }

  function setPill(on, text) {
    const pill = $("llmPill");
    pill.classList.toggle("on", on);
    pill.classList.toggle("off", !on);
    pill.querySelector(".pill-text").textContent = text;
    pill.title = text;
  }

  function sampleCategory(name) {
    const n = name.toLowerCase();
    if (n.includes("food")) return "Food & Beverage";
    if (n.includes("transport")) return "Transportation";
    if (n.includes("household")) return "Household & Apparel";
    if (n.includes("home")) return "Home & Hardware";
    if (n.includes("electr")) return "Electronics & Electrical";
    if (n.includes("recreation")) return "Recreation & Lifestyle";
    return "Others";
  }

  function renderSamples(samples) {
    if (!samples?.length) return;
    const grid = $("samplesGrid");
    grid.innerHTML = "";
    samples.forEach((s) => {
      const cat = sampleCategory(s.name);
      const b = document.createElement("button");
      b.type = "button";
      b.className = "sample";
      b.title = `Sample: ${cat}`;
      b.setAttribute("aria-label", `Scan sample ${cat} receipt`);
      b.innerHTML = `<img src="${esc(s.url)}" alt="" loading="lazy"><span class="sample-tag">${icon(cat, 13)}</span>`;
      b.addEventListener("click", async () => {
        if (polling) return;
        try {
          const blob = await (await fetch(s.url)).blob();
          handleFile(new File([blob], s.name, { type: blob.type || "image/jpeg" }));
          startScan();
        } catch (e) {
          showError("Could not load sample.", e.message);
        }
      });
      grid.appendChild(b);
    });
    $("samples").hidden = false;
  }

  // ─── File selection ─────────────────────────────────────────────────────
  function handleFile(file) {
    if (!file) return;
    const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
    const allowed = info?.allowed_extensions || [".png", ".jpg", ".jpeg", ".webp", ".bmp"];
    const maxMb = info?.max_file_mb || 15;
    if (!allowed.includes(ext) && !/^image\/(png|jpe?g|webp|bmp)$/.test(file.type)) {
      showError("Unsupported file type.", `Please use ${allowed.join(", ")}.`);
      return;
    }
    if (file.size > maxMb * 1024 * 1024) {
      showError("File is too large.", `Maximum size is ${maxMb} MB.`);
      return;
    }
    $("errorBox").hidden = true;
    selectedFile = file;
    if (originalUrl?.startsWith("blob:")) URL.revokeObjectURL(originalUrl);
    originalUrl = URL.createObjectURL(file);
    processedUrl = null;
    $("fileName").textContent = file.name;
    $("prepMeta").innerHTML = "";
    setTabsEnabled(false);
    setView("original");
    $("previewCard").hidden = false;
    $("scanBtn").disabled = false;
    resetSteps();
  }

  function clearFile() {
    selectedFile = null;
    originalUrl = processedUrl = null;
    $("previewCard").hidden = true;
    $("scanBtn").disabled = true;
    $("fileInput").value = $("cameraInput").value = "";
  }

  // ─── Image viewer ───────────────────────────────────────────────────────
  function setTabsEnabled(on) {
    document.querySelectorAll("[data-view]").forEach((t) => {
      if (t.dataset.view !== "original") t.disabled = !on;
    });
  }

  function setView(view) {
    currentView = view;
    document.querySelectorAll("[data-view]").forEach((t) =>
      t.setAttribute("aria-selected", String(t.dataset.view === view)));
    const viewer = $("viewer");
    viewer.classList.toggle("compare", view === "compare");
    const img = (src, alt) => `<img src="${esc(src)}" alt="${alt}" onerror="this.style.opacity=.25">`;
    if (view === "compare" && processedUrl) {
      viewer.innerHTML = img(originalUrl, "Original receipt") + img(processedUrl, "Preprocessed receipt");
    } else if (view === "processed" && processedUrl) {
      viewer.innerHTML = `<span class="zoom-hint">Used for OCR + CNN</span>` + img(processedUrl, "Preprocessed receipt");
    } else {
      viewer.innerHTML = originalUrl ? img(originalUrl, "Original receipt") : "";
    }
  }

  function openLightbox(src) {
    $("lightboxImg").src = src;
    $("lightbox").hidden = false;
  }

  // ─── Pipeline stepper ───────────────────────────────────────────────────
  function stages() { return info?.stages || Object.keys(STAGE_META); }

  function resetSteps() {
    $("steps").innerHTML = stages().map((s, i) => `
      <li class="step" data-stage="${s}">
        <div class="step-dot"><span>${i + 1}</span></div>
        <div><div class="step-name">${STAGE_META[s].name}</div><div class="step-desc">${STAGE_META[s].desc}</div></div>
        <div class="step-time"></div>
      </li>`).join("");
    $("stepsTotal").hidden = true;
    $("queueNote").hidden = true;
    $("pipelineState").textContent = selectedFile ? "Ready" : "Idle";
  }

  function updateSteps(job) {
    const order = stages();
    const activeIdx = job.status === "done" ? order.length : order.indexOf(job.stage);
    document.querySelectorAll(".step").forEach((el, i) => {
      const s = el.dataset.stage;
      el.classList.remove("active", "done", "warn", "failed");
      const t = job.timings?.[s];
      if (job.status === "error" && i === activeIdx) el.classList.add("failed");
      else if (i < activeIdx || (job.status === "done")) el.classList.add("done");
      else if (i === activeIdx && job.status === "running") el.classList.add("active");
      el.querySelector(".step-time").textContent = t != null ? secs(t) : i === activeIdx && job.status === "running" ? "running" : "";
    });
    const q = $("queueNote");
    if (job.status === "queued") {
      q.hidden = false;
      q.textContent = job.position > 1
        ? `Waiting in queue — ${job.position - 1} scan${job.position > 2 ? "s" : ""} ahead of you`
        : "Starting…";
    } else q.hidden = true;
    $("pipelineState").textContent = { queued: "Queued", running: "Running", done: "Complete", error: "Failed" }[job.status] || "";
    updateLive(job, activeIdx);
  }

  // Mirror of the stepper in the right panel, so progress is visible without scrolling
  let liveTimer = null;
  function showLive(on) {
    $("liveCard").hidden = !on;
    clearInterval(liveTimer);
    if (!on) return;
    const t0 = performance.now();
    $("liveFile").textContent = selectedFile?.name || "";
    $("liveBar").innerHTML = stages().map(() => `<div class="live-seg"></div>`).join("");
    $("liveLabels").innerHTML = stages().map((s) => `<span>${STAGE_META[s].name}</span>`).join("");
    liveTimer = setInterval(() => { $("liveElapsed").textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s"; }, 100);
  }

  function updateLive(job, activeIdx) {
    if ($("liveCard").hidden) return;
    const segs = $("liveBar").children, labels = $("liveLabels").children;
    [...segs].forEach((el, i) => {
      el.className = "live-seg" + (i < activeIdx ? " done" : i === activeIdx && job.status === "running" ? " active" : "");
      labels[i].classList.toggle("on", i <= activeIdx && job.status === "running");
    });
    if (job.status === "queued") {
      $("liveEyebrow").textContent = "Queued";
      $("liveTitle").textContent = "Waiting for the server…";
      $("liveSub").textContent = job.position > 1 ? `${job.position - 1} scan${job.position > 2 ? "s" : ""} ahead of you` : "Starting…";
    } else if (job.status === "running" && job.stage) {
      $("liveEyebrow").textContent = `Step ${activeIdx + 1} of ${stages().length}`;
      $("liveTitle").textContent = STAGE_META[job.stage].name + "…";
      $("liveSub").textContent = STAGE_META[job.stage].desc;
    }
  }

  function markLlmSteps(result) {
    if (result.llm.status === "ok") return;
    ["field_extraction", "safety_net"].forEach((s) => {
      const el = document.querySelector(`.step[data-stage="${s}"]`);
      if (!el) return;
      el.classList.remove("done");
      el.classList.add("warn");
      el.querySelector(".step-desc").textContent = result.final.rejected
        ? "Skipped — no text to verify"
        : `LLM unavailable — regex/DistilBERT only`;
    });
  }

  // ─── Scan ───────────────────────────────────────────────────────────────
  function setBusy(busy) {
    const btn = $("scanBtn");
    btn.disabled = busy || !selectedFile;
    btn.classList.toggle("busy", busy);
    $("scanLabel").textContent = busy ? "Scanning…" : "Scan receipt";
    $("dropZone").style.pointerEvents = busy ? "none" : "";
  }

  async function startScan() {
    if (!selectedFile || polling) return;
    $("errorBox").hidden = true;
    resetSteps();
    setBusy(true);
    polling = true;
    $("emptyState").hidden = true;
    $("results").hidden = true;
    showLive(true);
    if (window.matchMedia("(max-width: 900px)").matches) {
      $("liveCard").scrollIntoView({ behavior: "smooth", block: "start" });
    }

    const form = new FormData();
    form.append("file", selectedFile);
    const thumbPromise = makeThumb(originalUrl);

    try {
      const res = await fetch("/api/scan", { method: "POST", body: form });
      const data = await res.json().catch(() => ({ error: `Server returned ${res.status}` }));
      if (!res.ok || !data.success) throw new Error(data.error || `Server returned ${res.status}`);

      updateSteps({ status: "queued", position: data.position, timings: {} });
      const job = await pollJob(data.job_id);
      updateSteps(job);

      if (job.status === "error") throw new Error(job.error || "Processing failed");

      current = { result: job.result, name: selectedFile.name, thumb: await thumbPromise, ts: Date.now() };
      renderResult(current);
      saveHistory(current);
    } catch (e) {
      const offline = e instanceof TypeError;
      showError(offline ? "Could not reach the server." : "Scan failed.", offline ? "Check that app.py is still running." : e.message);
      $("pipelineState").textContent = "Failed";
      $("emptyState").hidden = false;
    } finally {
      polling = null;
      showLive(false);
      setBusy(false);
    }
  }

  async function pollJob(id) {
    let failures = 0;
    for (;;) {
      await sleep(450);
      try {
        const res = await fetch(`/api/jobs/${id}`, { cache: "no-store" });
        const job = await res.json();
        if (res.status === 404) throw new Error(job.error || "Job expired");
        failures = 0;
        if (job.status === "done" || job.status === "error") return job;
        updateSteps(job);
      } catch (e) {
        if (!(e instanceof TypeError) || ++failures > 20) throw e;   // tolerate brief network blips
      }
    }
  }

  // ─── Rendering ──────────────────────────────────────────────────────────
  function renderResult({ result, name }) {
    showLive(false);
    $("emptyState").hidden = true;
    $("results").hidden = false;

    // Images
    processedUrl = result.processed_image_url;
    if (!selectedFile || selectedFile.name !== name || !originalUrl) {
      originalUrl = result.image_url;
      $("fileName").textContent = name;
      $("previewCard").hidden = false;
    }
    setTabsEnabled(true);
    setView(currentView === "original" ? "compare" : currentView);
    const p = result.preprocess || {};
    $("prepMeta").innerHTML = [
      p.strategy && `<span class="chip">${esc(p.strategy)} strategy</span>`,
      p.sharpness != null && `<span class="chip">sharpness ${Math.round(p.sharpness)}</span>`,
      p.original_size && `<span class="chip">${p.original_size.join("×")} → ${p.processed_size.join("×")}</span>`,
    ].filter(Boolean).join("");

    renderVerdict(result);
    renderReceipt(result);
    renderAgreement(result);
    renderBars("bertBars", result.bert);
    renderBars("fusedBars", result.fused);
    renderBars("cnnBars", result.cnn);
    $("bertTopPred").textContent = `${result.bert.prediction} — ${pct(result.bert.confidence)}`;
    $("fusedTopPred").textContent = `${result.fused.prediction} — ${pct(result.fused.confidence)}`;
    $("cnnTopPred").textContent = `${result.cnn.prediction} — ${pct(result.cnn.confidence)}`;
    $("scoresNote").textContent = "raw model outputs — before the safety net";
    renderOcr(result);

    // Timings into the stepper (also when restored from history)
    updateSteps({ status: "done", timings: result.timings });
    markLlmSteps(result);
    $("totalTime").textContent = secs(result.timings.total);
    $("stepsTotal").hidden = false;

    if (window.matchMedia("(max-width: 900px)").matches) {
      $("results").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function renderVerdict(result) {
    const f = result.final;
    const cat = f.prediction;
    const v = $("verdict");
    v.style.setProperty("--accent", CATEGORY_COLORS[cat] || "#b8402f");
    $("verdictIcon").innerHTML = icon(cat, 30);
    $("heroCategory").textContent = f.rejected ? "Not a receipt" : cat;

    const subs = {
      distilbert: "Decided by DistilBERT (confident)",
      llm_confirmed: `DistilBERT, verified by ${llmName()}`,
      llm_override: `Overridden by ${llmName()} safety net`,
      fallback: "DistilBERT only — LLM unavailable",
      rejected: "Input validation rejected this image",
    };
    $("heroSub").textContent = subs[f.source] || "";

    // Confidence meter always shows DistilBERT's real confidence
    const meter = $("meter");
    meter.hidden = f.rejected;
    const conf = result.bert.confidence;
    $("meterLabel").textContent = `DistilBERT confidence in ${result.bert.prediction}`;
    $("meterValue").textContent = pct(conf);
    $("meterFill").style.width = "0";
    requestAnimationFrame(() => requestAnimationFrame(() => { $("meterFill").style.width = pct(conf); }));

    const badges = [];
    if (f.source === "llm_override") badges.push(`<span class="badge warn">LLM override</span>`);
    if (f.source === "llm_confirmed") badges.push(`<span class="badge ok">LLM verified</span>`);
    if (f.source === "distilbert") badges.push(`<span class="badge ok">High confidence</span>`);
    if (f.source === "fallback") badges.push(`<span class="badge warn">Safety net offline</span>`);
    if (f.rejected) badges.push(`<span class="badge bad">Rejected</span>`);
    if (!f.rejected && result.llm.status === "ok" && f.llm_category && f.llm_category !== result.bert.prediction && f.source === "distilbert")
      badges.push(`<span class="badge">LLM suggested ${esc(f.llm_category)}</span>`);
    badges.push(`<span class="badge">${secs(result.timings.total)} total</span>`);
    $("verdictBadges").innerHTML = badges.join("");

    $("trail").innerHTML = (f.trail || []).map((t) => `<li>${esc(t)}</li>`).join("");
  }

  function renderReceipt(result) {
    const fl = result.fields || {};
    const src = result.field_sources || {};
    const tag = (k) => src[k] === "llm" ? `<span class="src llm" title="Corrected or found by the LLM">LLM</span>`
      : src[k] === "regex" ? `<span class="src" title="Found by regex">RGX</span>` : "";
    const row = (k, cls = "") => {
      const v = fl[k];
      return `<div class="r-row ${cls}"><span class="k">${FIELD_LABELS[k]}</span><span class="dots"></span>
        <span class="v ${v ? "" : "empty"}">${v ? esc(v) : "not found"}${v ? tag(k) : ""}</span></div>`;
    };
    $("receiptCard").innerHTML = `
      <div class="store ${fl.store_name ? "" : "empty"}">${fl.store_name ? esc(fl.store_name) + tag("store_name") : "Store name not detected"}</div>
      ${row("transaction_date")}${row("transaction_time")}${row("tax_amount")}${row("total_amount", "total")}
      <div class="receipt-foot">*** ${esc(result.final.rejected ? "NO RECEIPT DATA" : result.final.prediction.toUpperCase())} ***</div>`;
  }

  function renderAgreement(result) {
    const f = result.final;
    const rows = [
      ["DistilBERT", "Text · primary", result.bert.prediction],
      ["Late fusion", "Image + text", result.fused.prediction],
      ["EfficientNet-B0", "Image only", result.cnn.prediction],
      [llmName(), "LLM safety net", result.llm.status === "ok" ? (f.llm_category || "Not a receipt") : null],
    ];
    const finalCat = f.rejected ? "Not a receipt" : f.prediction;
    const cmp = (p) => p == null ? "na" : (p === finalCat || (f.rejected && p === "Blank / Not a Receipt")) ? "yes" : "no";
    const votes = rows.filter((r) => r[2] != null);
    const agree = votes.filter((r) => cmp(r[2]) === "yes").length;

    $("agreeCard").innerHTML = `
      <div class="card-head" style="margin-bottom:4px"><span class="eyebrow">Agreement with final verdict</span></div>
      ${rows.map(([who, sub, p]) => `
        <div class="agree-row">
          <div class="who">${esc(who)}<small>${esc(sub)}</small></div>
          <div class="what">${p == null ? `<span style="color:var(--ink-faint)">${result.ocr_text ? "unavailable" : "not called"}</span>` : esc(p)}
            <span class="mark ${cmp(p)}">${{ yes: "✓", no: "✗", na: "–" }[cmp(p)]}</span></div>
        </div>`).join("")}
      <div class="agree-summary ${agree === votes.length && !f.rejected ? "" : "split"}">
        ${f.rejected ? "Rejected by input validation — model scores are shown for transparency only."
          : agree === votes.length ? `All ${votes.length} predictors agree.`
          : `${agree} of ${votes.length} predictors agree with the final verdict.`}
      </div>`;
  }

  function renderBars(id, model) {
    const el = $(id);
    el.innerHTML = Object.entries(model.scores).map(([label, score]) => `
      <div class="bar-row ${label === model.prediction ? "top" : ""}" title="${esc(label)}: ${pct(score)}">
        <span class="bar-ico">${icon(label, 15)}</span>
        <div><div class="bar-label">${esc(label)}</div><div class="bar-track"><div class="bar-fill" data-w="${(score * 100).toFixed(1)}%"></div></div></div>
        <span class="bar-pct">${pct(score)}</span>
      </div>`).join("");
    requestAnimationFrame(() => requestAnimationFrame(() =>
      el.querySelectorAll(".bar-fill").forEach((b) => { b.style.width = b.dataset.w; })));
  }

  function renderOcr(result) {
    const box = $("ocrBox");
    const lines = result.ocr_lines || [];
    if (!result.ocr_text) {
      box.textContent = "(No text detected)";
    } else if (ocrView === "lines" && lines.length) {
      box.innerHTML = `<ol>${lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ol>`;
    } else {
      box.textContent = result.ocr_text;
    }
    $("ocrMeta").textContent = `${lines.length} lines · ${result.ocr_text.length} chars`;
  }

  // ─── History (localStorage) ─────────────────────────────────────────────
  function loadHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
  }

  function saveHistory(entry) {
    let list = [entry, ...loadHistory()].slice(0, HISTORY_MAX);
    for (;;) {
      try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); break; }
      catch { if (list.length <= 1) break; list = list.slice(0, -1); }   // quota: drop oldest
    }
    updateHistoryCount();
  }

  function updateHistoryCount() { $("historyCount").textContent = loadHistory().length; }

  function renderHistory() {
    const list = loadHistory();
    const box = $("historyList");
    if (!list.length) {
      box.innerHTML = `<div class="history-empty">No scans yet. Results are kept in this browser only.</div>`;
      return;
    }
    box.innerHTML = "";
    list.forEach((h) => {
      const r = h.result;
      const b = document.createElement("button");
      b.type = "button";
      b.className = "history-item";
      const cat = r.final.rejected ? "Not a receipt" : r.final.prediction;
      b.innerHTML = `<img src="${esc(h.thumb || r.image_url)}" alt="">
        <div><b>${esc(r.fields.store_name || cat)}</b>
        <span>${esc(cat)}${r.fields.total_amount ? " · " + esc(r.fields.total_amount) : ""}</span>
        <span>${new Date(h.ts).toLocaleString()}</span></div>`;
      b.addEventListener("click", () => {
        closeDrawer();
        selectedFile = null;
        originalUrl = h.thumb || r.image_url;
        // Prefer the full-size upload if the server still has it
        const probe = new Image();
        probe.onload = () => { originalUrl = r.image_url; setView(currentView); };
        probe.src = r.image_url;
        current = h;
        $("errorBox").hidden = true;
        $("scanBtn").disabled = true;
        renderResult(h);
      });
      box.appendChild(b);
    });
  }

  function makeThumb(src) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const w = 120, h = Math.round(img.height * (w / img.width));
        const c = document.createElement("canvas");
        c.width = w; c.height = Math.min(h, 200);
        c.getContext("2d").drawImage(img, 0, 0, w, h);
        resolve(c.toDataURL("image/jpeg", 0.7));
      };
      img.onerror = () => resolve(null);
      img.src = src;
    });
  }

  // ─── About modal ────────────────────────────────────────────────────────
  function renderAbout() {
    const mc = info.model_card;
    const bestAcc = Math.max(...mc.models.map((m) => m.accuracy));
    const f1 = Object.entries(mc.bert_per_class_f1);
    $("aboutBody").innerHTML = `
      <h3>The pipeline</h3>
      <ol class="how-list">
        <li><b>Preprocessing</b> — the photo is converted to grayscale, resized to 1800px wide and cleaned. A sharpness check picks a light-sharpen filter for clean scans or adaptive thresholding for noisy phone photos.</li>
        <li><b>OCR</b> — PaddleOCR 3.x reads the text lines from the cleaned image.</li>
        <li><b>Field extraction</b> — regex finds the date, time, tax and total. ${esc(llmName())} then corrects OCR typos, finds the store name and gives its own category guess.</li>
        <li><b>Classification</b> — DistilBERT (fine-tuned on OCR text, max ${info.bert_max_len} tokens) is the primary classifier. EfficientNet-B0 classifies the cleaned image, and a late fusion (${info.fusion.cnn} × CNN + ${info.fusion.bert} × BERT) is shown for comparison.</li>
        <li><b>Safety net</b> — if DistilBERT is below ${Math.round(info.threshold * 100)}% confidence, or the LLM flags an unusual receipt, the LLM's category is used instead. Images with no receipt text are rejected. If the LLM is unreachable, the system falls back to DistilBERT.</li>
      </ol>

      <h3>Validation results</h3>
      <p>${mc.dataset.images.toLocaleString()} labelled receipts, ${mc.dataset.categories} categories, ${esc(mc.dataset.split)} split; measured on ${mc.dataset.val_images} validation images, with minority classes augmented in training only.</p>
      <table class="metric-table">
        <thead><tr><th>Model</th><th>Input</th><th class="num">Accuracy</th><th class="num">Macro F1</th><th class="num">Weighted F1</th></tr></thead>
        <tbody>${mc.models.map((m) => `
          <tr class="${m.accuracy === bestAcc ? "best" : ""}"><td>${esc(m.name)}<br><small style="color:var(--ink-soft)">${esc(m.role)}</small></td>
          <td>${esc(m.input)}</td><td class="num">${pct(m.accuracy)}</td><td class="num">${m.macro_f1.toFixed(3)}</td><td class="num">${m.weighted_f1.toFixed(3)}</td></tr>`).join("")}
        </tbody>
      </table>

      <h3>DistilBERT F1 by category</h3>
      ${f1.map(([cat, v]) => `
        <div class="f1-row">${icon(cat, 15)}<span>${esc(cat)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${v * 100}%;background:${CATEGORY_COLORS[cat]}"></div></div>
          <span class="num">${v.toFixed(3)}</span></div>`).join("")}
      <p style="margin-top:14px;color:var(--ink-soft);font-size:.82rem">Home &amp; Hardware and Household &amp; Apparel are confused most often: their receipts look alike and list similar physical goods.</p>`;
  }

  // ─── Drawers / modal ────────────────────────────────────────────────────
  function openDrawer() { renderHistory(); $("historyScrim").hidden = $("historyDrawer").hidden = false; }
  function closeDrawer() { $("historyScrim").hidden = $("historyDrawer").hidden = true; }
  function openAbout() { if (info) { $("aboutScrim").hidden = $("aboutModal").hidden = false; } }
  function closeAbout() { $("aboutScrim").hidden = $("aboutModal").hidden = true; }

  // ─── Export ─────────────────────────────────────────────────────────────
  function downloadJson() {
    if (!current) return;
    const r = current.result;
    const out = {
      file: current.name,
      scanned_at: new Date(current.ts).toISOString(),
      category: r.final.prediction,
      decision: { source: r.final.source, rejected: r.final.rejected, trail: r.final.trail },
      fields: r.fields,
      models: {
        distilbert: { prediction: r.bert.prediction, confidence: r.bert.confidence, scores: r.bert.scores },
        late_fusion: { prediction: r.fused.prediction, confidence: r.fused.confidence, scores: r.fused.scores },
        efficientnet_b0: { prediction: r.cnn.prediction, confidence: r.cnn.confidence, scores: r.cnn.scores },
        llm: { model: r.llm.model, status: r.llm.status, category: r.llm.category },
      },
      ocr_lines: r.ocr_lines,
      timings_seconds: r.timings,
    };
    const slug = (r.fields.store_name || "receipt").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "receipt";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], { type: "application/json" }));
    a.download = `${slug}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function fieldsText() {
    const r = current.result, f = r.fields;
    return [
      `Store: ${f.store_name || "-"}`,
      `Date: ${f.transaction_date || "-"}`,
      `Time: ${f.transaction_time || "-"}`,
      `Tax: ${f.tax_amount || "-"}`,
      `Total: ${f.total_amount || "-"}`,
      `Category: ${r.final.prediction}`,
    ].join("\n");
  }

  // ─── Events ─────────────────────────────────────────────────────────────
  function bind() {
    const dz = $("dropZone");
    dz.addEventListener("click", () => $("fileInput").click());
    dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("fileInput").click(); } });
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      if (!polling && e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    // Paste an image straight from the clipboard
    document.addEventListener("paste", (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
      if (item && !polling) {
        const f = item.getAsFile();
        handleFile(new File([f], `pasted.${f.type.split("/")[1] || "png"}`, { type: f.type }));
      }
    });

    $("browseBtn").addEventListener("click", () => $("fileInput").click());
    $("cameraBtn").addEventListener("click", () => $("cameraInput").click());
    $("fileInput").addEventListener("change", (e) => handleFile(e.target.files[0]));
    $("cameraInput").addEventListener("change", (e) => handleFile(e.target.files[0]));
    $("clearBtn").addEventListener("click", () => { if (!polling) { clearFile(); resetSteps(); } });
    $("scanBtn").addEventListener("click", startScan);

    document.querySelectorAll("[data-view]").forEach((t) => t.addEventListener("click", () => setView(t.dataset.view)));
    $("viewer").addEventListener("click", (e) => { if (e.target.tagName === "IMG") openLightbox(e.target.src); });
    $("lightbox").addEventListener("click", () => { $("lightbox").hidden = true; });

    document.querySelectorAll("[data-ocr]").forEach((t) => t.addEventListener("click", () => {
      ocrView = t.dataset.ocr;
      document.querySelectorAll("[data-ocr]").forEach((x) => x.setAttribute("aria-selected", String(x === t)));
      if (current) renderOcr(current.result);
    }));
    $("copyOcrBtn").addEventListener("click", () => current && copyText(
      ocrView === "lines" ? current.result.ocr_lines.join("\n") : current.result.ocr_text, "OCR text copied"));
    $("copyFieldsBtn").addEventListener("click", () => current && copyText(fieldsText(), "Fields copied"));
    $("downloadBtn").addEventListener("click", downloadJson);
    $("newScanBtn").addEventListener("click", () => {
      clearFile(); resetSteps();
      $("results").hidden = true; $("emptyState").hidden = false;
      window.scrollTo({ top: 0, behavior: "smooth" });
      $("fileInput").click();
    });

    $("historyBtn").addEventListener("click", openDrawer);
    $("closeHistoryBtn").addEventListener("click", closeDrawer);
    $("historyScrim").addEventListener("click", closeDrawer);
    $("clearHistoryBtn").addEventListener("click", () => { localStorage.removeItem(HISTORY_KEY); updateHistoryCount(); renderHistory(); });
    $("aboutBtn").addEventListener("click", openAbout);
    $("closeAboutBtn").addEventListener("click", closeAbout);
    $("aboutScrim").addEventListener("click", closeAbout);
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      $("lightbox").hidden = true; closeDrawer(); closeAbout();
    });
  }

  bind();
  resetSteps();
  updateHistoryCount();
  loadInfo().then(resetSteps);
})();
