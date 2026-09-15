// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

let state = { trips: [], auditEvents: [], stats: {} };
let selectedTripId = null;
let selectedExpenseId = null;
let health = {};
let activeTab = "workflow";

const $ = (id) => document.getElementById(id);

init();

async function init() {
  $("importBtn").addEventListener("click", importFolder);
  $("receiptFolder").addEventListener("change", renderFolderFileSummary);
  $("resetBtn").addEventListener("click", resetDemo);
  $("saveKeysBtn").addEventListener("click", saveKeys);
  $("clearKeysBtn").addEventListener("click", clearKeys);
  $("nvidiaApiKey").addEventListener("input", renderKeyStatus);
  for (const button of document.querySelectorAll("[data-tab]")) {
    button.addEventListener("click", () => setTab(button.getAttribute("data-tab") || "workflow"));
  }
  restoreKeys();
  health = await api("/api/health");
  $("healthMode").textContent = health.modelExecutionMode;
  $("statsMode").textContent = health.modelExecutionMode;
  $("metricsMode").textContent = health.modelExecutionMode;
  renderModelPath();
  renderKeyStatus();
  renderConfiguredModels();
  renderAgentControlFlow();
  await refresh();
}

async function refresh() {
  state = await api("/api/state");
  if (!selectedTripId && state.trips[0]) selectedTripId = state.trips[0].id;
  renderTrips();
  renderStats();
  renderPhaseMetrics();
  renderDetail();
}

function setTab(tab) {
  activeTab = tab;
  for (const button of document.querySelectorAll("[data-tab]")) {
    const isActive = button.getAttribute("data-tab") === tab;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  }
  for (const pane of document.querySelectorAll(".tab-pane")) pane.classList.toggle("active", pane.id === `${tab}Pane`);
}

async function importFolder() {
  const { supported: files, ignored } = selectedReceiptFiles();
  if (!files.length) return showProgress("Choose a receipt folder with JPG, JPEG, or PNG files first.", 0, true);
  $("importBtn").disabled = true;
  showProgress(`Creating trip for ${files.length} supported receipt images${ignored.length ? ` and ignoring ${ignored.length} unsupported/system file${ignored.length === 1 ? "" : "s"}` : ""}.`, 1);
  try {
    const { trip } = await api("/api/trips", {
      method: "POST",
      body: { employeeName: $("employeeName").value, tripName: $("tripName").value, tripPurpose: $("tripPurpose").value, totalFiles: files.length },
    });
    selectedTripId = trip.id;
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      showProgress(`Processing ${index + 1} of ${files.length}: ${file.webkitRelativePath || file.name}`, Math.round((index / files.length) * 100));
      const contentBase64 = await fileToBase64(file);
      await api(`/api/trips/${trip.id}/files`, { method: "POST", body: { fileName: file.name, mimeType: file.type || "image/png", contentBase64, lastModified: file.lastModified, modelApiKeys: sessionApiKeys() } });
      showProgress(`Processed ${index + 1} of ${files.length}: ${file.webkitRelativePath || file.name}`, Math.round(((index + 1) / files.length) * 100));
      await refresh();
    }
    const completed = await api(`/api/trips/${trip.id}/complete`, { method: "POST", body: {} });
    const status = completed.trip?.status === "blocked" ? "Human review is required for at least one receipt." : "Trip report is ready for review.";
    showProgress(`Finished ${files.length} receipt images. ${status}`, 100);
    await refresh();
  } catch (error) {
    showProgress(error.message || String(error), 100, true);
  } finally {
    $("importBtn").disabled = false;
  }
}

function selectedReceiptFiles() {
  const all = Array.from($("receiptFolder").files || []);
  const supported = all.filter((file) => isSupportedReceiptImage(file));
  const ignored = all.filter((file) => !isSupportedReceiptImage(file));
  return { all, supported, ignored };
}

function isSupportedReceiptImage(file) {
  return /\.(png|jpe?g)$/i.test(file.name);
}

function renderFolderFileSummary() {
  const { all, supported, ignored } = selectedReceiptFiles();
  const node = $("folderFileSummary");
  if (!all.length) {
    node.textContent = "No folder selected.";
    node.classList.remove("warn");
    return;
  }
  const ignoredNames = ignored.slice(0, 3).map((file) => file.name).join(", ");
  const suffix = ignored.length ? ` ${ignored.length} ignored unsupported/system file${ignored.length === 1 ? "" : "s"}${ignoredNames ? ` (${ignoredNames}${ignored.length > 3 ? ", ..." : ""})` : ""}.` : " No ignored files.";
  node.textContent = `${supported.length} supported receipt image${supported.length === 1 ? "" : "s"} will be imported from ${all.length} selected file${all.length === 1 ? "" : "s"}.${suffix}`;
  node.classList.toggle("warn", ignored.length > 0);
}

function showProgress(message, percent, error = false) {
  const node = $("progress");
  node.classList.remove("hidden");
  node.innerHTML = `<strong>${error ? "Processing error" : "Processing Receipt Folder"}</strong><div class="bar"><div style="width:${Math.max(0, Math.min(100, percent))}%"></div></div><p>${escapeHtml(message)}</p>`;
  node.style.background = error ? "#fff1f0" : "#ecfdf5";
  node.style.borderColor = error ? "#f4b4ad" : "#bee3d2";
}

function sessionApiKeys() {
  const nvidiaApiKey = $("nvidiaApiKey").value.trim();
  return { nvidiaApiKey };
}

function restoreKeys() {
  const stored = JSON.parse(sessionStorage.getItem("expenseIntelligence.apiKeys") || "{}");
  $("nvidiaApiKey").value = stored.nvidiaApiKey || "";
}

function saveKeys() {
  sessionStorage.setItem("expenseIntelligence.apiKeys", JSON.stringify(sessionApiKeys()));
  renderKeyStatus();
}

function clearKeys() {
  sessionStorage.removeItem("expenseIntelligence.apiKeys");
  $("nvidiaApiKey").value = "";
  renderKeyStatus();
}

function renderKeyStatus() {
  const keys = sessionApiKeys();
  const hasBrowserKey = Boolean(keys.nvidiaApiKey);
  $("keyStatus").innerHTML = `<strong>${hasBrowserKey ? "Browser-session key configured" : "No browser-session key configured"}</strong><p>${hasBrowserKey ? "Imports will send the key only with receipt-processing requests so the server can call Nemotron Parse and Omni." : "Imports will use server env keys if configured; otherwise they fail clearly. The default developer workflow does not silently use local fixture extraction."}</p><p>Server env key configured: <strong>${health.serverEnvKeyConfigured ? "yes" : "no"}</strong></p>`;
  renderConfiguredModels();
}

function renderConfiguredModels() {
  const keys = sessionApiKeys();
  const hasBrowserKey = Boolean(keys.nvidiaApiKey);
  const parseReady = Boolean(keys.nvidiaApiKey || health.serverEnvKeyConfigured);
  const omniReady = Boolean(keys.nvidiaApiKey || health.serverEnvKeyConfigured);
  $("configuredModels").innerHTML = [
    modelConfigRow("Parse", health.parseModel || "nvidia/nemotron-parse", health.parseBaseUrl || "https://integrate.api.nvidia.com", parseReady, hasBrowserKey),
    modelConfigRow("Omni", health.omniModel || "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", health.omniBaseUrl || "https://integrate.api.nvidia.com", omniReady, hasBrowserKey),
  ].join("");
}

function modelConfigRow(labelText, model, baseUrl, ready, hasBrowserKey) {
  return `<div class="model-config-row"><span class="row"><strong>${escapeHtml(labelText)}</strong><span class="badge ${ready ? "ready" : "warn"}">${ready ? "credential available" : "missing key"}</span></span><code>${escapeHtml(model)}</code><span>${escapeHtml(baseUrl)}</span><span>${hasBrowserKey ? "Browser-session key ready" : health.serverEnvKeyConfigured ? "Server env key configured" : "No key configured"}</span></div>`;
}

function renderTrips() {
  $("tripCount").textContent = state.trips.length;
  $("tripList").innerHTML = state.trips.length ? state.trips.map((trip) => `<div class="card ${trip.id === selectedTripId ? "active" : ""}" onclick="selectTrip('${trip.id}')"><span class="badge ${trip.status === "blocked" ? "blocked" : "ready"}">${label(trip.status)}</span><h3>${escapeHtml(trip.tripName)}</h3><p>${escapeHtml(trip.employeeName)} · ${trip.expenseIds.length} receipts · ${trip.skippedFiles.length} skipped</p></div>`).join("") : `<p class="hint">No trip reports yet.</p>`;
}

function renderStats() {
  const s = state.stats || {};
  const phase = s.phaseTotals || {};
  const auditStats = auditMetrics();
  $("stats").innerHTML = [
    stat("Receipts", s.receiptCount || 0, `${s.readyCount || 0} ready · ${s.blockedCount || 0} blocked`),
    stat("Avg confidence", `${s.averageConfidence || 0}%`, "field extraction confidence"),
    stat("Policy warnings", s.warningCount || 0, "ABC Company rules"),
    stat("Total time", `${s.totalProcessingMs || 0} ms`, "all receipts"),
    stat("Nemotron Parse", auditStats.parseCalls, `${auditStats.parseLatencyMs} ms observed latency`),
    stat("Nemotron Omni", auditStats.omniCalls, `${auditStats.omniLatencyMs} ms observed latency`),
    stat("Runtime checks", auditStats.runtimeChecks, "NemoClaw-style decisions"),
    stat("Audit events", auditStats.auditEvents, "governance records"),
    stat("Parse + Omni", `${phase.parse_omni_repair || 0} ms`, "model/fixture and repair phase"),
    stat("Policy", `${phase.abc_policy || 0} ms`, "policy phase"),
  ].join("");
}

function auditMetrics() {
  const events = state.auditEvents || [];
  const modelEvents = events.filter((event) => event.type === "model.call" || event.type === "model.routing" || event.action?.includes("nemotron"));
  const parseEvents = modelEvents.filter((event) => event.action?.includes("parse") || JSON.stringify(event.details || {}).toLowerCase().includes("parse"));
  const omniEvents = modelEvents.filter((event) => event.action?.includes("omni") || JSON.stringify(event.details || {}).toLowerCase().includes("omni"));
  const runtimeEvents = events.filter((event) => event.type?.startsWith("runtime."));
  const latency = (items) => items.reduce((sum, event) => sum + (Number(event.details?.latencyMs) || 0), 0);
  return {
    parseCalls: parseEvents.length,
    omniCalls: omniEvents.length,
    parseLatencyMs: latency(parseEvents),
    omniLatencyMs: latency(omniEvents),
    runtimeChecks: runtimeEvents.length,
    auditEvents: events.length,
  };
}

function renderModelPath() {
  $("modelPath").innerHTML = [
    pathStep("1. Nemotron Parse", health.parseModel || "nvidia/nemotron-parse", `Endpoint: ${health.parseBaseUrl || "https://integrate.api.nvidia.com"}`),
    pathStep("2. Nemotron Omni", health.omniModel || "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", `Endpoint: ${health.omniBaseUrl || "https://integrate.api.nvidia.com"}`),
    pathStep("3. NemoClaw-style controls", "local runtime policy", "Tool allowlist, outbound host checks, secret redaction, audit, human approval gate."),
  ].join("");
}

function pathStep(title, value, note) {
  return `<div class="path-step"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(value)}</span><p>${escapeHtml(note)}</p></div>`;
}

function renderPhaseMetrics() {
  const trip = state.trips.find((candidate) => candidate.id === selectedTripId) || state.trips[0];
  const container = $("phaseMetrics");
  if (!trip) {
    container.innerHTML = `<p class="body-copy">Import receipts to see per-phase timing for intake, Parse/Omni extraction, schema repair, policy evaluation, and review readiness.</p>`;
    return;
  }
  const rows = [];
  for (const expense of trip.expenses) {
    for (const phase of expense.performance) rows.push(`<tr><td>${escapeHtml(expense.fileName)}</td><td>${escapeHtml(phase.name)}</td><td>${phase.durationMs} ms</td></tr>`);
    rows.push(`<tr><td>${escapeHtml(expense.fileName)}</td><td>model provider</td><td>${escapeHtml(expense.extracted.provider)} · ${escapeHtml(expense.extracted.model)}</td></tr>`);
  }
  container.innerHTML = rows.length ? `<table><thead><tr><th>Receipt</th><th>Phase</th><th>Value</th></tr></thead><tbody>${rows.join("")}</tbody></table>` : `<p class="body-copy">No receipt metrics yet.</p>`;
}

function renderAgentControlFlow() {
  const stages = [
    ["Receipt folder", "Web intake", "Browser uploads JPG, JPEG, or PNG receipts as normalized file events."],
    ["Orchestration", "Expense agent", "Creates one trip, processes each receipt, tracks progress, and coordinates extraction, repair, policy, review, and export."],
    ["Runtime controls", "NemoClaw-style policy", "Authorizes tools, checks outbound model hosts, redacts secrets, records audit events, and requires human approval."],
    ["Model tools", "Nemotron Parse + Omni", "Parse reads receipt evidence; Omni normalizes fields, confidence, provenance, and rationale."],
    ["Scoped memory", "Workflow context", "Stores per-receipt Parse/Omni outputs, saved corrections, policy results, audit events, and trip-level context that systems such as Oracle Fusion can use to understand the full expense workflow."],
    ["Human review", "Developer / employee", "Reviews image beside fields, fixes AI mistakes, saves values, and approves downstream handoff."],
  ];
  $("agentControlFlow").innerHTML = `<div class="workflow-diagram">${stages.map((stage, index) => `<div class="workflow-node ${index === 1 || index === 2 ? "is-agentic" : ""}"><span class="workflow-index">${index + 1}</span><strong>${escapeHtml(stage[0])}</strong><span>${escapeHtml(stage[1])}</span><p>${escapeHtml(stage[2])}</p></div>${index < stages.length - 1 ? `<span class="workflow-arrow">→</span>` : ""}`).join("")}</div><p class="scroll-hint">Agentic behavior is the orchestration loop: route model calls, validate evidence, repair missing fields, apply policy, update state, and decide whether to continue or ask for human review.</p>`;
}

function stat(labelText, value, note) {
  return `<div class="stat"><span>${escapeHtml(labelText)}</span><strong>${escapeHtml(String(value))}</strong><span>${escapeHtml(note)}</span></div>`;
}

function renderDetail() {
  const trip = state.trips.find((candidate) => candidate.id === selectedTripId);
  const detail = $("detail");
  if (!trip) { detail.className = "detail empty"; detail.textContent = "Import a receipt folder to create the first governed trip report."; return; }
  detail.className = "detail";
  const warnings = trip.expenses.flatMap((expense) => expense.policyChecks.filter((check) => check.decision === "warn").map((check) => ({ ...check, fileName: expense.fileName })));
  const audits = state.auditEvents.filter((event) => event.tripId === trip.id).slice(0, 80);
  detail.innerHTML = `
    <details open><summary>Trip Report</summary><div class="section-body"><h2>${escapeHtml(trip.tripName)}</h2><p>${escapeHtml(trip.employeeName)} · ${escapeHtml(trip.tripPurpose)}</p><p><span class="badge ${trip.status === "blocked" ? "blocked" : "ready"}">${label(trip.status)}</span>${trip.processedFiles}/${trip.totalFiles} files processed</p><div class="actions"><button onclick="approveTrip('${trip.id}')">Approve downstream handoff</button><button class="secondary" onclick="downloadCsv('${trip.id}')">Download CSV</button></div></div></details>
    <div class="review-workspace">
      <details class="receipt-list-panel" open><summary>Receipts To Review</summary><div class="section-body receipts">${trip.expenses.map(renderReceiptRow).join("")}</div></details>
      <div class="receipt-review-panel">${renderExpenseReview(trip.expenses.find((expense) => expense.id === selectedExpenseId)) || renderReceiptReviewPlaceholder()}</div>
    </div>
    <details><summary>Expense CSV Export</summary><div class="section-body"><p>Download uses the saved fields currently visible in the trip report.</p><button onclick="downloadCsv('${trip.id}')">Download CSV</button></div></details>
    <details><summary>ABC Company Policy Warnings</summary><div class="section-body">${warnings.length ? warnings.map((w) => `<p><span class="badge warn">warn</span><strong>${escapeHtml(w.fileName)}</strong>: ${escapeHtml(w.reason)}</p>`).join("") : "<p>No warnings.</p>"}</div></details>
    <details><summary>Agent Control Trace</summary><div class="section-body"><table><tbody>${trip.agentTrace.steps.map((step) => `<tr><td>${escapeHtml(step.toolAction)}</td><td>${escapeHtml(step.status)}</td><td>${escapeHtml(step.summary)}</td></tr>`).join("")}</tbody></table></div></details>
    <details><summary>Governance Audit Timeline</summary><div class="section-body"><table><tbody>${audits.map((event) => `<tr><td>${escapeHtml(event.timestamp)}</td><td>${escapeHtml(event.action)}</td><td>${escapeHtml(event.severity)}</td></tr>`).join("")}</tbody></table></div></details>`;
}

function renderReceiptRow(expense) {
  const f = expense.savedFields;
  const warnings = expense.policyChecks.filter((check) => check.decision === "warn").length;
  return `<div class="receipt-row ${expense.id === selectedExpenseId ? "active" : ""}" onclick="selectExpense('${expense.id}')"><div><span class="badge ${expense.status === "blocked" ? "blocked" : "ready"}">${label(expense.status)}</span>${warnings ? `<span class="badge warn">${warnings} warnings</span>` : ""}<h3>${escapeHtml(f.merchant || expense.fileName)}</h3><p>${escapeHtml([f.transactionDate, money(f.amount, f.currency), f.category, f.paymentMethod].filter(Boolean).join(" · "))}</p></div><strong>Review</strong></div>`;
}

function renderExpenseReview(expense) {
  if (!expense) return "";
  const f = expense.savedFields;
  return `<details open><summary>Receipt Review: ${escapeHtml(expense.fileName)}</summary><div class="section-body review"><div><img class="receipt-img" src="${escapeHtml(f.receiptFileRef || "")}" alt="Receipt image" />${extractionSourceNote(expense)}</div><div><h3>Saved Fields</h3><p class="body-copy">Review AI-extracted values, correct mistakes, then save. Saved values are used in policy checks, metrics, and CSV export.</p><div class="field-grid">${fieldInput("merchant", f.merchant, expense)}${fieldInput("transactionDate", f.transactionDate, expense, "date")}${fieldInput("amount", f.amount, expense, "number")}${fieldInput("currency", f.currency, expense)}${fieldInput("tax", f.tax, expense, "number")}${fieldInput("tip", f.tip, expense, "number")}${fieldInput("location", f.location, expense)}${fieldInput("category", f.category, expense)}${fieldInput("paymentMethod", f.paymentMethod, expense)}${fieldInput("checkInDate", f.checkInDate, expense, "date")}${fieldInput("checkOutDate", f.checkOutDate, expense, "date")}</div><button onclick="saveFields('${expense.id}')">Save fields</button><h3>Before / After Agent Phases</h3>${renderAgentPhases(expense)}</div></div></details>`;
}

function renderReceiptReviewPlaceholder() {
  return `<div class="receipt-review-placeholder"><h3>Select a receipt to review</h3><p>Click any receipt line on the left to render the original image, extracted fields, confidence notes, and before/after agent phases here.</p></div>`;
}

function extractionSourceNote(expense) {
  if (expense.extracted.provider === "nvidia-nemotron") {
    return `<div class="model-pill">Extraction source: Nemotron Parse + Omni<br /><span>${escapeHtml(expense.extracted.model)}</span></div>`;
  }
  return `<div class="model-pill warn">Extraction source: local deterministic fixture evidence<br /><span>Nemotron was not called because this receipt was processed in explicit local fixture mode. For the default demo, add a build.nvidia.com key in API Keys and re-import.</span></div>`;
}

function renderAgentPhases(expense) {
  const phases = {
    "1. Receipt image": expense.fileName,
    "2. Nemotron Parse evidence": expense.extracted.rawText.slice(0, 1800),
    "3. Nemotron Omni structured output": expense.extracted.fields,
    "4. Schema guard + policy-ready saved fields": expense.savedFields,
    "5. Display-safe rationale": expense.extracted.reasoning.slice(0, 8),
  };
  return Object.entries(phases).map(([title, value]) => `<details class="phase" open><summary>${escapeHtml(title)}</summary><pre>${escapeHtml(typeof value === "string" ? value : JSON.stringify(value, null, 2))}</pre></details>`).join("");
}

function fieldInput(name, value, expense, type = "text") {
  const confidence = expense.extracted.confidence[name];
  const reasoning = expense.extracted.reasoning.find((item) => item.field === name);
  const note = confidence ? `Confidence ${Math.round(confidence * 100)}%. ${reasoning?.summary || ""}` : reasoning?.summary || "";
  return `<label>${label(name)}${confidence ? `<span class="badge">${Math.round(confidence * 100)}%</span>` : ""}<input data-field="${name}" type="${type}" value="${escapeHtml(value ?? "")}" />${note ? `<span class="field-note">${escapeHtml(note)}</span>` : ""}</label>`;
}

async function saveFields(expenseId) {
  const fields = {};
  document.querySelectorAll(`[data-field]`).forEach((input) => {
    const key = input.getAttribute("data-field");
    if (!key) return;
    const value = input.type === "number" && input.value !== "" ? Number(input.value) : input.value || null;
    fields[key] = value;
  });
  await api(`/api/expenses/${expenseId}/fields`, { method: "PATCH", body: { fields } });
  await refresh();
}

async function approveTrip(tripId) {
  await api(`/api/trips/${tripId}/approve`, { method: "POST", body: { approvedBy: "ABC Expense Reviewer" } });
  await refresh();
}

function downloadCsv(tripId) {
  window.location.href = `/api/trips/${tripId}/export.csv`;
}

async function resetDemo() {
  await api("/api/reset", { method: "POST", body: {} });
  selectedTripId = null;
  selectedExpenseId = null;
  $("progress").classList.add("hidden");
  await refresh();
}

function selectTrip(id) { selectedTripId = id; selectedExpenseId = null; renderTrips(); renderDetail(); }
function selectExpense(id) { selectedExpenseId = id; renderDetail(); }
window.selectTrip = selectTrip;
window.selectExpense = selectExpense;
window.saveFields = saveFields;
window.approveTrip = approveTrip;
window.downloadCsv = downloadCsv;

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs || 240000);
  try {
    const response = await fetch(path, { method: options.method || "GET", headers: { "content-type": "application/json" }, body: options.body ? JSON.stringify(options.body) : undefined, signal: controller.signal });
    const text = await response.text();
    const payload = text ? JSON.parse(text) : {};
    if (!response.ok || payload.error) throw new Error(payload.error || response.statusText);
    return payload;
  } catch (error) {
    if (error?.name === "AbortError") throw new Error("Request timed out while waiting for receipt processing.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function money(amount, currency = "USD") { return typeof amount === "number" ? `${currency || "USD"} ${amount.toFixed(2)}` : ""; }
function label(value) { return String(value || "").replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase()); }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[ch])); }
