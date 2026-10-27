(() => {
  "use strict";

  const state = { report: null, profile: null, jobs: [], activeJobId: null, pollTimer: null };
  const $ = (selector) => document.querySelector(selector);
  const elements = {
    healthPill: $("#health-pill"), healthLabel: $("#health-label"), footerHealth: $("#footer-health"),
    message: $("#global-message"), jobForm: $("#job-form"), jobDescription: $("#job-description"),
    count: $("#character-count"), clearJob: $("#clear-job"), analyze: $("#analyze-button"),
    report: $("#report-content"), profile: $("#profile-content"), generationForm: $("#generation-form"),
    includeCoverLetter: $("#include-cover-letter"), generate: $("#generate-button"), jobs: $("#jobs-content")
  };

  const create = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  };
  const valueAt = (object, paths, fallback = "") => {
    for (const path of paths) {
      const value = path.split(".").reduce((result, key) => result && result[key], object);
      if (value !== undefined && value !== null && value !== "") return value;
    }
    return fallback;
  };
  const asArray = (value) => Array.isArray(value) ? value : [];
  const pretty = (value) => String(value || "").replaceAll("_", " ");
  const statusText = (value) => pretty(value).replace(/\b\w/g, (letter) => letter.toUpperCase());

  function setMessage(text, kind = "error") {
    elements.message.hidden = !text;
    elements.message.textContent = text || "";
    elements.message.classList.toggle("is-info", kind === "info");
  }

  async function api(path, options = {}) {
    const response = await fetch(path, { headers: { "Accept": "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}) }, ...options });
    let payload = null;
    try { payload = await response.json(); } catch (_) { payload = null; }
    if (!response.ok) {
      const nestedError = payload?.error;
      const detail = typeof nestedError === "object" && nestedError !== null
        ? nestedError.message || nestedError.code
        : valueAt(payload || {}, ["detail", "message", "error"], `Request failed (${response.status})`);
      throw new Error(String(detail || `Request failed (${response.status})`));
    }
    return payload;
  }

  function setButtonBusy(button, busy, busyText) {
    button.disabled = busy;
    if (busy) {
      button.dataset.idleLabel = button.querySelector("span")?.textContent || "";
      if (button.querySelector("span")) button.querySelector("span").textContent = busyText;
    } else if (button.dataset.idleLabel && button.querySelector("span")) {
      button.querySelector("span").textContent = button.dataset.idleLabel;
    }
  }

  function updateCount() { elements.count.textContent = `${elements.jobDescription.value.length.toLocaleString()} / 30,000`; }

  function loadSavedDraft() {
    try {
      const saved = window.localStorage.getItem("rie.jobDescription");
      if (saved && !elements.jobDescription.value) elements.jobDescription.value = saved;
    } catch (_) { /* Private browsing or disabled storage is still a valid mode. */ }
    updateCount();
  }

  function saveDraft() {
    try { window.localStorage.setItem("rie.jobDescription", elements.jobDescription.value); } catch (_) { /* Ignore storage restrictions. */ }
  }

  function renderHealth(payload, offline = false) {
    const healthy = !offline && (payload?.status === "ok" || payload?.healthy === true || payload?.ok === true || payload?.status === "healthy");
    const configured = payload?.configured === false || payload?.provider_configured === false;
    const dot = elements.healthPill.querySelector(".status-dot");
    dot.className = `status-dot ${offline ? "status-dot--offline" : configured ? "status-dot--loading" : healthy ? "" : "status-dot--error"}`;
    const label = offline ? "Engine unreachable" : configured ? "Local engine · configure provider" : healthy ? "Local engine ready" : "Engine needs attention";
    elements.healthLabel.textContent = label;
    elements.footerHealth.textContent = offline ? "offline" : healthy ? "ready" : "needs attention";
  }

  function renderProfile(profile) {
    state.profile = profile || {};
    elements.profile.replaceChildren();
    const isSummary = profile && (Object.prototype.hasOwnProperty.call(profile, "name") || Object.prototype.hasOwnProperty.call(profile, "target_roles"));
    const info = profile?.personal_info || profile?.personal || (isSummary ? profile : {});
    const positioning = profile?.positioning && typeof profile.positioning === "object" ? profile.positioning : {};
    const summaryPositioning = typeof profile?.positioning === "string" ? profile.positioning : "Evidence-backed profile loaded locally.";
    const targetRoles = asArray(profile?.target_roles || positioning?.target_roles);
    const preferences = profile?.candidate_preferences || {};
    const cardMain = create("div", "profile-main");
    cardMain.append(create("p", "profile-name", valueAt(info, ["name", "full_name", "profile_owner"], "Local candidate")));
    cardMain.append(create("p", "profile-positioning", valueAt(positioning, ["core_identity", "public_current_focus"], summaryPositioning)));
    const cardMeta = create("div", "profile-meta");
    const dl = create("dl");
    const addMeta = (label, value) => { const dt = create("dt", null, label); const dd = create("dd", null, value || "Not specified"); dl.append(dt, dd); };
    addMeta("Location", valueAt(info, ["location"], valueAt(preferences, ["current_location"], "Not specified")));
    addMeta("Focus", targetRoles.slice(0, 2).join(" · ") || valueAt(profile, ["current_level"], valueAt(positioning, ["current_level"], "Evidence pool")));
    addMeta("Workflow", "Local-first · human review");
    cardMeta.append(dl); cardMain && elements.profile.append(cardMain, cardMeta);
  }

  function renderReport(report) {
    state.report = report;
    elements.report.replaceChildren();
    const summary = report?.summary || {};
    const requirements = asArray(report?.requirements || report?.requirement_matches);
    const metrics = create("div", "report-grid");
    [["matched", "Strong evidence", "metric-card--warm"], ["partial", "Provisional", ""], ["missing", "Needs evidence", ""]].forEach(([key, label, className]) => {
      const card = create("div", `metric-card ${className}`); card.append(create("span", "metric-value", summary[key] ?? requirements.filter((item) => item.status === key).length), create("span", "metric-label", label)); metrics.append(card);
    });
    const panels = create("div", "report-panels");
    const evidencePanel = create("div", "panel");
    const evidenceHeading = create("div", "panel-heading"); evidenceHeading.append(create("h3", null, "Requirements & evidence"), create("span", "panel-note", `${requirements.length} detected`)); evidencePanel.append(evidenceHeading);
    if (!requirements.length) {
      evidencePanel.append(create("p", "evidence-item", "No explicit skill requirements were detected in this description."));
    } else {
      const list = create("ul", "requirement-list");
      requirements.forEach((item) => {
        const li = create("li", "requirement");
        const top = create("div", "requirement-top"); top.append(create("span", "requirement-name", statusText(item.requirement || item.name || "Requirement")), create("span", `status-label status-label--${item.status || "missing"}`, statusText(item.status || "unknown"))); li.append(top);
        const evidence = [...asArray(item.evidence), ...asArray(item.provisional_evidence)];
        evidence.slice(0, 3).forEach((entry) => { const p = create("p", "evidence-item"); p.append(create("span", "evidence-path", entry.path || entry.source || "Evidence"), ` — ${entry.excerpt || entry.claim || "linked support"}`); li.append(p); });
        list.append(li);
      }); evidencePanel.append(list);
    }
    const contextPanel = create("div", "panel");
    const contextHeading = create("div", "panel-heading"); contextHeading.append(create("h3", null, "Review context"), create("span", "panel-note", "Human decision")); contextPanel.append(contextHeading);
    const roles = asArray(report?.role_families); if (roles.length) { contextPanel.append(create("p", "evidence-item", "Role families")); const tags = create("div", "role-tags"); roles.forEach((role) => tags.append(create("span", "tag", statusText(role)))); contextPanel.append(tags); }
    const selection = report?.resume_selection || {};
    const selected = selection?.selected && typeof selection.selected === "object" ? selection.selected : {};
    if (Object.keys(selected).length) {
      const selectionBlock = create("div", "selection-block");
      const selectionHeading = create("div", "panel-heading");
      selectionHeading.append(create("h3", null, "Resume evidence plan"), create("span", "panel-note", `${selection?.summary?.selected_items ?? 0} selected`));
      selectionBlock.append(selectionHeading);
      const order = asArray(selection.section_order).map(statusText).join(" → ");
      if (order) selectionBlock.append(create("p", "evidence-item", `Section order · ${order}`));
      ["experience", "projects", "certifications"].forEach((section) => {
        const items = asArray(selected[section]);
        if (!items.length) return;
        const row = create("div", "selection-row");
        row.append(create("span", "selection-label", statusText(section)));
        const names = create("div", "role-tags");
        items.slice(0, 4).forEach((item) => names.append(create("span", "tag", item.label || item.key || "Selected evidence")));
        row.append(names);
        selectionBlock.append(row);
      });
      const suppressedTechnical = asArray(selection.suppressed).filter((item) => item.section === "experience" && /non-technical|communication\/community|direct JD bridge/i.test(item.reason || ""));
      if (suppressedTechnical.length) {
        selectionBlock.append(create("p", "selection-suppressed", `Suppressed for this JD · ${suppressedTechnical.slice(0, 3).map((item) => item.label).join(" · ")}`));
      }
      contextPanel.append(selectionBlock);
    }
    const dl = create("dl", "review-list");
    const addReview = (label, value, note) => { const item = create("div", "review-item"); const dt = create("dt", null, label); const dd = create("dd", null, value); if (note) dd.append(create("small", null, note)); item.append(dt, dd); dl.append(item); };
    const eligibility = report?.eligibility || {}; const gate = report?.experience_gate || {};
    addReview("Eligibility", statusText(eligibility.status || "unknown"), eligibility.reason); addReview("Experience", statusText(gate.status || "unknown"), gate.reason);
    const compensation = asArray(report?.compensation_mentions).map((item) => typeof item === "string" ? item : item.text).filter(Boolean); addReview("Compensation", compensation.join(" · ") || "Not detected");
    const ats = report?.ats_validation;
    if (ats?.summary) addReview("ATS checks", statusText(ats.status || "review"), `${ats.summary.passed} pass · ${ats.summary.review} review · ${ats.summary.failed} fail`);
    contextPanel.append(dl); panels.append(evidencePanel, contextPanel); elements.report.append(metrics, panels);
  }

  function extractMatchReport(payload) {
    if (payload?.match_report && typeof payload.match_report === "object") return payload.match_report;
    if (payload?.report && typeof payload.report === "object") return payload.report;
    if (payload?.data?.match_report && typeof payload.data.match_report === "object") return payload.data.match_report;
    return payload;
  }

  function normalizeJobs(payload) {
    const list = Array.isArray(payload) ? payload : payload?.jobs || payload?.items || payload?.results || [];
    return Array.isArray(list) ? list : [];
  }

  function jobStatus(job) { return String(job?.status || job?.state || job?.phase || "queued").toLowerCase(); }

  function isPreviewableArtifact(key) {
    return key === "resume" || key === "cover_letter";
  }

  async function copyText(value) {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return;
    }
    const helper = document.createElement("textarea");
    helper.value = value;
    helper.setAttribute("readonly", "");
    helper.style.position = "fixed";
    helper.style.opacity = "0";
    document.body.append(helper);
    helper.select();
    const copied = document.execCommand("copy");
    helper.remove();
    if (!copied) throw new Error("Clipboard access is unavailable.");
  }

  function createArtifactCard(key, url) {
    const card = create("div", "result-card result-card--artifact");
    card.append(create("span", "result-card-label", statusText(key)));
    const actions = create("div", "artifact-actions");
    if (isPreviewableArtifact(key)) {
      const previewButton = create("button", "artifact-button", "Preview code");
      previewButton.type = "button";
      const separator = create("span", "artifact-separator", "·");
      const rawLink = document.createElement("a");
      rawLink.href = url;
      rawLink.target = "_blank";
      rawLink.rel = "noreferrer";
      rawLink.textContent = "Open raw ↗";
      actions.append(previewButton, separator, rawLink);

      const preview = create("div", "code-preview");
      preview.hidden = true;
      const toolbar = create("div", "code-preview-toolbar");
      const filename = create("span", null, key === "resume" ? "resume.tex" : "cover_letter.tex");
      const copyButton = create("button", "artifact-button", "Copy code");
      copyButton.type = "button";
      copyButton.disabled = true;
      toolbar.append(filename, copyButton);
      const pre = create("pre");
      const code = create("code", null, "Preview loads the generated LaTeX here.");
      pre.append(code);
      preview.append(toolbar, pre);
      card.append(actions, preview);

      previewButton.addEventListener("click", async () => {
        const opening = preview.hidden;
        preview.hidden = !opening;
        previewButton.textContent = opening ? "Hide code" : "Preview code";
        if (!opening || preview.dataset.loaded === "true") return;
        code.textContent = "Loading code…";
        try {
          const response = await fetch(url, { headers: { "Accept": "text/plain" }, cache: "no-store" });
          if (!response.ok) throw new Error(`Could not load the artifact (${response.status}).`);
          code.textContent = await response.text();
          preview.dataset.loaded = "true";
          copyButton.disabled = false;
        } catch (error) {
          preview.classList.add("is-error");
          code.textContent = error.message || "Could not load the generated code.";
        }
      });

      copyButton.addEventListener("click", async () => {
        if (preview.dataset.loaded !== "true") return;
        const original = copyButton.textContent;
        try {
          await copyText(code.textContent);
          copyButton.textContent = "Copied";
        } catch (error) {
          copyButton.textContent = "Copy failed";
        }
        window.setTimeout(() => { copyButton.textContent = original; }, 1400);
      });
    } else {
      const rawLink = document.createElement("a");
      rawLink.href = url;
      rawLink.target = "_blank";
      rawLink.rel = "noreferrer";
      rawLink.textContent = "Open output ↗";
      actions.append(rawLink);
      card.append(actions);
    }
    return card;
  }

  function renderJobs(jobs) {
    state.jobs = jobs;
    elements.jobs.replaceChildren();
    if (!jobs.length) { const empty = create("div", "empty-state empty-state--compact"); empty.append(create("h3", null, "No generation runs yet"), create("p", null, "Your completed packets and active jobs will collect here.")); elements.jobs.append(empty); return; }
    jobs.slice(0, 8).forEach((job) => {
       const run = create("article", "job-run"); const header = create("div", "job-run-header");
       const title = valueAt(job, ["title", "role", "kind"], "Resume packet"); header.append(create("h3", "job-run-title", title), create("time", "job-run-time", job.created_at || job.created || "Local run")); run.append(header);
      const status = jobStatus(job); const progress = Number(job.progress ?? (status === "completed" || status === "complete" || status === "succeeded" ? 100 : status === "running" ? 55 : status === "failed" ? 100 : 15));
      const track = create("div", "progress-track"); const bar = create("div", "progress-bar"); bar.style.width = `${Math.max(5, Math.min(progress, 100))}%`; track.append(bar); run.append(track);
      const detail = valueAt(job, ["message", "detail", "error"], status === "failed" ? "Generation failed. Review the engine message and try again." : status === "completed" || status === "complete" || status === "succeeded" ? "Packet ready for review." : `Status · ${statusText(status)}`); run.append(create("p", "job-run-detail", detail));
        const result = job.result || job.output || job.artifacts; if (result && (status === "completed" || status === "complete" || status === "succeeded")) { const grid = create("div", "result-grid"); const artifactUrls = result.artifact_urls && typeof result.artifact_urls === "object" ? result.artifact_urls : {}; Object.entries(artifactUrls).forEach(([key, value]) => { if (value) grid.append(createArtifactCard(key, value)); }); Object.entries(result).filter(([key, value]) => key !== "artifact_urls" && value && typeof value !== "object").slice(0, 4).forEach(([key, value]) => { const card = create("div", "result-card"); card.append(create("span", "result-card-label", statusText(key))); card.append(create("span", "result-card-value", value)); grid.append(card); }); run.append(grid); }
      elements.jobs.append(run);
    });
  }

  async function refreshJobs() { try { renderJobs(normalizeJobs(await api("/api/jobs"))); } catch (error) { if (!state.jobs.length) setMessage(`Could not load generation runs: ${error.message}`); } }

  async function pollJob(id) {
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(id)}`); const merged = state.jobs.map((item) => (item.id || item.job_id) === id ? { ...item, ...job } : item); renderJobs(merged.length ? merged : [job]);
       const status = jobStatus(job); if (["queued", "pending", "running", "processing", "started"].includes(status)) state.pollTimer = window.setTimeout(() => pollJob(id), 1600); else { state.activeJobId = null; setButtonBusy(elements.generate, false); }
    } catch (error) { setMessage(`Job progress could not be refreshed: ${error.message}`); state.pollTimer = window.setTimeout(() => pollJob(id), 3000); }
  }

  async function onAnalyze(event) {
    event.preventDefault(); const text = elements.jobDescription.value.trim(); if (!text) { setMessage("Paste a job description before analyzing.", "info"); elements.jobDescription.focus(); return; }
    setMessage(""); setButtonBusy(elements.analyze, true, "Reading…"); elements.report.innerHTML = '<div class="empty-state"><div class="empty-glyph" aria-hidden="true">…</div><h3>Reading the role</h3><p>Matching explicit signals to the local evidence pool.</p></div>';
    try {
      const payload = await api("/api/analyze", { method: "POST", body: JSON.stringify({ job_description: text }) });
      const report = extractMatchReport(payload);
      if (!report || typeof report !== "object" || !report.summary) {
        throw new Error("The local matcher returned an incomplete report.");
      }
      renderReport(report);
      document.querySelector("#match-report").scrollIntoView({ behavior: "smooth", block: "start" });
    }
    catch (error) { elements.report.replaceChildren(); const empty = create("div", "empty-state"); empty.append(create("div", "empty-glyph", "!"), create("h3", null, "Analysis unavailable"), create("p", null, error.message || "The local matcher could not read this description.")); elements.report.append(empty); setMessage(`Analysis failed: ${error.message}`); }
    finally { setButtonBusy(elements.analyze, false); }
  }

  async function onGenerate(event) {
    event.preventDefault(); const text = elements.jobDescription.value.trim(); if (!text) { setMessage("Add and analyze a job description before starting generation.", "info"); elements.jobDescription.focus(); return; }
    setMessage(""); setButtonBusy(elements.generate, true, "Starting…");
    try { const payload = await api("/api/jobs", { method: "POST", body: JSON.stringify({ kind: "generate", job_description: text, include_cover_letter: elements.includeCoverLetter.checked }) }); const job = payload?.job || payload; const id = job?.id || job?.job_id; if (!id) throw new Error("The engine did not return a generation id."); state.activeJobId = id; renderJobs([job, ...state.jobs.filter((item) => (item.id || item.job_id) !== id)]); pollJob(id); document.querySelector("#runs").scrollIntoView({ behavior: "smooth", block: "start" }); }
    catch (error) { setButtonBusy(elements.generate, false); setMessage(`Generation could not start: ${error.message}`); }
  }

  function setupNavigation() {
    document.querySelectorAll("[data-scroll-target]").forEach((link) => link.addEventListener("click", () => { document.querySelectorAll(".nav-link").forEach((item) => item.classList.toggle("is-active", item === link)); }));
    document.addEventListener("keydown", (event) => { if (event.target.matches("textarea, input")) return; const key = event.key.toLowerCase(); const target = key === "w" ? "workbench" : key === "p" ? "profile" : key === "g" ? "runs" : null; if (target) { event.preventDefault(); document.querySelector(`#${target}`).scrollIntoView({ behavior: "smooth" }); } });
  }

  elements.jobDescription.addEventListener("input", () => { updateCount(); saveDraft(); });
  elements.jobDescription.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") elements.jobForm.requestSubmit(); });
  elements.clearJob.addEventListener("click", () => { elements.jobDescription.value = ""; saveDraft(); updateCount(); elements.jobDescription.focus(); setMessage(""); });
  elements.jobForm.addEventListener("submit", onAnalyze); elements.generationForm.addEventListener("submit", onGenerate); setupNavigation();

  loadSavedDraft();

  (async function bootstrap() {
    try { renderHealth(await api("/api/health")); } catch (_) { renderHealth(null, true); }
    try { renderProfile(await api("/api/profile")); } catch (error) { elements.profile.replaceChildren(); const empty = create("div", "empty-state empty-state--compact"); empty.append(create("h3", null, "Profile unavailable"), create("p", null, "The workbench is still usable, but profile context could not be loaded.")); elements.profile.append(empty); }
    await refreshJobs();
  })();
})();
