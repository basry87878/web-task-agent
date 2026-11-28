(function () {
  // Set this to your deployed Render backend URL before publishing to GitHub Pages,
  // e.g. "https://jobtruth-api.onrender.com" (no trailing slash).
const API_BASE = "https://web-task-agent.onrender.com";

  const FALLBACK_PRESETS = [
    {
      id: "b2b_leads",
      label: "B2B Lead List",
      badge: "Prospecting",
      description:
        "Discover target companies with positioning, size/metrics, outreach angles, and verified URLs.",
      prompt:
        "Build a B2B lead list of 6 European B2B SaaS companies specializing in AI workflow automation or operations automation. For each company, provide their target segment, core offering, pricing model or team/funding signals, a tailored B2B outreach angle, and their verified official website URL.",
    },
    {
      id: "competitor_pricing",
      label: "Competitor Pricing Matrix",
      badge: "Competitive Intel",
      description:
        "Extract live plan tiers, monthly pricing, and limits directly from official pricing pages.",
      prompt:
        "Create a competitor pricing matrix comparing 5 leading AI meeting assistant platforms (Otter.ai, Fireflies.ai, Fathom, tl;dv, and Grain). Use Tavily Extract on their official pricing pages to capture free vs paid tiers, monthly per-seat pricing, key plan limits, and verified pricing page URLs.",
    },
    {
      id: "vendor_shortlist",
      label: "Vendor Shortlist",
      badge: "Procurement",
      description:
        "Compare top B2B vendors on capabilities, pricing model, trade-offs, and best-fit use cases.",
      prompt:
        "Shortlist 5 managed cloud vector database vendors suitable for a mid-size engineering team (such as Pinecone, Qdrant Cloud, Weaviate Cloud, Zilliz Cloud, and Turso/pgvector providers). Compare deployment model, pricing structure, standout strengths, trade-offs, and verified official URLs.",
    },
  ];

  const sessionId =
    sessionStorage.getItem("jobtruth_sid") ||
    (crypto.randomUUID ? crypto.randomUUID() : "sid-" + Date.now());
  sessionStorage.setItem("jobtruth_sid", sessionId);

  const DEFAULT_MAX_CUSTOM_RUNS = 20;

  const state = {
    presets: FALLBACK_PRESETS,
    history: [],
    activeIndex: -1,
    isRunning: false,
    runsRemaining: DEFAULT_MAX_CUSTOM_RUNS,
    deepMode: false,
    deepTargetRows: 100,
    deepJobId: null,
    deepPollTimer: null,
    deepActiveProviders: [],
    deepLastState: null,
  };

  // DOM references
  const presetsGrid = document.getElementById("presets-grid");
  const customForm = document.getElementById("custom-form");
  const promptInput = document.getElementById("custom-prompt");
  const charCounter = document.getElementById("char-counter");
  const runCustomBtn = document.getElementById("run-custom-btn");
  const runsBadge = document.getElementById("runs-badge");
  const deepModeToggle = document.getElementById("deep-mode-toggle");
  const deepControls = document.getElementById("deep-controls");
  const deepTargetPills = document.getElementById("deep-target-pills");
  const deepPanel = document.getElementById("deep-panel");
  const deepStatusBadge = document.getElementById("deep-status-badge");
  const deepStatusText = document.getElementById("deep-status-text");
  const deepJobMeta = document.getElementById("deep-job-meta");
  const deepPanelTitle = document.getElementById("deep-panel-title");
  const deepStopBtn = document.getElementById("deep-stop-btn");
  const deepProgressFill = document.getElementById("deep-progress-fill");
  const deepProgressCaption = document.getElementById("deep-progress-caption");
  const deepMetricWave = document.getElementById("deep-metric-wave");
  const deepMetricDomains = document.getElementById("deep-metric-domains");
  const deepMetricRows = document.getElementById("deep-metric-rows");
  const deepMetricElapsed = document.getElementById("deep-metric-elapsed");
  const deepProvidersList = document.getElementById("deep-providers-list");
  const deepErrorBanner = document.getElementById("deep-error-banner");
  const deepExportRow = document.getElementById("deep-export-row");
  const deepDownloadXlsxBtn = document.getElementById("deep-download-xlsx-btn");
  const deepDownloadCsvBtn = document.getElementById("deep-download-csv-btn");
  const deepPreviewTbody = document.getElementById("deep-preview-tbody");
  const promptAppliedBanner = document.getElementById("prompt-applied-banner");
  const statusBanner = document.getElementById("status-banner");
  const statusBannerText = document.getElementById("status-banner-text");
  const errorBanner = document.getElementById("error-banner");
  const resultsSection = document.getElementById("results-section");
  const reportTitleEl = document.getElementById("report-title");
  const reportSummaryEl = document.getElementById("report-summary");
  const metricRowsEl = document.getElementById("metric-rows");
  const metricUrlsEl = document.getElementById("metric-urls");
  const metricModelEl = document.getElementById("metric-model");
  const metricTimeEl = document.getElementById("metric-time");
  const guardrailNoteEl = document.getElementById("guardrail-note");
  const flaggedBannerEl = document.getElementById("flagged-banner");
  const resultsThead = document.getElementById("results-thead");
  const resultsTbody = document.getElementById("results-tbody");
  const downloadXlsxBtn = document.getElementById("download-xlsx-btn");
  const downloadCsvBtn = document.getElementById("download-csv-btn");
  const historyListEl = document.getElementById("history-list");
  const apiStatusPill = document.getElementById("api-status-pill");
  const apiStatusText = document.getElementById("api-status-text");

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function isSafeHttpUrl(url) {
    try {
      const parsed = new URL(url);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
      return false;
    }
  }

  function setApiStatus(mode, text) {
    if (!apiStatusPill || !apiStatusText) return;
    const dot = apiStatusPill.querySelector(".status-dot");
    if (dot) {
      dot.className = `status-dot ${mode}`;
    }
    apiStatusText.textContent = text;
  }

  function updateCharCounter() {
    const len = (promptInput.value || "").length;
    charCounter.textContent = `${len} / 500`;
  }

  function updateRunsBadge() {
    if (state.runsRemaining === null || state.runsRemaining === undefined) {
      runsBadge.textContent = "";
      return;
    }
    if (state.runsRemaining > 0) {
      runsBadge.textContent = `${state.runsRemaining} custom run${state.runsRemaining === 1 ? "" : "s"} remaining this session · Presets are cached`;
    } else {
      runsBadge.textContent = "Session custom run limit reached · Presets & exports remain available";
    }
  }

  function setRunning(running, initialMsg) {
    state.isRunning = running;
    runCustomBtn.disabled = running;
    promptInput.disabled = running;
    document.querySelectorAll(".btn-preset").forEach((b) => {
      b.disabled = running;
    });
    if (running) {
      errorBanner.classList.add("hidden");
      statusBannerText.textContent =
        initialMsg || "Running live Tavily research & structuring report…";
      statusBanner.classList.remove("hidden");
    } else {
      statusBanner.classList.add("hidden");
    }
  }

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.remove("hidden");
  }

  function renderPresets() {
    presetsGrid.innerHTML = state.presets
      .map(
        (preset) => `
        <div class="preset-card">
          <div>
            <span class="preset-badge">${escapeHtml(preset.badge)}</span>
            <h3 class="preset-title">${escapeHtml(preset.label)}</h3>
            <p class="preset-desc">${escapeHtml(preset.description)}</p>
          </div>
          <button
            type="button"
            class="btn btn-preset"
            data-preset-id="${escapeHtml(preset.id)}"
          >
            Run ${escapeHtml(preset.label)}
          </button>
        </div>`
      )
      .join("");

    presetsGrid.querySelectorAll(".btn-preset").forEach((btn) => {
      btn.addEventListener("click", () => {
        const pid = btn.getAttribute("data-preset-id");
        const preset = state.presets.find((p) => p.id === pid);
        if (preset) executeResearch({ prompt: preset.prompt, presetId: preset.id });
      });
    });
  }

  function renderHistory() {
    if (!state.history.length) {
      historyListEl.innerHTML =
        '<div class="history-empty">Run a preset or custom brief to generate a source-verified Excel report.</div>';
      return;
    }

    historyListEl.innerHTML = state.history
      .map((entry, idx) => {
        const activeCls = idx === state.activeIndex ? " active" : "";
        const label = entry.title.length > 34 ? entry.title.slice(0, 34) + "…" : entry.title;
        return `<button type="button" class="history-btn${activeCls}" data-hist-idx="${idx}" title="${escapeHtml(entry.title)}">📄 ${escapeHtml(label)}</button>`;
      })
      .reverse()
      .join("");

    historyListEl.querySelectorAll(".history-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.getAttribute("data-hist-idx"));
        if (!Number.isNaN(idx) && state.history[idx]) {
          state.activeIndex = idx;
          renderHistory();
          renderActiveReport();
        }
      });
    });
  }

  function renderActiveReport() {
    const report = state.history[state.activeIndex];
    if (!report) {
      resultsSection.classList.add("hidden");
      return;
    }

    resultsSection.classList.remove("hidden");
    reportTitleEl.textContent = report.title;
    reportSummaryEl.textContent = report.executive_summary;
    metricRowsEl.textContent = String(report.rows.length);
    metricUrlsEl.textContent = String(report.verified_url_count);
    metricModelEl.textContent = report.model_used || "—";
    metricTimeEl.textContent =
      report.cached ? `${Number(report.duration_sec).toFixed(1)}s (cached)` : `${Number(report.duration_sec).toFixed(1)}s`;

    if (report.unverified_dropped > 0) {
      guardrailNoteEl.textContent = `Guardrail note: ${report.unverified_dropped} row(s) with unverified links were filtered out automatically.`;
      guardrailNoteEl.classList.remove("hidden");
    } else {
      guardrailNoteEl.classList.add("hidden");
    }

    const flagMap = new Map();
    (report.flagged_rows || []).forEach((f) => {
      flagMap.set(f.row_index, f.reasons || []);
    });

    if (flagMap.size > 0) {
      flaggedBannerEl.textContent = `⚠️ ${flagMap.size} row(s) highlighted in amber need a quick check (e.g., pricing not publicly listed or domain-level URL match).`;
      flaggedBannerEl.classList.remove("hidden");
    } else {
      flaggedBannerEl.classList.add("hidden");
    }

    const cols = report.columns || [
      "Name",
      "Category",
      "Key Details",
      "Pricing / Metrics",
      "Notes",
      "Source URL",
    ];
    resultsThead.innerHTML = `<tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr>`;

    const colClass = {
      Name: "col-name",
      Category: "col-category",
      "Key Details": "col-details",
      "Pricing / Metrics": "col-metrics",
      Notes: "col-notes",
      "Source URL": "col-url",
    };

    if (!report.rows.length) {
      resultsTbody.innerHTML = `
        <tr>
          <td colspan="${cols.length}" class="deep-preview-empty">
            No verified records strictly matched all criteria for this request. See the summary above for details.
          </td>
        </tr>`;
      downloadXlsxBtn.disabled = true;
      downloadCsvBtn.disabled = true;
      downloadXlsxBtn.textContent = "⬇️ No rows to export";
    } else {
      downloadXlsxBtn.disabled = false;
      downloadCsvBtn.disabled = false;
      resultsTbody.innerHTML = report.rows
        .map((row, rowIdx) => {
          const reasons = flagMap.get(rowIdx);
          const trClass = reasons ? ' class="row-flagged"' : "";
          const cells = cols
            .map((col, colIdx) => {
              const val = row[col] || "";
              const cls = colClass[col] ? ` class="${colClass[col]}"` : "";
              if (col === "Source URL") {
                if (isSafeHttpUrl(val)) {
                  return `<td${cls}><a class="source-link" href="${escapeHtml(val)}" target="_blank" rel="noopener noreferrer">Open source ↗</a></td>`;
                }
                return `<td${cls}>—</td>`;
              }
              let extra = "";
              if (colIdx === 0 && reasons && reasons.length) {
                extra = `<div><span class="flag-pill" title="${escapeHtml(reasons.join("; "))}">Flagged: ${escapeHtml(reasons[0])}</span></div>`;
              }
              return `<td${cls}>${escapeHtml(val)}${extra}</td>`;
            })
            .join("");
          return `<tr${trClass}>${cells}</tr>`;
        })
        .join("");
      downloadXlsxBtn.textContent = `⬇️ Download Excel Report (${report.filename || "report.xlsx"})`;
    }
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function executeResearch({ prompt, presetId = null }) {
    if (state.isRunning) return;
    const cleaned = (prompt || "").trim();
    if (!presetId && cleaned.length < 10) {
      showError("Please enter a slightly more detailed research brief (at least 10 characters).");
      return;
    }

    setRunning(true, "Running live Tavily research & structuring report…");

    // Friendly progressive wake-up notice if Render free tier is cold-starting
    const coldStartTimer = setTimeout(() => {
      if (state.isRunning) {
        statusBannerText.textContent =
          "Waking up the cloud research server & gathering live 2026 web sources (first request after idle may take ~30–45s)…";
      }
    }, 6500);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 95000);

    try {
      const res = await fetch(`${API_BASE}/api/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: cleaned,
          preset_id: presetId,
          session_id: sessionId,
        }),
        signal: controller.signal,
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Server returned HTTP ${res.status}. Please try again.`);
      }

      setApiStatus("online", "Research engine ready");
      state.runsRemaining = data.runs_remaining;
      updateRunsBadge();

      state.history.push(data);
      state.activeIndex = state.history.length - 1;
      renderHistory();
      renderActiveReport();
    } catch (err) {
      if (err && err.name === "AbortError") {
        showError(
          "The research request timed out while the cloud server was waking up. Please click Run again — the server should now be warm."
        );
      } else if (err instanceof TypeError) {
        showError(
          `Could not reach the research backend at ${API_BASE}. If deployed on Render free tier, wait a few seconds for cold start and try again.`
        );
        setApiStatus("offline", "Backend unreachable");
      } else {
        showError(err.message || "Could not complete research. Please try again.");
      }
    } finally {
      clearTimeout(coldStartTimer);
      clearTimeout(timeoutId);
      setRunning(false);
    }
  }

  async function downloadExport(format) {
    const report = state.history[state.activeIndex];
    if (!report) return;

    const btn = format === "csv" ? downloadCsvBtn : downloadXlsxBtn;
    const origText = btn.textContent;
    btn.disabled = true;
    btn.textContent = format === "csv" ? "Preparing CSV…" : "Preparing Excel…";

    try {
      const res = await fetch(`${API_BASE}/api/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: report.title,
          executive_summary: report.executive_summary || "",
          rows: report.rows,
          format,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Export failed (HTTP ${res.status}).`);
      }

      const blob = await res.blob();
      const disp = res.headers.get("Content-Disposition") || "";
      const match = disp.match(/filename="?([^"]+)"?/i);
      const fallbackName =
        format === "csv"
          ? (report.filename || "research_report.xlsx").replace(/\.xlsx$/i, ".csv")
          : report.filename || "research_report.xlsx";
      const rawName = match ? match[1] : fallbackName;
      const downloadName = rawName.replace(/[^a-zA-Z0-9._-]/g, "_") || "research_report.xlsx";

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = downloadName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
      showError(err.message || "Download failed. Please try again.");
    } finally {
      btn.disabled = false;
      btn.textContent = origText;
    }
  }

  // Silent non-blocking wake-up + preset/GA4 hydration on page load
  async function bootApp() {
    renderPresets();
    renderHistory();
    updateCharCounter();
    updateRunsBadge();

    // 1. Silent non-blocking /health wake-up ping for Render free tier
    fetch(`${API_BASE}/health`, { method: "GET" })
      .then((r) => {
        if (r.ok) setApiStatus("online", "Research engine ready");
        else setApiStatus("warming", "Research engine waking up…");
      })
      .catch(() => {
        setApiStatus("warming", "Backend waking up or offline");
      });

    // 2. Hydrate presets, remaining quota, and optional GA4 measurement ID
    try {
      const res = await fetch(
        `${API_BASE}/api/presets?session_id=${encodeURIComponent(sessionId)}`
      );
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data.presets) && data.presets.length) {
          state.presets = data.presets;
          renderPresets();
        }
        if (typeof data.runs_remaining === "number") {
          state.runsRemaining = data.runs_remaining;
          updateRunsBadge();
        }
        if (data.ga_measurement_id && typeof window.initJobTruthGA === "function") {
          window.initJobTruthGA(data.ga_measurement_id);
        }
        setApiStatus("online", "Research engine ready");
      }
    } catch {
      // Fallback presets already rendered cleanly
    }
  }

  // Expose bridge for floating coach-widget.js ("Use this prompt" 1-click button)
  window.JobTruthApp = {
    getCustomPrompt() {
      return (promptInput.value || "").trim();
    },
    setCustomPrompt(text) {
      promptInput.value = (text || "").slice(0, 500);
      updateCharCounter();
      promptAppliedBanner.textContent =
        "Loaded the improved prompt into the Custom Research Brief box below.";
      promptAppliedBanner.classList.remove("hidden");
      promptInput.scrollIntoView({ behavior: "smooth", block: "center" });
      promptInput.focus();
      setTimeout(() => promptAppliedBanner.classList.add("hidden"), 6000);
    },
  };

  function fireGA4Event(eventName, params) {
    try {
      if (typeof window.gtag === "function") {
        window.gtag("event", eventName, params || {});
      }
    } catch {
      // ignore analytics errors
    }
  }

  function updateDeepModeUI() {
    if (!deepModeToggle || !deepControls) return;
    deepModeToggle.classList.toggle("active", state.deepMode);
    deepModeToggle.setAttribute("aria-pressed", state.deepMode ? "true" : "false");
    deepControls.classList.toggle("hidden", !state.deepMode);
    deepControls.setAttribute("aria-hidden", state.deepMode ? "false" : "true");
    runCustomBtn.textContent = state.deepMode ? "Launch Deep Research" : "Run Custom Research";
    runsBadge.classList.toggle("hidden", state.deepMode);
  }

  function stopDeepPolling() {
    if (state.deepPollTimer) {
      clearInterval(state.deepPollTimer);
      state.deepPollTimer = null;
    }
  }

  function formatDeepStatusLabel(rawStatus) {
    const map = {
      running: "Planning",
      planning: "Planning",
      discovering: "Discovering",
      extracting: "Extracting",
      completed: "Completed",
      stopped: "Stopped",
      error: "Error",
    };
    return map[(rawStatus || "").toLowerCase()] || "Running";
  }

  function renderDeepPanel(jobData) {
    if (!deepPanel || !jobData) return;
    state.deepLastState = jobData;
    deepPanel.classList.remove("hidden");

    const allowedStatuses = new Set(["planning", "discovering", "extracting", "completed", "stopped", "error"]);
    const rawStatus = (jobData.status || "planning").toLowerCase();
    const statusKey = allowedStatuses.has(rawStatus) ? rawStatus : "planning";
    const isTerminal = ["completed", "stopped", "error"].includes(statusKey);

    deepStatusText.textContent = formatDeepStatusLabel(statusKey);
    deepStatusBadge.className = `deep-status-badge status-${statusKey}${isTerminal ? "" : " is-live"}`;
    deepJobMeta.textContent = `Wave ${jobData.wave || 0} · ${jobData.queries_completed || 0} queries completed`;
    deepPanelTitle.textContent = jobData.title || "Deep Research Engine";

    deepStopBtn.classList.toggle("hidden", isTerminal);
    deepStopBtn.disabled = isTerminal;

    const collected = jobData.rows_collected || 0;
    const target = jobData.target_rows;
    if (target && target > 0) {
      const pct = isTerminal && statusKey === "completed" ? 100 : Math.min(100, Math.max(4, Math.round((collected / target) * 100)));
      deepProgressFill.classList.remove("continuous-pulse");
      deepProgressFill.style.width = `${pct}%`;
      deepProgressCaption.textContent = `${collected.toLocaleString()} / ${target.toLocaleString()} validated rows (${pct}%)`;
    } else {
      if (isTerminal) {
        deepProgressFill.classList.remove("continuous-pulse");
        deepProgressFill.style.width = "100%";
      } else {
        deepProgressFill.classList.add("continuous-pulse");
      }
      deepProgressCaption.textContent = `${collected.toLocaleString()} validated rows collected · Continuous mode (stop anytime)`;
    }

    deepMetricWave.textContent = `Wave ${jobData.wave || 0} · ${jobData.queries_completed || 0}`;
    deepMetricDomains.textContent = String(jobData.urls_discovered || 0);
    deepMetricRows.textContent = target && target > 0 ? `${collected} / ${target}` : String(collected);
    deepMetricElapsed.textContent = `${Number(jobData.elapsed_seconds || 0).toFixed(1)}s`;

    // Render provider pills
    const usedSet = new Set(jobData.providers_used || []);
    const exhaustedSet = new Set(jobData.providers_exhausted || []);
    const allNames = Array.from(
      new Set([...(state.deepActiveProviders || []), ...usedSet, ...exhaustedSet])
    );
    if (allNames.length) {
      deepProvidersList.innerHTML = allNames
        .map((name) => {
          const cls = exhaustedSet.has(name)
            ? "provider-chip exhausted"
            : usedSet.has(name)
            ? "provider-chip active"
            : "provider-chip";
          return `<span class="${cls}">${escapeHtml(name)}</span>`;
        })
        .join("");
    } else {
      deepProvidersList.innerHTML = '<span class="provider-chip">Multi-Engine Router</span>';
    }

    if (jobData.error) {
      deepErrorBanner.textContent = jobData.error;
      deepErrorBanner.classList.remove("hidden");
    } else {
      deepErrorBanner.classList.add("hidden");
    }

    if (jobData.has_export) {
      deepExportRow.classList.remove("hidden");
    } else {
      deepExportRow.classList.add("hidden");
    }

    const preview = jobData.preview_rows || [];
    if (!preview.length) {
      deepPreviewTbody.innerHTML = `
        <tr>
          <td colspan="6" class="deep-preview-empty">
            ${
              isTerminal
                ? "No verified rows were extracted for this run."
                : "Discovering unique domains & extracting verified rows…"
            }
          </td>
        </tr>`;
    } else {
      deepPreviewTbody.innerHTML = preview
        .slice()
        .reverse()
        .map((row) => {
          const url = row["Source URL"] || "";
          const urlCell = isSafeHttpUrl(url)
            ? `<a class="source-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">Open source ↗</a>`
            : "—";
          return `
            <tr>
              <td class="col-name">${escapeHtml(row["Name"] || "")}</td>
              <td class="col-category">${escapeHtml(row["Category"] || "")}</td>
              <td class="col-details">${escapeHtml(row["Key Details"] || "")}</td>
              <td class="col-metrics">${escapeHtml(row["Pricing / Metrics"] || "")}</td>
              <td class="col-notes">${escapeHtml(row["Notes"] || "")}</td>
              <td class="col-url">${urlCell}</td>
            </tr>`;
        })
        .join("");
    }
  }

  async function pollDeepJob() {
    if (!state.deepJobId) return;
    try {
      const res = await fetch(`${API_BASE}/api/deep-research/${encodeURIComponent(state.deepJobId)}`);
      if (!res.ok) return;
      const data = await res.json();
      renderDeepPanel(data);
      const st = (data.status || "").toLowerCase();
      if (st === "completed" || st === "stopped" || st === "error") {
        stopDeepPolling();
        setRunning(false);
        if (st === "completed") {
          fireGA4Event("deep_research_completed", {
            rows_collected: data.rows_collected || 0,
          });
        }
      }
    } catch {
      // transient network blip while polling; keep retrying on next 3s tick
    }
  }

  async function startDeepResearch(promptText) {
    const cleaned = (promptText || "").trim();
    if (cleaned.length < 10) {
      showError("Please enter a slightly more detailed research brief (at least 10 characters).");
      return;
    }

    stopDeepPolling();
    errorBanner.classList.add("hidden");
    deepErrorBanner.classList.add("hidden");
    setRunning(true, "Launching multi-engine Deep Research wave planner…");

    try {
      const targetPayload = state.deepTargetRows > 0 ? state.deepTargetRows : 0;
      const res = await fetch(`${API_BASE}/api/deep-research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: cleaned,
          target_rows: targetPayload,
          session_id: sessionId,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Could not start Deep Research (HTTP ${res.status}).`);
      }

      state.deepJobId = data.job_id;
      state.deepActiveProviders = data.active_providers || [];
      statusBanner.classList.add("hidden");

      fireGA4Event("deep_research_started", {
        target_rows: targetPayload,
      });

      renderDeepPanel({
        job_id: data.job_id,
        status: "planning",
        wave: 0,
        queries_completed: 0,
        urls_discovered: 0,
        rows_collected: 0,
        target_rows: targetPayload > 0 ? targetPayload : null,
        providers_used: [],
        providers_exhausted: [],
        preview_rows: [],
        has_export: false,
        elapsed_seconds: 0,
        title: `Deep Research: ${cleaned.slice(0, 68)}`,
      });
      deepPanel.scrollIntoView({ behavior: "smooth", block: "start" });

      // Poll every 3 seconds (also keeps Render free service warm during long runs)
      state.deepPollTimer = setInterval(pollDeepJob, 3000);
      pollDeepJob();
    } catch (err) {
      setRunning(false);
      showError(err.message || "Could not start Deep Research. Please try again.");
    }
  }

  async function stopDeepResearch() {
    if (!state.deepJobId) return;
    deepStopBtn.disabled = true;
    deepStopBtn.textContent = "Stopping & Finalizing…";
    try {
      const res = await fetch(
        `${API_BASE}/api/deep-research/${encodeURIComponent(state.deepJobId)}/stop`,
        { method: "POST" }
      );
      const data = await res.json().catch(() => null);
      stopDeepPolling();
      setRunning(false);
      if (data) {
        renderDeepPanel(data);
        fireGA4Event("deep_research_stopped", {
          rows_collected: data.rows_collected || 0,
        });
      }
    } catch {
      deepStopBtn.disabled = false;
    } finally {
      deepStopBtn.textContent = "⏹ Stop & Export Now";
    }
  }

  async function downloadDeepExport(format) {
    if (!state.deepJobId) return;
    const btn = format === "csv" ? deepDownloadCsvBtn : deepDownloadXlsxBtn;
    const orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = format === "csv" ? "Preparing CSV…" : "Preparing Excel…";
    try {
      const res = await fetch(
        `${API_BASE}/api/deep-research/${encodeURIComponent(state.deepJobId)}/download?format=${encodeURIComponent(format)}`
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Download failed (HTTP ${res.status}).`);
      }
      const blob = await res.blob();
      const disp = res.headers.get("Content-Disposition") || "";
      const match = disp.match(/filename="?([^"]+)"?/i);
      const rawName = match
        ? match[1]
        : `deep_research_${state.deepJobId}.${format}`;
      const downloadName = rawName.replace(/[^a-zA-Z0-9._-]/g, "_") || `deep_research.${format}`;

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = downloadName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);

      fireGA4Event("deep_research_downloaded", { format });
    } catch (err) {
      deepErrorBanner.textContent = err.message || "Download failed.";
      deepErrorBanner.classList.remove("hidden");
    } finally {
      btn.disabled = false;
      btn.textContent = orig;
    }
  }

  promptInput.addEventListener("input", updateCharCounter);

  if (deepModeToggle) {
    deepModeToggle.addEventListener("click", () => {
      state.deepMode = !state.deepMode;
      updateDeepModeUI();
    });
  }

  if (deepTargetPills) {
    deepTargetPills.querySelectorAll(".target-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        state.deepTargetRows = Number(pill.getAttribute("data-target") || "100");
        deepTargetPills.querySelectorAll(".target-pill").forEach((p) => {
          const active = p === pill;
          p.classList.toggle("active", active);
          p.setAttribute("aria-checked", active ? "true" : "false");
        });
      });
    });
  }

  if (deepStopBtn) {
    deepStopBtn.addEventListener("click", stopDeepResearch);
  }
  if (deepDownloadXlsxBtn) {
    deepDownloadXlsxBtn.addEventListener("click", () => downloadDeepExport("xlsx"));
  }
  if (deepDownloadCsvBtn) {
    deepDownloadCsvBtn.addEventListener("click", () => downloadDeepExport("csv"));
  }

  customForm.addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.deepMode) {
      startDeepResearch(promptInput.value);
    } else {
      executeResearch({ prompt: promptInput.value, presetId: null });
    }
  });

  downloadXlsxBtn.addEventListener("click", () => downloadExport("xlsx"));
  downloadCsvBtn.addEventListener("click", () => downloadExport("csv"));

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      updateDeepModeUI();
      bootApp();
    });
  } else {
    updateDeepModeUI();
    bootApp();
  }
})();
