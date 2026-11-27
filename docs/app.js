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

  const state = {
    presets: FALLBACK_PRESETS,
    history: [],
    activeIndex: -1,
    isRunning: false,
    runsRemaining: null,
  };

  // DOM references
  const presetsGrid = document.getElementById("presets-grid");
  const customForm = document.getElementById("custom-form");
  const promptInput = document.getElementById("custom-prompt");
  const charCounter = document.getElementById("char-counter");
  const runCustomBtn = document.getElementById("run-custom-btn");
  const runsBadge = document.getElementById("runs-badge");
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
      const downloadName = match ? match[1] : fallbackName;

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

  promptInput.addEventListener("input", updateCharCounter);

  customForm.addEventListener("submit", (e) => {
    e.preventDefault();
    executeResearch({ prompt: promptInput.value, presetId: null });
  });

  downloadXlsxBtn.addEventListener("click", () => downloadExport("xlsx"));
  downloadCsvBtn.addEventListener("click", () => downloadExport("csv"));

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootApp);
  } else {
    bootApp();
  }
})();
