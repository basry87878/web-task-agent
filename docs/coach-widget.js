(function () {
  // Set this to your deployed Render backend URL before publishing to GitHub Pages,
  // e.g. "https://jobtruth-api.onrender.com" (no trailing slash).
  const API_BASE = "https://web-task-agent.onrender.com";

  function initCoachWidget() {
    if (document.getElementById("jobtruth-coach-host")) return;

    const sessionId =
      sessionStorage.getItem("jobtruth_sid") ||
      (crypto.randomUUID ? crypto.randomUUID() : "sid-" + Date.now());
    sessionStorage.setItem("jobtruth_sid", sessionId);

    const host = document.createElement("div");
    host.id = "jobtruth-coach-host";
    document.body.appendChild(host);

    const shadow = host.attachShadow({ mode: "open" });

    shadow.innerHTML = `
      <style>
        * { box-sizing: border-box; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }

        @keyframes coachFloatBob {
          0%, 100% { transform: translateY(0) scale(1.05); }
          50% { transform: translateY(-6px) scale(1.08); }
        }

        .toggle-btn {
          position: fixed; right: 24px; bottom: 24px; z-index: 99999;
          width: 56px; height: 56px; border-radius: 50%; border: none;
          background: #111827; color: #ffffff; cursor: pointer;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.24);
          display: flex; align-items: center; justify-content: center;
          transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
        }
        .toggle-btn:hover {
          background: #1f2937;
          box-shadow: 0 12px 28px rgba(0, 0, 0, 0.34);
          animation: coachFloatBob 0.85s ease-in-out infinite;
        }
        .toggle-btn svg {
          width: 24px; height: 24px; fill: none; stroke: #ffffff; stroke-width: 2;
          stroke-linecap: round; stroke-linejoin: round;
        }

        .panel {
          position: fixed; right: 24px; bottom: 92px; z-index: 99998;
          width: 370px; max-width: calc(100vw - 32px); height: 500px; max-height: 78vh;
          background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px;
          box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
          display: none; flex-direction: column; overflow: hidden;
        }
        .panel.open { display: flex; }

        .header {
          background: #111827; color: #ffffff; padding: 13px 16px;
          display: flex; justify-content: space-between; align-items: center;
        }
        .header-title { font-weight: 600; font-size: 15px; line-height: 1.25; }
        .header-sub { font-size: 12px; color: #9ca3af; margin-top: 2px; line-height: 1.25; }
        .close-btn {
          background: transparent; border: none; color: #d1d5db;
          font-size: 22px; line-height: 1; cursor: pointer; padding: 4px 8px;
          border-radius: 8px;
        }
        .close-btn:hover { color: #ffffff; background: rgba(255, 255, 255, 0.1); }

        .messages {
          flex: 1; padding: 14px; overflow-y: auto;
          display: flex; flex-direction: column; gap: 10px; background: #f9fafb;
        }
        .msg {
          max-width: 88%; padding: 10px 13px; border-radius: 12px;
          font-size: 13.5px; line-height: 1.45; word-wrap: break-word;
        }
        .msg.bot {
          align-self: flex-start; background: #ffffff; color: #111827;
          border: 1px solid #e5e7eb;
        }
        .msg.user {
          align-self: flex-end; background: #111827; color: #ffffff;
        }
        .msg a { color: #2563eb; text-decoration: underline; }
        .msg.user a { color: #93c5fd; }
        .quote-box {
          margin: 6px 0; padding: 7px 10px;
          border-left: 3px solid #111827; background: #f3f4f6;
          border-radius: 6px; font-size: 12.5px; color: #111827;
        }

        .use-prompt-btn {
          align-self: flex-start;
          border-radius: 999px; border: 1px solid #bfdbfe;
          background: #eff6ff; color: #1d4ed8;
          font-size: 12px; font-weight: 600;
          padding: 5px 12px; cursor: pointer;
          transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        }
        .use-prompt-btn:hover {
          background: #1d4ed8; color: #ffffff; border-color: #1d4ed8;
        }

        .chips {
          display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 12px;
          background: #f9fafb; border-top: 1px solid #f3f4f6;
        }
        .chip {
          font-size: 12px; font-weight: 500; padding: 6px 11px; border-radius: 999px;
          border: 1px solid #d1d5db; background: #ffffff; color: #111827;
          cursor: pointer; transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        }
        .chip:hover:not(:disabled) {
          background: #111827; color: #ffffff; border-color: #111827;
        }
        .chip:disabled { opacity: 0.45; cursor: not-allowed; }

        .form {
          display: flex; gap: 8px; padding: 10px;
          border-top: 1px solid #e5e7eb; background: #ffffff;
        }
        .input {
          flex: 1; padding: 9px 12px; border: 1px solid #d1d5db;
          border-radius: 10px; font-size: 13.5px; outline: none;
          color: #111827; background: #ffffff;
        }
        .input:focus { border-color: #111827; }
        .send-btn {
          padding: 0 14px; border: none; border-radius: 10px;
          background: #111827; color: #ffffff; font-size: 13px; font-weight: 600;
          cursor: pointer; transition: background 0.15s ease;
        }
        .send-btn:hover:not(:disabled) { background: #1f2937; }
        .send-btn:disabled { opacity: 0.45; cursor: not-allowed; }
      </style>

      <button class="toggle-btn" id="toggle" aria-label="Open Research Assistant">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
        </svg>
      </button>

      <div class="panel" id="panel" role="dialog" aria-label="Research Assistant">
        <div class="header">
          <div>
            <div class="header-title">Research Assistant</div>
            <div class="header-sub">Prompt Coach · Tool Guide</div>
          </div>
          <button class="close-btn" id="close" aria-label="Close">×</button>
        </div>
        <div class="messages" id="messages">
          <div class="msg bot">
            Hi! 👋 Paste a rough research idea and I'll sharpen it into a 2026 brief, or ask how JobTruth works.
          </div>
        </div>
        <div class="chips" id="chips">
          <button class="chip" type="button" data-action="improve">Improve my prompt</button>
          <button class="chip" type="button" data-action="how">How does this work?</button>
        </div>
        <form class="form" id="form">
          <input class="input" id="input" type="text" placeholder="Write a message..." maxlength="400" required />
          <button class="send-btn" id="send" type="submit">Send</button>
        </form>
      </div>
    `;

    const toggleBtn = shadow.getElementById("toggle");
    const closeBtn = shadow.getElementById("close");
    const panel = shadow.getElementById("panel");
    const messagesEl = shadow.getElementById("messages");
    const chipsEl = shadow.getElementById("chips");
    const form = shadow.getElementById("form");
    const input = shadow.getElementById("input");
    const sendBtn = shadow.getElementById("send");

    const history = [];
    let busy = false;

    function formatSafeHtml(text) {
      const escaped = String(text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
      const lines = escaped.split(/\r?\n/).map((line) => {
        if (line.startsWith("&gt; ")) {
          return `<div class="quote-box">${line.slice(5)}</div>`;
        }
        return line;
      });
      return lines
        .join("<br>")
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/(https?:\/\/[^\s<>"'&)]+)/g, (match) => {
          try {
            const u = new URL(match);
            if (u.protocol !== "http:" && u.protocol !== "https:") return match;
            return `<a href="${u.href}" target="_blank" rel="noopener noreferrer">${match}</a>`;
          } catch {
            return match;
          }
        })
        .replace(
          /([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})/g,
          '<a href="mailto:$1">$1</a>'
        );
    }

    function updateControls() {
      input.disabled = busy;
      sendBtn.disabled = busy;
      chipsEl.querySelectorAll(".chip").forEach((c) => {
        c.disabled = busy;
      });
    }

    function addMessage(text, role) {
      const div = document.createElement("div");
      div.className = `msg ${role}`;
      div.innerHTML = formatSafeHtml(text);
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return div;
    }

    function addUsePromptButton(promptText) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "use-prompt-btn";
      btn.textContent = "✨ Use this prompt";
      btn.addEventListener("click", () => {
        if (window.JobTruthApp && typeof window.JobTruthApp.setCustomPrompt === "function") {
          window.JobTruthApp.setCustomPrompt(promptText);
          panel.classList.remove("open");
        }
      });
      messagesEl.appendChild(btn);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    async function sendText(text) {
      const cleaned = (text || "").trim();
      if (!cleaned || busy) return;

      busy = true;
      updateControls();
      addMessage(cleaned, "user");
      input.value = "";
      history.push({ role: "user", content: cleaned });

      const typingEl = addMessage("Thinking…", "bot");

      try {
        const res = await fetch(`${API_BASE}/api/coach`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: history.slice(-8),
            session_id: sessionId,
          }),
        });

        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          history.pop();
          typingEl.innerHTML = formatSafeHtml(
            data.detail || "Could not get a reply right now. Please try again."
          );
        } else {
          typingEl.innerHTML = formatSafeHtml(data.reply);
          history.push({ role: "assistant", content: data.reply });
          if (data.suggested_prompt) {
            addUsePromptButton(data.suggested_prompt);
          }
        }
      } catch {
        history.pop();
        typingEl.innerHTML =
          "Could not reach the assistant right now. If the cloud server is waking up, please retry in a few seconds.";
      } finally {
        busy = false;
        updateControls();
        input.focus();
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    }

    toggleBtn.addEventListener("click", () => {
      panel.classList.toggle("open");
      if (panel.classList.contains("open") && !input.disabled) input.focus();
    });

    closeBtn.addEventListener("click", () => panel.classList.remove("open"));

    chipsEl.querySelectorAll(".chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        const action = btn.getAttribute("data-action");
        if (action === "how") {
          sendText("How does this work?");
        } else if (action === "improve") {
          const draft =
            window.JobTruthApp && typeof window.JobTruthApp.getCustomPrompt === "function"
              ? window.JobTruthApp.getCustomPrompt()
              : "";
          if (draft) {
            const prefix = "Improve this 2026 B2B research prompt: ";
            sendText(prefix + draft.slice(0, 400 - prefix.length));
          } else {
            sendText(
              "Improve my prompt — share one sharp, ready-to-run 2026 B2B research brief example and ask what industry I want to research."
            );
          }
        }
      });
    });

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      sendText(input.value);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initCoachWidget);
  } else {
    initCoachWidget();
  }
})();
