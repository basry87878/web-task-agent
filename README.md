# 📊 AI Market Research & Excel Builder

A lean, zero-billing, cloud-deployable AI research web app built with **Streamlit**, **LangGraph**, **Google Gemini**, **Tavily**, and **Pydantic**.

Turn any B2B research brief into a **source-verified, spreadsheet-ready `.xlsx` report** with clickable source links—no local Ollama models, no shared-disk state, and zero malformed rows.

---

## ✨ Key Features

* **Cloud-Only Gemini Free-Tier Fallback Chain:** Routes requests through `gemini-4-hydrogen` (primary) and falls back automatically to `gemini-3.5-flash-lite` on rate limits or transient errors.
* **Tavily Search + Extract:** Uses Tavily Search for live web discovery and Tavily Extract for full-page pricing tables, specs, and vendor details.
* **Strict Pydantic Structured Outputs:** Every run validates rows through `ResearchReport` / `ResearchItem` schemas and verifies every `source_url` against URLs actually returned by Tavily in that run.
* **In-Memory `.xlsx` Exports:** Generates formatted Excel spreadsheets (`Research` + `Summary` tabs, styled headers, clickable hyperlinks) in memory via `io.BytesIO`—multi-user safe with zero disk writes.
* **Polished Demo UI & Guardrails:**
  * 3 clickable business presets (**B2B Lead List**, **Competitor Pricing Matrix**, **Vendor Shortlist**) cached for 1 hour to protect free quotas.
  * Custom research brief input capped at **3 custom runs per session**.
  * Inline interactive preview table (`st.dataframe` with clickable source links) shown directly above the `.xlsx` download button.

---

## 🛠️ Local Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
4. Launch the app:
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Free Deployment on Render (`https://jobtruth.top/`)

This repository includes a ready-to-use [`render.yaml`](render.yaml) blueprint for Render's **Free** Python Web Service tier:

1. Push this repo to GitHub (make sure `.env` is **not** committed—it is listed in `.gitignore`).
2. In the [Render Dashboard](https://dashboard.render.com/), click **New + → Blueprint** (or **Web Service**) and connect your GitHub repository.
3. Under **Environment Variables**, add:
   * `GEMINI_API_KEY`
   * `TAVILY_API_KEY`
4. Deploy! Render runs:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```
5. **Connect Custom Domain (`jobtruth.top`):**
   * In your Render service → **Settings → Custom Domains**, add `jobtruth.top` (and `www.jobtruth.top` if desired).
   * Add the `A` / `ALIAS` and `CNAME` records Render provides at your DNS provider. Render provisions a free TLS certificate automatically.

---

## 📂 Project Structure

```text
web-task-agent/
├── app.py                  # Streamlit UI, 3 business presets, session rate limit, inline preview & download
├── agent.py                # Cloud-only Gemini fallback + Tavily Search/Extract + 2-stage LangGraph pipeline
├── schemas.py              # Strict Pydantic v2 schemas, URL verification, and in-memory .xlsx exporter
├── config.py               # Centralized env/secrets loading and guardrail settings
├── render.yaml             # Render free-tier Python web service blueprint
├── .streamlit/config.toml  # Headless server & clean UI theme config
├── requirements.txt        # Lean cloud-only Python dependencies
├── .env.example            # Template for GEMINI_API_KEY and TAVILY_API_KEY
└── .gitignore              # Excludes .env, .venv, *.xlsx, and local artifacts
```
