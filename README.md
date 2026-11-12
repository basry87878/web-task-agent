# 🌐 Web-to-Task AI Agent

<img width="1897" height="853" alt="image" src="https://github.com/user-attachments/assets/56426783-46dc-4ed2-9c9e-c96bac12231f" />


A custom, production-ready autonomous AI agent built with **LangGraph**, **LangChain**, and **Streamlit**. Designed to bridge live web research and actionable productivity, this agent surfs the web via DuckDuckGo, extracts live URLs, logs actionable checklists into local text files, and exports structured research findings directly into downloadable Excel (`.xlsx`) spreadsheets.

To ensure high availability and zero downtime during API rate limits, the system uses a **multi-model fallback architecture**—dynamically routing queries through Google Gemini cloud tiers and failing over automatically to local offline LLMs via **Ollama**.

---

## ✨ Key Features

* **Autonomous Web Research:** Searches the live web using a locked DuckDuckGo backend (`DuckDuckGoSearchAPIWrapper`) to retrieve titles, detailed snippets, and source URLs.
* **Multi-Model Fallback Pipeline:** Automatically cascades through four model tiers if an API quota or timeout occurs:
  1. `gemini-flash-lite-latest` (Cloud Primary)
  2. `gemini-2.5-flash` (Cloud Fallback)
  3. `qwen3:14b` (Local Ollama Fallback)
  4. `phi4-mini` (Local Lightweight Ollama Fallback)
* **Automated Task Logging:** Converts research into structured to-do items saved locally in `my_tasks.txt`, complete with a live sidebar checklist and one-click clearing in the UI.
* **Excel Data Export:** Compiles structured research tables (such as professional profiles, news summaries, or job listings with links) into `.xlsx` files using `pandas` and `openpyxl`, ready for instant download from the Streamlit sidebar.
* **Persistent Conversation Memory:** Maintains full multi-turn context across the session so follow-up commands (e.g., *"Export those findings to Excel"*) seamlessly reference earlier searches.

---

## 🛠️ Tech Stack

* **Orchestration & Agents:** LangGraph, LangChain (`langchain-core`, `langchain-community`)
* **LLM Providers:** Google Gemini API (`langchain-google-genai`), Ollama (`langchain-ollama`)
* **Frontend UI:** Streamlit
* **Tools & Data Processing:** DuckDuckGo Search (`ddgs`), Pandas, OpenPyXL, Python-Dotenv

---

## 📂 Project Structure

```text
web-task-agent/
├── agent.py              # Core LangGraph ReAct agent, tools, and multi-model fallback array
├── app.py                # Streamlit web interface, session memory, and sidebar managers
├── run_app.bat           # One-click Windows batch launcher
├── requirements.txt      # Project Python dependencies
├── .env.example          # Template for environment variables
└── .gitignore            # Excludes .env, .venv, and local output files
