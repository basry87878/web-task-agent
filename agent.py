import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

models = [
    ("gemini-flash-lite-latest", ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest", temperature=0, timeout=90, max_retries=1)),
    ("gemini-2.5-flash", ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, timeout=90, max_retries=1)),
    ("ollama qwen3:14b (local)", ChatOllama(
        model="qwen3:14b", temperature=0)),
    ("ollama phi4-mini (local)", ChatOllama(
        model="phi4-mini", temperature=0)),
]

wrapper = DuckDuckGoSearchAPIWrapper(backend="duckduckgo")

@tool
def web_search_tool(query: str) -> str:
    """Searches the web and returns titles, snippets, and exact URLs/links."""
    try:
        results = wrapper.results(query, max_results=5)
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title')}\nInfo: {r.get('snippet')}\nLink: {r.get('link')}\n")
        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Search temporarily unavailable ({e})."

@tool
def create_task(title: str, notes: str) -> str:
    """Creates a new task in my_tasks.txt with a title and detailed notes."""
    with open("my_tasks.txt", "a", encoding="utf-8") as f:
        f.write(f"- [ ] {title} | Notes: {notes}\n")
    return f"Successfully created task: '{title}'"

@tool
def save_to_excel(rows: list[dict], filename: str = "findings.xlsx") -> str:
    """Saves a list of dictionaries (e.g. [{'Profile': '...', 'Info': '...', 'Link': '...'}]) to an Excel (.xlsx) file."""
    if not filename.endswith(".xlsx"):
        filename += ".xlsx"
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)
    return f"Successfully saved {len(rows)} rows to {filename}"

tools = [web_search_tool, create_task, save_to_excel]

system_prompt = (
    "You are a helpful assistant with conversation memory. "
    "Use web_search_tool to find up-to-date information and URLs. "
    "Use create_task when asked to log a to-do task. "
    "Use save_to_excel when asked to save or export data/findings into an Excel file, "
    "using information from earlier in the conversation if already searched."
)

agents = [
    (name, create_react_agent(llm, tools, prompt=system_prompt))
    for name, llm in models
]

agent = agents[0][1]