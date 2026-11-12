import os
import streamlit as st
from agent import agents

st.set_page_config(page_title="Web-to-Task AI Agent", page_icon="🌐", layout="wide")

st.title("🌐 Web-to-Task AI Agent")
st.write("Type a research prompt below, and the agent will surf the web and log tasks for you.")

# Sidebar to view saved tasks and download generated Excel files
with st.sidebar:
    st.header("📋 Saved Tasks")
    if os.path.exists("my_tasks.txt"):
        with open("my_tasks.txt", "r", encoding="utf-8") as f:
            tasks_content = f.read().strip()
        if tasks_content:
            st.markdown(tasks_content)
            if st.button("🗑️ Clear All Tasks"):
                open("my_tasks.txt", "w", encoding="utf-8").close()
                st.rerun()
        else:
            st.info("No tasks logged yet.")
    else:
        st.info("No tasks logged yet.")

    st.divider()
    st.header("📊 Excel Exports")
    xlsx_files = [f for f in os.listdir(".") if f.endswith(".xlsx")]
    if xlsx_files:
        for file in xlsx_files:
            with open(file, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {file}",
                    data=f,
                    file_name=file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=file
                )
    else:
        st.info("No Excel files generated yet.")

# Maintain chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)

# User input box
if prompt := st.chat_input("e.g., Search for Nvidia news and save an abstract as a task"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text = ""
        used_model = ""

        # Build full conversation history from session state
        chat_history = [
            (m["role"], m["content"]) for m in st.session_state["messages"]
        ]

        with st.status("Surfing the web and logging tasks...", expanded=True) as status:
            for name, current_agent in agents:
                st.write(f"🔄 Trying model: **{name}**")
                try:
                    for step in current_agent.stream(
                        {"messages": chat_history}, stream_mode="values"
                    ):
                        msg = step["messages"][-1]
                        if msg.type == "ai" and getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                st.write(f"🛠️ Calling `{tc['name']}`: `{tc['args']}`")
                        elif msg.type == "ai" and msg.content:
                            response_text = extract_text(msg.content)
                    used_model = name
                    status.update(label=f"Completed using {used_model}", state="complete", expanded=False)
                    break
                except Exception as e:
                    st.warning(f"{name} failed: {str(e)[:120]}... trying next model.")

        if response_text:
            st.markdown(response_text)
            st.session_state["messages"].append({"role": "assistant", "content": response_text})
            st.rerun()
        else:
            st.error("All models failed to complete the request.")