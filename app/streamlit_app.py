import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="SentinelAI", page_icon="📄")
st.title("SentinelAI")
st.caption("Ask questions about your uploaded documents.")

with st.sidebar:
    st.header("Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "md", "pdf"],
    )

    if uploaded_file and st.button("Upload and index"):
        with st.spinner("Uploading and indexing..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                    timeout=120,
                )
                response.raise_for_status()
                st.success(response.json()["message"])
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()
    st.subheader("Available documents")

    if st.button("Refresh list"):
        st.rerun()

    try:
        response = requests.get(f"{API_URL}/documents", timeout=15)
        response.raise_for_status()
        documents = response.json()["documents"]

        if not documents:
            st.caption("No documents found.")

        for document in documents:
            filename = document["filename"]

            with st.container(border=True):
                st.write(f"📄 {filename}")

                if st.button("Delete", key=f"delete_{filename}"):
                    try:
                        delete_response = requests.delete(
                            f"{API_URL}/documents/{filename}",
                            timeout=30,
                        )
                        delete_response.raise_for_status()
                        st.success(f"Deleted {filename}")
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(f"Delete failed: {exc}")

    except requests.RequestException:
        st.warning("Could not load documents. Is FastAPI running?")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(source)

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

                answer = result["answer"]
                # Your graph currently returns answer as a list.
                if isinstance(answer, list):
                    answer = "\n\n".join(
                        item.get("text", str(item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in answer
                    )

                st.markdown(answer)

                sources = result.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            if isinstance(source, dict):
                                st.write(source.get("source", source))
                                if source.get("text"):
                                    st.caption(source["text"])
                            else:
                                st.write(source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.RequestException as exc:
                st.error(f"Could not reach SentinelAI API: {exc}")