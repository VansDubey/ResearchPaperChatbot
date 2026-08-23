import os
import streamlit as st
from rag import create_conversational_rag_chain
from utils import generate_random_id, ask_question


def show_ui(chain, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"👋 Hello! I'm Archie. I just read the paper you shared. {prompt_to_user}",
            }
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(f"{message['content']}")

    # User-provided prompt
    if query := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking.."):
                    response = st.write_stream(
                        ask_question(chain, query, st.session_state["session_id"])
                    )
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


def main():
    st.set_page_config(page_title="Archie — arXiv Assistant", page_icon="📚")
    st.title("📚 Archie - The Arxiv Assistant")
    st.divider()

    if "ready" not in st.session_state:
        st.session_state["ready"] = False

    with st.sidebar:
        st.divider()
        st.header("Ollama Settings")
        ollama_base_url = st.text_input(
            "Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        chat_model = st.text_input(
            "Chat model", value=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
        )
        embedding_model = st.text_input(
            "Embedding model",
            value=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        )
        st.caption("Ollama must be running and both models must be downloaded.")
        st.divider()
        st.header("Add Research Paper")
        pdf_url = st.text_input(
            "🔗 Enter Arxiv PDF URL:", value="https://arxiv.org/pdf/2403.11703v1"
        )
        if "generated_session_id" not in st.session_state:
            st.session_state.generated_session_id = generate_random_id()
        session_id = st.text_input(
            "🆔 Enter Session ID:", value=st.session_state.generated_session_id
        )

        if st.button("🚀 Create Chatbot"):
            try:
                if not ollama_base_url.strip():
                    st.error("Ollama URL cannot be empty.")
                elif not chat_model.strip() or not embedding_model.strip():
                    st.error("Chat and embedding model names cannot be empty.")
                elif not session_id.strip():
                    st.error("Session ID cannot be empty.")
                else:
                    with st.spinner("Reading and indexing the paper..."):
                        chain = create_conversational_rag_chain(
                            pdf_url=pdf_url,
                            ollama_base_url=ollama_base_url.rstrip("/"),
                            chat_model=chat_model.strip(),
                            embedding_model=embedding_model.strip(),
                        )
                    st.session_state["conversational_rag_chain"] = chain
                    st.session_state["session_id"] = session_id
                    st.session_state.pop("messages", None)
                    st.success("Chatbot created successfully!")
                    st.session_state["ready"] = True

            except Exception as e:
                st.error(f"Could not create the chatbot: {e}")
        st.divider()
        st.header("Quick Guide")
        st.write(
            """
            1. Start Ollama and download both models.
            2. Enter the arXiv PDF URL.
            3. Optional: choose a Session ID for memory.
            4. Click 'Create Chatbot' to begin.
            5. Ask questions in the chat.
        """
        )

    if st.session_state["ready"]:
        show_ui(st.session_state["conversational_rag_chain"])


if __name__ == "__main__":
    main()
