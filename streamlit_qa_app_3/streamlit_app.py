import streamlit as st
from langchain_core.runnables import RunnableSequence

from streamlit_qa_app_3.di.singleton import DEPENDENCIES_SINGLETON
from streamlit_qa_app_3.query_explicit import query

def init_chat_history():
    """Initialize chat history with a system message."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]


def start_chat(internal_chat_history: list[str], rag_chain: RunnableSequence):
    """Start the chatbot conversation."""
    # Display chat messages from history on app rerun
    with chat_placeholder.container():
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from Chat models
        response = query(
            query=prompt,
            chat_history=internal_chat_history,
            rag_chain=rag_chain,
        )

        # message_placeholder.markdown(response)
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

st.title("🤖 Q&A App")
chat_placeholder = st.empty()

if __name__ == "__main__":
    print('Reloading the main loop')
    init_chat_history()
    start_chat(
        internal_chat_history=DEPENDENCIES_SINGLETON.chat_history,
        rag_chain=DEPENDENCIES_SINGLETON.rag_chain,
    )