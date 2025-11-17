from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONTEXTUALIZE_Q_SYSTEM_PROMPT_TEXT = """Given a chat history and the latest user question {input} \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT_TEXT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
