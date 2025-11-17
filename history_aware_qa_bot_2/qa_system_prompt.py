from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

QA_SYSTEM_PROMPT_TEXT = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question {input}. \
based on {context}.
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
"""
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT_TEXT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
