import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from streamlit_qa_app_3.contextualize_prompt import CONTEXTUALIZE_Q_PROMPT
from streamlit_qa_app_3.di.container import ChatBotDependenciesContainer
from streamlit_qa_app_3.query_explicit import create_full_rag_chain
from streamlit_qa_app_3.vector_retriever import create_vector_retriever


def create_default_dependencies() -> ChatBotDependenciesContainer:
    print("creating a new di container")

    chat_history = []
    llm = create_chat_model()
    embeddings_model = create_embeddings_model()
    knowledge_retriever = create_vector_retriever(
        faq_path=os.environ["FAQ_PATH"],
        embeddings_model=embeddings_model,
    )
    rag_chain = create_full_rag_chain(
        contextualize_question_prompt=CONTEXTUALIZE_Q_PROMPT,
        llm=llm,
        knowledge_retriever=knowledge_retriever,
    )
    return ChatBotDependenciesContainer(chat_history, knowledge_retriever, rag_chain, llm)

def create_chat_model():
    gemini_api_key = os.environ["GOOGLE_API_KEY"]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=gemini_api_key,
    )
    return llm


def create_embeddings_model():
    gemini_api_key = os.environ["GOOGLE_API_KEY"]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key,
    )
    return embeddings