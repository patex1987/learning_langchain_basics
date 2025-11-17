import os
from dataclasses import dataclass

from colorama import Fore
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from history_aware_qa_bot_2.contextualize_prompt import CONTEXTUALIZE_Q_PROMPT
from history_aware_qa_bot_2.query_explicit import query, create_full_rag_chain
from history_aware_qa_bot_2.vector_retriever import create_vector_retriever


@dataclass
class ChatBotDependenciesContainer:
    chat_history: list[str]
    knowledge_retriever: BaseRetriever
    rag_chain: RunnableSequence
    llm: BaseChatModel


def create_default_dependencies() -> ChatBotDependenciesContainer:
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


def start():
    instructions = """Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"""
    print(Fore.BLUE + "\n\x1b[3m" + instructions + "\x1b[0m" + Fore.RESET)

    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice")
        start()


def ask():
    dependencies = create_default_dependencies()

    while True:
        user_input = input("Q: ")
        # Exit
        if user_input == "x":
            start()
        else:
            response = query(
                query=user_input,
                chat_history=dependencies.chat_history,
                rag_chain=dependencies.rag_chain,
            )
            print(Fore.BLUE + "A: " + response["answer"] + Fore.RESET)
            print(Fore.WHITE + "\n-------------------------------------------------")


if __name__ == "__main__":
    start()
