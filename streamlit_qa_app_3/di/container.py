from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableSequence


@dataclass
class ChatBotDependenciesContainer:
    chat_history: list[str]
    knowledge_retriever: BaseRetriever
    rag_chain: RunnableSequence
    llm: BaseChatModel