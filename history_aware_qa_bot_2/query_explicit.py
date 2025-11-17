"""
The same as query_lcel, but using an explicit notation

- User asks a question
- LangChain rewrites the question using chat history (so it becomes standalone)
- The rewritten question is sent to the vector retriever
- Retriever returns relevant docs from Chroma
- LLM answers using:
    - retrieved docs
    - chat history
    - the current question
- The answer is appended to the history
- Next question uses the history again

So LangChain is doing two LLM calls per user query:
- rewrite question using history
- answer question using retrieved docs + history
"""
from functools import partial
from typing import Any

from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableSequence, RunnableMap, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from history_aware_qa_bot.qa_system_prompt import QA_PROMPT


def query(
    query: str,
    chat_history: list[str],
    rag_chain: RunnableSequence,
):
    """
    Query and generate a response.

    TODO: You need to provide all other dependencies as well
    """
    response = generate_response(rag_chain=rag_chain, query=query, chat_history=chat_history)
    chat_history.extend(
        [
            HumanMessage(content=query),
            # we store the assistant's answer as AIMessage
            # (langchain will accept a bare string, but AIMessage is more explicit)
            # AIMessage(content=response["answer"])
            # if you want to keep it simple:
            response["answer"],
        ]
    )
    return response


def generate_response(rag_chain, query: str, chat_history):
    """Generate a response to a user query."""
    return rag_chain.invoke(
        {
            "chat_history": chat_history,
            "input": query,
        }
    )


def create_full_rag_chain(
    contextualize_question_prompt: ChatPromptTemplate,
    llm: BaseChatModel,
    knowledge_retriever: BaseRetriever,
) -> RunnableSequence:
    """
    Main chain the receives the input, enriches and creates a standalone question based on history, and answers it.


    What this returns on .invoke(...):
    Input:
        (chat_history, input)
    Output:
        {
          "answer": "<string from LLM>",
          "context": [...docs...],
          "input": "<original user query>",
          "chat_history": [...],
        }


    :return:
    """
    contextualized_q_chain = create_contextualized_question_chain(
        contextualize_question_prompt=contextualize_question_prompt, llm=llm
    )

    history_aware_retriever = create_history_aware_retriever(contextualized_q_chain, knowledge_retriever)
    question_answer_chain = create_qa_chain(QA_PROMPT, llm)

    rag_chain = RunnableSequence(
        *[
            history_aware_retriever,
            RunnableMap(
                {
                    "answer": question_answer_chain,  # returns string
                    "context": lambda x: x["context"],
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: x["chat_history"],
                }
            ),
        ]
    )
    return rag_chain


def create_contextualized_question_chain(
    contextualize_question_prompt: ChatPromptTemplate, llm: BaseChatModel
) -> RunnableSequence:
    """
    Convert the user query and chat history into a standalone question

    :return:
    """
    runnable_steps = [
        contextualize_question_prompt,  # uses "chat_history" + "input"
        llm,  # ChatOpenAI
        StrOutputParser(),  # convert AIMessage → str
    ]

    contextualize_q_chain = RunnableSequence(*runnable_steps)
    return contextualize_q_chain


def create_history_aware_retriever(contextualize_q_chain: RunnableSequence, vector_retriever: BaseRetriever) -> RunnableSequence:
    """
    Create a history aware retriever.

    Create a standalone question enriched by the chat history.
     (chat_history, input) → context docs

    :param contextualize_q_chain: the chain to use to contextualize the question
    :param vector_retriever: knowledge retriever from the vector store
    :return: Output format is

        {
          "context": [Document(...), ...],
          "chat_history": [...],
          "input": "Who wrote it?"
        }

    """
    wrapped_knowledge_retriever = VectorRetrieverWrapper(vector_retriever)

    sequence = RunnableSequence(
        *[
            # Step A: build a dict with what contextualization needs
            RunnableLambda(
                lambda x: {
                    "chat_history": x["chat_history"],
                    "input": x["input"],
                }
            ),
            # Step B: run contextualization chain
            RunnableMap(
                {
                    "standalone_question": contextualize_q_chain,
                    "chat_history": lambda x: x["chat_history"],
                    "input": lambda x: x["input"],
                }
            ),
            # Step C: actually call the vector retriever with standalone_question
            RunnableLambda(wrapped_knowledge_retriever),
        ]
    )
    return sequence

class VectorRetrieverWrapper:
    """
    creating just for the sake of fun, as i don't want to have lambdas all over the place
    Plus this makes my debugging experience better, i can stop whenever i want to see vector
    retriever is getting called
    """
    def __init__(self, vector_retriever: BaseRetriever):
        self.vector_retriever = vector_retriever

    def __call__(self, raw_input_data: dict[str, Any]):
        context = self.vector_retriever.invoke(raw_input_data["standalone_question"])
        return {
            "context": context,
            "chat_history": raw_input_data["chat_history"],
            "input": raw_input_data["input"],
        }

def create_qa_chain(qa_prompt, llm):
    """

    qa_prompt expects: { "context": ..., "chat_history": ..., "input": ... }

    :param qa_prompt:
    :param llm:
    :return:
    """
    question_answer_chain = RunnableSequence(
        *[
            qa_prompt,
            llm,
            StrOutputParser(),
        ]
    )
    return question_answer_chain
