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
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableSequence, RunnableMap, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


CHAT_HISTORY = []

def query(query: str, chat_history: list[str]):
    """Query and generate a response."""
    rag_chain = create_full_rag_chain()

    response = generate_response(rag_chain=rag_chain, query=query, chat_history=)
    chat_history.extend([
        HumanMessage(content=query),
        # we store the assistant's answer as AIMessage
        # (langchain will accept a bare string, but AIMessage is more explicit)
        # AIMessage(content=response["answer"])
        # if you want to keep it simple:
        response["answer"],
    ])
    return response

def generate_response(rag_chain, query: str, chat_history):
    """Generate a response to a user query."""
    return rag_chain.invoke({
        "chat_history": chat_history,
        "input": query,
    })

def create_full_rag_chain():
    """

    What this returns on .invoke(...):

        {
          "answer": "<string from LLM>",
          "context": [...docs...],
          "input": "<original user query>",
          "chat_history": [...],
        }


    :return:
    """
    contextualized_q_chain = create_contextualized_question_chain(question_prompt=, llm=)

    history_aware_retriever = create_history_aware_retriever(
        llm, contextualized_q_chain, VECTOR_RETRIEVER
    )
    question_answer_chain = create_qa_chain(QA_PROMPT, llm)

    # 4. Full RAG chain: (chat_history, input) → {answer, context, input}
    rag_chain = RunnableSequence([
        # Step 1: history-aware retrieval (adds "context")
        history_aware_retriever,
        # Step 2: run QA on {context, chat_history, input}
        RunnableMap({
            "answer": question_answer_chain,          # returns string
            "context": lambda x: x["context"],
            "input":   lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }),
    ])
    return rag_chain

def create_contextualized_question_chain(question_prompt, llm) -> RunnableSequence:
    """
    Convert the user query and chat history into a standalone question

    :return:
    """
    contextualize_q_chain = RunnableSequence([
        question_prompt,  # uses "chat_history" + "input"
        llm,                     # ChatOpenAI
        StrOutputParser(),       # convert AIMessage → str
    ])
    return contextualize_q_chain

def create_history_aware_retriever(llm, contextualize_q_chain, vector_retriever) -> RunnableSequence:
    """
    Create a history aware retriever

     (chat_history, input) → context docs

    :param llm: the LLM to use
    :param retriever: the retriever to use
    :param contextualize_q_chain: the chain to use to contextualize the question
    :param vector_retriever: knowledge retriever from the vector store
    :return: Output format is

        {
          "context": [Document(...), ...],
          "chat_history": [...],
          "input": "Who wrote it?"
        }

    """
    sequence = RunnableSequence([
        # Step A: build a dict with what contextualization needs
        RunnableLambda(lambda x: {
            "chat_history": x["chat_history"],
            "input": x["input"],
        }),
        # Step B: run contextualization chain
        RunnableMap({
            "standalone_question": contextualize_q_chain,
            "chat_history": lambda x: x["chat_history"],
            "input": lambda x: x["input"],
        }),
        # Step C: actually call the vector retriever with standalone_question
        RunnableLambda(lambda x: {
            "context": vector_retriever.invoke(x["standalone_question"]),
            "chat_history": x["chat_history"],
            "input": x["input"],
        }),
    ])
    return sequence

def create_qa_chain(qa_prompt, llm):
    """

    qa_prompt expects: { "context": ..., "chat_history": ..., "input": ... }

    :param qa_prompt:
    :param llm:
    :return:
    """
    question_answer_chain = RunnableSequence([
        qa_prompt,
        llm,
        StrOutputParser(),
    ])
    return question_answer_chain


