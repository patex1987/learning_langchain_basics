"""
This code builds a History-aware RAG pipeline:

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

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv
import warnings
import os

warnings.filterwarnings("ignore")

load_dotenv()

llm = ChatOpenAI()
chat_history = []


# historical messages and the latest user question, and reformulates the question if it makes reference to any information in the historical information
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

# build the full QA chain
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


# indexing
def create_vector_retriever():
    documents = TextLoader("./docs/faq.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
    splits = text_splitter.split_documents(documents)
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    retriever = db.as_retriever()
    return retriever


VECTOR_RETRIEVER = create_vector_retriever()
# Retrieve chat history
history_aware_retriever = create_history_aware_retriever(llm, VECTOR_RETRIEVER, CONTEXTUALIZE_Q_PROMPT)

# Retrieve and generate
question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)


def generate_response(query):
    """Generate a response to a user query"""
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain.invoke({"chat_history": chat_history, "input": query})


def query(query):
    """Query and generate a response"""
    response = generate_response(query)
    chat_history.extend([HumanMessage(content=query), response["answer"]])
    return response
