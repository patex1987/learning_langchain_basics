"""
A very simple showcase how to leverage vector databased in an FAQ chatbot.

Using explicit notations for creating chains, instead of the LCEL notation.
Motivation: to learn and understand how langchain works under the hood.
"""

from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableMap,
    RunnableSequence,
)

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_core.runnables.utils import Output
from langchain_core.vectorstores import VectorStoreRetriever
from openai import OpenAI
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# init model
model = ChatOpenAI()


SYSTEM_TEMPLATE_TEXT: str = """
    You are a customer support specialist 
    question: {question}. 
    You assist users with general inquiries based on {context} 
    and  technical issues. 
    """

# define prompt
SYSTEM_PROMPT_TEMPLATE = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE_TEXT)
USER_PROMPT_TEMPLATE = HumanMessagePromptTemplate.from_template(
    input_variables=["question", "context"],
    template="{question}"
)
CUSTOMER_SPECIALIST_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SYSTEM_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE
])

def get_embedding(text_to_embed):
    response = client.embeddings.create(
        model= "text-embedding-ada-002",
        input=[text_to_embed]
    )
    print(response.data[0].embedding)




# indexing
def load_split_documents() -> list[Document]:
    """Load a file from path, split it into chunks, embed each chunk and load it into the vector store."""
    raw_text = TextLoader("./docs/faq.txt.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=30, chunk_overlap=0, separator=".")
    chunks = text_splitter.split_documents(raw_text)
    # print(f"number of chunks {len(chunks)}")
    # print(chunks[0])
    return chunks


# convert to embeddings
def create_embeddings_retriever(documents, user_query) -> VectorStoreRetriever:
    """Create a vector store from a set of documents."""
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embeddings)
    docs = db.similarity_search(user_query)
    # print(docs)
    # get_embedding(user_query)
    # _ = [get_embedding(doc.page_content) for doc in docs]
    return db.as_retriever()


def build_chain(retriever) -> RunnableSequence:
    """
    Build a chain for an FAQ application vectorizing the faq.txt documents.

    Build the chain with the explicit notation.

    :param retriever:
    :return:
    """

    # Step 1: Split the input into {context, question}
    map_inputs = RunnableMap({
        "context": retriever,
        "question": RunnablePassthrough(),
    })

    # Step 2: Build the sequence manually
    chain_sequence = [
        map_inputs,
        CUSTOMER_SPECIALIST_CHAT_PROMPT,
        model,
        StrOutputParser(),
    ]
    chain = RunnableSequence(*chain_sequence)

    return chain

def generate_response(retriever, query) -> Output:
    """Generate a response to a user query."""
    chain = build_chain(retriever)
    return chain.invoke(query)


def query(query) -> Output:
    """Query the model with a user query."""
    documents = load_split_documents()
    retriever = create_embeddings_retriever(documents, query)
    return generate_response(retriever, query)

# query("what is the return policy?")


