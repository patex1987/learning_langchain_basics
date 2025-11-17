from langchain_community.document_loaders import TextLoader

# from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter


def create_vector_retriever(faq_path: str, embeddings_model: Embeddings):
    print("Creating vector embeddings")
    documents = TextLoader(faq_path).load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
    chunks = text_splitter.split_documents(documents)
    # db = Chroma.from_documents(chunks, embedding=embeddings_model)
    db = InMemoryVectorStore.from_documents(chunks, embeddings_model)
    retriever = db.as_retriever()
    return retriever
