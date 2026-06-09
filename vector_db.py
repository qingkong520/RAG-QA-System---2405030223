import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents_from_folder(folder_path):
    documents = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
    
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_db(documents, persist_directory=CHROMA_DB_PATH):
    chunks = split_documents(documents)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_db, len(chunks)


def load_vector_db(persist_directory=CHROMA_DB_PATH):
    if not os.path.exists(persist_directory):
        return None
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_db


def retrieve_similar_documents(vector_db, query, k=3):
    results = vector_db.similarity_search(query, k=k)
    return results


def add_documents_to_vector_db(vector_db, documents):
    chunks = split_documents(documents)
    vector_db.add_documents(chunks)
    return len(chunks)


def get_vector_db_stats(vector_db):
    collection = vector_db._collection
    count = collection.count()
    return {"chunk_count": count}


if __name__ == "__main__":
    print("测试文档加载和向量数据库...")
    pass
