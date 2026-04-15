import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = "./data"
VECTORSTORE_DIR = "./vectorstore"

def load_documents(folder):
    docs = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    return docs

def main():
    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} pages")

    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Embedding and storing...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=VECTORSTORE_DIR)
    print("Done! Vector store saved.")

if __name__ == "__main__":
    main()