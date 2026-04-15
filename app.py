import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

VECTORSTORE_DIR = "./vectorstore"

# Format retrieved docs into context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    print("🔄 Loading vector store...")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load vector DB
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )

    # Retriever (Top 3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM (Groq)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # Prompt
    prompt = PromptTemplate.from_template("""
You are a document-based Q&A assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say:
"I couldn't find relevant information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")

    # RAG Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n📚 RAG Q&A Bot Ready! Type 'exit' to quit.\n")

    while True:
        question = input("❓ Your question: ").strip()

        if question.lower() == "exit":
            print("👋 Exiting...")
            break

        # Retrieve documents
        docs = retriever.invoke(question)

        if not docs:
            print("\n⚠️ No relevant information found.\n")
            continue

        # Generate answer
        answer = chain.invoke(question)

        print("\n✅ Answer:\n")
        print(answer)

        # Show unique sources
        print("\n📌 Top Sources:")
        unique_sources = set()

        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "N/A")
            unique_sources.add((source, page))

        for i, (src, pg) in enumerate(unique_sources):
            print(f"{i+1}. {src} (Page {pg})")

        # Show retrieved chunks
        print("\n📖 Retrieved Context:")

        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "N/A")
            content = doc.page_content[:200].replace("\n", " ")

            print(f"[{i+1}] {source} (Page {page})")
            print(f"→ {content}...\n")

        print("-" * 60)


if __name__ == "__main__":
    main()