import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Get API key safely from environment variable (or .env file)
api_key = os.getenv("GROQ_API_KEY")

def process_pdf_and_query(pdf_path: str, user_query: str):
    """
    Core RAG function: Loads a PDF, chunks text, creates local embeddings,
    builds a Chroma vector store, and queries the Groq LLM with context citations.
    """
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Preserve file metadata
    for doc in documents:
        doc.metadata["source_name"] = os.path.basename(pdf_path)

    # 2. Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(documents)

    # 3. Create Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Retrieve Chunks
    relevant_docs = retriever.invoke(user_query)

    # 5. Build Context with Citations
    context_str = ""
    for doc in relevant_docs:
        page = doc.metadata.get("page", 0) + 1
        source = doc.metadata.get("source_name", "PDF")
        context_str += f"\n[Source: {source}, Page {page}]:\n{doc.page_content}\n"

    # 6. Query Groq LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    prompt = f"""Answer the question concisely based ONLY on the context below.
If you cite facts, mention the page number where possible.

Context:
{context_str}

Question: {user_query}
"""
    
    response = llm.invoke(prompt)
    return response.content, relevant_docs


if __name__ == "__main__":
    # Local terminal testing
    sample_pdf = "sample.pdf"
    if os.path.exists(sample_pdf):
        query = "What are the key points in this document?"
        answer, docs = process_pdf_and_query(sample_pdf, query)
        print("--- Answer ---")
        print(answer)
    else:
        print(f"Please place a '{sample_pdf}' file in this directory to test locally.")