import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ---------------------------------------------------------
# 1. Page Configuration & Title
# ---------------------------------------------------------
st.set_page_config(page_title="Advanced Groq RAG Assistant", page_icon="⚡", layout="wide")
st.title("⚡ Advanced Multi-PDF RAG Assistant (Powered by Groq)")

# ---------------------------------------------------------
# 2. Sidebar Configuration (API Key & File Upload)
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Groq API Key:", type="password", help="Get a free key from console.groq.com")
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    process_btn = st.button("Process Documents", type="primary")

# ---------------------------------------------------------
# 3. Document Processing Pipeline
# ---------------------------------------------------------
if process_btn:
    if not api_key:
        st.error("Please enter your Groq API key!")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF document.")
    else:
        os.environ["GROQ_API_KEY"] = api_key
        
        with st.spinner("Processing PDFs, chunking text, and building vector index..."):
            documents = []
            for file in uploaded_files:
                # Save uploaded file temporarily to read metadata
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file.read())
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                # Preserve original filename in metadata
                for doc in docs:
                    doc.metadata["source_name"] = file.name
                documents.extend(docs)
                os.remove(tmp_path)

            # Step A: Chunking
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = text_splitter.split_documents(documents)

            # Step B: Local Embeddings & Vector Store
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Step C: Automatic Executive Summary Generation
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
            sample_text = "\n".join([d.page_content[:400] for d in documents[:4]])
            summary_prompt = f"Provide a clean, bulleted executive summary of these documents based on this excerpt:\n\n{sample_text}"
            summary_res = llm.invoke(summary_prompt)
            st.session_state.summary = summary_res.content
            
            st.success("Documents processed successfully!")

# ---------------------------------------------------------
# 4. Executive Summary Section
# ---------------------------------------------------------
if "summary" in st.session_state:
    st.subheader("💡 Executive Summary")
    st.info(st.session_state.summary)

# ---------------------------------------------------------
# 5. Interactive Chat & Citation Interface
# ---------------------------------------------------------
if "retriever" in st.session_state:
    st.subheader("💬 Ask Questions Across Your Uploaded Documents")
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render Past Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Process User Input
    user_query = st.chat_input("Ask a question about your uploaded PDFs...")
    
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
            
            # Retrieve Relevant Chunks
            relevant_docs = st.session_state.retriever.invoke(user_query)
            
            # Construct Context with File & Page Citations
            context_str = ""
            sources_used = []
            for doc in relevant_docs:
                page = doc.metadata.get("page", 0) + 1
                source = doc.metadata.get("source_name", "PDF")
                context_str += f"\n[Source: {source}, Page {page}]:\n{doc.page_content}\n"
                sources_used.append(f"**{source}** (Page {page})")

            prompt = f"""Answer the question concisely based ONLY on the context below.
If you cite facts, mention the page number where possible.

Context:
{context_str}

Question: {user_query}
"""
            response = llm.invoke(prompt)
            st.write(response.content)
            
            # Display Sources in an Expander Dropdown
            with st.expander("📌 View Retrived Context & Page Citations"):
                st.write("Information retrieved from:")
                for s in set(sources_used):
                    st.markdown(f"* {s}")

        st.session_state.messages.append({"role": "assistant", "content": response.content})