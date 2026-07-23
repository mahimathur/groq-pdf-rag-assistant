import os
import tempfile
import streamlit as st

# Load local .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_core import process_pdf_and_query

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Groq-Powered PDF RAG Assistant")
st.write("Upload a PDF to get an instant summary, ask questions, and view page-level citations.")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Fetch API key from Environment or Streamlit Secrets
env_api_key = os.getenv("GROQ_API_KEY")
if not env_api_key and "GROQ_API_KEY" in st.secrets:
    env_api_key = st.secrets["GROQ_API_KEY"]

# If API Key is found in environment/secrets, use it automatically!
if env_api_key:
    os.environ["GROQ_API_KEY"] = env_api_key
    st.sidebar.success("🔑 API Key configured successfully!")
else:
    # Only show the input box if NO key was found in secrets/environment
    api_key_input = st.sidebar.text_input(
        "Enter Groq API Key",
        type="password",
        help="Paste your API key here to enable document processing."
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please enter your Groq API Key in the sidebar to proceed.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        st.sidebar.info(f"File '{uploaded_file.name}' ready for processing.")

        tab1, tab2 = st.tabs(["💬 Document Q&A", "⚡ Executive Summary"])

        with tab1:
            st.subheader("Ask Questions About Your PDF")
            user_query = st.text_input(
                "Enter your question:",
                placeholder="e.g., What are the main findings or key points in this report?"
            )

            if user_query:
                with st.spinner("Retrieving relevant context and generating answer..."):
                    try:
                        answer, relevant_docs = process_pdf_and_query(tmp_path, user_query)
                        
                        st.markdown("### Answer")
                        st.write(answer)

                        st.markdown("---")
                        st.markdown("### Source Citations")
                        for idx, doc in enumerate(relevant_docs, 1):
                            page_num = doc.metadata.get("page", 0) + 1
                            with st.expander(f"Reference {idx} — Page {page_num}"):
                                st.write(doc.page_content)

                    except Exception as e:
                        st.error(f"An error occurred during query processing: {e}")

        with tab2:
            st.subheader("Instant Document Summary")
            if st.button("Generate Executive Summary"):
                with st.spinner("Summarizing document..."):
                    try:
                        summary_query = "Provide a concise executive summary of this document, highlighting the key objectives, findings, and conclusions."
                        summary, _ = process_pdf_and_query(tmp_path, summary_query)
                        st.markdown("### Executive Summary")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {e}")
else:
    st.info("👈 Upload a PDF file from the sidebar to get started!")