import streamlit as st
import fitz  # PyMuPDF for PDF parsing
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define the model name (change as needed)
MODEL_NAME = "gpt2"  # Replace with your preferred local model

def load_huggingface_llm(model_name):
    """Load a local HuggingFace model for text generation with optimizations."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        if torch.cuda.is_available():
            model = model.half().cuda()  # Load model in half precision
        else:
            model = model.float()  # Use float precision if CUDA is unavailable

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=1024,
            temperature=0.7
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        st.error(f"Error loading HuggingFace LLM: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file efficiently."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
        return text if text.strip() else "No readable text found in PDF."
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def create_vector_db(text):
    """Create a FAISS vector database from extracted text."""
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_texts(chunks, embeddings)
        return vector_db
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        return None

def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("📄💬 PDF Chatbot (Local AI)")
    st.sidebar.header("Upload your PDF")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing PDF... Please wait."):
            text = extract_text_from_pdf(uploaded_file)
            if not text or text.startswith("Error"):
                return

            vector_db = create_vector_db(text)
            if not vector_db:
                return

            retriever = vector_db.as_retriever()
            llm = load_huggingface_llm(MODEL_NAME)
            if not llm:
                return

            try:
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                st.success("✅ PDF successfully processed! You can now ask questions.")
            except Exception as e:
                st.error(f"Error setting up QA chain: {e}")
                return

        user_query = st.text_input("Ask something about the document:")
        if user_query:
            with st.spinner("Generating response..."):
                try:
                    response = qa_chain.run(user_query)
                    st.write("**Answer:**", response)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
