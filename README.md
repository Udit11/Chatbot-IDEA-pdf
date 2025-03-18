PDF Chatbot 📚💬 (Local AI)

This project is a local AI-powered chatbot that allows users to upload PDFs and ask questions based on the document's content. It utilizes FAISS for vector search and a HuggingFace model for response generation.

Features

📚 Upload and process PDF files

🎮 AI-powered Q&A based on the document content

⚡ Fast vector search using FAISS

🔍 Supports local HuggingFace models

Installation

Clone the repository:

git clone https://github.com/Udit11/Chatbot-IDEA-pdf.git
cd Chatbot-IDEA-pdf

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Usage

Run the chatbot:

streamlit run app.py

Upload a PDF file via the sidebar.

Ask questions about the document.

Dependencies

Python (>=3.8)

streamlit

pymupdf (for PDF parsing)

transformers, torch

faiss-cpu

langchain

Deployment Guide

Local Deployment: Follow the usage steps above.

Cloud Deployment (Streamlit Sharing, Hugging Face Spaces, etc.):

Ensure requirements.txt is complete.

Deploy using Streamlit’s cloud services.

Contributing

Feel free to fork the repo and submit PRs.

License

MIT License.

Author: Udit Srivastava

