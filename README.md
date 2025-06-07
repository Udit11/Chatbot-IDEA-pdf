# PDF Chatbot 📚💬 (Local AI)

A streamlined, AI-powered chatbot designed for querying PDF documents locally. This application enables users to upload PDF files and interact with their contents through natural language queries.

## 🔧 Key Features

- 📚 **PDF Upload & Parsing** – Seamlessly upload and process PDF documents.
- 💬 **AI-Powered Q&A** – Ask context-aware questions based on the document’s content.
- ⚡ **Efficient Vector Search** – Leveraging FAISS for high-speed semantic retrieval.
- 🤖 **Local Model Support** – Utilizes HuggingFace transformers for generating responses without requiring cloud APIs.

## 🚀 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Udit11/Chatbot-IDEA-pdf.git
cd Chatbot-IDEA-pdf
```

2. **(Optional) Create and activate a virtual environment:**

```bash
python -m venv venv
# On Unix or MacOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. **Install required packages:**

```bash
pip install -r requirements.txt
```

## 🧠 Usage

Start the chatbot with:

```bash
streamlit run app.py
```

- Use the sidebar to upload a PDF document.
- Interact with the chatbot by asking questions related to the uploaded content.

## 📦 Dependencies

- Python >= 3.8
- `streamlit`
- `pymupdf`
- `transformers`, `torch`
- `faiss-cpu`
- `langchain`

## 🌐 Deployment Options

- **Local Deployment**: Follow the instructions under “Usage.”
- **Cloud Deployment** (Streamlit Sharing, Hugging Face Spaces, etc.):
  - Ensure `requirements.txt` includes all dependencies.
  - Follow the respective cloud platform’s deployment instructions.

## 🤝 Contributing

Contributions are welcome! Fork the repository and submit a pull request with your improvements.

## 📄 License

This project is licensed under the MIT License.

---

**Author**: Udit Srivastava  
GitHub: [Udit11](https://github.com/Udit11)
