import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")

# -------------------------------------------------------
# 384-dim embeddings (MATCHES main.py)
# -------------------------------------------------------
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------------------------------------------
# Optimized text splitter
# -------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

# -------------------------------------------------------
# Smart PDF loader (handles text PDFs + scanned PDFs)
# -------------------------------------------------------
def load_pdf_smart(path: str):
    """Load a PDF using PyPDFLoader first, fallback to UnstructuredPDFLoader."""
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()

        # Check if PyPDF extracted text or if pages are empty
        if all(len(d.page_content.strip()) == 0 for d in docs):
            print("⚠️ Empty/Scanned PDF detected — switching to Unstructured loader.")
            raise ValueError("Empty PDF")

        return docs

    except Exception:
        loader = UnstructuredPDFLoader(
            path,
            strategy="fast",
            infer_table_structure=False,
            ocr_languages=None
        )
        return loader.load()


# -------------------------------------------------------
# FULL VECTORIZE PIPELINE
# -------------------------------------------------------
def vectorize_all_pdfs():
    print(f"\n📁 Scanning PDFs in: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print("❌ DATA_DIR does not exist!")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("❌ No PDF files found inside /data folder.")
        return

    for pdf in pdf_files:
        subject = os.path.splitext(pdf)[0]
        file_path = os.path.join(DATA_DIR, pdf)

        print(f"\n📘 Processing: {pdf}  | Subject: '{subject}'")

        # 1) Load PDF
        pages = load_pdf_smart(file_path)

        # Add metadata
        for p in pages:
            p.metadata["subject"] = subject

        # 2) Chunk PDF
        chunks = text_splitter.split_documents(pages)

        if not chunks:
            print(f"⚠️ Skipped {pdf} — no text extracted.")
            continue

        # 3) Create subject folder
        subject_db_path = os.path.join(VECTOR_DB_DIR, subject)
        os.makedirs(subject_db_path, exist_ok=True)

        # 4) Store vectors
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=subject_db_path
        )

        print(f"✅ Saved {len(chunks)} chunks → {subject_db_path}")

    print("\n🎉 SUCCESS: All PDFs vectorized into the new 384-dim system!")
