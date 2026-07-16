import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import os



load_dotenv(Path(__file__).parent / ".env")


BASE_DIR = Path(__file__).parent

db_name = str(BASE_DIR / "vector_pdf")


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

pdfs=glob.glob(str(BASE_DIR / "pdf" / "*"))

pdf_documents = []

for folder in pdfs:
    doc_type = Path(folder).stem

    loader = DirectoryLoader(
        folder,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    pdf_docs = loader.load()

    for pdf in pdf_docs:
        pdf.metadata["doc_type"] = doc_type
        pdf_documents.append(pdf)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(pdf_documents)


if os.path.exists(db_name):
    import shutil
    shutil.rmtree(db_name)

vector = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_name
)

print("Vector database created successfully!")