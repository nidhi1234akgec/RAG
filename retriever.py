from pathlib import Path
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_classic.retrievers import EnsembleRetriever

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "deeplearning.pdf"

VECTOR_DB_PATH = BASE_DIR / "vectorstore"

# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY,
)

# --------------------------------------------------
# Load and Split Documents
# --------------------------------------------------

print("Loading PDF...")

docs = PyPDFLoader(str(PDF_PATH)).load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
).split_documents(docs)

# Add chunk number metadata
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk"] = i + 1

print(f"Total chunks created: {len(chunks)}")

# --------------------------------------------------
# BM25 Retriever
# --------------------------------------------------

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10

# --------------------------------------------------
# FAISS Vector Store
# --------------------------------------------------

if VECTOR_DB_PATH.exists():

    print("Loading existing FAISS index...")

    vector_store = FAISS.load_local(
        str(VECTOR_DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

else:

    print("Creating FAISS index...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    VECTOR_DB_PATH.mkdir(exist_ok=True)

    vector_store.save_local(str(VECTOR_DB_PATH))

    print("FAISS index saved successfully!")

# --------------------------------------------------
# FAISS Retriever
# --------------------------------------------------

faiss_retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)

# --------------------------------------------------
# Hybrid Retriever
# --------------------------------------------------

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        faiss_retriever,
    ],
    weights=[
        0.4,  # BM25
        0.6,  # FAISS
    ],
)

print("Hybrid Retriever initialized successfully.")