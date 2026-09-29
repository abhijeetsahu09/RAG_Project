from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

PDF_PATH = "sample.pdf"

CHROMA_DIR = "./chroma_db"

TOP_K = 4

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


# ============================================================
# 1. LOAD PDF
# ============================================================

def load_pdf(pdf_path: str):

    print("\n[1] Loading PDF...")

    loader = PyPDFLoader(pdf_path)

    pages = loader.load()

    print(f"Loaded {len(pages)} pages.")

    return pages


# ============================================================
# 2. SPLIT DOCUMENT INTO CHUNKS
# ============================================================

def split_documents(pages):

    print("\n[2] Splitting document into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    # Add clean page numbers
    for chunk in chunks:

        # PyPDFLoader uses zero-based page numbers
        chunk.metadata["page"] = (
            int(chunk.metadata.get("page", 0)) + 1
        )

    print(f"Created {len(chunks)} chunks.")

    return chunks


# ============================================================
# 3. CREATE LOCAL EMBEDDING MODEL
# ============================================================

def create_embeddings():

    print("\n[3] Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model ready.")

    return embeddings


# ============================================================
# 4. CREATE VECTOR DATABASE
# ============================================================

def create_vectorstore(chunks, embeddings):

    print("\n[4] Creating vector database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="insurance_documents",
    )

    print("Vector database ready.")

    return vectorstore


# ============================================================
# 5. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve(vectorstore, question):

    print("\n[5] Searching for relevant chunks...")

    documents = vectorstore.similarity_search(
        question,
        k=TOP_K,
    )

    return documents


# ============================================================
# 6. SEND CONTEXT + QUESTION TO LLM
# ============================================================

def generate_answer(question, documents):

    print("\n[6] Generating answer...")

    llm = ChatOpenAI(
    model="openrouter/free",
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

    context = ""

    for i, doc in enumerate(documents):

        page = doc.metadata.get("page", "unknown")

        context += f"""
--- SOURCE {i + 1} ---
Page: {page}

{doc.page_content}

"""

    prompt = f"""
You are an insurance document analysis assistant.

Answer the user's question using ONLY the provided document
context.

Do not use outside knowledge.

If the answer cannot be found in the context, say:
 
"I could not find the answer in the provided document."

Always mention the page number when possible.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# 7. COMPLETE RAG PIPELINE
# ============================================================

def build_rag():

    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"\nPDF not found: {pdf_path.absolute()}\n"
        )

    pages = load_pdf(str(pdf_path))

    chunks = split_documents(pages)

    embeddings = create_embeddings()

    vectorstore = create_vectorstore(
        chunks,
        embeddings,
    )

    return vectorstore


# ============================================================
# 8. TERMINAL CHAT
# ============================================================

def main():

    print("\n======================================")
    print("       SIMPLE PDF RAG SYSTEM")
    print("======================================")

    vectorstore = build_rag()

    print("\nRAG system is ready!")

    print("\nAsk questions about your PDF.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":

            print("\nGoodbye!")

            break

        if not question:

            continue

        documents = retrieve(
            vectorstore,
            question,
        )

        if not documents:

            print("\nAssistant:")
            print(
                "I could not find relevant information "
                "in the document."
            )

            continue

        answer = generate_answer(
            question,
            documents,
        )

        print("\nAssistant:")
        print(answer)

        print("\nRetrieved sources:")

        for i, doc in enumerate(documents):

            page = doc.metadata.get(
                "page",
                "unknown"
            )

            print(
                f"  {i + 1}. Page {page}"
            )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()