import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

app = FastAPI()

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ----------------------------
# CONFIG
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data"  # Add your PDFs here

# ----------------------------
# EMBEDDINGS
# ----------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embedding_model = get_embedding_model()
vectorstore = None

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

def build_vectorstore():
    """Build FAISS DB from PDFs in DATA_PATH"""
    documents = load_pdf_files(DATA_PATH)
    if not documents:
        print(f"⚠️ No PDFs found in {DATA_PATH}. Add your PDFs there first.")
        return None
    chunks = create_chunks(documents)
    db = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ FAISS index created with {len(chunks)} chunks.")
    return db

# ----------------------------
# LOAD OR CREATE VECTORSTORE
# ----------------------------
if os.path.exists(DB_FAISS_PATH) and os.path.isdir(DB_FAISS_PATH):
    try:
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("✅ FAISS index loaded.")
    except Exception as e:
        print(f"⚠️ Could not load FAISS index: {e}")
        vectorstore = build_vectorstore()
else:
    vectorstore = build_vectorstore()

# ----------------------------
# QA CHAIN
# ----------------------------
def set_custom_prompt():
    custom_template = """
    Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, just say you don't know.
    Only use the given context.

    Context: {context}
    Question: {question}

    Start the answer directly.
    """
    return PromptTemplate(template=custom_template, input_variables=["context", "question"])

qa_chain = None
if vectorstore is not None:
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.0,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt()},
    )
else:
    print("⚠️ QA chain not initialized. No vectorstore available.")

# ----------------------------
# ROUTES
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    if qa_chain is None:
        return JSONResponse({"error": "Vectorstore not available. Add PDFs to build knowledge base."}, status_code=500)
    try:
        response = qa_chain.invoke({"query": query.question})
        return JSONResponse({"answer": response["result"]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)