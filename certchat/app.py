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
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

load_dotenv()

app = FastAPI()

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ----------------------------
# CONFIG
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data"  # PDFs folder

# ----------------------------
# EMBEDDINGS
# ----------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embedding_model = get_embedding_model()
vectorstore: FAISS | None = None

# ----------------------------
# HELPERS
# ----------------------------
def load_pdf_files(data_path: str) -> list[Document]:
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def create_chunks(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def build_vectorstore() -> FAISS | None:
    documents = load_pdf_files(DATA_PATH)
    if not documents:
        print(f"⚠️ No PDFs found in {DATA_PATH}.")
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
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model)
        print("✅ FAISS index loaded.")
    except Exception as e:
        print(f"⚠️ Could not load FAISS index: {e}")
        vectorstore = build_vectorstore()
else:
    vectorstore = build_vectorstore()

# ----------------------------
# PROMPT
# ----------------------------
def get_custom_prompt() -> PromptTemplate:
    template = """
Use the following context to answer the question.
If the answer is not contained, respond: "I don't know."

Context: {context}
Question: {question}

Answer directly:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ----------------------------
# LLM + RETRIEVAL SETUP
# ----------------------------
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

retriever: VectorStoreRetriever | None = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None
prompt = get_custom_prompt()

# Simple query function combining retriever + LLM
async def answer_query(question: str) -> str:
    if retriever is None:
        return "Vectorstore not available. Add PDFs first."
    
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])
    
    # Create LLM input using prompt template
    input_text = prompt.format(context=context, question=question)
    
    # Get answer from LLM
    return llm.generate([input_text]).generations[0][0].text

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
    try:
        answer = await answer_query(query.question)
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
