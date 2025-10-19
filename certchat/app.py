import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.human import HumanMessage


load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data"

# ---------------- EMBEDDINGS ----------------
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embedding_model = get_embedding_model()
vectorstore = None

# ---------------- PDF LOADING ----------------
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def build_vectorstore():
    documents = load_pdf_files(DATA_PATH)
    if not documents:
        print(f"⚠️ No PDFs found in {DATA_PATH}. Add PDFs there first.")
        return None
    chunks = create_chunks(documents)
    db = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ FAISS index created with {len(chunks)} chunks.")
    return db

# ---------------- LOAD OR CREATE VECTORSTORE ----------------
if os.path.exists(DB_FAISS_PATH) and os.path.isdir(DB_FAISS_PATH):
    try:
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("✅ FAISS index loaded.")
    except Exception as e:
        print(f"⚠️ Could not load FAISS index: {e}")
        vectorstore = build_vectorstore()
else:
    vectorstore = build_vectorstore()

# ---------------- LCEL / ChatPipeline ----------------
qa_pipeline = None
if vectorstore:

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.0,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    # ----- Prompt in old style: Question -> Context -> Answer -----
    prompt = ChatPromptTemplate.from_template("""
Question:
{question}

Context:
{context}

Answer directly:
""")

    # Fetch top-k relevant documents from FAISS
    def retrieve_and_format_docs(question: str, k: int = 3):
        docs = vectorstore.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

    # Main pipeline
    def ask_question(question: str) -> str:
        context = retrieve_and_format_docs(question)
        if not context:
            return "No documents found for the question."
        human_msg = HumanMessage(content=prompt.format_prompt(context=context, question=question).to_string())
        response = llm.generate([[human_msg]])  # use generate with list of messages
        return response.generations[0][0].text

    qa_pipeline = ask_question
else:
    print("⚠️ QA pipeline not initialized. No vectorstore available.")

# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    if qa_pipeline is None:
        return JSONResponse({"error": "Vectorstore not available. Add PDFs to build knowledge base."}, status_code=500)
    try:
        answer = qa_pipeline(query.question)
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
