import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Impor Model & Database
from langchain_ollama import ChatOllama # Berubah dari OllamaLLM ke ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Impor Chain untuk RAG (Retrieval Chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


app = FastAPI(title="Local RAG API - Retrieval Chain Mode")

# Konfigurasi Path Penyimpanan
UPLOAD_DIR = "uploads"
DB_DIR = "db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# 1. Inisialisasi Model AI & Embeddings
# Menggunakan ChatOllama agar lebih kompatibel dengan Retrieval Chain
llm = ChatOllama(model="llama3")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# Skema Request untuk Chat
class ChatRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "API RAG Lokal (Retrieval Chain) berjalan!"}

# --- ENDPOINT UPLOAD PDF ---
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya menerima file PDF")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Teks PDF tidak terbaca")
            
        vector_store.add_documents(chunks)
        return {"message": "Upload Berhasil", "filename": file.filename, "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT CHAT (RETRIEVAL CHAIN MODE) ---
@app.post("/chat")
def chat_with_ai(request: ChatRequest):
    try:
        # A. Setup Dasar
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        system_prompt = (
            "Anda adalah asisten cerdas. Gunakan konteks di bawah ini untuk menjawab pertanyaan.\n\n"
            "KONTEKS:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        # B. Terdapat Rangkaian Chain LCEL
        # Di sini kita menyambungkan komponen secara eksplisit
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
        # Rangkaian: Cari Dokumen -> Format Jadi Teks -> Gabungkan ke Prompt -> Kirim ke Llama 3
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
        )
        # C. Eksekusi
        print(f"DEBUG: Processing query with LCEL: {request.query}", flush=True)
        response = rag_chain.invoke(request.query)
        
        # Jika responnya dalam bentuk objek ChatMessage, ambil .content-nya
        final_answer = response.content if hasattr(response, 'content') else str(response)
        return {
            "query": request.query,
            "answer": final_answer,
            "sources": "Data berhasil ditarik dari dokumen." # Bisa kita detailkan lagi nanti
        }
        
    except Exception as e:
        print(f"!!! LCEL ERROR: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
