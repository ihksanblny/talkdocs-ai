import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="Local RAG API")

# Konfigurasi Path
UPLOAD_DIR = "uploads"
DB_DIR = "db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# 1. Inisialisasi Model AI (Ollama)
llm = OllamaLLM(model="llama3")

# 2. Inisialisasi Model Embedding (Untuk mengubah teks jadi vektor)
# all-MiniLM-L6-v2 sangat ringan dan bagus untuk performa lokal
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Inisialisasi Vector Database (ChromaDB)
vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

@app.get("/")
def read_root():
    return {"message": "API RAG Lokal berjalan dengan sukses!"}

# --- ENDPOINT BARU: UNTUK UPLOAD PDF ---
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya menerima file PDF")
    
    # Simpan file PDF ke folder uploads
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # A. Membaca isi file PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # B. Memecah teks besar menjadi potongan kecil (Chunking)
        # Tujuannya agar AI bisa mencari potongan info dengan lebih akurat
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # C. Menyimpan ke Database Vektor (ChromaDB)
        if not chunks:
            raise HTTPException(status_code=400, detail="Tidak ada teks yang bisa diekstrak dari PDF. Pastikan PDF bukan hasil scan/gambar.")
        
        vector_store.add_documents(chunks)
        
        return {
            "message": "File berhasil diproses dan disimpan ke database ingatan AI!",
            "filename": file.filename,
            "total_chunks_saved": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    query: str
    
@app.post("/chat")
def chat_with_ai(request: ChatRequest):
    try:
        # 1. Cari potongan teks yang relevan
        # Kita lakukan pencarian manual agar lebih stabil
        docs = vector_store.similarity_search(request.query, k=3)
        
        if not docs:
            return {"answer": "Saya tidak menemukan informasi tersebut di dokumen Anda.", "sources": []}
            
        # 2. Gabungkan teks dari dokumen sebagai konteks
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # 3. Susun Prompt secara manual
        full_prompt = (
            f"Anda adalah asisten AI dari Personal Knowledge Base.\n"
            f"Gunakan konteks berikut untuk menjawab pertanyaan pengguna.\n"
            f"Jika tidak ada di konteks, katakan tidak tahu.\n\n"
            f"KONTEKS:\n{context_text}\n\n"
            f"PERTANYAAN: {request.query}\n\n"
            f"JAWABAN:"
        )
        
        # 4. Panggil Ollama secara langsung
        print(f"DEBUG: Mengirim pertanyaan ke Ollama...", flush=True)
        response = llm.invoke(full_prompt)
        
        return {
            "query": request.query,
            "answer": response,
            "sources": [doc.page_content[:200] for doc in docs]
        }
        
    except Exception as e:
        print(f"!!! ERROR: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
