# TalkDocs AI - Local RAG Knowledge Base

TalkDocs AI adalah sistem tanya jawab dokumen (RAG - Retrieval-Augmented Generation) yang berjalan sepenuhnya secara lokal menggunakan **FastAPI**, **LangChain**, dan **Ollama (Llama 3)**. Dengan ini, Anda bisa mengunggah dokumen PDF dan melakukan tanya jawab secara privat tanpa perlu koneksi internet.

## 🚀 Fitur Utama

- **100% Privat**: Data Anda (PDF dan Database) tidak pernah keluar dari mesin lokal.
- **FastAPI Backend**: Performa tinggi dan dokumentasi API otomatis.
- **Ollama Integration**: Menggunakan model Llama 3 yang canggih secara lokal.
- **Vector Search**: Menggunakan ChromaDB untuk pencarian informasi yang cepat dan relevan.

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python)
- **AI Orchestration**: LangChain
- **LLM**: Ollama (Llama 3)
- **Embedding**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB

## 📦 Persyaratan Sistem

1. **Python 3.10+**
2. **Ollama** terinstal di komputer.
3. **Llama 3 Model**: Jalankan perintah `ollama pull llama3` di terminal.

## ⚙️ Instalasi & Setup API

### 1. Clone Repositori
```bash
git clone https://github.com/ihksanblny/talkdocs-ai.git
cd talkdocs-ai
```

### 2. Buat Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Untuk Linux/Mac
.venv\Scripts\activate     # Untuk Windows
```

### 3. Instal Dependencies
```bash
pip install -r api/requirements.txt
```

### 4. Jalankan Server API
Masuk ke folder `api` dan jalankan uvicorn:
```bash
cd api
uvicorn main:app --reload
```
API akan berjalan di `http://127.0.0.1:8000`.

## 📖 Dokumentasi Endpoints

### 1. Cek Koneksi
- **Method**: `GET`
- **URL**: `/`
- **Tujuan**: Memastikan API berjalan dengan sukses.

### 2. Upload PDF
- **Method**: `POST`
- **URL**: `/upload`
- **Format**: `multipart/form-data`
- **Body**: `file` (File PDF)
- **Fungsi**: File akan dibaca, dipecah menjadi *chunks*, dan disimpan ke database vektor (`api/db`).

### 3. Tanya Jawab (Chat)
- **Method**: `POST`
- **URL**: `/chat`
- **Body (JSON)**:
  ```json
  {
    "query": "Apa isi dari dokumen yang saya upload?"
  }
  ```
- **Fungsi**: AI akan mencari konteks yang relevan di database dan menjawab berdasarkan dokumen tersebut.

## 📂 Struktur Folder
- `api/main.py`: Logika utama FastAPI & LangChain.
- `api/db/`: Penyimpanan database vektor (Diabaikan oleh Git).
- `api/uploads/`: Folder penyimpanan file PDF sementara (Diabaikan oleh Git).
- `.gitignore`: Mengatur file yang tidak perlu di-upload ke GitHub.
- `main.py`: File API dengan setup backend menggunakan LCEL
- `main1.py`: File API dengan setup backend Manual
### Note : Ubah salah satu file dengan nama main.py

---
Dikembangkan oleh Ihksan menggunakan Ollama & LangChain.
