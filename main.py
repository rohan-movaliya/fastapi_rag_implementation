from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import os
import shutil

from utils_functions import *


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()


@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    """Handle file uploads and process them into FAISS index."""
    saved_files = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)

    if not saved_files:
        return {"message": "❌ No valid files uploaded."}

    documents = load_documents(saved_files)

    if not documents:
        return {"message": "❌ No valid documents processed."}

    create_vector_store(documents)

    # Cleanup uploads directory
    shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    return {
        "message": "✅ Files uploaded and FAISS index created successfully!",
        "files": [file.filename for file in files]
    }


@app.get("/list_documents/")
async def list_documents():
    """Retrieve stored document chunks and metadata."""
    try:
        vector_store = load_vector_store()
        vector_df = store_to_df(vector_store)

        # Extract unique document names
        unique_documents = vector_df["document"].unique().tolist()

        return {"documents": unique_documents}
    
    except Exception as e:
        return {"error": f"Failed to retrieve documents: {str(e)}"}
    

@app.delete("/delete_document/{document_name}")
async def delete_document_api(document_name: str):
    """Delete a document and its related chunks from FAISS."""
    try:
        vector_store = load_vector_store()
        delete_document(vector_store,document_name)
        return {"message": f"✅ Document '{document_name}' deleted successfully!"}
    except Exception as e:
        return {"error": f"Failed to delete document: {str(e)}"}


@app.put("/update_document/")
async def update_document(files: List[UploadFile] = File(...)):
    """Update FAISS by adding a new document."""
        
    saved_files = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)

    if not saved_files:
        return {"message": "❌ No valid files uploaded."}

    documents = load_documents(saved_files)

    # Load existing FAISS store
    vector_store = load_vector_store()

    # Add new document
    add_to_vector_store(documents, vector_store)

    # Cleanup uploads directory
    shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    return {
        "message": f"✅ Document added successfully!",
        "files": [file.filename for file in files]
    }


@app.post("/query/")
async def query_documents_api(question: str = Form(...), source: str = Form(None)):
    """Query stored embeddings, filtering by source document."""
    
    vector_store = load_vector_store()
    
    if vector_store is None:
        return {"detail": "❌ No documents available. Please upload files first."}
    
    if source:
        # Filter documents by source
        try:
            filtered_documents = filter_documents_by_source(vector_store, source)
        except ValueError as e:
            return {"detail": str(e)}

        if not filtered_documents:
            return {"detail": f"❌ No documents found for source: {source}"}

        # Query only the filtered documents
        response = query_documents(filtered_documents, question)

        return {"answer": response["answer"]}
    
    else:
        # Query with all the documents
        response = query_documents(vector_store, question)
        source = response["context"][0].metadata["source"].split("/")[-1]
        
        return {"answer": response["answer"], "source": source}
