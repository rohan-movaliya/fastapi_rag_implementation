from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List, Optional
import os
import shutil
from datetime import datetime, timedelta
from jose import JWTError, jwt

from utils_functions import *

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

# Secret Key for signing JWT
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

fake_users_db = {
    "admin_user": {
        "username": "admin_user",
        "email": "admin@example.com",
        "password": "admin123", 
        "role": "admin",
        "disabled": False,
    },
    "normal_user": {
        "username": "normal_user",
        "email": "user@example.com",
        "password": "user123", 
        "role": "user",
        "disabled": False,
    }
}

# Function to get user from fake database
def get_user(db, username: str):
    return db.get(username)

# Authenticate user
def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return user

# Create access token with role
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return {"access_token": access_token, "token_type": "bearer"}

# Get current user from token
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = get_user(fake_users_db, username)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return {"username": user["username"], "email": user["email"], "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Role-based access control (RBAC)
def admin_required(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied: Admins only")
    return current_user

def user_required(current_user: dict = Depends(get_current_user)):
    if current_user["role"] not in ["admin", "user"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return current_user



# --- API Endpoints ---

@app.post("/upload/", dependencies=[Depends(admin_required)])
async def upload_files(files: List[UploadFile] = File(...)):
    """Handle file uploads (Admin Only)"""
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

    return {"message": "✅ Files uploaded and FAISS index created successfully!", "files": [file.filename for file in files]}


@app.get("/list_documents/", dependencies=[Depends(user_required)])
async def list_documents():
    """Retrieve stored document chunks (Admin Only)"""
    try:
        vector_store = load_vector_store()
        vector_df = store_to_df(vector_store)
        unique_documents = vector_df["document"].unique().tolist()
        return {"documents": unique_documents}
    except Exception as e:
        return {"error": f"Failed to retrieve documents: {str(e)}"}
    

@app.delete("/delete_document/{document_name}", dependencies=[Depends(admin_required)])
async def delete_document_api(document_name: str):
    """Delete a document (Admin Only)"""
    try:
        vector_store = load_vector_store()
        vector_df = store_to_df(vector_store)
        chunk_list = vector_df.loc[vector_df["document"] == document_name]["chunk_id"].tolist()
        if not chunk_list:
            return {"message": f"❌ Document '{document_name}' not found in FAISS."}
        vector_store.delete(chunk_list)
        vector_store.save_local(FAISS_INDEX_DIR)
        return {"message": f"✅ Document '{document_name}' deleted successfully!"}
    except Exception as e:
        return {"error": f"Failed to delete document: {str(e)}"}


@app.put("/update_document/", dependencies=[Depends(admin_required)])
async def update_document(files: List[UploadFile] = File(...)):
    """Update FAISS index (Admin Only)"""
    saved_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)

    if not saved_files:
        return {"message": "❌ No valid files uploaded."}

    documents = load_documents(saved_files)
    vector_store = load_vector_store()
    add_to_vector_store(documents, vector_store)
    vector_store.save_local(FAISS_INDEX_DIR)

    # Cleanup uploads directory
    shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    return {"message": "✅ Document added successfully!", "files": [file.filename for file in files]}


@app.post("/query/", dependencies=[Depends(user_required)])
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