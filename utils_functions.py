import pandas as pd
from langchain_community.document_loaders import TextLoader, PyPDFLoader,Docx2txtLoader,UnstructuredPowerPointLoader,UnstructuredHTMLLoader,UnstructuredXMLLoader,CSVLoader,UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


FAISS_INDEX_DIR = "faiss_store" 
GOOGLE_API_KEY = "AIzaSyBjWN3WOME3Q3PLHEG1T0brQ-Uykfe0NRU"


def load_documents(files):
    """Load documents from text and PDF files."""
    documents = []
    for file_path in files:
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_path.endswith(".html"):
            loader = UnstructuredHTMLLoader(file_path)
        elif file_path.endswith(".xml"):
            loader = UnstructuredXMLLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif file_path.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file_path)
        else:
            print(f"❌ Unsupported file format: {file_path}")
            continue
        documents.extend(loader.load())
    return documents


def create_vector_store(documents):
    """Create FAISS index and save it."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_documents = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    db = FAISS.from_documents(split_documents, embeddings)
    db.save_local(FAISS_INDEX_DIR)


def load_vector_store():
    """Load FAISS vector vector_store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    vector_store =  FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True) 
    remove_duplicate_documents(vector_store)   
    return vector_store


def delete_document(store, document):
    vector_df = store_to_df(store)
    chunk_list = vector_df.loc[vector_df["document"] == document]["chunk_id"].tolist()
    if chunk_list:
        store.delete(chunk_list)
    store.save_local(FAISS_INDEX_DIR)    


def store_to_df(vector_store):
    """Convert FAISS vector_store to DataFrame."""
    v_dict = vector_store.docstore._dict  # Access the document vector_store dictionary
    data_rows = []

    for k, v in v_dict.items():
        metadata = v.metadata  # Extract metadata safely
        doc_name = metadata.get("source", "Unknown").split("/")[-1]  # Get filename safely
        page_number = metadata.get("page", 0) + 1  # Avoid key errors, default page 1
        content = v.page_content

        data_rows.append({
            "chunk_id": k,
            "document": doc_name,
            "page": page_number,
            "content": content
        })

    return pd.DataFrame(data_rows)  # Convert to DataFrame


def add_to_vector_store(documents, vector_store):
    """Load new document, split it, and add it to FAISS vector_store."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_documents = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    db = FAISS.from_documents(split_documents, embeddings)

    # Merge with existing FAISS index
    vector_store.merge_from(db)
    vector_store.save_local(FAISS_INDEX_DIR)


def remove_duplicate_documents(vector_store):
    vector_df = store_to_df(vector_store)
    duplicates = vector_df[vector_df.duplicated(subset=["content"], keep="first")]["chunk_id"].to_list()
    if duplicates:
        vector_store.delete(duplicates)
    vector_store.save_local(FAISS_INDEX_DIR)    


def filter_documents_by_source(vector_store, source):
    """Filter FAISS documents based on source metadata."""
    filtered_docs = []

    # Ensure vector vector_store has metadata
    if not hasattr(vector_store, "index_to_docstore_id") or not hasattr(vector_store, "docstore"):
        raise ValueError("FAISS vector_store does not contain document metadata.")

    # Loop through stored document IDs
    for doc_id in vector_store.index_to_docstore_id.values():
        doc = vector_store.docstore.search(doc_id)  # Retrieve document using ID
        metadata_source = doc.metadata["source"].split("/")[-1]
        if metadata_source == source:
            filtered_docs.append(doc)

    if not filtered_docs:
        raise ValueError(f"❌ No documents found for source: {source}")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    new_vector_store = FAISS.from_documents(filtered_docs, embeddings)

    return new_vector_store 
        

def query_documents(vector_store,question):

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        template="""
        Answer the following question based only on the provided context.
        Think step by step before providing a detailed answer.

        Context:
        {context}

        Question: {input}
        """
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": question})
    
    return response
