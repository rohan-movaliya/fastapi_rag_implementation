1. User uploads multiple files
2. Extract & Preprocess data from files
3. Store document embeddings in FAISS VectorDB
4. User ask a question related to the store data
5. Retrive relevent chunks using FAISS
6. Send retrived context to Gemini API for response generation
7. Return the response to the user via FastAPI
