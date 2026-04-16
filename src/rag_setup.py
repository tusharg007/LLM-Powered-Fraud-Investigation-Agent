import sqlite3
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os

def setup_rag():
    print("Loading historical fraud cases from SQLite...")
    conn = sqlite3.connect("data/fraud_cases.db")
    # In real world, we'd load only recent or complex cases
    df = pd.read_sql("SELECT * FROM historical_fraud LIMIT 1000", conn)
    conn.close()

    print(f"Loaded {len(df)} historical fraud cases.")

    # Convert to LangChain Documents
    documents = []
    for _, row in df.iterrows():
        # Represent the case as a text summary
        content = (
            f"User ID: {row['user_id']} | "
            f"Amount: ${row['purchase_value']} | "
            f"Device: {row['device_id']} | "
            f"IP: {row['ip_address']} | "
            f"Browser: {row['browser']} | "
            f"Source: {row['source']} | "
            f"Time since signup: {(pd.to_datetime(row['purchase_time']) - pd.to_datetime(row['signup_time'])).total_seconds()/3600:.1f} hours\n"
            f"Resolution: {row['resolution_notes']}"
        )
        # Store raw data in metadata for retrieval filtering if needed
        metadata = {
            "user_id": str(row['user_id']),
            "purchase_value": float(row['purchase_value']),
            "is_fraud": int(row['class'])
        }
        documents.append(Document(page_content=content, metadata=metadata))

    print("Initializing HuggingFace internal embeddings (all-MiniLM-L6-v2)...")
    # Using a fast, local embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Persist directory for Chroma
    persist_directory = "models/chroma_db"
    
    # Optional: Clear old DB if it exists (for clean runs)
    # Note: For production you wouldn't wipe it
    
    print("Creating Chroma Vector Store...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    print(f"RAG vector database created and stored at {persist_directory}")

if __name__ == "__main__":
    setup_rag()
