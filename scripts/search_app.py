import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from corenn_py import CoreNN
import os

# --- Configuration ---
MODEL_ID = "IIC/MEL"
DATA_FILE_PATH = "data/gdpr-export-spain.json" # Path to the original data
EMBEDDINGS_FILE_PATH = "data/gdpr_embeddings.npz"
DB_PATH = "data/gdpr_db"
EMBEDDING_DIM = 1024

# --- Model and Tokenizer for Querying ---
print("Loading tokenizer and model for querying...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
except TypeError:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded successfully.")

def get_query_embedding(text):
    """Generates a single embedding for a query text."""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy().astype(np.float32)

# NEW FUNCTION: Loads the original text into a dictionary for easy lookup.
def load_documents_for_lookup(filepath):
    """Loads document texts into a dictionary with ID as the key."""
    doc_lookup = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i, entry in enumerate(data):
            content = entry.get("content", {})
            text_data = content.get("text", {})
            
            summary = text_data.get("summary", "")
            facts = text_data.get("facts", "")
            holding = text_data.get("holding", "")
            
            full_text = f"Summary: {summary}\nFacts: {facts}\nHolding: {holding}"
            doc_id = content.get("case_nr_name", f"Unknown_ID_{i}")
            
            doc_lookup[doc_id] = full_text
    return doc_lookup

def main():
    """Main function to load the database and perform searches."""
    
    # MODIFIED: Load document text for lookup
    print(f"Loading original document text from '{DATA_FILE_PATH}'...")
    document_lookup = load_documents_for_lookup(DATA_FILE_PATH)
    print(f"Loaded text for {len(document_lookup)} documents.")

    if not os.path.exists(DB_PATH):
        print(f"Database not found at '{DB_PATH}'. Building from '{EMBEDDINGS_FILE_PATH}'...")
        
        if not os.path.exists(EMBEDDINGS_FILE_PATH):
            print(f"Error: Embeddings file '{EMBEDDINGS_FILE_PATH}' not found.")
            print("Please run 'generate_embeddings.py' first.")
            return

        data = np.load(EMBEDDINGS_FILE_PATH)
        doc_ids = data['ids'].tolist()
        doc_vectors = data['vectors'].astype(np.float32)

        db = CoreNN.create(DB_PATH, {"dim": EMBEDDING_DIM})
        db.insert_f32(doc_ids, doc_vectors)
        print("Database created and indexed successfully.")
    else:
        print(f"Found existing database at '{DB_PATH}'. Opening it...")
        db = CoreNN.open(DB_PATH)

    print(db)

    print("\n--- Performing Search ---")
    query_text = "complaint against Facebook for sharing user data"
    query_text = input(f"Enter your query (default: '{query_text}'): ") or query_text
    
    query_vector = get_query_embedding(query_text)
    
    results = db.query_f32(query_vector, 3)

    print(f"\nTop 3 results for query: '{query_text}'\n")
    for i, (doc_id, distance) in enumerate(results[0], 1):
        full_text = document_lookup.get(doc_id, "Full text not found for this ID.")
        
        snippet = full_text.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."

        print(f"--- Result {i} ---")
        print(f"  ID:       {doc_id}")
        print(f"  Distance: {distance:.4f}")
        print(f"  Snippet:  {snippet}")
        print("-" * (len(str(i)) + 12))
        print()

if __name__ == "__main__":
    main()