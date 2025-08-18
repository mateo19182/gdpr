import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os

# --- Configuration ---
MODEL_ID = "IIC/MEL"
DATA_FILE_PATH = "data/gdpr-export-spain.json"
OUTPUT_FILE_PATH = "gdpr_embeddings.npz"

def load_documents(filepath, amount=10):
    """Loads and processes documents from the GDPRhub JSON file."""
    print(f"Loading documents from {filepath}...")
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data[:amount]:
            content = entry.get("content", {})
            text_data = content.get("text", {})
            
            summary = text_data.get("summary", "")
            facts = text_data.get("facts", "")
            holding = text_data.get("holding", "")
            
            full_text = f"Summary: {summary}\nFacts: {facts}\nHolding: {holding}"
            doc_id = content.get("case_nr_name", "Unknown_ID_" + str(len(documents)))
            
            documents.append({"id": doc_id, "text": full_text})
    print(f"Loaded and processed {len(documents)} documents.")
    return documents

def generate_embeddings(texts, batch_size=8):
    """Generates embeddings for a list of texts using the specified model."""
    print(f"Initializing model '{MODEL_ID}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID)
    
    model.eval()
    print("Model loaded. Starting embedding generation...")
    
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(batch_embeddings.cpu().numpy())
            print(f"  Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
    
    return np.vstack(all_embeddings)

def main():
    """Main function to load data, generate embeddings, and save them."""
    if os.path.exists(OUTPUT_FILE_PATH):
        print(f"Output file '{OUTPUT_FILE_PATH}' already exists. Skipping generation.")
        return

    # 1. Load documents
    documents = load_documents(DATA_FILE_PATH, 10)# Truncate to 512 characters for embedding
    doc_texts = [doc['text'] for doc in documents]
    doc_ids = np.array([doc['id'] for doc in documents]) # Convert to numpy array for saving

    # 2. Generate embeddings
    doc_vectors = generate_embeddings(doc_texts)
    
    # 3. Save the IDs and vectors to a compressed file
    print(f"Saving document IDs and vectors to '{OUTPUT_FILE_PATH}'...")
    np.savez_compressed(
        OUTPUT_FILE_PATH,
        ids=doc_ids,
        vectors=doc_vectors
    )
    print("Embeddings generated and saved successfully.")

if __name__ == "__main__":
    main()