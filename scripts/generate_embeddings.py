import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# --- Configuration ---
CONFIG = {
    "data_file": "data/gdpr-export-spain.json",
    "output_file": "data/gdpr_embeddings.npz",
    "model_id": "IIC/MEL",
    "num_docs": 0,  # Number of documents to process. 0 for all.
    "batch_size": 8,
    "force": False,  # Force regeneration even if the output file exists.
}

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_documents(filepath: Path, amount: int = 0) -> List[Dict[str, str]]:
    """
    Loads and processes documents from a JSON file.

    Args:
        filepath: Path to the JSON data file.
        amount: The number of documents to load. If 0, all documents are loaded.

    Returns:
        A list of dictionaries, where each dictionary represents a document.
    """
    logging.info(f"Loading documents from {filepath}...")
    if not filepath.exists():
        logging.error(f"Data file not found at: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if amount > 0:
        data = data[:amount]

    documents = []
    for i, entry in enumerate(data):
        content = entry.get("content", {})
        text_data = content.get("text", {})
        
        summary = text_data.get("summary", "")
        facts = text_data.get("facts", "")
        holding = text_data.get("holding", "")
        
        full_text = f"Summary: {summary}\nFacts: {facts}\nHolding: {holding}"
        doc_id = content.get("case_nr_name", f"Unknown_ID_{i}")
        
        documents.append({"id": doc_id, "text": full_text})
        
    logging.info(f"Loaded and processed {len(documents)} documents.")
    return documents

def generate_embeddings(texts: List[str], model_id: str, batch_size: int = 8) -> np.ndarray:
    """
    Generates embeddings for a list of texts using a specified model.

    Args:
        texts: A list of strings to embed.
        model_id: The Hugging Face model identifier.
        batch_size: The number of texts to process in a single batch.

    Returns:
        A numpy array of embeddings.
    """
    logging.info(f"Initializing model '{model_id}'...")
    # Use a try-except block for models that may require `trust_remote_code=True`
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    except (ValueError, TypeError):
        logging.warning("Failed to load with `trust_remote_code=True`, trying without.")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
    
    model.eval()
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Model loaded on {device}. Starting embedding generation...")
    
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

def main():
    """Main function to load data, generate embeddings, and save them."""
    output_path = Path(CONFIG["output_file"])
    if output_path.exists() and not CONFIG["force"]:
        logging.info(f"Output file '{output_path}' already exists. Set 'force' in CONFIG to overwrite. Skipping generation.")
        return

    # 1. Load documents
    documents = load_documents(Path(CONFIG["data_file"]), CONFIG["num_docs"])
    if not documents:
        return
        
    doc_texts = [doc['text'] for doc in documents]
    doc_ids = np.array([doc['id'] for doc in documents])

    # 2. Generate embeddings
    doc_vectors = generate_embeddings(doc_texts, CONFIG["model_id"], CONFIG["batch_size"])
    
    # 3. Save the IDs and vectors to a compressed file
    logging.info(f"Saving document IDs and vectors to '{output_path}'...")
    np.savez_compressed(
        output_path,
        ids=doc_ids,
        vectors=doc_vectors
    )
    logging.info("Embeddings generated and saved successfully.")

if __name__ == "__main__":
    main()