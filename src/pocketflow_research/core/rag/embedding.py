# utils/embedding.py
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from model2vec import StaticModel

logging.basicConfig(level=logging.INFO)

# Note: You need to install sentence-transformers:
# pip install sentence-transformers

# --- Global Variables ---
# Load the model globally to avoid reloading it every time the function is called.
# This assumes the function will be called multiple times within the same process.
# If memory is a concern or you run this in serverless functions, adjust accordingly.
_model = None
# _model_name = "bge-m3-distilled" # Specify the bge-m3 model
_model_name = "FlukeTJ/bge-m3-m2v-distilled-256"
# _model_name = "jaeyong2/bge-m3-Thai"

def _load_model():
    """Loads the Sentence Transformer model."""
    global _model
    if _model is None:
        try:
            logging.info(f"Loading embedding model: {_model_name}...")
            # _model = SentenceTransformer(_model_name, cache_folder="cache_models")
            _model = StaticModel.from_pretrained(_model_name, path="cache_models")
            logging.info("Embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Sentence Transformer model '{_model_name}': {e}")
            _model = None # Ensure it's None if loading failed
    return _model

def get_embedding(text: Union[str, List[str]], normalize: bool = True) -> Union[np.ndarray, List[np.ndarray], None]:
    model = _load_model()
    if model is None:
        logging.error("Embedding model is not available.")
        return None

    try:
        embeddings = model.encode(text, normalize_embeddings=normalize)
        return embeddings
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return None

if __name__ == "__main__":
    print("Testing embedding generation...")

    # Test with a single sentence
    sentence1 = "This is a test sentence."
    print(f"\nEmbedding single sentence: '{sentence1}'")
    embedding1 = get_embedding(sentence1)
    if embedding1 is not None:
        print(f"  Embedding shape: {embedding1.shape}")
        print(f"  Embedding norm: {np.linalg.norm(embedding1)}")
    else:
        print("  Failed to generate embedding.")

    # Test with a list of sentences
    sentences = [
        "The weather is nice today.",
        "Large language models are powerful.",
        "arXiv is a repository for preprints."
    ]
    print(f"\nEmbedding list of {len(sentences)} sentences...")
    embeddings_list = get_embedding(sentences)
    if embeddings_list is not None:
        print(f"  Generated {len(embeddings_list)} embeddings.")
        print(f"  Shape of first embedding: {embeddings_list[0].shape}")
        print(f"  Norm of first embedding: {np.linalg.norm(embeddings_list[0])}")
    else:
        print("  Failed to generate embeddings for the list.")
