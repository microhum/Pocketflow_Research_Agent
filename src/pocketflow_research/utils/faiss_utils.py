# utils/faiss_utils.py
import faiss
import numpy as np
import logging
import os
import pickle
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)

# Note: You need to install faiss and numpy:
# pip install faiss-cpu numpy  # or faiss-gpu if you have CUDA

# --- Constants ---
DEFAULT_INDEX_FILENAME = "faiss_index.idx"
DEFAULT_MAP_FILENAME = "faiss_text_map.pkl"

# --- Core Functions ---

def build_faiss_index(embeddings: np.ndarray) -> Optional[faiss.Index]:
    """
    Builds a FAISS index from a list of embeddings.

    Args:
        embeddings (np.ndarray): A 2D numpy array where each row is an embedding.

    Returns:
        Optional[faiss.Index]: The created FAISS index, or None if input is invalid.
    """
    if embeddings is None or len(embeddings) == 0:
        logging.warning("Cannot build FAISS index from empty or None embeddings.")
        return None
    try:
        embeddings = np.asarray(embeddings, dtype='float32') # Ensure float32 for FAISS
        dimension = embeddings.shape[1]
        # Using IndexFlatL2 for simplicity. Other index types (e.g., IndexIVFFlat)
        # might be better for very large datasets but require training.
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logging.info(f"Built FAISS IndexFlatL2 with {index.ntotal} vectors of dimension {dimension}.")
        return index
    except Exception as e:
        logging.error(f"Error building FAISS index: {e}")
        return None

def save_faiss_index(index: faiss.Index,
                     text_map: Dict[int, str],
                     save_dir: str,
                     index_filename: str = DEFAULT_INDEX_FILENAME,
                     map_filename: str = DEFAULT_MAP_FILENAME):
    """
    Saves the FAISS index and the corresponding ID-to-text map to disk.

    Args:
        index (faiss.Index): The FAISS index to save.
        text_map (Dict[int, str]): Dictionary mapping index IDs (0 to n-1) to text chunks.
        save_dir (str): Directory where the files will be saved.
        index_filename (str): Filename for the FAISS index.
        map_filename (str): Filename for the text map pickle file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logging.info(f"Created save directory: {save_dir}")

    index_path = os.path.join(save_dir, index_filename)
    map_path = os.path.join(save_dir, map_filename)

    try:
        logging.info(f"Saving FAISS index to {index_path}...")
        faiss.write_index(index, index_path)
        logging.info("FAISS index saved successfully.")

        logging.info(f"Saving text map to {map_path}...")
        with open(map_path, 'wb') as f:
            pickle.dump(text_map, f)
        logging.info("Text map saved successfully.")

    except Exception as e:
        logging.error(f"Error saving FAISS index or text map: {e}")

def load_faiss_index(save_dir: str,
                     index_filename: str = DEFAULT_INDEX_FILENAME,
                     map_filename: str = DEFAULT_MAP_FILENAME) -> Tuple[Optional[faiss.Index], Optional[Dict[int, str]]]:
    """
    Loads a FAISS index and its corresponding text map from disk.

    Args:
        save_dir (str): Directory where the files are saved.
        index_filename (str): Filename of the FAISS index.
        map_filename (str): Filename of the text map pickle file.

    Returns:
        Tuple[Optional[faiss.Index], Optional[Dict[int, str]]]: Loaded index and text map,
                                                                or (None, None) if loading fails.
    """
    index_path = os.path.join(save_dir, index_filename)
    map_path = os.path.join(save_dir, map_filename)

    loaded_index = None
    loaded_map = None

    if not os.path.exists(index_path) or not os.path.exists(map_path):
        logging.warning(f"Index file ({index_path}) or map file ({map_path}) not found in {save_dir}.")
        return None, None

    try:
        logging.info(f"Loading FAISS index from {index_path}...")
        loaded_index = faiss.read_index(index_path)
        logging.info(f"FAISS index loaded successfully ({loaded_index.ntotal} vectors).")

        logging.info(f"Loading text map from {map_path}...")
        with open(map_path, 'rb') as f:
            loaded_map = pickle.load(f)
        logging.info(f"Text map loaded successfully ({len(loaded_map)} entries).")

        # Sanity check
        if loaded_index.ntotal != len(loaded_map):
             logging.warning(f"Index size ({loaded_index.ntotal}) does not match map size ({len(loaded_map)}). There might be inconsistencies.")

    except Exception as e:
        logging.error(f"Error loading FAISS index or text map: {e}")
        return None, None

    return loaded_index, loaded_map

def search_faiss_index(index: faiss.Index,
                       query_embedding: np.ndarray,
                       k: int = 5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Searches the FAISS index for the top k nearest neighbors to the query embedding.

    Args:
        index (faiss.Index): The FAISS index to search.
        query_embedding (np.ndarray): The query embedding (1D numpy array).
        k (int): The number of nearest neighbors to retrieve.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            - Distances (1D array of size k)
            - Indices (1D array of size k)
            Or (None, None) if an error occurs or k is invalid.
    """
    if index is None or query_embedding is None:
        logging.error("Index or query embedding is None.")
        return None, None
    if k <= 0:
        logging.error("k must be greater than 0 for search.")
        return None, None
    if k > index.ntotal:
        logging.warning(f"k ({k}) is greater than the number of vectors in the index ({index.ntotal}). Setting k to {index.ntotal}.")
        k = index.ntotal
        if k == 0: return None, None # Cannot search empty index

    try:
        query_embedding = np.asarray(query_embedding, dtype='float32').reshape(1, -1) # Ensure 2D for search
        distances, indices = index.search(query_embedding, k)
        # Result is 2D array [ [distances] ], [ [indices] ], flatten them
        return distances[0], indices[0]
    except Exception as e:
        logging.error(f"Error searching FAISS index: {e}")
        return None, None

# --- Example Usage ---
if __name__ == "__main__":
    print("Testing FAISS utilities...")

    # 1. Create dummy data
    dim = 8 # Low dimension for testing
    num_vectors = 100
    dummy_embeddings = np.random.rand(num_vectors, dim).astype('float32')
    dummy_text_map = {i: f"This is text chunk number {i}." for i in range(num_vectors)}

    # 2. Build index
    print("\n--- Building Index ---")
    index = build_faiss_index(dummy_embeddings)
    if index:
        print(f"Index built successfully: {index.ntotal} vectors, dimension {index.d}")

        # 3. Save index and map
        print("\n--- Saving Index and Map ---")
        save_directory = "./faiss_test_data"
        save_faiss_index(index, dummy_text_map, save_directory)

        # 4. Load index and map
        print("\n--- Loading Index and Map ---")
        loaded_idx, loaded_map = load_faiss_index(save_directory)

        if loaded_idx and loaded_map:
            print(f"Loaded index: {loaded_idx.ntotal} vectors")
            print(f"Loaded map: {len(loaded_map)} entries")

            # 5. Search index
            print("\n--- Searching Index ---")
            query_vec = np.random.rand(1, dim).astype('float32')
            k_neighbors = 5
            distances, indices = search_faiss_index(loaded_idx, query_vec, k=k_neighbors)

            if distances is not None and indices is not None:
                print(f"Search results for k={k_neighbors}:")
                for i in range(k_neighbors):
                    idx = indices[i]
                    dist = distances[i]
                    text = loaded_map.get(idx, "Text not found!")
                    print(f"  Rank {i+1}: Index={idx}, Distance={dist:.4f}, Text='{text[:30]}...'")
            else:
                print("Search failed.")
        else:
            print("Loading failed.")

        # Clean up test directory
        # import shutil
        # if os.path.exists(save_directory):
        #     shutil.rmtree(save_directory)
        #     print(f"\nCleaned up test directory: {save_directory}")

    else:
        print("Index building failed.")
