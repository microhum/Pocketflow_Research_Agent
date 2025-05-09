# utils/chunk_text.py
import logging
import re

logging.basicConfig(level=logging.INFO)

# --- Helper Functions ---

def _chunk_by_char(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    """Splits text by character count with overlap."""
    if not text: return []
    if chunk_overlap >= chunk_size: raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    start_index = 0
    text_len = len(text)
    while start_index < text_len:
        end_index = min(start_index + chunk_size, text_len)
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
        if start_index >= end_index and end_index != text_len:
             logging.warning("Character chunking stopped due to overlap configuration.")
             break
        if start_index >= text_len: break
    logging.info(f"Split text (length {text_len}) into {len(chunks)} chunks (method: char, size ~{chunk_size}, overlap {chunk_overlap}).")
    return chunks

def _chunk_by_rule_paragraph(text: str, min_chunk_size: int = 50) -> list[str]:
    """Splits text by paragraphs (double newlines) or single newlines as fallback."""
    if not text: return []
    # Split by double newline first, then filter out empty strings
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        # Fallback to single newline if double newline yields nothing
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    # Further split very long paragraphs if needed (optional, simple split here)
    final_chunks = []
    for para in paragraphs:
        if len(para) > min_chunk_size * 5: # Arbitrary threshold for splitting long paragraphs
             # Simple split for demonstration; could use character chunking here too
             sub_chunks = _chunk_by_char(para, chunk_size=min_chunk_size*3, chunk_overlap=min_chunk_size//2)
             final_chunks.extend(sub_chunks)
        elif len(para) >= min_chunk_size:
             final_chunks.append(para)
        else:  # Handle very short paragraphs by merging with previous one
            if final_chunks and len(para) < min_chunk_size:
                final_chunks[-1] = final_chunks[-1] + " " + para
            else: final_chunks.append(para)

    # Basic filtering of potentially empty chunks after processing
    final_chunks = [chunk for chunk in final_chunks if chunk]

    logging.info(f"Split text (length {len(text)}) into {len(final_chunks)} chunks (method: rule-paragraph, min_size ~{min_chunk_size}).")
    return final_chunks

def _chunk_by_semantic(text: str, model_name: str = 'all-MiniLM-L6-v2', threshold: float = 0.5) -> list[str]:
    """Placeholder for semantic chunking using embeddings."""
    raise NotImplementedError

def chunk_text(text: str, method: str = 'rule', **kwargs) -> list[str]:
    """
    Splits text into chunks using the specified method.

    Args:
        text (str): The input text.
        method (str): Chunking method ('char', 'rule', 'semantic').
        **kwargs: Additional arguments for the specific chunking method.
            - 'char': chunk_size (int, default 1000), chunk_overlap (int, default 100)
            - 'rule': min_chunk_size (int, default 50)
            - 'semantic': model_name (str, default 'all-MiniLM-L6-v2'), threshold (float, default 0.5)

    Returns:
        list[str]: A list of text chunks.
    """
    if method == 'char':
        return _chunk_by_char(text,
                              chunk_size=kwargs.get('chunk_size', 1000),
                              chunk_overlap=kwargs.get('chunk_overlap', 100))
    elif method == 'rule':
        return _chunk_by_rule_paragraph(text,
                                        min_chunk_size=kwargs.get('min_chunk_size', 50))
    elif method == 'semantic':
        return _chunk_by_semantic(text,
                                  model_name=kwargs.get('model_name', 'all-MiniLM-L6-v2'),
                                  threshold=kwargs.get('threshold', 0.5))
    else:
        logging.error(f"Unknown chunking method: {method}. Falling back to 'rule'.")
        return _chunk_by_rule_paragraph(text)


if __name__ == "__main__":
    test_text = """This is the first paragraph. It contains several sentences about chunking.
    We are testing different methods.

    This is the second paragraph. It is separated by a double newline. Rule-based chunking should identify this.
    It might also be quite long, potentially requiring further splitting depending on the implementation details.

    This is a third, shorter paragraph.

    A final paragraph to conclude the test text. Let's see how the methods handle it.
    """

    print("--- Testing Method: char ---")
    char_chunks = chunk_text(test_text, method='char', chunk_size=100, chunk_overlap=20)
    for i, chunk in enumerate(char_chunks): print(f"Char Chunk {i+1} ({len(chunk)}): '{chunk}'")

    print("\n--- Testing Method: rule ---")
    rule_chunks = chunk_text(test_text, method='rule', min_chunk_size=40)
    for i, chunk in enumerate(rule_chunks): print(f"Rule Chunk {i+1} ({len(chunk)}): '{chunk}'")

    # print("\n--- Testing Method: semantic (Placeholder) ---")
    # semantic_chunks = chunk_text(test_text, method='semantic')
    # for i, chunk in enumerate(semantic_chunks): print(f"Semantic Chunk {i+1} ({len(chunk)}): '{chunk}'")
