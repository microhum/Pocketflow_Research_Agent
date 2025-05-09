# Clean up imports by removing unused typing imports
import logging
from pocketflow import Node, BatchNode
import os
from typing import List, Optional, Dict, Any, Tuple, Literal
import numpy as np
from datetime import datetime
import arxiv
from models import (
    SharedStore,
    FetchedPaperInfo,
    ChunkSourceInfo,
    RetrievedChunk,
    FaissIndexObject,
)

# Import utility functions
from utils.call_llm import call_llm
from core.rag.fetch_arxiv_papers import fetch_arxiv_papers
from core.rag.fetch_thaijo_papers import fetch_thaijo_papers # Added ThaiJO fetcher
from core.rag.process_pdf import download_and_extract_text
from core.rag.chunk_text import chunk_text
from core.rag.embedding import get_embedding
from core.rag.faiss_utils import (
    build_faiss_index,
    load_faiss_index,
    search_faiss_index,
    save_faiss_index,
    DEFAULT_INDEX_FILENAME,
    DEFAULT_MAP_FILENAME
)

logging.basicConfig(level=logging.INFO)

# --- Configuration ---
PERSISTENT_INDEX_DIR = "./persistent_faiss_index"
CHUNK_METHOD = 'rule' # 'char', 'rule', or 'semantic' (placeholder)
CHUNK_KWARGS = {'min_chunk_size': 200} # Adjust based on method
MAX_PAPERS_TO_FETCH = 20
TOP_K_RESULTS = 5 # Total relevant chunks to retrieve

# --- Flow ---
# GetTopicNode >> QueryIntentClassifierNode >> KeywordExtractorNode >> FetchArxivNode/FetchThaijoNode

# --- Nodes ---
# START OF FLOW
class GetTopicNode(Node):
    """Gets the research topic from the shared dictionary."""
    def prep(self, shared: SharedStore) -> Optional[str]:
        topic: Optional[str] = shared.get("topic")
        if not topic:
            logging.warning("Topic not found in shared data for GetTopicNode.")
            return None
        return topic

    def exec(self, topic: Optional[str]) -> Optional[str]:
        return topic

    def post(self, shared: SharedStore, prep_res: Optional[str], exec_res: Optional[str]) -> Optional[str]:
        if exec_res:
            logging.info(f"Using research topic from shared data: {exec_res}")
        else:
            logging.error("GetTopicNode executed without a valid topic.")
            return "end_early"
        return "default"

class QueryIntentClassifierNode(Node):
    """Classifies the user query intent (fetch new vs. QA on current)."""
    def prep(self, shared: SharedStore) -> Optional[str]:
        return shared.get("topic")

    def exec(self, topic: Optional[str]) -> Literal["FETCH_NEW", "QA_CURRENT", "unknown"]:
        if not topic:
            logging.error("No topic provided for intent classification.")
            return "unknown"

        prompt = f"""Analyze the following user query: "{topic}"
Determine the user's primary intent. Is the user asking a general question that requires searching for new documents, or are they asking a specific question likely answerable from previously discussed or indexed documents?
Respond with ONLY one of the following labels:
FETCH_NEW - If the query requires searching for and processing new documents.
QA_CURRENT - If the query seems to be asking about documents already in context or indexed.
Query: "{topic}"
Intent Label:"""
        try:
            intent_label_str = call_llm(prompt).strip().upper()
            logging.info(f"LLM classified intent for '{topic}' as: {intent_label_str}")
            if intent_label_str == "FETCH_NEW":
                return "FETCH_NEW"
            elif intent_label_str == "QA_CURRENT":
                return "QA_CURRENT"
            else:
                logging.warning(f"LLM returned unexpected intent label: {intent_label_str}. Defaulting to FETCH_NEW.")
                return "FETCH_NEW"
        except Exception as e:
            logging.error(f"Error during intent classification LLM call: {e}", exc_info=True)
            return "FETCH_NEW"

    def post(self, shared: SharedStore, prep_res: Optional[str], exec_res: Literal["FETCH_NEW", "QA_CURRENT", "unknown"]) -> Optional[str]:
        shared["query_intent"] = exec_res
        logging.info(f"Query intent classified as: {exec_res}")
        if exec_res == "FETCH_NEW":
            return "fetch_new"
        elif exec_res == "QA_CURRENT":
            return "qa_current"
        else:
            return "fetch_new"

class KeywordExtractorNode(Node):
    """Extracts relevant keywords from the user query for searching."""
    def prep(self, shared: SharedStore) -> Optional[str]:
        return shared.get("topic")

    def exec(self, topic: Optional[str]) -> Optional[str]:
        if not topic:
            logging.error("No topic provided for keyword extraction.")
            return topic
        prompt = f"""Extract the essential keywords or a concise search phrase from the following user query, suitable for searching academic databases like ArXiv or ThaiJO. Focus on the core concepts.
User Query: "{topic}"
Return ONLY the keywords/search phrase. Examples:
Query: "Summarize the latest advancements in large language models for code generation" -> Keywords: large language models code generation
Query: "What did the paper 'Attention is All You Need' say about transformer architectures?" -> Keywords: Attention is All You Need transformer architectures
Query: "Compare reinforcement learning methods for robotics" -> Keywords: reinforcement learning robotics comparison
Query: "ปลานิลพบเจอที่ไหนบ้างในประเทศไทย?" -> Keywords: ปลานิล ไทย
Query: "{topic}"
Keywords/Search Phrase:"""
        try:
            extracted_keywords = call_llm(prompt).strip()
            logging.info(f"Extracted keywords for '{topic}': {extracted_keywords}")
            if topic and extracted_keywords and len(extracted_keywords) < len(topic) * 1.5:
                return extracted_keywords
            else:
                logging.warning(f"Keyword extraction might have failed or returned poor result: '{extracted_keywords}'. Using original topic.")
                return topic
        except Exception as e:
            logging.error(f"Error during keyword extraction LLM call: {e}", exc_info=True)
            return topic

    def post(self, shared: SharedStore, prep_res: Optional[str], exec_res: Optional[str]) -> Optional[str]:
        shared["search_keywords"] = exec_res
        logging.info(f"Search keywords set to: {exec_res}")
        return "default"

class FetchArxivNode(Node):
    """Fetches paper metadata from arXiv based on search keywords."""
    def prep(self, shared: SharedStore) -> Optional[str]:
        keywords: Optional[str] = shared.get("search_keywords", shared.get("topic"))
        return keywords

    def exec(self, search_query: Optional[str]) -> List[FetchedPaperInfo]:
        if not search_query:
            logging.error("No search query provided to FetchArxivNode.")
            return []
        
        raw_papers_arxiv: List[arxiv.Result] = [] # Renamed to avoid conflict
        try:
            print(f"Fetching papers from arXiv for keywords: {search_query}")
            raw_papers_arxiv = fetch_arxiv_papers(search_query, max_results=MAX_PAPERS_TO_FETCH)
            logging.info(f"Raw results from fetch_arxiv_papers: {len(raw_papers_arxiv)} papers.")
        except Exception as e:
            logging.error(f"Error calling fetch_arxiv_papers: {e}", exc_info=True)
            return []

        processed_papers: List[FetchedPaperInfo] = []
        for p in raw_papers_arxiv:
            try:
                if p.pdf_url:
                    paper_data = {
                        "title": p.title,
                        "source_url": p.pdf_url,
                        "source": "arxiv",
                        "authors": [author.name for author in p.authors] if p.authors else [],
                        "published_date": p.published
                    }
                    processed_papers.append(FetchedPaperInfo(**paper_data))
                else:
                    logging.warning(f"Paper '{p.title}' skipped, missing pdf_url.")
            except AttributeError as ae:
                logging.warning(f"Paper skipped due to missing attribute: {ae}. Paper data: {p}")
            except Exception as ex: 
                logging.error(f"Error processing or validating ArXiv paper '{p.title}': {ex}", exc_info=True)
        
        logging.info(f"Papers after filtering for pdf_url and processing: {len(processed_papers)} papers.")
        return processed_papers

    def post(self, shared: SharedStore, prep_res: Optional[str], exec_res: List[FetchedPaperInfo]) -> Optional[str]:
        shared["fetched_papers"] = exec_res
        num_fetched = len(exec_res)
        logging.info(f"Fetched {num_fetched} papers from ArXiv.")
        if num_fetched > 0:
            print(f"\n--- Found {num_fetched} ArXiv papers (showing first {min(num_fetched, 5)}): ---")
            for i, paper in enumerate(exec_res[:5]):
                # paper is FetchedPaperInfo, so use attribute access
                print(f"  {i+1}. {paper.title if paper else 'N/A'}")
            print("----------------------------------------")
        elif not exec_res:
            logging.warning("No ArXiv papers found or fetched. RAG process might yield no results.")
            return "end_early"
        return "default"

class FetchThaijoNode(Node):
    """Fetches paper metadata from ThaiJO based on search keywords."""
    def prep(self, shared: SharedStore) -> Optional[str]:
        keywords: Optional[str] = shared.get("search_keywords", shared.get("topic"))
        return keywords

    def exec(self, search_query: Optional[str]) -> List[FetchedPaperInfo]:
        if not search_query:
            logging.error("No search query provided to FetchThaijoNode.")
            return []
        
        raw_papers_thaijo: List[Dict[str, Any]] = [] # Renamed
        try:
            print(f"Fetching papers from ThaiJO for keywords: {search_query}")
            raw_papers_thaijo = fetch_thaijo_papers(search_query, size=MAX_PAPERS_TO_FETCH)
        except Exception as e:
            logging.error(f"Error calling fetch_thaijo_papers: {e}", exc_info=True)
            return []

        processed_papers: List[FetchedPaperInfo] = []
        for p_dict in raw_papers_thaijo:
            try:
                if p_dict.get("url"): 
                    paper_data = {
                        "title": p_dict.get("title", "N/A"),
                        "source_url": p_dict.get("url"),
                        "source": p_dict.get("source", "thaijo"), 
                        "authors": p_dict.get("authors", []), 
                        "published_date": p_dict.get("published_date")
                    }
                    processed_papers.append(FetchedPaperInfo(**paper_data))
                else:
                    logging.warning(f"ThaiJO Paper '{p_dict.get('title', 'N/A')}' skipped, missing url.")
            except Exception as ex: 
                logging.error(f"Error processing or validating ThaiJO paper '{p_dict.get('title', 'N/A')}': {ex}", exc_info=True)
        return processed_papers

    def post(self, shared: SharedStore, prep_res: Optional[str], exec_res: List[FetchedPaperInfo]) -> Optional[str]:
        shared["fetched_papers"] = exec_res
        num_fetched = len(exec_res)
        logging.info(f"Fetched {num_fetched} papers from ThaiJO.")
        if num_fetched > 0:
            print(f"\n--- Found {num_fetched} ThaiJO papers (showing first {min(num_fetched, 5)}): ---")
            for i, paper in enumerate(exec_res[:5]):
                print(f"  {i+1}. {paper.title if paper else 'N/A'}")
            print("----------------------------------------")
        elif not exec_res:
            logging.warning("No ThaiJO papers found or fetched. RAG process might yield no results.")
            return "end_early"
        return "default"

# Type for the result of ProcessPapersBatchNode.exec
ProcessedPaperResult = Dict[str, Any] # Contains title, source, chunks (List[str])

class ProcessPapersBatchNode(BatchNode):
    """Downloads, extracts text, and chunks text for each paper."""
    def prep(self, shared: SharedStore) -> List[FetchedPaperInfo]:
        papers: List[FetchedPaperInfo] = shared.get("fetched_papers", [])
        return papers

    def exec(self, paper_info: FetchedPaperInfo) -> ProcessedPaperResult:
        logging.info(f"Processing paper_info: title={paper_info.title}, source={paper_info.source}")
        
        source_url = paper_info.source_url
        title = paper_info.title
        source = paper_info.source

        if not source_url:
            return {"title": title, "source": source, "chunks": []}

        logging.info(f"Processing paper from {source}: {title}")
        extracted_text = download_and_extract_text(source_url)
        if not extracted_text:
            logging.warning(f"Failed to extract text for paper: {title} from {source_url}")
            return {"title": title, "source": source, "chunks": []}

        chunks = chunk_text(extracted_text, method=CHUNK_METHOD, **CHUNK_KWARGS)
        logging.info(f"Generated {len(chunks)} chunks for paper: {title} ({source})")
        return {"title": title, "source": source, "chunks": chunks}

    def post(self, shared: SharedStore, prep_res: List[FetchedPaperInfo], exec_res_list: List[ProcessedPaperResult]) -> Optional[str]:
        all_chunks: List[str] = []
        chunk_source_map: Dict[int, ChunkSourceInfo] = {}
        chunk_idx_counter = 0
        for paper_result in exec_res_list:
            title = paper_result["title"]
            source = paper_result["source"]
            paper_chunks = paper_result["chunks"]
            for chunk_text_item in paper_chunks:
                all_chunks.append(chunk_text_item)
                chunk_source_map[chunk_idx_counter] = ChunkSourceInfo(title=title, source=source)
                chunk_idx_counter += 1

        shared["all_chunks"] = all_chunks
        shared["chunk_source_map"] = chunk_source_map
        logging.info(f"Total chunks collected from all papers: {len(all_chunks)}")
        if not all_chunks:
            logging.warning("No chunks were generated from any paper.")
            return "end_early"
        return "default"

class EmbedChunksNode(Node):
    """Embeds all collected text chunks."""
    def prep(self, shared: SharedStore) -> List[str]:
        return shared.get("all_chunks", [])

    def exec(self, chunks: List[str]) -> Tuple[Optional[np.ndarray], Dict[int, str]]:
        if not chunks:
            return None, {}
        embeddings: Optional[np.ndarray] = get_embedding(chunks)
        if embeddings is None:
             logging.error("Embedding generation failed.")
             return None, {}
        text_map: Dict[int, str] = {i: chunk for i, chunk in enumerate(chunks)}
        return embeddings, text_map

    def post(self, shared: SharedStore, prep_res: List[str], exec_res: Tuple[Optional[np.ndarray], Dict[int, str]]) -> Optional[str]:
        embeddings, text_map = exec_res
        if embeddings is None or len(embeddings) == 0: # np.np.ndarray can be empty
             logging.error("No embeddings generated in EmbedChunksNode.")
             shared["temp_embeddings"] = None # Or np.array([])
             shared["temp_text_map"] = {}
             return "end_early"

        shared["temp_embeddings"] = embeddings
        shared["temp_text_map"] = text_map
        logging.info(f"Generated {len(embeddings)} embeddings for temporary index.")
        return "default"

class BuildTempIndexNode(Node):
    """Builds the temporary FAISS index."""
    def prep(self, shared: SharedStore) -> Optional[np.ndarray]:
        return shared.get("temp_embeddings")

    def exec(self, embeddings: Optional[np.ndarray]) -> Optional[FaissIndexObject]:
        if embeddings is None:
            return None
        index: FaissIndexObject = build_faiss_index(embeddings)
        return index

    def post(self, shared: SharedStore, prep_res: Optional[np.ndarray], exec_res: Optional[FaissIndexObject]) -> Optional[str]:
        shared["temp_index"] = exec_res
        if exec_res is None:
            logging.error("Failed to build temporary FAISS index.")
            return "end_early"
        logging.info("Temporary FAISS index built.")

        temp_text_map: Optional[Dict[int, str]] = shared.get("temp_text_map", {})
        if temp_text_map and exec_res is not None: # exec_res is the index
            logging.info(f"Saving index and map to persistent directory: {PERSISTENT_INDEX_DIR}")
            save_faiss_index(exec_res, temp_text_map, PERSISTENT_INDEX_DIR)
        else:
            logging.warning("Temporary text map is empty or index is None. Skipping persistent save.")
        return "default"

class EmbedQueryNode(Node):
    """Embeds the user's original query."""
    def prep(self, shared: SharedStore) -> Optional[str]:
        return shared.get("topic")

    def exec(self, topic: Optional[str]) -> Optional[np.ndarray]:
        if not topic: 
            return None
        embedding: Optional[np.ndarray] = get_embedding(topic)
        return embedding

    def post(self, shared: SharedStore, prep_res: Optional[str], exec_res: Optional[np.ndarray]) -> Optional[str]:
        shared["query_embedding"] = exec_res
        if exec_res is None: 
            logging.error("Failed to embed query.")
            return "end_early"
        logging.info("Query embedded successfully.")
        return "default"

class RetrieveChunksNode(Node):
    """Loads persistent index, searches both indexes, combines results."""
    def prep(self, shared: SharedStore) -> Tuple[Optional[np.ndarray], Optional[FaissIndexObject], Dict[int, str], Dict[int, ChunkSourceInfo]]:
        query_embedding: Optional[np.ndarray] = shared.get("query_embedding")
        temp_index: Optional[FaissIndexObject] = shared.get("temp_index")
        temp_map: Dict[int, str] = shared.get("temp_text_map", {})
        chunk_source_map: Dict[int, ChunkSourceInfo] = shared.get("chunk_source_map", {})
        return query_embedding, temp_index, temp_map, chunk_source_map

    def exec(self, inputs: Tuple[Optional[np.ndarray], Optional[FaissIndexObject], Dict[int, str], Dict[int, ChunkSourceInfo]]) -> List[RetrievedChunk]:
        query_embedding, temp_index, temp_map, chunk_source_map = inputs
        if query_embedding is None:
            logging.error("Query embedding missing for retrieval.")
            return []

        results: Dict[float, Tuple[int, str]] = {} # {distance: (chunk_index, source_type)}

        persistent_index, persistent_map = load_faiss_index(PERSISTENT_INDEX_DIR)
        if persistent_index and persistent_map:
            logging.info("Searching persistent index...")
            distances, indices = search_faiss_index(persistent_index, query_embedding, k=TOP_K_RESULTS)
            if distances is not None and indices is not None:
                for dist, idx in zip(distances, indices):
                    if idx in persistent_map:
                        results[dist] = (idx, "persistent")
                    else:
                         logging.warning(f"Index {idx} found in persistent index but not in map.")
        else:
            logging.info("Persistent index not found or failed to load. Skipping.")

        if temp_index and temp_map:
            logging.info("Searching temporary index...")
            distances, indices = search_faiss_index(temp_index, query_embedding, k=TOP_K_RESULTS)
            if distances is not None and indices is not None:
                for dist, idx in zip(distances, indices):
                    if idx in temp_map:
                         results[dist] = (idx, "temporary")
                    else:
                         logging.warning(f"Index {idx} found in temporary index but not in map.")
        else:
            logging.info("Temporary index not available. Skipping.")

        sorted_distances = sorted(results.keys())
        final_retrieved_chunks: List[RetrievedChunk] = []
        seen_indices_types: set[Tuple[int, str]] = set()

        for dist in sorted_distances:
            original_index, source_type = results[dist]
            identifier = (original_index, source_type)

            if identifier not in seen_indices_types:
                current_map_for_text = temp_map if source_type == "temporary" else persistent_map

                if original_index in current_map_for_text:
                    chunk_text_content = current_map_for_text[original_index]
                    
                    # For temporary chunks, their source info is in chunk_source_map
                    # For persistent chunks, we don't have their original paper info easily here unless saved with the persistent_map
                    # Assuming chunk_source_map primarily refers to temp chunks from current session.
                    # If persistent_map items also need rich source info, that map needs to store it.
                    # For now, if source_type is 'persistent', we create a generic ChunkSourceInfo.
                    if source_type == "temporary":
                        original_paper_info: ChunkSourceInfo = chunk_source_map.get(original_index, ChunkSourceInfo(title="Unknown Persistent", source="persistent_store"))
                    else: # persistent
                        # This is a simplification. Ideally, persistent_map would store richer source info.
                        original_paper_info = ChunkSourceInfo(title=f"Persistent Chunk {original_index}", source="persistent_store")


                    final_retrieved_chunks.append(
                        RetrievedChunk(text=chunk_text_content, original_source=original_paper_info)
                    )
                    seen_indices_types.add(identifier)
                    logging.info(f"Retrieved chunk (Dist: {dist:.4f}, Source: {source_type}, Paper: {original_paper_info.title}): '{chunk_text_content[:80]}...'")

                    if len(final_retrieved_chunks) >= TOP_K_RESULTS:
                        break
                else:
                    logging.warning(f"Index {original_index} from {source_type} source not found in corresponding text map.")
        return final_retrieved_chunks

    def post(self, shared: SharedStore, prep_res: Any, exec_res: List[RetrievedChunk]) -> Optional[str]:
        shared["retrieved_chunks_with_source"] = exec_res
        logging.info(f"Retrieved {len(exec_res)} relevant chunks with source info for context.")
        if not exec_res:
            logging.warning("No relevant chunks found from either index.")
        return "default"

class GenerateResponseNode(Node):
    """Generates the final response using LLM with retrieved context."""
    def prep(self, shared: SharedStore) -> Tuple[Optional[str], List[RetrievedChunk]]:
        topic: Optional[str] = shared.get("topic")
        chunks_with_source: List[RetrievedChunk] = shared.get("retrieved_chunks_with_source", [])
        return topic, chunks_with_source

    def exec(self, inputs: Tuple[Optional[str], List[RetrievedChunk]]) -> Tuple[str, List[ChunkSourceInfo]]:
        topic, retrieved_chunks = inputs
        if not topic:
            return "Error: No topic provided.", []

        chunk_texts: List[str] = [item.text for item in retrieved_chunks]
        context = "\n\n---\n\n".join(chunk_texts)

        unique_sources: List[ChunkSourceInfo] = []
        seen_titles: set[str] = set()
        for item in retrieved_chunks:
            source_info = item.original_source # This is already ChunkSourceInfo
            if source_info.title and source_info.title != "Unknown Title" and source_info.title not in seen_titles: # Added check for "Unknown Title"
                unique_sources.append(source_info)
                seen_titles.add(source_info.title)

        if not context:
            logging.warning("Generating response without retrieved context.")
            prompt = f"Please provide information about the topic: {topic}"
        else:
            prompt = f"Based on the following context from research papers, please answer the question or summarize the topic: {topic}\n\nContext:\n{context}\n\nAnswer/Summary:"
        
        answer_text = call_llm(prompt)
        return answer_text, unique_sources

    def post(self, shared: SharedStore, prep_res: Any, exec_res: Tuple[str, List[ChunkSourceInfo]]) -> Optional[str]:
        answer_text, unique_sources = exec_res
        shared["answer_text"] = answer_text
        shared["answer_sources"] = unique_sources
        logging.info(f"Generated initial answer text and identified {len(unique_sources)} unique sources.")
        return "default"

class CiteSourcesNode(Node):
    """Formats citations for sources used in the answer and appends them."""
    def prep(self, shared: SharedStore) -> Tuple[str, List[ChunkSourceInfo], List[FetchedPaperInfo]]:
        answer_text: str = shared.get("answer_text", "")
        answer_sources: List[ChunkSourceInfo] = shared.get("answer_sources", [])
        fetched_papers: List[FetchedPaperInfo] = shared.get("fetched_papers", [])
        return answer_text, answer_sources, fetched_papers

    def exec(self, inputs: Tuple[str, List[ChunkSourceInfo], List[FetchedPaperInfo]]) -> Tuple[str, str]: # Returns (final_answer, citations_section)
        answer_text, answer_sources, fetched_papers = inputs

        if not answer_sources:
            return answer_text, "" # No sources, no citation section

        citations_list: List[str] = []
        # Create a lookup map from FetchedPaperInfo for easier access by title
        papers_lookup: Dict[str, FetchedPaperInfo] = {p.title: p for p in fetched_papers if p.title}

        for source_info_item in answer_sources: # source_info_item is ChunkSourceInfo
            title = source_info_item.title
            if not title or title == "N/A" or title == "Unknown Title" or title.startswith("Persistent Chunk"): # Skip generic persistent chunks
                continue

            paper_meta: Optional[FetchedPaperInfo] = papers_lookup.get(title)
            if not paper_meta:
                logging.warning(f"Could not find full metadata for cited source: {title}")
                citations_list.append(f"- {title} [{source_info_item.source}]")
                continue

            authors_str = ", ".join(paper_meta.authors) if paper_meta.authors else "Unknown Author"
            
            year = "N/A"
            if paper_meta.published_date:
                try:
                    if isinstance(paper_meta.published_date, str):
                        # Try to parse YYYY from YYYY-MM-DD...
                        year = paper_meta.published_date.split('-')[0]
                        if not year.isdigit() or len(year) != 4: # Basic validation
                             year = paper_meta.published_date[:4] # Fallback to first 4 chars
                    elif isinstance(paper_meta.published_date, datetime):
                        year = str(paper_meta.published_date.year)
                except Exception:
                    year = "N/A"

            url = paper_meta.source_url if paper_meta.source_url else "No URL available"
            source_type_str = paper_meta.source.capitalize()

            citation_str = f"- {authors_str}. ({year}). {title}. [{source_type_str}]. Retrieved from {url}"
            citations_list.append(citation_str)

        citations_section_str = ""
        if citations_list:
            citations_section_str = "\n\nReferences:\n" + "\n".join(citations_list)
            final_answer_str = answer_text + citations_section_str
        else:
            final_answer_str = answer_text
        
        return final_answer_str, citations_section_str

    def post(self, shared: SharedStore, prep_res: Any, exec_res: Tuple[str, str]) -> Optional[str]:
        final_answer_with_citations, _ = exec_res # Unpack tuple
        shared["final_answer"] = final_answer_with_citations # Store only the answer string
        logging.info("Appended citations to the final answer.")
        return "default"

class EarlyEndReporterNode(Node):
    """Logs that the flow ended early and stops."""
    def prep(self, shared: SharedStore) -> Optional[str]:
        # Optionally, could retrieve a reason for ending early if stored in shared
        reason: Optional[str] = shared.get("early_end_reason", "An 'end_early' signal was received by a node.")
        return reason

    def exec(self, reason: Optional[str]) -> str:
        message = f"Flow ended early. Reason: {reason if reason else 'Unknown reason, end_early triggered.'}"
        logging.warning(message)
        # Potentially, set a flag in shared store
        # shared["flow_status"] = "ended_early"
        return message # The message itself can be the exec_res

    def post(self, shared: SharedStore, prep_res: Optional[str], exec_res: str) -> Optional[str]:
        # No further action, flow stops.
        # We could also store the exec_res (the message) in shared if needed.
        shared["final_status_message"] = exec_res
        shared["final_answer"] = shared.get("final_answer", f"Flow ended early: {exec_res}") # Ensure final_answer reflects this
        return None # Stops the flow

if __name__ == "__main__":
    print("\n--- Running Node Tests ---")

    shared_data_test: SharedStore = {}
    test_topic_main = "ปลานิลสามารถพบเจอได้ที่ไหนในไทย" 

    print("\n--- Testing GetTopicNode (Simulated) ---")
    get_topic_node_test = KeywordExtractorNode()
    shared_data_test["topic"] = test_topic_main
    keywords = get_topic_node_test.run(shared_data_test)

    print(f"Topic: {test_topic_main}")
    print(f"Fetched Keywords: {keywords}")
    print("----------------------------")

    print("\n--- Testing FetchThaiJoNode ---")
    fetch_arxiv_node_test = FetchThaijoNode()
    fetch_arxiv_node_test.run(shared_data_test) # run will call prep, exec, post
    print(f"ThaiJo Papers Fetched: {len(shared_data_test.get('fetched_papers', []))}")
    arxiv_papers_test: List[FetchedPaperInfo] = shared_data_test.get('fetched_papers', [])
    print("----------------------------")

    if arxiv_papers_test:
        print("\n--- Testing ProcessPapersBatchNode (ThaiJo) ---")
        process_papers_node_test = ProcessPapersBatchNode()
        
        # Manually simulate BatchNode execution for testing as run() might not be suitable for BatchNode directly
        prep_res_batch = process_papers_node_test.prep(shared_data_test)
        exec_res_list_batch: List[ProcessedPaperResult] = []
        if prep_res_batch:
            for paper_item_test in prep_res_batch: # paper_item_test is FetchedPaperInfo
                try:
                    exec_res_item = process_papers_node_test.exec(paper_item_test)
                    exec_res_list_batch.append(exec_res_item)
                except Exception as e_batch:
                    logging.error(f"Error processing paper item in test: {paper_item_test.title}. Error: {e_batch}", exc_info=True)
                    exec_res_list_batch.append({"title": paper_item_test.title, "source": paper_item_test.source, "chunks": []})
        process_papers_node_test.post(shared_data_test, prep_res_batch, exec_res_list_batch)

        print(f"Total Chunks Generated: {len(shared_data_test.get('all_chunks', []))}")
        print(f"Chunk Source Map Entries: {len(shared_data_test.get('chunk_source_map', {}))}")
        print("----------------------------")
    else:
        print("\n--- Skipping ProcessPapersBatchNode (No ThaiJo papers) ---")
        print("----------------------------")

    if shared_data_test.get('all_chunks'):
        print("\n--- Testing EmbedChunksNode ---")
        embed_chunks_node_test = EmbedChunksNode()
        embed_chunks_node_test.run(shared_data_test)
        embeddings_test: Optional[np.ndarray] = shared_data_test.get('temp_embeddings')
        text_map_test: Dict[int, str] = shared_data_test.get('temp_text_map', {})
        print(f"Embeddings Generated: {'Yes' if embeddings_test is not None else 'No'}, Shape: {embeddings_test.shape if embeddings_test is not None and hasattr(embeddings_test, 'shape') else 'N/A'}")
        print(f"Text Map Entries: {len(text_map_test)}")
        print("----------------------------")
    else:
        print("\n--- Skipping EmbedChunksNode (No chunks) ---")
        print("----------------------------")

    if shared_data_test.get('temp_embeddings') is not None:
        print("\n--- Testing BuildTempIndexNode ---")
        build_index_node_test = BuildTempIndexNode()
        build_index_node_test.run(shared_data_test)
        temp_index_test: Optional[FaissIndexObject] = shared_data_test.get('temp_index')
        print(f"Temporary Index Built: {'Yes' if temp_index_test is not None else 'No'}")
        index_file_test = os.path.join(PERSISTENT_INDEX_DIR, DEFAULT_INDEX_FILENAME)
        map_file_test = os.path.join(PERSISTENT_INDEX_DIR, DEFAULT_MAP_FILENAME)
        print(f"Persistent Index File Exists: {os.path.exists(index_file_test)}")
        print(f"Persistent Map File Exists: {os.path.exists(map_file_test)}")
        print("----------------------------")
    else:
        print("\n--- Skipping BuildTempIndexNode (No embeddings) ---")
        print("----------------------------")

    print("\n--- Testing EmbedQueryNode ---")
    embed_query_node_test = EmbedQueryNode()
    embed_query_node_test.run(shared_data_test)
    query_embedding_test: Optional[np.ndarray] = shared_data_test.get('query_embedding')
    print(f"Query Embedding Generated: {'Yes' if query_embedding_test is not None else 'No'}")
    print("----------------------------")

    if shared_data_test.get('query_embedding') is not None:
        print("\n--- Testing RetrieveChunksNode ---")
        retrieve_chunks_node_test = RetrieveChunksNode()
        retrieve_chunks_node_test.run(shared_data_test)
        retrieved_test: List[RetrievedChunk] = shared_data_test.get('retrieved_chunks_with_source', [])
        print(f"Retrieved Chunks: {len(retrieved_test)}")
        if retrieved_test:
            print("Example Retrieved Chunk Source:", retrieved_test[0].original_source) # original_source is ChunkSourceInfo
        print("----------------------------")
    else:
        print("\n--- Skipping RetrieveChunksNode (No query embedding) ---")
        print("----------------------------")

    print("\n--- Testing GenerateResponseNode ---")
    generate_response_node_test = GenerateResponseNode()
    generate_response_node_test.run(shared_data_test)
    answer_text_test: Optional[str] = shared_data_test.get('answer_text')
    answer_sources_test: List[ChunkSourceInfo] = shared_data_test.get('answer_sources', [])
    print(f"Generated Answer Text: {answer_text_test[:100] if answer_text_test else 'N/A'}...")
    print(f"Identified Answer Sources: {len(answer_sources_test)}")
    print("----------------------------")

    print("\n--- Testing CiteSourcesNode ---")
    cite_sources_node_test = CiteSourcesNode()
    shared_data_test['fetched_papers'] = arxiv_papers_test # Ensure it has the papers fetched earlier
    cite_sources_node_test.run(shared_data_test)
    final_answer_test: Optional[str] = shared_data_test.get('final_answer')
    print(f"Final Answer with Citations: {final_answer_test if final_answer_test else 'N/A'}")
    print("----------------------------\n")
