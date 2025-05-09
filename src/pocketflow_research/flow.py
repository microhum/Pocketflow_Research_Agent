# flow.py
import logging
from pocketflow import Flow
from pocketflow_research.nodes import (
    GetTopicNode,
    QueryIntentClassifierNode,
    KeywordExtractorNode,
    FetchArxivNode,
    FetchThaijoNode,
    ProcessPapersBatchNode,
    EmbedChunksNode,
    BuildTempIndexNode,
    EmbedQueryNode,
    RetrieveChunksNode,
    GenerateResponseNode,
    CiteSourcesNode,
    EarlyEndReporterNode # Added EarlyEndReporterNode
)
from pocketflow_research.models import SharedStore # Import SharedStore
from typing import cast # For casting dict to SharedStore

logging.basicConfig(level=logging.INFO)

def create_research_agent_flow(source: str = 'arxiv') -> Flow:
    """
    Creates and returns the Research Agent flow.

    Args:
        source (str): The paper source to use ('arxiv' or 'thaijo'). Defaults to 'arxiv'.
    
    Returns:
        Flow: The configured PocketFlow object.
    """
    if source not in ['arxiv', 'thaijo']:
        raise ValueError("Invalid source specified. Choose 'arxiv' or 'thaijo'.")

    # Instantiate nodes
    get_topic = GetTopicNode()
    classify_intent = QueryIntentClassifierNode()
    extract_keywords = KeywordExtractorNode()
    
    paper_fetching_node = FetchArxivNode() if source == 'arxiv' else FetchThaijoNode()
    process_papers = ProcessPapersBatchNode()
    embed_chunks = EmbedChunksNode()
    build_temp_index = BuildTempIndexNode()
    embed_query = EmbedQueryNode()
    retrieve_chunks = RetrieveChunksNode()
    generate_response = GenerateResponseNode()
    cite_sources = CiteSourcesNode()
    early_end_reporter = EarlyEndReporterNode() # Instantiate the new node


    # Define the flow connections 

    get_topic >> classify_intent

    # Path for "FETCH_NEW" intent
    classify_intent - "fetch_new" >> extract_keywords
    extract_keywords >> paper_fetching_node
    paper_fetching_node >> process_papers
    process_papers >> embed_chunks
    embed_chunks >> build_temp_index
    build_temp_index >> embed_query # Embed query after building new index
    embed_query - "default" >> retrieve_chunks # Default transition after embedding query

    # Path for "QA_CURRENT" intent
    # If QA_CURRENT, we still need to embed the current query (topic)
    # to search the existing persistent index.
    classify_intent - "qa_current" >> embed_query 
    # After embedding query in QA_CURRENT path, go to retrieve_chunks
    # No new papers are fetched or processed for QA_CURRENT.
    # RetrieveChunksNode will rely on the persistent index.
    embed_query - "default" >> retrieve_chunks # This ensures retrieve_chunks is hit from both paths

    # Common path after retrieval (from either FETCH_NEW or QA_CURRENT)
    retrieve_chunks >> generate_response
    generate_response >> cite_sources

    # Map the final answer to the shared data store
    nodes_that_can_end_early = [
        get_topic,
        paper_fetching_node,
        process_papers,
        embed_chunks,
        build_temp_index,
        embed_query
    ]

    for node_instance in nodes_that_can_end_early:
        node_instance - "end_early" >> early_end_reporter

    logging.info(f"Research Agent flow connections defined (Source: {source}), including end_early handling.")

    return Flow(start=get_topic)

if __name__ == "__main__":
    # print("--- Testing Research Agent Flow (ArXiv Source) ---")
    # arxiv_flow_instance = create_research_agent_flow(source='arxiv')
    # if arxiv_flow_instance:
    #     print("ArXiv flow created.")
    #     shared_arxiv_data: SharedStore = cast(SharedStore, {"topic": "Large Language Models"})
    #     arxiv_flow_instance.run(shared_arxiv_data)
    #     print("\nArXiv Flow Run Complete.")
    #     print(f"Final Answer (ArXiv): {shared_arxiv_data.get('final_answer', 'Not generated')}")

    print("\n--- Testing Research Agent Flow (ThaiJO Source) ---")
    thaijo_flow_instance = create_research_agent_flow(source='thaijo')
    if thaijo_flow_instance:
        print("ThaiJO flow created.")
        shared_thaijo_data: SharedStore = cast(SharedStore, {"topic": "ปลานิลพบเจอที่ไหนในไทยบ้าง"})
        thaijo_flow_instance.run(shared_thaijo_data)
        print("\nThaiJO Flow Run Complete.")
        # print(f"Shared Data (ThaiJO): {shared_thaijo_data}")
        print(f"Final Answer (ThaiJO): {shared_thaijo_data.get('final_answer', 'Not generated')}")
