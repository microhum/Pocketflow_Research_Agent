# utils/fetch_arxiv_papers.py
import arxiv
import logging
from typing import List
from pocketflow_research.models import ArxivPaperSchema

logging.basicConfig(level=logging.INFO)

def fetch_arxiv_papers(query: str, max_results: int = 10, days_back: int = 30) -> List[ArxivPaperSchema]:
    """
    Fetches paper metadata from arXiv based on a query.

    Args:
        query (str): The search query (e.g., topic, keywords).
        max_results (int): Maximum number of papers to retrieve.
        days_back (int): How many days back to search for recent papers.

    Returns:
        List[ArxivPaperSchema]: A list of arXiv paper schema objects.
              Returns an empty list if an error occurs.
    """
    search_results: List[ArxivPaperSchema] = []
    try:
        full_query = query
        logging.info(f"Searching arXiv with query: '{full_query}', max_results={max_results}")

        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance # Sort by relevance
        )

        results = list(search.results())
        logging.info(f"Found {len(results)} papers on arXiv.")
        # Convert arxiv.Result objects to ArxivPaperSchema objects
        search_results = [ArxivPaperSchema.from_arxiv_result(result) for result in results]

    except Exception as e:
        logging.error(f"Error fetching papers from arXiv: {e}")
        search_results = []

    return search_results

def print_papers(papers: List[ArxivPaperSchema]):
    """Prints the details of fetched arXiv papers."""
    if papers:
        print(f"\nFetched {len(papers)} papers:")
        for i, paper in enumerate(papers):
            print(f"  {i+1}. ID: {paper.entry_id}")
            print(f"     Title: {paper.title}")
            print(f"     Published: {paper.published}")
            print(f"     PDF URL: {paper.pdf_url}")
            print(f"     Abstract: {paper.summary[:100]}...")
    else:
        print("No papers found or an error occurred.")

if __name__ == "__main__":
    test_query = "Large Language Model Agent"
    print(f"Testing fetch_arxiv_papers with query: '{test_query}'")
    papers = fetch_arxiv_papers(test_query, max_results=5)
    print_papers(papers)
