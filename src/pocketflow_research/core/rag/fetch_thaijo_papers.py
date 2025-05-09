import re
from bs4 import BeautifulSoup
import requests
import json
import logging
from typing import List, Optional
from pocketflow_research.models import ThaiJoPaperSchema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

THAIJO_API_URL = "https://www.tci-thaijo.org/api/articles/search/"

# ThaiJO API URL for fetching paper link from href of PDF Button
def GetActualPDFUrl(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    pattern = re.compile(fr'{url}.*')
    pdf_link = soup.find_all('a', href=pattern)[0]

    if pdf_link is not None:
        actual_pdf_url = pdf_link.get('href').replace('view', 'download'); print(actual_pdf_url)
    else:
        print("Element not found.")

    return actual_pdf_url

def fetch_thaijo_papers(term: str, page: int = 1, size: int = 10, strict: bool = True) -> List[ThaiJoPaperSchema]:
    """
    Fetches research paper details from the ThaiJO API based on a search term.

    Args:
        term (str): The search keyword.
        page (int, optional): The page number for pagination. Defaults to 1.
        size (int, optional): The number of results per page. Defaults to 10.
        strict (bool, optional): Whether to perform strict matching. Defaults to True.

    Returns:
        List[ThaiJoPaperSchema]: A list of ThaiJo paper schema objects.
              Returns an empty list if the request fails or no results are found.
    """
    payload = {
        "term": term,
        "page": page,
        "size": size,
        "strict": strict,
        "title": True,
        "author": True,
        "abstract": True
    }
    headers = {'Content-Type': 'application/json'}

    try:
        logging.info(f"Fetching ThaiJO papers for term: '{term}', page: {page}, size: {size}")
        response = requests.post(THAIJO_API_URL, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()

        if data and data.get("total", 0) > 0 and "result" in data:
            articles: List[ThaiJoPaperSchema] = []
            for item in data["result"]:
                title = item.get("title", {}).get("th_TH") or item.get("title", {}).get("en_US", "N/A")
                abstract = item.get("abstract_clean", {}).get("th_TH") or item.get("abstract_clean", {}).get("en_US", "N/A")
                if isinstance(abstract, str):
                    abstract = abstract.replace('\n', ' ').strip()

                authors_list = [
                    author.get("full_name", {}).get("th_TH") or author.get("full_name", {}).get("en_US", "N/A")
                    for author in item.get("authors", [])
                ]

                article_schema = ThaiJoPaperSchema(
                    id=item.get("id"),
                    title=title,
                    abstract=abstract,
                    authors=authors_list,
                    url=GetActualPDFUrl(item.get("articleUrl")),
                    published_date=item.get("datePublished"),
                    source="thaijo"
                )
                articles.append(article_schema)
            logging.info(f"Successfully fetched {len(articles)} articles from ThaiJO.")
            return articles
        else:
            logging.info(f"No results found on ThaiJO for term: '{term}'")
            return []

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from ThaiJO API: {e}")
        return []
    except json.JSONDecodeError:
        logging.error("Error decoding JSON response from ThaiJO API.")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred during ThaiJO fetch: {e}")
        return []

def print_papers(papers_list: List[ThaiJoPaperSchema], term: str):
    """Prints the details of fetched papers."""
    if papers_list:
        print(f"\nFound {len(papers_list)} papers on ThaiJO for '{term}':")
        for i, paper in enumerate(papers_list):
            print(f"\n--- Paper {i+1} ---")
            print(f"  ID: {paper.id}")
            print(f"  Title: {paper.title}")
            abstract_preview = paper.abstract[:150] + "..." if paper.abstract else 'N/A'
            print(f"  Abstract: {abstract_preview}")
            print(f"  Authors: {', '.join(paper.authors) if paper.authors else 'N/A'}")
            print(f"  URL: {paper.url if paper.url else 'N/A'}")
            print(f"  Published: {paper.published_date if paper.published_date else 'N/A'}")
            print(f"  Source: {paper.source}")
    else:
        print(f"Could not fetch papers or no papers found for '{term}' on ThaiJO.")

if __name__ == "__main__":
    search_term = str(input())
    papers = fetch_thaijo_papers(search_term, size=5)
    print_papers(papers, search_term)
