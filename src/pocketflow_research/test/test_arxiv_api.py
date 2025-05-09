# test/test_arxiv_api.py
import unittest
import arxiv
import sys
import os

# Add the parent directory (project root) to the Python path
# to allow importing from the 'utils' directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from utils.fetch_arxiv_papers import fetch_arxiv_papers

# Optional: Remove the path modification after import if needed,
# though generally safe to leave for script execution duration.
# sys.path.pop(0)

class TestArxivFetch(unittest.TestCase):

    def test_fetch_papers_valid_query(self):
        """Test fetching papers with a query known to return results."""
        query = "quantum computing"
        max_results = 5
        papers = fetch_arxiv_papers(query, max_results=max_results)

        # Check if the result is a list
        self.assertIsInstance(papers, list, "Should return a list.")

        # Check if the list is not empty (assuming the query is broad enough)
        # Note: This might occasionally fail if arXiv API has issues or query yields no recent results
        self.assertTrue(len(papers) > 0, f"Expected > 0 papers for query '{query}', got {len(papers)}.")

        # Check if the number of results is at most max_results
        self.assertTrue(len(papers) <= max_results, f"Expected <= {max_results} papers, got {len(papers)}.")

        # Check if all items in the list are arxiv.Result objects
        if papers: # Only check if the list is not empty
            self.assertTrue(
                all(isinstance(p, arxiv.Result) for p in papers),
                "All items in the list should be arxiv.Result objects."
            )
            # Optionally check some attributes
            first_paper = papers[0]
            self.assertTrue(hasattr(first_paper, 'entry_id'))
            self.assertTrue(hasattr(first_paper, 'title'))
            self.assertTrue(hasattr(first_paper, 'pdf_url'))
            self.assertTrue(hasattr(first_paper, 'published'))

    def test_fetch_papers_unlikely_query(self):
        """Test fetching papers with a query unlikely to return results."""
        query = "nonexistent_topic_xyz123_for_testing"
        papers = fetch_arxiv_papers(query, max_results=5)

        # Check if the result is a list
        self.assertIsInstance(papers, list, "Should return a list even for no results.")

        # Check if the list is empty
        self.assertEqual(len(papers), 0, f"Expected 0 papers for unlikely query '{query}', got {len(papers)}.")

    # You could add more tests, e.g., mocking the arxiv library for offline testing
    # or testing different sort criteria if needed.

if __name__ == '__main__':
    unittest.main()
