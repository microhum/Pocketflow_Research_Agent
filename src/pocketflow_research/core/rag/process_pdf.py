# utils/process_pdf.py
import time
import requests
import fitz  # PyMuPDF
import io
import logging
from utils.proxy_rotatation_request import get_random_proxy_request

logging.basicConfig(level=logging.INFO)

def download_and_extract_text(pdf_url: str) -> str | None:
    """
    Downloads a PDF from a URL and extracts its text content using PyMuPDF.

    Args:
        pdf_url (str): The URL of the PDF file.

    Returns:
        str | None: The extracted text content, or None if an error occurs.
    """
    extracted_text = None
    try:
        logging.info(f"Downloading PDF from: {pdf_url}")

        # Use get_random_proxy_request to prevent 429 errors
        # response = get_random_proxy_request(pdf_url, timeout=10) # Added timeout and random user-agent
        # ThaiJo 429 error handling which occur every 5 seconds
        
        response = requests.get(pdf_url, timeout=20)
        if response.status_code == 429:
            time.sleep(5)
            response = requests.get(pdf_url, timeout=20)

        response.raise_for_status()  # Raise an exception for bad status codes

        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' not in content_type:
            logging.warning(f"URL {pdf_url} did not return PDF content (Content-Type: {content_type}). Skipping extraction.")
            return None

        logging.info("Download complete. Extracting text using PyMuPDF...")
        pdf_stream = io.BytesIO(response.content)

        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        num_pages = doc.page_count
        logging.info(f"PDF has {num_pages} pages.")

        text_parts = []
        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            text_parts.append(page.get_text("text") or "")

        extracted_text = "\n".join(text_parts)
        doc.close()
        logging.info(f"Successfully extracted text (length: {len(extracted_text)}).")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading PDF from {pdf_url}: {e}")
        return None
    except fitz.FileDataError as e: # Catch specific PyMuPDF errors
        logging.error(f"Error reading PDF content from {pdf_url} with PyMuPDF: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {pdf_url}: {e}", exc_info=True) # Log traceback for unexpected errors
        return None

    return extracted_text

if __name__ == "__main__":
    test_url = "https://arxiv.org/pdf/1706.03762.pdf" # Attention is All You Need paper
    print(f"Testing download_and_extract_text with URL: {test_url}")

    text = download_and_extract_text(test_url)

    if text:
        print("\nSuccessfully extracted text (first 500 chars):")
        print(text[:500] + "...")
        print(f"\nTotal characters extracted: {len(text)}")
    else:
        print("\nFailed to download or extract text from the PDF.")
