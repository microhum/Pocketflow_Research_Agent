import requests
import json

url = "https://www.tci-thaijo.org/api/articles/search/"
payload = {
    "term": "ปลากัด",
    "page": 1,
    "size": 20,
    "strict": True,
    "title": True,
    "author": True,
    "abstract": True
}
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    # Save the response content to a JSON file
    with open("thaijo_response.json", "w", encoding="utf-8") as f:
        json.dump(response.json(), f, ensure_ascii=False, indent=4)

    print("Successfully fetched data and saved to thaijo_response.json")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
except json.JSONDecodeError:
    print("Failed to decode JSON response.")
    # Optionally save the raw response text for debugging
    with open("thaijo_response_raw.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Raw response saved to thaijo_response_raw.txt")
