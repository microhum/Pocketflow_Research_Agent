import requests
from bs4 import BeautifulSoup
import random

# https://github.com/robert4digital/get-free-proxies/blob/master/getProxies_BeautifulSoup.py + adjustments
def get_proxies():
    proxies = []
    res = requests.get('https://free-proxy-list.net/', headers={'User-Agent': 'EHEHEH'})
    soup = BeautifulSoup(res.text, "lxml")

    for items in soup.select("tbody tr"):
        cols = items.select("td")
        if len(cols) < 7:
            continue

        anonymity = cols[4].text.strip().lower()
        https = cols[6].text.strip().lower()

        if anonymity == 'elite proxy' and https == 'yes':
            proxy = f"{cols[0].text.strip()}:{cols[1].text.strip()}"
            proxies.append(proxy)

    return proxies

def get_random_proxy(proxy_list):
    return random.choice(proxy_list)

def get_random_proxy_request(url, timeout=30):
    proxy_list = get_proxies()
    if not proxy_list:
        raise Exception("No proxies fetched. Cannot proceed.")
    
    proxy = {
        "http": get_random_proxy(proxy_list),
        "https": get_random_proxy(proxy_list)
    }
    print(f"Using proxy: {proxy}")
    
    response = requests.get(url, proxies=proxy, timeout=timeout)
    return response



if __name__ == "__main__":
    print('Number of proxies:',len(get_proxies()))
    print(get_proxies())

    try:
        res = get_random_proxy_request("https://so08.tci-thaijo.org/index.php/romyoongthong/article/download/4963/3544")
        print(res.json())
    except Exception as e:
        print(f"Request failed: {e}")
