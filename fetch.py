import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
import time
import requests

def fetch_html(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch {url}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None
    finally:
        time.sleep(5.0)  # Ensure this runs after every request


def fetch_articles():
    base_url = 'https://www.dawn.com/authors/774/nadeem-f-paracha'
    article_data = []

    # Fetch the first page to determine the total number of pages
    html_content = fetch_html(f"{base_url}/1")
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the total number of pages from pagination links
    pagination_links = soup.select('ul.pagination li a')
    if pagination_links:
        last_page_link = pagination_links[-2]  # Second last link contains the last page number
        total_pages = int(last_page_link.text)
    else:
        total_pages = 1  # Default to 1 if no pagination is found
    
    total_pages = 1

    for page_number in range(1, total_pages + 1):

        url = f"{base_url}/{page_number}"
        html_content = fetch_html(url)
        soup = BeautifulSoup(html_content, 'html.parser')

        articles_on_page = soup.find_all('article', class_='story')

        for i, article in enumerate(articles_on_page[:2]):
            # print(f"article no: {i}")
            link_tag = article.find('a', class_='story__link')
            date_tag = article.find('span', class_='timestamp__calendar')

            if link_tag and date_tag:
                href = link_tag['href']
                date_published = date_tag.text.strip()
                article_data.append({'url': href, 'date': date_published})
                # print(f"{href} {date_published}")

    return article_data