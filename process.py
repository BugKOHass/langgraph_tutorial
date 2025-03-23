# import nltk
import aiohttp
import sys
from bs4 import BeautifulSoup
from groq import Groq

from fetch import fetch_html

# Initialize NLTK resources (download if not already done)
# nltk.download('punkt')

# Placeholder function to simulate analysis using deepseek-r1-distill-llama-70b from Groq
def analyze_article(subject, article_text):
    # This is where you'd integrate with the actual LLM model from Groq.
    # For demonstration purposes, we'll return dummy data.
    # return {
    #     "beliefs": "Secularism, Democracy",
    #     "political_parties": ["Pakistan Peoples Party"],
    #     "political_systems": ["Parliamentary democracy"],
    #     "relevant_words": {"democracy": 5, "secularism": 3}
    # }
    client = Groq(api_key="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")

    prompt = "Give a summary of the text in one sentence."
    chat_completion = client.chat.completions.create(
    messages=[
    {"role": "system", "content": prompt},
    {"role": "user", "content": article_text}
    ],
    model="llama-3.3-70b-versatile",
    temperature=0.3
    )
    analysis = chat_completion.choices[0].message.content
    result = f"\n{subject}\n{analysis}\n"
    return result


def process_article(article_url):

    html_content = fetch_html(article_url)
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the subject/title from meta tags or JSON-LD script
    title_tag = soup.find('meta', {'name': 'twitter:title'}) or \
                soup.find('meta', {'property': 'og:title'}) or \
                soup.find('meta', {'itemprop': 'title'})
    
    if title_tag:
        subject = title_tag['content']
    else:
        # Fallback to JSON-LD parsing for headline
        json_ld_script = soup.find('script', type='application/ld+json')
        if json_ld_script:
            import json
            data = json.loads(json_ld_script.string)
            subject = data.get('headline', 'No Title Found')
        else:
            subject = 'No Title Found'

    # Extract the article text from <p> tags
    paragraphs = soup.find_all('p')

    article_text_parts = []

    for num, p in enumerate(paragraphs):


        if p.get("class") is not None: continue
        
        if p.find('em'):
            continue

        article_text_parts.append(p.get_text(strip=True))

    article_text = ' '.join(article_text_parts)

    result = analyze_article(subject, article_text)
    return result