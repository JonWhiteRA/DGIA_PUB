import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_chapter(num):
    if 1 <= num <= 4:
        return 1
    elif 5 <= num <= 11:
        return 2
    elif 12 <= num <= 23:
        return 3
    elif 24 <= num <= 43:
        return 4
    elif 44 <= num <= 50:
        return 5
    elif 51 <= num <= 59:
        return 6
    elif 60 <= num <= 76:
        return 7
    elif 77 <= num <= 84:
        return 8
    elif 85 <= num <= 91:
        return 9
    elif 92 <= num <= 93:
        return 10
    elif 94 <= num <= 99:
        return 11
    else:
        return None

def extract_contents(li, article_number, title, parent_index, url):
    contents = []
    content_text = li.get_text(strip=True)

    # Extract links from the current <li> item
    links = [a['href'] for a in li.find_all('a', href=True)]
    links_string = ', '.join(links) if links else ''

    # Append the current <li> content with chapter
    chapter = get_chapter(article_number)
    contents.append((chapter, article_number, f"{parent_index}", title, content_text, url, links_string))

    # Check for nested <ol> elements
    nested_ol = li.find('ol')
    if nested_ol:
        nested_lis = nested_ol.find_all('li')
        for index, nested_li in enumerate(nested_lis, start=1):
            # Recursively extract contents from nested <li>
            contents += extract_contents(nested_li, article_number, title, f"{parent_index}.{index}", url)

    return contents

def extract_paragraphs(entry_content_div, article_number, title, url):
    contents = []
    footnote_div = entry_content_div.find(id='wpcf-field-fussnote')  # Locate the footnote div

    paragraphs = entry_content_div.find_all('p')
    for paragraph in paragraphs:
        if footnote_div and paragraph in footnote_div.find_all('p'):
            continue  # Skip paragraphs that are in the footnotes

        content_text = paragraph.get_text(strip=True)
        
        # Extract links from the current <p> item
        links = [a['href'] for a in paragraph.find_all('a', href=True)]
        links_string = ', '.join(links) if links else ''
        
        # Append the <p> content with chapter and index 0
        chapter = get_chapter(article_number)
        contents.append((chapter, article_number, '0', title, content_text, url, links_string))
    
    return contents

def scrape_gdpr_article(num):
    url = f"https://gdpr-info.eu/art-{num}-gdpr/"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to retrieve article {num}. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the article number
    article_number = num

    # Extract the title
    title = soup.find('span', class_='dsgvo-title').get_text(strip=True)

    # Extract the contents and links
    contents = []
    entry_content_div = soup.find('div', class_='entry-content')
    
    if entry_content_div:
        # Extract paragraphs
        contents += extract_paragraphs(entry_content_div, article_number, title, url)

        # Process <ol> elements
        ol = entry_content_div.find('ol')
        if ol:
            lis = ol.find_all('li')
            for index, li in enumerate(lis, start=1):
                # Use the recursive function to extract contents
                contents += extract_contents(li, article_number, title, str(index), url)

    return contents

def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['Chapter', 'Article Number', 'Paragraph Num', 'Title', 'Content', 'Links', 'Source'])
    df.to_csv(filename, index=False, mode='a', header=not pd.io.common.file_exists(filename))

def main():
    # Change this range according to how many articles you want to scrape
    for num in range(1, 100):  # Example range from 60 to 65
        print(f"Processing article {num}...")  # Output the current article number
        contents = scrape_gdpr_article(num)
        if contents:  # Ensure there's content to save
            save_to_csv(contents, '/output/gdpr_articles.csv')
        else:
            print(f"No content found for article {num}.")

if __name__ == '__main__':
    main()
