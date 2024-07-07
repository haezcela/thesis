import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm

def generate_month_urls(base_url, year):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct' 'Nov', 'Dec']
    month_urls = []
    for month in months:
        month_url = f"{base_url}/thebookshelf/docmonth/{month}/{year}/1"
        month_urls.append((month, month_url))  # Store both month name and URL
    return month_urls

def extract_and_modify_links(url):
    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()

    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all links starting with the specified URL pattern
    links = soup.find_all("a", href=lambda href: href.startswith("https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/1/"))

    modified_links = add_friendly_to_showdocs([link.get("href") for link in links])

    return modified_links

def add_friendly_to_showdocs(links):
    modified_links = []
    for link in links:
        # Split the URL into parts
        parts = link.split('/')
        # Get the characters before the last '/' and use it as the filename
        filename = parts[-1]
        # Modify the URL to include 'showdocsfriendly'
        parts[parts.index('showdocs')] = 'showdocsfriendly'
        modified_link = '/'.join(parts)
        modified_links.append((modified_link, filename))
    return modified_links

def scrape_content_from_link(link):
    # Send a GET request to the link
    response = requests.get(link)
    response.raise_for_status()

    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Get all text content from the page (excluding HTML tags)
    text_content = soup.get_text(separator='\n')

    return text_content.strip()  # Remove leading and trailing whitespace

def save_data_to_txt(data, folder, filename):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Write data to a TXT file
    with open(os.path.join(folder, filename), mode='w', encoding='utf-8') as file:
        file.write(data)

# Example usage:
base_url = "https://elibrary.judiciary.gov.ph"
year = "2023"

month_urls = generate_month_urls(base_url, year)

for month_name, month_url in tqdm(month_urls):
    print(f"Extracting links for {month_name} {year}...")
    modified_links = extract_and_modify_links(month_url)
    for link, filename in modified_links:
        data = scrape_content_from_link(link)
        folder = f'cases/{year}/{month_name}'  # Folder path for the year and month
        save_data_to_txt(data, folder, f'{filename}.txt')
        print(f"Content from link {link} saved to {folder}/{filename}.txt")
