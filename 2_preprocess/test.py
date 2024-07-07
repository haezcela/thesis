import requests
from bs4 import BeautifulSoup
import os

# URL of the library website
url = 'https://elibrary.judiciary.gov.ph/thebookshelf/docmonth/Jan/1998/1'

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links that point to PDF files
pdf_links = soup.find_all('a', href=lambda href: (href and href.endswith('.pdf')))

# Create a directory to save the PDF files
os.makedirs('pdf_files', exist_ok=True)

# Check if any PDF links are found
if not pdf_links:
    print("No PDF files found on the webpage.")
else:
    # Download up to 10 PDF files
    counter = 0
    for link in pdf_links:
        if counter >= 10:
            break
        pdf_url = link.get('href')
        pdf_filename = os.path.join('pdf_files', pdf_url.split('/')[-1])
        with open(pdf_filename, 'wb') as f:
            pdf_response = requests.get(pdf_url)
            f.write(pdf_response.content)
            print(f"Downloaded {pdf_filename}")
        counter += 1
