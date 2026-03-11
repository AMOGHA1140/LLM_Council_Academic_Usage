import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote

def sanitize_filename(name):
    # Remove characters that are invalid in Windows/Mac/Linux filenames
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def download_cs230_projects(stop_term="Spring 2019"):
    base_url = "https://cs230.stanford.edu/past-projects/"
    print(f"Fetching project list from {base_url}...")
    
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    outstanding_projects = {}
    normal_projects = {}

    current_term = None
    current_category = None
    encountered_terms = []

    # Iterate through all elements that might contain relevant data
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'li', 'p']):
        text = element.get_text(strip=True)
        
        # Check if the text matches a term format (e.g., "Fall 2021", "Winter 2018")
        if re.match(r'^(Fall|Winter|Spring|Summer)\s+\d{4}$', text, re.IGNORECASE):
            current_term = text
            if current_term not in encountered_terms:
                encountered_terms.append(current_term)
            continue

        if not current_term:
            continue

        # Determine if we are in the Outstanding or Submissions (Normal) section
        if "Outstanding Projects" in text:
            current_category = "outstanding"
            continue
        elif "Submissions" in text:
            current_category = "normal"
            continue

        # Find project links
        link = element.find('a')
        if link and link.get('href') and link.get('href').endswith('.pdf'):
            pdf_url = urljoin(base_url, link.get('href'))
            
            # Extract the raw title (usually appears before " by " in the text)
            raw_title = text.split(' by ')[0]
            if raw_title == text:
                # Fallback if " by " isn't present
                raw_title = text.replace(link.get_text(), '').strip()
            
            title = sanitize_filename(raw_title)
            actual_filename = unquote(pdf_url.split('/')[-1])
            
            project_data = {
                "term": current_term,
                "title": title,
                "pdf_url": pdf_url,
                "filename": actual_filename
            }
            
            if current_category == "outstanding":
                outstanding_projects[pdf_url] = project_data
            elif current_category == "normal":
                normal_projects[pdf_url] = project_data

    # Determine which terms to download based on the stop_term
    valid_terms = set()
    for term in encountered_terms:
        valid_terms.add(term)
        if term.lower() == stop_term.lower():
            break

    # Setup directories
    outstanding_dir = "Outstanding Projects"
    normal_dir = "Normal Projects"
    os.makedirs(outstanding_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    def download_file(project, folder):
        term = project['term']
        if term not in valid_terms:
            return
            
        file_name = f"{term} - {project['title']} - {project['filename']}"
        file_path = os.path.join(folder, file_name)
        
        if not os.path.exists(file_path):
            print(f"Downloading: {file_name}")
            try:
                r = requests.get(project['pdf_url'])
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(r.content)
            except Exception as e:
                print(f"Failed to download {project['pdf_url']}: {e}")
        else:
            print(f"Already exists: {file_name}")

    print(f"\nDownloading Outstanding Projects (up to {stop_term})...")
    for url, proj in outstanding_projects.items():
        download_file(proj, outstanding_dir)
        
    print(f"\nDownloading Normal Projects (up to {stop_term})...")
    for url, proj in normal_projects.items():
        # Enforce exact PDF URL match to prevent normal overwrite of outstanding
        if url not in outstanding_projects:
            download_file(proj, normal_dir)

    print("\nDownload process completed.")

if __name__ == "__main__":
    # Change this variable to control the stop term
    target_stop_term = "Spring 2019"
    download_cs230_projects(stop_term=target_stop_term)