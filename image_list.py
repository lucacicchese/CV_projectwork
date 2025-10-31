import re
import bs4
import pandas as pd

# Read the HTML file
with open('alinari1.txt', 'r', encoding='utf-8') as f:
    html = f.read()

# Parse the HTML
soup = bs4.BeautifulSoup(html, 'html.parser')

# Extract data
items = []
for item_box in soup.find_all('div', class_='itemBox'):
    # Full ID
    full_id_elem = item_box.find('p', class_='inventary_code')
    full_id = full_id_elem.find('span').text if full_id_elem else 'N/A'
    
    # Short ID (6-digit code from data-src URL)
    img_elem = item_box.find('img')
    data_src = img_elem.get('data-src', '') if img_elem else ''
    match = re.search(r'Image(\d{6})\.jpg', data_src)
    short_id = match.group(1) if match else 'N/A'
    
    # Caption
    caption_elem = item_box.find('h2', class_='caption')
    caption = caption_elem.text.strip() if caption_elem else 'N/A'
    
    # Photographer
    photographer_elem = item_box.find('p', class_='autorefotografiadc')
    if photographer_elem:
        photofield = photographer_elem.find('em', class_='field')
        field_text = photofield.text.strip() if photofield else ''
        value_span = photographer_elem.find('span', class_='value')
        photographer = value_span.text.strip() if value_span else 'N/A'
        photographer_full = f"{field_text} {photographer}".strip()
    else:
        photographer_full = 'N/A'
    
    # Date
    date_elem = item_box.find('p', class_='datascatto_cr')
    date = date_elem.find('span').text.strip() if date_elem and date_elem.find('span') else ''
    
    # Additional metadata
    xsize = item_box.get('xsize', 'N/A')
    ysize = item_box.get('ysize', 'N/A')
    position = item_box.get('position', 'N/A')
    archive = img_elem.get('data-archive', 'N/A') if img_elem else 'N/A'
    
    # Detail URL (using Short ID)
    detail_url = f"https://www.alinari.it/item/en/1/{short_id}" if short_id != 'N/A' else 'N/A'
    
    items.append({
        'Full ID': full_id,
        'LNR ID': short_id,
        'Caption': caption,
        'Photographer': photographer_full,
        'Date': date,
        'X Size': xsize,
        'Y Size': ysize,
        'Position': position,
        'Archive': archive,
        'URL': detail_url
    })

# Create DataFrame
df = pd.DataFrame(items)

# Print first 10 rows
print(df.head(10))

# Save to CSV
df.to_csv('alinari_images_detailed.csv', index=False)
