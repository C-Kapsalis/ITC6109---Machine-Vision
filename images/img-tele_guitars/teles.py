import os
import csv
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

# Define folder structure
master_folder = os.getcwd()
st_guitars_folder = os.path.join(master_folder, "details")
images_folder = os.path.join(st_guitars_folder, "images")
mapping_folder = os.path.join(st_guitars_folder, "mapping")

# Create the folders if they don't exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(mapping_folder, exist_ok=True)

# CSV file path for mappings
csv_file_path = os.path.join(mapping_folder, "guitar_mapping.csv")

# Normalize file names
def normalize_filename(name):
    return name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")

# Extract guitar name and file name from href
def parse_guitar_name(href):
    base_name = href.split(".htm")[0]  # Remove ".htm"
    guitar_name = base_name.replace("_", " ")  # Replace underscores with spaces
    file_name = normalize_filename(base_name)  # Normalize for the file name
    return guitar_name, file_name

# Main scraping function
def scrape_guitar_images(page):
    base_url = "https://www.thomann.de/gr/"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Turn off headless for debugging
        page = browser.new_page()
        page.goto(base_url + f"t_models.html?ls=100&pg={page}")

        # Wait for the product list to load
        page.wait_for_selector("a.product__content.no-underline")

        # Collect product page URLs
        product_anchors = page.query_selector_all("a.product__content.no-underline")
        product_urls = [urljoin(base_url, anchor.get_attribute("href")) for anchor in product_anchors]

        # Open CSV for writing
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["model_name", "image_file_name", "image_url", "product_url"])

            # Iterate over product pages
            for product_url in product_urls:
                href = product_url.split("/")[-1]
                guitar_name, file_name = parse_guitar_name(href)

                # Navigate to the product page
                page.goto(product_url)
                page.wait_for_load_state("load")

                # Scroll iteratively to load all content
                for _ in range(5):  # Try scrolling multiple times
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    page.wait_for_timeout(1000)  # Wait for images to load

                # Check for dynamic images
                img_url = None
                try:
                    img_element = page.query_selector('div[data-zg-index="1"] img')
                    if img_element:
                        img_url = img_element.get_attribute("src")
                except Exception as e:
                    print(f"Error finding image for {guitar_name}: {e}")

                if img_url:
                    img_path = os.path.join(images_folder, f"{file_name}.jpg")

                    # Download the image
                    response = page.request.get(img_url)
                    with open(img_path, "wb") as img_file:
                        img_file.write(response.body())

                    # Write to CSV
                    writer.writerow([guitar_name, f"{file_name}.jpg", img_url, product_url])
                    print(f"Successfully downloaded image for {guitar_name}")
                else:
                    # Log missing images
                    writer.writerow([guitar_name, "MISSING_IMAGE", "N/A", product_url])
                    print(f"Image not found for {guitar_name}")

        browser.close()

# Run the script
scrape_guitar_images(page=1)

print(f"Images saved in: {images_folder}")
print(f"CSV mapping file saved at: {csv_file_path}")

