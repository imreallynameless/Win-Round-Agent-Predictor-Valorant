import asyncio
from playwright.async_api import async_playwright
import json
import os
import argparse

async def get_slugs(url, page):
    print(f"Navigating to {url}...")
    try:
        await page.goto(url, timeout=60000)
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return {}

    # Wait for match links to appear
    print("  Waiting for matches to load...")
    try:
        await page.wait_for_selector('a[href*="match="]', timeout=30000)
    except:
        print("  No match links found after 30s timeout")
        return {}
    
    # Additional wait for React to fully render
    await asyncio.sleep(2)

    # Scroll to load more content
    for _ in range(3):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1)

    links = await page.evaluate('''() => {
        const anchors = Array.from(document.querySelectorAll('a[href*="match="]'));
        return anchors.map(a => a.href);
    }''')

    mapping = {}
    for link in links:
        if "match=" in link and "/series/" in link:
            try:
                # Link format: .../series/<slug>/<series_id>?match=<match_id>
                part = link.split("/series/")[1]
                slug = part.split("/")[0]
                
                match_id = link.split("match=")[1].split("&")[0]
                
                mapping[match_id] = slug
            except:
                continue
    
    print(f"  Found {len(mapping)} matches.")
    return mapping

async def main(folder):
    # Extract mode (training/test) from folder name
    if folder.startswith("training"):
        mode = "training"
    elif folder.startswith("test"):
        mode = "test"
    else:
        mode = folder.split("_")[0]
    
    search_links_file = os.path.join(folder, f"{mode}_searchlinks.txt")
    mapping_file = os.path.join(folder, f"{mode}_id_slug_map.json")

    if not os.path.exists(search_links_file):
        print(f"No search links file found at {search_links_file}")
        return

    with open(search_links_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs to process\n")

    full_mapping = {}
    
    async with async_playwright() as p:
        # headless=False shows the browser window
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Processing...")
            m = await get_slugs(url, page)
            full_mapping.update(m)

        await browser.close()

    print(f"\nTotal mapped: {len(full_mapping)}")
    
    with open(mapping_file, "w") as f:
        json.dump(full_mapping, f, indent=2)
    print(f"Saved mapping to {mapping_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch slugs from a data folder.")
    parser.add_argument("folder", help="Data folder name (e.g., 'training_data_haven', 'test_data_abyss')")
    args = parser.parse_args()

    asyncio.run(main(args.folder))
