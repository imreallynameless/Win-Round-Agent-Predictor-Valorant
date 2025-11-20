import asyncio
from playwright.async_api import async_playwright
import json
import os
import argparse

async def get_slugs(url, page):
    print(f"Navigating to {url}...")
    try:
        await page.goto(url, timeout=60000)
        # Wait for either match links or 'No matches'
        try:
            await page.wait_for_selector('a[href*="match="]', timeout=10000)
        except:
            print("  (No match links found immediately, might be empty or loading slow)")
            
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return {}

    # Scroll a bit
    for _ in range(3):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(0.5)

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

async def main(mode):
    base_dir = f"{mode}_data"
    search_links_file = os.path.join(base_dir, f"{mode}_searchlinks.txt")
    mapping_file = os.path.join(base_dir, f"{mode}_id_slug_map.json")

    if not os.path.exists(search_links_file):
        print(f"No search links file found at {search_links_file}")
        return

    with open(search_links_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    full_mapping = {}
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Processing...")
            m = await get_slugs(url, page)
            full_mapping.update(m)

        await browser.close()

    print(f"Total mapped: {len(full_mapping)}")
    
    with open(mapping_file, "w") as f:
        json.dump(full_mapping, f, indent=2)
    print(f"Saved mapping to {mapping_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch slugs for training or test data.")
    parser.add_argument("mode", choices=["training", "test"], help="Mode: 'training' or 'test'")
    args = parser.parse_args()

    asyncio.run(main(args.mode))
