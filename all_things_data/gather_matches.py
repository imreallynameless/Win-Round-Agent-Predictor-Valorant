import asyncio
import time
from playwright.async_api import async_playwright
import os
import argparse

async def get_match_ids_from_url(url, page):
    match_ids = set()
    print(f"Navigating to {url}...")
    try:
        await page.goto(url, timeout=60000)
        # Wait for any match link to appear
        try:
            await page.wait_for_selector('a[href*="match="]', timeout=20000)
        except:
            print(f"No matches found or timeout for {url}")
            return []
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return []

    # Scroll a bit to load more matches (basic infinite scroll handling)
    for _ in range(5): 
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(2)

    # Extract match links
    links = await page.evaluate('''() => {
        const anchors = Array.from(document.querySelectorAll('a[href*="match="]'));
        return anchors.map(a => a.href);
    }''')

    for link in links:
        if "match=" in link:
            try:
                match_id = link.split("match=")[1].split("&")[0]
                match_ids.add(match_id)
            except:
                continue
    
    print(f"Found {len(match_ids)} matches for {url}")
    return list(match_ids)

async def main(mode):
    base_dir = f"{mode}_data"
    search_links_file = os.path.join(base_dir, f"{mode}_searchlinks.txt")
    output_file = os.path.join(base_dir, f"{mode}_matches.txt")

    if not os.path.exists(search_links_file):
        print(f"No search links file found at {search_links_file}")
        return

    all_ids = set()
    
    # Read URLs from file
    with open(search_links_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        for url in urls:
            ids = await get_match_ids_from_url(url, page)
            for m in ids:
                all_ids.add(m)
        
        await browser.close()
            
    # Save to file
    print(f"Total unique matches found: {len(all_ids)}")
    with open(output_file, "w") as f:
        for m in all_ids:
            f.write(f"{m}\n")
    print(f"Saved match IDs to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather match IDs for training or test data.")
    parser.add_argument("mode", choices=["training", "test"], help="Mode: 'training' or 'test'")
    args = parser.parse_args()
    
    asyncio.run(main(args.mode))
