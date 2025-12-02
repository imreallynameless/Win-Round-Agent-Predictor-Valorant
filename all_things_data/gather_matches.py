import asyncio
from playwright.async_api import async_playwright
import os
import argparse

# Map ID to Map Name mapping (from rib.gg URL parameters)
MAP_ID_TO_NAME = {
    "1": "Ascent",
    "2": "Bind",
    "4": "Split",
    "7": "Haven",
    "8": "Icebox",
    "10": "Breeze",
    "11": "Fracture",
    "12": "Pearl",
    "13": "Abyss",
    "14": "Lotus",
    "15": "Sunset",
}

MAP_NAME_TO_ID = {v: k for k, v in MAP_ID_TO_NAME.items()}


async def get_match_ids_from_url(url: str, page, map_name: str) -> list[str]:
    """
    Navigate to URL and extract match IDs for the specific map.
    Waits for actual content to load rather than fixed timeouts.
    """
    print(f"Navigating to {url}...")
    print(f"  Looking for map: {map_name}")
    
    # Step 1: Navigate to the URL
    try:
        await page.goto(url, timeout=60000)
    except Exception as e:
        print(f"  Error loading page: {e}")
        return []
    
    # Step 2: Wait for match links to appear (with long timeout for slow React hydration)
    print("  Waiting for matches to load...")
    try:
        await page.wait_for_selector('a[href*="match="]', timeout=30000)
    except:
        print("  No match links found after 30s timeout")
        return []
    
    # Step 3: Additional wait for React to fully render all content
    await asyncio.sleep(2)
    
    # Step 4: Scroll to load more content
    print("  Scrolling to load more content...")
    for _ in range(3):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1)
    
    # Step 5: Extract match IDs - only for the specific map
    match_ids = await page.evaluate('''(mapName) => {
        const links = Array.from(document.querySelectorAll('a[href*="match="]'));
        const ids = new Set();
        
        links.forEach(link => {
            if (link.textContent.trim() === mapName) {
                const url = link.href;
                const match = url.match(/match=(\\d+)/);
                if (match) {
                    ids.add(match[1]);
                }
            }
        });
        
        return Array.from(ids);
    }''', map_name)
    
    print(f"  Found {len(match_ids)} {map_name} matches")
    return match_ids


async def main(folder: str):
    # Extract mode and map name from folder name
    parts = folder.rstrip('/').split('_')
    
    if folder.startswith("training"):
        mode = "training"
    elif folder.startswith("test"):
        mode = "test"
    else:
        mode = parts[0]
    
    map_name = parts[-1].capitalize() if len(parts) >= 3 else None
    if not map_name:
        print(f"Could not determine map name from folder: {folder}")
        return
    
    print(f"Map: {map_name}")
    if map_name in MAP_NAME_TO_ID:
        print(f"Map ID: {MAP_NAME_TO_ID[map_name]}")
    
    search_links_file = os.path.join(folder, f"{mode}_searchlinks.txt")
    output_file = os.path.join(folder, f"{mode}_matches.txt")

    if not os.path.exists(search_links_file):
        print(f"No search links file found at {search_links_file}")
        return

    with open(search_links_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs to process\n")

    all_ids = set()

    async with async_playwright() as p:
        # headless=False shows the browser window
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Processing...")
            ids = await get_match_ids_from_url(url, page, map_name)
            all_ids.update(ids)
        
        await browser.close()
    
    print(f"\n{'='*40}")
    print(f"Total unique matches found: {len(all_ids)}")
    
    with open(output_file, "w") as f:
        for match_id in sorted(all_ids):
            f.write(f"{match_id}\n")
    
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather match IDs from rib.gg")
    parser.add_argument("folder", help="Data folder (e.g., 'training_data_ascent')")
    args = parser.parse_args()
    
    asyncio.run(main(args.folder))
