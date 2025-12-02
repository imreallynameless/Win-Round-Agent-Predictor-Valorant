import os
import json
import argparse

def main(folder):
    # Extract mode (training/test) from folder name
    if folder.startswith("training"):
        mode = "training"
    elif folder.startswith("test"):
        mode = "test"
    else:
        mode = folder.split("_")[0]
    
    mapping_file = os.path.join(folder, f"{mode}_id_slug_map.json")
    data_dir = os.path.join(folder, f"{mode}_data_json")

    if not os.path.exists(mapping_file):
        print(f"No mapping file found at {mapping_file}")
        return
        
    if not os.path.exists(data_dir):
        print(f"No data directory found at {data_dir}")
        return

    with open(mapping_file, "r") as f:
        id_slug_map = json.load(f)

    files = os.listdir(data_dir)

    renamed_count = 0
    for filename in files:
        if not filename.endswith(".json"):
            continue
        
        # extract match id (assuming format "12345.json" or already renamed)
        if "-" in filename:
            continue # already renamed

        match_id = filename.replace(".json", "")
        
        if match_id in id_slug_map:
            slug = id_slug_map[match_id]
            new_filename = f"{slug}-{match_id}.json"
            
            old_path = os.path.join(data_dir, filename)
            new_path = os.path.join(data_dir, new_filename)
            
            os.rename(old_path, new_path)
            print(f"Renamed {filename} -> {new_filename}")
            renamed_count += 1
        else:
            # Optional: print skipping only if verbose
            # print(f"Skipping {filename} (no slug found)")
            pass

    print(f"Renamed {renamed_count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename JSON files in a data folder.")
    parser.add_argument("folder", help="Data folder name (e.g., 'training_data_haven', 'test_data_abyss')")
    args = parser.parse_args()

    main(args.folder)
