import pandas as pd
import os
import json
from pathlib import Path
import sys
from ipfs_stac import client

module_path = module_path = os.path.abspath(Path(__file__).parent.parent.resolve())
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass

with open(
    MISC_DIR / "selected_cid_mapping.json",
    "r",
) as file:
    data_payload = json.load(file)

# Initialize the IPFS client
ipfs_node = client.Web3(local_gateway="127.0.0.1")


# Extract a list of pinned items from the IPFS node
CIDs_ON_NODE = ipfs_node.pinned_list()
for item in data_payload:
    print(f"Downloading {item['filename']} with CID: {item['cid']}")
    if item["cid"] not in CIDs_ON_NODE:
        print(f"{item['cid']} not found. Unable to download {item['filename']}")
        continue
    # Create a directory to store content by scene name if it does not exist
    SCENE_NAME_DIR = Path(TILE_DIR / item["filename"].rsplit(".", 2)[0])
    SCENE_NAME_DIR.mkdir(parents=True, exist_ok=True)

    ipfs_node.writeCID(item["cid"], SCENE_NAME_DIR / item["filename"])
