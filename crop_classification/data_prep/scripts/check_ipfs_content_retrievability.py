import json
import subprocess
import pandas as pd
import tqdm
from pathlib import Path

import sys

module_path = Path(__file__).parent.parent.parent.resolve().as_posix()
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass


with open(Path(MISC_DIR, "tile_payloads.json"), "r") as file:
    data = json.load(file)

# Open the output file
with open(Path(MISC_DIR, "missing_cids.json"), "w") as output_file:

    for scene in tqdm.tqdm(data.keys()):
        # get the scene data
        scene_data = data[scene]

        # get the CID for each band
        for asset in scene_data:
            # check if the CID is in IPFS
            result = subprocess.run(
                ["ipfs", "cat", asset["cid"]],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Check if the error message is in the output
            if (
                "Error: block was not found locally (offline): ipld: could not find"
                in result.stderr.decode("utf-8")
            ):
                # if the CID is not in IPFS, write it to the output file
                missing_cid_info = {
                    "scene": scene,
                    "band": asset["band_name"],
                    "filename": asset["filename"],
                    "cid": asset["cid"],
                }
                output_file.write(json.dumps(missing_cid_info) + "\n")

# Return a count of the number of missing CIDs
missing_cids = pd.read_json(Path(MISC_DIR, "missing_cids.json"), lines=True)
print("Finished scanning...")
if len(missing_cids) > 0:
    print(
        f"""
{len(missing_cids)} CID(s) could not be retrieved. IPFS nodes that contain content may be offline.


The missing CIDs have been saved to {Path(MISC_DIR, "missing_cids.json")}.
"""
    )
else:
    print("Content that was requested is retrievable.")
