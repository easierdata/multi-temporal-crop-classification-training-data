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


def main() -> None:
    """
    Main function to check the retrievability of IPFS content.
    This function reads a JSON file containing tile payloads, checks if the content
    identified by CIDs (Content Identifiers) is available in the IPFS network, and
    writes any missing CIDs to an output JSON file. It also prints the number of
    missing CIDs and their details.
    The function performs the following steps:
    1. Reads the tile payloads from "tile_payloads.json".
    2. Iterates through each scene and its assets to check if the CID is available in IPFS.
    3. If a CID is not found, writes the missing CID information to "missing_cids.json".
    4. Prints the total number of missing CIDs and their details.
    Note:
        The function assumes the presence of the IPFS command-line tool and that the
        IPFS daemon is running.
    Raises:
        FileNotFoundError: If the input file "tile_payloads.json" or the output file
                           "missing_cids.json" cannot be found or opened.
        json.JSONDecodeError: If the input JSON file cannot be parsed.
    """
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
                    ["ipfs", "dag", "stat", asset["cid"]],
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


if __name__ == "__main__":
    main()
