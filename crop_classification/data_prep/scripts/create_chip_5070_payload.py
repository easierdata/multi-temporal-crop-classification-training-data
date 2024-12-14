import json
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
    Main function to create a filtered GeoJSON payload.
    This function performs the following steps:
    1. Loads a GeoJSON file containing bounding box chip payloads and extracts feature IDs.
    2. Loads a pre-generated GeoJSON file with ESPG:5070 coordinates.
    3. Filters the features in the pre-generated GeoJSON file based on the extracted feature IDs.
    4. Creates a new FeatureCollection with the filtered features.
    5. Saves the new FeatureCollection to a specified GeoJSON file.
    Raises:
        FileNotFoundError: If any of the input files are not found.
        json.JSONDecodeError: If there is an error decoding the JSON files.
    """
    # Load bb_chip_payload.geojson and extract feature IDs
    with open(BB_CHIP_PAYLOAD, "r") as f:
        bb_chip_payload = json.load(f)
        payload_ids = {
            feature["properties"]["id"] for feature in bb_chip_payload["features"]
        }

    # Load pregenerated chip_bbox geojson with ESPG:5070 coordinates
    with open(PREGEN_BB_CHIP_5070_PAYLOAD, "r") as f:
        chip_bbox_original = json.load(f)

    # Filter features that are in payload_ids
    filtered_features = [
        feature
        for feature in chip_bbox_original["features"]
        if feature["properties"]["id"] in payload_ids
    ]

    # Create new FeatureCollection
    filtered_feature_collection = {
        "type": "FeatureCollection",
        "features": filtered_features,
    }

    # Save to bb_chip_5070_payload.geojson
    with open(BB_CHIP_5070_PAYLOAD, "w") as f:
        json.dump(filtered_feature_collection, f, indent=4)


if __name__ == "__main__":
    main()
