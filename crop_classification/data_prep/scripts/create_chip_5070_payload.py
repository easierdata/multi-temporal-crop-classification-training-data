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
