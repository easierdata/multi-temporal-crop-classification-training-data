import yaml
from pathlib import Path

with open(
    Path(__file__).parent.parent.parent / "config" / "default_config.yaml", "r"
) as file:
    config = yaml.safe_load(file)

DATA_DIR = config["data_dir"]
TRAINING_DATASET_DIR_NAME = config["train_dataset_name"]

# Construct the path to the training dataset directory by joining the data_dir and train_dataset_name
TRAINING_DATASET_PATH = Path(DATA_DIR) / TRAINING_DATASET_DIR_NAME
MISC_DIR = Path(DATA_DIR) / "misc"
REQUIRED_SOURCES = Path(DATA_DIR) / "required_sources"

# Define the paths to the training dataset directories
TILE_DIR = Path(TRAINING_DATASET_PATH) / "tiles"
TILE_REPROJECTED_DIR = Path(TRAINING_DATASET_PATH) / "tiles_reprojected"
CHIP_DIR = Path(TRAINING_DATASET_PATH) / "chips"
FMASK_DIR = Path(TRAINING_DATASET_PATH) / "chips_fmask"
FILTERED_DIR = Path(TRAINING_DATASET_PATH) / "chips_filtered"

# Define files paths to processing data
CHIPS_ID_JSON = Path(DATA_DIR) / "chips_id.json"
CHIPS_DF_PKL = Path(DATA_DIR) / "chips_df.pkl"
TILES_DF_PKL = Path(DATA_DIR) / "tiles_df.pkl"
TILES_DF_CSV = Path(DATA_DIR) / "tiles.csv"
SELECTED_TILES_PKL = Path(DATA_DIR) / "selected_tiles_df.pkl"
SELECTED_TILES_CSV = Path(DATA_DIR) / "selected_tiles.csv"


### --------------------------------------------------------------------------------------------------------------------
# Files used to create the bounding box for the chips
# These files are generated using the gen_chip_bbox.ipynb notebook
# Coordinates in ESPG:4326
BB_CHIP_PAYLOAD = Path(REQUIRED_SOURCES) / "bb_chip_payload.geojson"
# Coordinates in ESPG:5070
BB_CHIP_5070_PAYLOAD = Path(REQUIRED_SOURCES) / "bb_chip_5070_payload.geojson"

# Source files referenced to create the training dataset. These files can be acquired from the links provided if the
# files are not already present in the `required_sources` directory.
# https://sentinel.esa.int/web/sentinel/missions/sentinel-2/data-products
HLS_KML_FILE = Path(REQUIRED_SOURCES) / "sentinel_tile_grid.kml"
# https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php
CDL_SOURCE = Path(REQUIRED_SOURCES) / "2022_30m_cdls.tif"
# CDL class properties: https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php#what.7
CLD_CLASS_PROPERTIES = Path(REQUIRED_SOURCES) / "cdl_classes.csv"
# Updated CDL class properties with new defined class values See gen_chip_bbox.ipynb for more details
CLD_RECLASS_PROPERTIES = Path(REQUIRED_SOURCES) / "cdl_total_dst.csv"
### --------------------------------------------------------------------------------------------------------------------

# CSV File's used to track the progress and details on the chips and tiles
TRACK_CHIPS = Path(DATA_DIR) / "track_chips.csv"
TRACK_TILES = Path(DATA_DIR) / "track_tiles.csv"
# Resultant csv file containing the class weights for each chip. See `chip_weight` function in gen_chip_bbox.ipynb for more details
CHIPS_TO_SAMPLE = Path(MISC_DIR) / "chips_df_to_sample.pkl"
# CSV file containing the class weights for each class in the training dataset. See `Generate BBoxes` section in gen_chip_bbox.ipynb for more details
TASK_CLASS_SAMPLES = Path(MISC_DIR) / "class_weights.csv"


# Ensure that training dataset directories exist
if not CHIP_DIR.exists():
    CHIP_DIR.mkdir(parents=True)
if not TILE_DIR.exists():
    TILE_DIR.mkdir(parents=True)
if not FILTERED_DIR.exists():
    FILTERED_DIR.mkdir(parents=True)
if not FMASK_DIR.exists():
    FMASK_DIR.mkdir(parents=True)
if not TILE_REPROJECTED_DIR.exists():
    TILE_REPROJECTED_DIR.mkdir(parents=True)


# Ensure that the training dataset files exist
if not CHIPS_ID_JSON.exists():
    CHIPS_ID_JSON.touch()
if not BB_CHIP_PAYLOAD.exists():
    BB_CHIP_PAYLOAD.touch()
if not TRACK_CHIPS.exists():
    TRACK_CHIPS.touch()
if not TRACK_TILES.exists():
    TRACK_TILES.touch()
if not HLS_KML_FILE.exists():
    HLS_KML_FILE.touch()
if not CDL_SOURCE.exists():
    CDL_SOURCE.touch()
if not CLD_RECLASS_PROPERTIES.exists():
    CLD_RECLASS_PROPERTIES.touch()
