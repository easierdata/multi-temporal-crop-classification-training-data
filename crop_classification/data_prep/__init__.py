import yaml
from pathlib import Path

with open(
    Path(__file__).parent.parent.parent / "config" / "default_config.yaml", "r"
) as file:
    config = yaml.safe_load(file)

# ipfs-stac properties
IPFS_STAC = config["ipfs_stac_params"]

# Construct the paths to the core directories in the `DATA_DIR`
DATA_DIR = config["data_dir"]
TRAINING_DATA_ROOT_PATH = Path(DATA_DIR, "training_datasets")
MISC_DIR = Path(DATA_DIR, "misc")
REQUIRED_SOURCES = Path(DATA_DIR, "required")

### --------------------------------------------------------------------------------------------------------------------
# Define the paths to the directories used to store the training dataset files
TRAINING_DATASET_DIR_NAME = config["train_dataset_name"]
TRAINING_DATASET_PATH = Path(TRAINING_DATA_ROOT_PATH, TRAINING_DATASET_DIR_NAME)
TILE_DIR = Path(TRAINING_DATASET_PATH) / "tiles"
TILE_REPROJECTED_DIR = Path(TRAINING_DATASET_PATH) / "tiles_reprojected"
CHIP_DIR = Path(TRAINING_DATASET_PATH) / "chips"
FMASK_DIR = Path(TRAINING_DATASET_PATH) / "chips_fmask"
FILTERED_DIR = Path(TRAINING_DATASET_PATH) / "chips_filtered"

# Ensure that training dataset directories exist
CHIP_DIR.mkdir(parents=True, exist_ok=True)
TILE_DIR.mkdir(parents=True, exist_ok=True)
FILTERED_DIR.mkdir(parents=True, exist_ok=True)
FMASK_DIR.mkdir(parents=True, exist_ok=True)
TILE_REPROJECTED_DIR.mkdir(parents=True, exist_ok=True)
# MISC_DIR.mkdir(parents=True, exist_ok=True)
### --------------------------------------------------------------------------------------------------------------------


### --------------------------------------------------------------------------------------------------------------------
# Pregenerated files used for the data preparation process
PREPROCESSED_SOURCES = Path(REQUIRED_SOURCES, "preprocessed_data")
CLD_CLASS_PROPERTIES = Path(PREPROCESSED_SOURCES, "cdl_classes.csv")
# These files are generated using the gen_chip_bbox.ipynb notebook
CLD_RECLASS_PROPERTIES = Path(PREPROCESSED_SOURCES, "cdl_total_dst.csv")
PREGEN_BB_CHIP_PAYLOAD = Path(
    PREPROCESSED_SOURCES, "pregen_bb_chip_payload.geojson"
)  # Coordinates in ESPG:4326
PREGEN_BB_CHIP_5070_PAYLOAD = Path(
    PREPROCESSED_SOURCES, "pregen_bb_chip_5070_payload.geojson"
)  # Coordinates in ESPG:5070
### --------------------------------------------------------------------------------------------------------------------


### --------------------------------------------------------------------------------------------------------------------
# User-defined AOI files
# Note: chip_payload is in ESPG:4326 and chip_5070_payload is in ESPG:5070
USER_AOI_DIR = Path(REQUIRED_SOURCES, "aoi_selections")
USER_DEFINED_AOI_FILE = config.get("chip_payload_filename", "")

# Use user-defined AOI file if it exists, otherwise use pre-generated AOI files
if USER_DEFINED_AOI_FILE:
    # Remove the .geojson extension to get the file name
    AOI_NAME_FILENAME = USER_DEFINED_AOI_FILE.split(".geojson")[0]
    BB_CHIP_5070_PAYLOAD = Path(USER_AOI_DIR, f"{AOI_NAME_FILENAME}_5070.geojson")
    BB_CHIP_PAYLOAD = Path(USER_AOI_DIR, f"{AOI_NAME_FILENAME}.geojson")
else:
    BB_CHIP_PAYLOAD = PREGEN_BB_CHIP_PAYLOAD
    BB_CHIP_5070_PAYLOAD = PREGEN_BB_CHIP_5070_PAYLOAD
### --------------------------------------------------------------------------------------------------------------------


### --------------------------------------------------------------------------------------------------------------------
# Downstream files that are generated during the data preparation process
CHIPS_ID_JSON = Path(DATA_DIR) / "chips_id.json"
CHIPS_DF_PKL = Path(DATA_DIR) / "chips_df.pkl"
SELECTED_TILES_PKL = Path(DATA_DIR) / "selected_tiles_df.pkl"
SELECTED_TILES_CSV = Path(DATA_DIR) / "selected_tiles.csv"
# CSV File's used to track the progress and details on the chips and tiles
TRACK_CHIPS = Path(DATA_DIR) / "track_chips.csv"
TRACK_TILES = Path(DATA_DIR) / "track_tiles.csv"
### --------------------------------------------------------------------------------------------------------------------


### --------------------------------------------------------------------------------------------------------------------
# Properties specific to the bounding box generation process
N_CHIPS = 500
# Resultant csv file containing the class weights for each chip.
# See `chip_weight` function in gen_chip_bbox.ipynb for more details
CHIPS_TO_SAMPLE = Path(MISC_DIR) / "chips_df_to_sample.pkl"
# CSV file containing the class weights for each class in the training dataset.
# See `Generate BBoxes` section in gen_chip_bbox.ipynb for more details
TASK_CLASS_SAMPLES = Path(MISC_DIR) / "class_weights.csv"
### --------------------------------------------------------------------------------------------------------------------

# Ensure that the training dataset files exist
CHIPS_ID_JSON.touch(exist_ok=True)
BB_CHIP_PAYLOAD.touch(exist_ok=True)
TRACK_CHIPS.touch(exist_ok=True)
TRACK_TILES.touch(exist_ok=True)
### --------------------------------------------------------------------------------------------------------------------


### --------------------------------------------------------------------------------------------------------------------
# Source files referenced to create the training dataset. These files can be acquired from the links provided if the
# files are not already present in the `required_sources` directory.
EXTERNAL_CONTENT = Path(REQUIRED_SOURCES, "external_content")
HLS_KML_LINK = "https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip"
HLS_KML_FILE = Path(EXTERNAL_CONTENT, "sentinel_tile_grid.kml")
CDL_SOURCE_LINK = (
    "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php"
)
CDL_SOURCE = Path(EXTERNAL_CONTENT, "2022_30m_cdls.tif")
# CDL class properties: https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php#what.7
CLD_CLASS_LINK = (
    "https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php#what.7"
)

# Ensure that the required source files exist
if not HLS_KML_FILE.exists():
    print(
        f"Please download the Sentinel Tile from {HLS_KML_LINK} and place it in {EXTERNAL_CONTENT.resolve()}"
    )
if not CDL_SOURCE.exists():
    print(
        f"Please download the National Cropland data layer from {CDL_SOURCE_LINK} and place it in {EXTERNAL_CONTENT.resolve()}"
    )
if not CLD_RECLASS_PROPERTIES.exists():
    print(
        f"Please download the CDL class properties from {CLD_CLASS_LINK} and place it in {EXTERNAL_CONTENT.resolve()}"
    )
### --------------------------------------------------------------------------------------------------------------------
