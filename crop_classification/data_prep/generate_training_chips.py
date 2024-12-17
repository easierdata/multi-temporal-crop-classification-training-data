import shutil
import sys
import numpy as np
import rasterio
import rasterio.mask
import pandas as pd
from pathlib import Path
import tqdm

import json


# The code used to add the src directory to the Python path, making
# it possible to import modules from that directory.

module_path = Path(__file__).parent.parent.resolve().as_posix()
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass

## set up CDL reclass
cdl_class_df = pd.read_csv(CLD_RECLASS_PROPERTIES)
CROP_DICT = dict(zip(cdl_class_df.old_class_value, cdl_class_df.new_class_value))

# Remove Fmask value from HLS_BANDS since we do not want
# to merge it into the final merged imagery chip
_HLS_BANDS = [band for band in HLS_BANDS if band != "Fmask"]


def crop_multi(x):
    return CROP_DICT[x]


def get_tile_info(chip_tile, all_tiles_df):
    tile_info_df = all_tiles_df[all_tiles_df.tile_id == chip_tile]
    # Ensure that we have 3 images for the tile
    if len(tile_info_df) != 3:
        raise AssertionError(
            f"Expected 3 images for tile {chip_tile}, but found {len(tile_info_df)}"
        )
    first_image_date = tile_info_df.iloc[0].date
    second_image_date = tile_info_df.iloc[1].date
    third_image_date = tile_info_df.iloc[2].date
    return (
        tile_info_df,
        first_image_date,
        second_image_date,
        third_image_date,
    )


def get_image_paths(tile_info_df):
    all_date_images = []
    all_date_qa = []

    for i in range(3):
        filename = tile_info_df.iloc[i].title_id

        # Get the paths to the images for each band
        all_date_images.extend(
            [Path(TILE_REPROJECTED_DIR, f"{filename}.{band}.tif") for band in HLS_BANDS]
        )

        # Add the path to the QA band for each date
        all_date_qa.extend([Path(TILE_REPROJECTED_DIR, f"{filename}.Fmask.tif")])
    return all_date_images, all_date_qa


def check_all_qa(all_date_qa, shape):
    valid_first, bad_pct_first, qa_first = check_qa(all_date_qa[0], shape)
    valid_second, bad_pct_second, qa_second = check_qa(all_date_qa[1], shape)
    valid_third, bad_pct_third, qa_third = check_qa(all_date_qa[2], shape)
    qa_bands = []
    qa_bands.append(qa_first)
    qa_bands.append(qa_second)
    qa_bands.append(qa_third)
    qa_bands = np.array(qa_bands).astype(np.uint8)
    return (
        valid_first,
        bad_pct_first,
        qa_first,
        valid_second,
        bad_pct_second,
        qa_second,
        valid_third,
        bad_pct_third,
        qa_third,
        qa_bands,
    )


def check_qa(
    qa_path,
    shape,
    valid_qa=[0, 4, 32, 36, 64, 68, 96, 100, 128, 132, 160, 164, 192, 196, 224, 228],
):
    """
    This function receives a path to a qa file, and a geometry. It clips the QA file to the geometry.
    It returns the number of valid QA pixels in the geometry, and the clipped values.

    Assumptions: The valid_qa values are taken from Ben Mack's post:
    https://benmack.github.io/nasa_hls/build/html/tutorials/Working_with_HLS_datasets_and_nasa_hls.html

    Inputs:
    - qa_path: full path to reprojected QA tif file
    - shape: 'geometry' property of single polygon feature read by fiona
    - valid_qa: list of integer values that are 'valid' for QA band.
    """
    try:
        with rasterio.open(qa_path) as src:
            out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
    except ValueError as e:
        if str(e) == "Input shapes do not overlap raster.":
            print("Warning: Input shapes do not overlap raster.")
            return (0, 0, None)
        else:
            raise
    vals = out_image.flatten()
    unique, counts = np.unique(vals, return_counts=True)
    qa_df = pd.DataFrame({"qa_val": unique, "counts": counts})
    qa_df[~qa_df.qa_val.isin(valid_qa)].sort_values(["counts"], ascending=False)
    qa_df["pct"] = (100 * qa_df["counts"]) / (224.0 * 224.0)

    bad_qa = qa_df[~qa_df.qa_val.isin(valid_qa)].sort_values(
        ["counts"], ascending=False
    )
    if len(bad_qa) > 0:
        highest_invalid_percent = bad_qa.pct.tolist()[0]
    else:
        highest_invalid_percent = 0
    # ncell = len(vals)
    valid_count = sum(x in valid_qa for x in vals)
    return (valid_count, highest_invalid_percent, out_image[0])


def get_out_bands(all_date_images, shape):
    out_bands = []
    for img in all_date_images:
        with rasterio.open(img) as src:
            out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
            out_meta = src.meta
            out_bands.append(out_image[0])
    out_bands = np.array(out_bands)
    # na_count = sum(out_bands.flatten() == -1000)
    # out_bands = np.clip(out_bands, 0, None)
    return out_bands, out_transform, out_meta


def write_hls_chips(chip_id, out_bands, out_transform, out_meta):
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_bands.shape[1],
            "width": out_bands.shape[2],
            "count": out_bands.shape[0],
            "transform": out_transform,
        }
    )

    na_count = sum(out_bands.flatten() == -1000)
    out_bands = np.clip(out_bands, 0, None)
    with rasterio.open(
        Path(CHIP_DIR) / f"{chip_id}_merged.tif", "w", **out_meta
    ) as dest:
        dest.write(out_bands)

    return out_meta, na_count


def write_qa_bands(chip_id, qa_bands, out_transform, out_meta):
    out_meta.update(
        {
            "driver": "GTiff",
            "height": qa_bands.shape[1],
            "width": qa_bands.shape[2],
            "count": qa_bands.shape[0],
            "transform": out_transform,
        }
    )
    with rasterio.open(
        Path(FMASK_DIR) / f"{chip_id}_Fmask.tif", "w", **out_meta
    ) as dest:
        dest.write(qa_bands)

    # return out_meta


def clip_cdl_to_chip(shape):
    with rasterio.open(CDL_SOURCE) as src:
        out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
        out_meta = src.meta
        colormap = src.colormap(1)
    return out_image, out_transform, out_meta, colormap


def write_cdl_chips(chip_id, out_image, out_transform, out_meta, colormap):
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )
    c_multi = np.vectorize(crop_multi)
    out_image_multi = c_multi(out_image).astype(np.uint8)
    with rasterio.open(Path(CHIP_DIR) / f"{chip_id}.mask.tif", "w", **out_meta) as dest:
        dest.write(out_image_multi)
        dest.write_colormap(1, colormap)


def process_chip(chip_id, chip_tile, shape, all_tiles):
    try:
        # Select a tile and check to ensure that the tile has 3 images
        (
            selected_tile,
            first_image_date,
            second_image_date,
            third_image_date,
        ) = get_tile_info(chip_tile, all_tiles)
    except AssertionError as e:
        print(f"Error processing tile {chip_tile}: {e}")

    # Get the paths to the images and QA files for the tile
    all_date_images, all_date_qa = get_image_paths(selected_tile)

    (
        valid_first,
        bad_pct_first,
        qa_first,
        valid_second,
        bad_pct_second,
        qa_second,
        valid_third,
        bad_pct_third,
        qa_third,
        qa_bands,
    ) = check_all_qa(all_date_qa, shape)

    # If any of the QA bands are invalid, return
    if not valid_first or not valid_second or not valid_third:
        return

    out_bands, out_transform, ouput_metadata = get_out_bands(all_date_images, shape)

    ouput_metadata, na_count = write_hls_chips(
        chip_id, out_bands, out_transform, ouput_metadata
    )

    write_qa_bands(chip_id, qa_bands, out_transform, ouput_metadata)

    out_image, out_transform, ouput_metadata, colormap = clip_cdl_to_chip(shape)

    write_cdl_chips(chip_id, out_image, out_transform, ouput_metadata, colormap)

    return {
        "valid_first": valid_first,
        "valid_second": valid_second,
        "valid_third": valid_third,
        "bad_pct_first": bad_pct_first,
        "bad_pct_second": bad_pct_second,
        "bad_pct_third": bad_pct_third,
        "na_count": na_count,
        "first_image_date": first_image_date,
        "second_image_date": second_image_date,
        "third_image_date": third_image_date,
    }


def main():

    # Define missing variables
    tiles_to_chip = []
    chip_df = None
    chip_ids = []
    chipping_js = None
    track_df = None

    # Load in details about the chips
    chip_df = pd.read_pickle(CHIPS_DF_PKL)
    with open(CHIPS_ID_JSON) as f:
        chip_ids = json.load(f)
    with open(BB_CHIP_PAYLOAD, "r") as file:
        chips = json.load(file)
    with open(BB_CHIP_5070_PAYLOAD, "r") as file_chip:
        chipping_js = json.load(file_chip)

    # Load in the selected tiles that were reprojected and get a list of the tiles to chip
    selected_tiles_df = pd.read_pickle(SELECTED_TILES_PKL)
    tiles_to_chip = selected_tiles_df.tile_id.unique().tolist()

    # If the track chips file exists, load it in and identify if any additional tiles need to be processed
    if TRACK_CHIPS.exists():
        track_df = pd.read_csv(TRACK_CHIPS)
        # Convert the already processed tiles back to the list, `chip_data`
        chip_data = track_df.to_dict(orient="records")
        # Get the tiles that have already been processed
        tiles_already_processed = track_df.tile.unique().tolist()
        tiles_to_chip = list(set(tiles_to_chip) - set(tiles_already_processed))
        print(
            f"""
        The file, {TRACK_CHIPS.name} already exists with {len(tiles_already_processed)} tile(s) already processed.

        Continuing with the remaining {len(tiles_to_chip)} tile(s).

        NOTE: If you would like to reprocess all tiles, please delete the file {TRACK_CHIPS.name}.
        """
        )

    # Export failed tiles to a csv file
    if failed_tiles:
        failed_tiles_df = pd.DataFrame(failed_tiles)
        failed_tiles_df.to_csv(
            Path(TRAINING_DATASET_PATH, "failed_tiles_to_chip.csv"), index=False
        )

    chip_data_df = pd.DataFrame(chip_data)
    chip_data_df["bad_pct_max"] = chip_data_df[
        ["bad_pct_first", "bad_pct_second", "bad_pct_third"]
    ].max(axis=1)
    chip_data_df.to_csv(TRACK_CHIPS, index=False)

    # Filter chips with bad_pct_max > 5 and na_count > 0
    for tile in tiles_to_chip:
        # select the subset of rows in the dataframe that match the tile
        filtered_chips = chip_data_df[tile == chip_data_df.tile]
        # filter the chips based on the conditions
        filtered_chips = filtered_chips[
            (filtered_chips.bad_pct_max < 5) & (filtered_chips.na_count == 0)
        ].chip_id.tolist()
        for chip_id in filtered_chips:
            chip_files = CHIP_DIR.glob(f"*{chip_id}*")
            for file in chip_files:
                shutil.copyfile(file, Path(FILTERED_DIR, file.name))


if __name__ == "__main__":
    main()
