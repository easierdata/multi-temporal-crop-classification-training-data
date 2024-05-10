import os
import sys
import numpy as np
import rasterio
import rasterio.mask
import pandas as pd
from pathlib import Path


import json


# The code cell is used to add the src directory to the Python path, making
# it possible to import modules from that directory.

module_path = module_path = os.path.abspath(Path(__file__).parent.parent.resolve())
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass


def crop_multi(x):
    return crop_dict[x]


def get_tile_info(chip_tile, track_csv):
    tile_info_df = track_csv[track_csv.tile == chip_tile]
    selected_image_folders = tile_info_df.save_path.to_list()
    assert len(selected_image_folders) == 3
    first_image_date = tile_info_df.iloc[0].date
    second_image_date = tile_info_df.iloc[1].date
    third_image_date = tile_info_df.iloc[2].date
    return (
        tile_info_df,
        selected_image_folders,
        first_image_date,
        second_image_date,
        third_image_date,
    )


def get_image_paths(tile_info_df, bands):
    all_date_images = []
    all_date_qa = []
    for i in range(3):
        for band in bands:
            all_date_images.append(
                tile_info_df.iloc[i].save_path
                + f"{tile_info_df.iloc[i].filename}.{band}.reproject.tif"
            )
        all_date_qa.append(
            tile_info_df.iloc[i].save_path
            + f"{tile_info_df.iloc[i].filename}.Fmask.reproject.tif"
        )
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
    with rasterio.open(qa_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
        vals = out_image.flatten()
        unique, counts = np.unique(vals, return_counts=True)
        qa_df = pd.DataFrame({"qa_val": unique, "counts": counts})
        qa_df
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
    na_count = sum(out_bands.flatten() == -1000)
    out_bands = np.clip(out_bands, 0, None)
    return out_bands, out_transform, out_meta, na_count


def write_hls_chips(chip_id, out_bands, out_transform, out_meta, chip_dir):
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_bands.shape[1],
            "width": out_bands.shape[2],
            "count": out_bands.shape[0],
            "transform": out_transform,
        }
    )
    with rasterio.open(
        chip_dir + str(chip_id) + "_merged.tif", "w", **out_meta
    ) as dest:
        dest.write(out_bands)


def write_qa_bands(chip_id, qa_bands, out_transform, out_meta, chip_fmask_dir):
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
        chip_fmask_dir + str(chip_id) + "_Fmask.tif", "w", **out_meta
    ) as dest:
        dest.write(qa_bands)


def clip_cdl_to_chip(cdl_file, shape):
    with rasterio.open(cdl_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
        out_meta = src.meta
        colormap = src.colormap(1)
    return out_image, out_transform, out_meta, colormap


def write_cdl_chips(
    chip_id, out_image_multi, out_transform, out_meta, colormap, chip_dir
):
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image_multi.shape[1],
            "width": out_image_multi.shape[2],
            "transform": out_transform,
        }
    )
    with rasterio.open(chip_dir + str(chip_id) + ".mask.tif", "w", **out_meta) as dest:
        dest.write(out_image_multi)
        dest.write_colormap(1, colormap)


def process_chip(
    chip_id,
    chip_tile,
    shape,
    track_csv,
    cdl_file,
    bands=["B02", "B03", "B04", "B8A", "B11", "B12"],
):
    (
        tile_info_df,
        selected_image_folders,
        first_image_date,
        second_image_date,
        third_image_date,
    ) = get_tile_info(chip_tile, track_csv)

    all_date_images, all_date_qa = get_image_paths(tile_info_df, bands)

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

    out_bands, out_transform, out_meta, na_count = get_out_bands(all_date_images, shape)

    write_hls_chips(chip_id, out_bands, out_transform, out_meta, CHIP_DIR)

    write_qa_bands(chip_id, qa_bands, out_transform, out_meta, FMASK_DIR)

    out_image, out_transform, out_meta, colormap = clip_cdl_to_chip(cdl_file, shape)

    write_cdl_chips(chip_id, out_image, out_transform, out_meta, colormap, CHIP_DIR)

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


def c_multi(out_image):
    # Add your implementation of the c_multi function here
    np.vectorize(crop_multi)


def main():

    # Define missing variables
    tiles_to_chip = []
    chip_df = None
    chip_ids = []
    chipping_js = None
    track_df = None
    ## process chips
    failed_tiles = []

    ## set up CDL reclass
    cdl_class_df = pd.read_csv(CLD_RECLASS_PROPERTIES)
    crop_dict = dict(zip(cdl_class_df.old_class_value, cdl_class_df.new_class_value))

    failed_tiles = []
    chip_data = []

    chip_df = pd.read_pickle(CHIPS_DF_PKL)
    with open(CHIPS_ID_JSON) as f:
        chip_ids = json.load(f)
    track_df = pd.read_pickle(TILES_DF_PKL)
    with open(BB_CHIP_PAYLOAD, "r") as file:
        chips = json.load(file)

    tiles_to_chip = track_df.tile.unique().tolist()
    with open(BB_CHIP_1570_PAYLOAD, "r") as file_chip:
        chipping_js = json.load(file_chip)

    for tile in tiles_to_chip:
        chips_to_process = chip_df[chip_df.tile == tile[1:]].reset_index(drop=True)
        for k in range(len(chips_to_process)):
            current_id = chips_to_process.chip_id[k]
            chip_tile = chips_to_process.tile[k]
            chip_index = chip_ids.index(current_id)

            chip_feature = chipping_js["features"][chip_index]

            shape = [chip_feature["geometry"]]
            full_tile_name = "T" + chip_tile

            try:
                chip_info = process_chip(
                    current_id, full_tile_name, shape, track_df, CDL_SOURCE
                )
                chip_info["chip_id"] = current_id
                chip_data.append(chip_info)
            except:
                failed_tiles.append(tile)

    chip_df = pd.DataFrame(chip_data)
    chip_df["bad_pct_max"] = chip_df[
        ["bad_pct_first", "bad_pct_second", "bad_pct_third"]
    ].max(axis=1)
    chip_df.to_csv(TRACK_CHIPS, index=False)


if __name__ == "__main__":
    main()