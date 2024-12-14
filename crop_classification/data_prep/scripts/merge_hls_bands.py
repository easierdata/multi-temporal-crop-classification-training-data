import rasterio
import numpy as np
from pathlib import Path


def merge_hls_bands(folder_path):
    band_images = []

    for file_path in folder_path.glob("*.tif"):
        # If the file ends with Fmask.tif, skip it
        if file_path.name.endswith("Fmask.tif"):
            continue
        with rasterio.open(file_path) as src:
            band_images.append(src.read(1))  # Read the first band

    if band_images:
        merged_image = np.mean(band_images, axis=0)  # Merge by averaging
        return merged_image
    return None


def save_merged_image(merged_image, output_path):
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=merged_image.shape[0],
        width=merged_image.shape[1],
        count=1,
        dtype=merged_image.dtype,
    ) as dst:
        dst.write(merged_image, 1)


def main():
    base_dir = Path("./data/download/training_data/tiles")
    for folder in base_dir.iterdir():
        if folder.is_dir():
            merged_image = merge_hls_bands(folder)
            if merged_image is not None:
                output_path = folder / f"{folder.name}_merged.tif"
                save_merged_image(merged_image, output_path)


if __name__ == "__main__":
    main()
