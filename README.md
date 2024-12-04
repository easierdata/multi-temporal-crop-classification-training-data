# Multi-Temporal Crop Classification: Data generation pipeline

## Introduction

This repository, originally [based](https://github.com/ClarkCGA/multi-temporal-crop-classification-training-data) from [Clark Center for Geospatial Analytics](https://www.clarku.edu/centers/geospatial-analytics/), contains the pipeline to generate data for input into the [multi-temporal crop classification model pipeline](https://github.com/ClarkCGA/multi-temporal-crop-classification-baseline) to fine-tune and test the baseline supervised CNN model or creating a new baseline model from scratch. To generate the data, we use the USDA Cropland Data Layer (CDL) to curate labels and scenes from [NASAs Harmonized Landsat and Sentinel-2](https://hls.gsfc.nasa.gov/) (HLS) product to capture snapshots of time across the growing season.

> The dataset is published as part of the [Prithvi 100M](https://arxiv.org/abs/2310.18660) foundation model release on [HuggingFace](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification) and [Source Cooperative](https://beta.source.coop/repositories/clarkcga/multi-temporal-crop-classification/) with an open-access license.

## Dataset Overview

The primary purpose of the data generation pipeline is for training geospatial segmentation machine learning models. The datasets used in the data generation pipeline consist of the HLS product, a NASA initiative that provides compatible surface reflectance data from Landsat-8 and Sentinel-2 satellites, enabling global land observation every 2-3 days at a 30-meter spatial resolution, and the CDL product,an annual, geo-referenced, crop-specific land cover raster layer produced by the USDA National Agricultural Statistics Service using satellite imagery and extensive agricultural ground reference data.

### Data Access

A selection of scenes from HLS, is available on decentralized networks such as [IPFS](https://ipfs.io/) and [Filecoin](https://filecoin.io/) for resilient and accessible open science collaboration while The python library, ipfs-stac, leverages the [STAC spec](https://stacspec.org/en) to discover content via the [Easier STAC API](https://stac.easierdata.info/) and IPFS for content retrieval.

> < WHAT OTHER DETAILS DO I NEED TO ADD???>

 The CDL data is available for download from the [USDA NASS website](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/). The data generation pipeline requires the HLS data to be downloaded in HDF format and the CDL data to be downloaded in GeoTIFF format.

## Data Generation Pipeline

The data generation pipeline is a multi-step process that involves identifying, determining, downloading and processing HLS scenes. The final output of the processed HLS scenes is the input that for the multi-temporal crop classification model.

Below is an outline of the processes that take place within the data generation pipeline:

1. Specify an area of interest (AOI) representing a region for which training data will be generated*
2. Prepare CDL chips and identify intersecting HLS scenes that correspond to each chip
3. Determine candidate scenes that meet cloud cover and spatial coverage criteria
4. Select and download scenes from IPFS
5. Reproject each tile based on the CDL projection
6. Merge scene bands and clip to chip boundaries
7. Discard clipped results that do not meet QA and NA criteria

For a detailed explanation of each pipeline process, please refer to this [document](./doc/Training%20Data%20Overview.md).

> :exclamation: Setting an AOI is optional, purely for running this pipeline on a smaller subset of the intended data. The USDA CDL sample chips that are referenced were defined based by on a bounding box, ensuring a representative sampling across the CONUS for projects run at Clark University's Center for Geospatial Analytics. By default, the full set of 5000 sample chips will be used, therefore expect a long time frame to finish running the pipeline.

## Running Data Generation Pipeline

### Prerequisites

This project uses [Poetry](https://python-poetry.org/) for dependency management and is compatible with Python versions >=3.10. To install all necessary dependencies, follow these steps:

1. Install Poetry if you haven't already:

   ```shell
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:

   ```shell
   git clone https://github.com/easierdata/multi-temporal-crop-classification-training-data.git
   cd multi-temporal-crop-classification-training-data
   ```

3. Install the dependencies:

   ```shell
   poetry install
   ```

4. Install the IPFS desktop app or Kubo CLI client as this will will allow you to start up a IPFS local node on your machine.

### Configuration

Optionally, you can override the default pipeline coverage of the CONUS with [Select-AOI](Select-AOI.html) by drawing a square or polygon feature. A geojson file is exported, selecting all the intersecting CLD sample  chips.

A configuration file is included to optionally override default properties, located in the [config](./config) directory named `config.yaml`. The configuration file contains the following settings:

- `data_dir`: The directory where the HLS and CDL data is stored
- `train_dataset_name`: Name of the directory where the training data will be stored
- `val_csv_path`: Path to the CSV file containing the validation data
- `test_csv_path`: Path to the CSV file containing the test data
- `aoi_selection`: Path to the geojson file containing the AOI selection from `Select-AOI`

### Running the Pipeline

The pipeline starts by running the following scripts [found here](./crop_classification/data_prep):

1. [prepare_async.py](./crop_classification/data_prep/prepare_async.py) - This script prepares the CDL chips and identifies intersecting HLS scenes that correspond to each chip. The output of this script is a CSV file containing the HLS scenes that intersect with each CDL chip.
2. <ADD DOWNLOAD SCRIPT> - This script downloads the HLS scenes from IPFS based on the CSV file generated in the previous step.
3. [reproject.py](./crop_classification/data_prep/reproject.py) - This script reprojects scene based on the CDL projection.
4. [process_chips.py](./crop_classification/data_prep/process_chips.py) - This script merges scene bands and clips to chip boundaries. It also discards clipped results that do not meet QA and NA criteria.

## Assumptions
<br />

Here are the 5 steps for chip generation in `workflow.ipynb`.
- Download HLS files in HDF format (downloaded for specified months, for all HLS tiles with chips)
- Convert HDF files to GeoTIFF (3 images converted per HLS tile based on cloud cover and spatial coverage criteria. The cloud cover criteria is < 5% total cloud in entire image. The spatial coverage criteria first attempts to find 3 candidate images with 100% spatial coverage, and then decreases this threshold to 90%, 80%... 50%.) Of the candidate images that meet these criteria, the algorithm converts the first, last, and middle image.
- Reproject GeoTIFF files to CDL projection
- Chip HLS and CDL images based on chip geojson. (Outputs are in `chips`, `chips_binary`, and `chips_multi`. Each of these folders contains a 12 band HLS image e.g. `chip_000_000_merged.tif` per chip, with band order R, G, B, NIR for first date, then second date, then third date. The CDL chip is contained in the `chip_000_000.mask` file. In the `chips` directory, the CDL chip contains raw CDL values. In `chips_binary`, the CDL values are reclassified so 0 is non-crop and 1 is crop. In `chips_multi`, the CDL values are reclassified to 13 classes as in `cdl_freq.csv`)
- Filter chips based on QA values and NA values. (The QA values and NA values per chip are tracked in `chip_tracker.csv`. The filter logic excludes chips that have >5% image values (for any of the three HLS image dates) for a single bad QA class. The filter logic also excludes any chips that have 1 or more NA pixels in any HLS image, in any band.)

Also, when first determining which HLS tiles to use in the pipeline, please check that there are erroneous HLS tiles (see step 0a in `workflow.ipynb`). In our use case, we found that certain chips in southern CONUS were associated with HLS tile `01SBU`, which is wrong.

<br />

## Build/Run Docker Environment
<br />

Build the Docker image as following:
```
docker build -t cdl-data .
```

Run the Docker as following (this should run after changing folder to the current folder of the repo):
```
docker run -it -v <local_path_HLS_data_folder>:/data/ -v "$(pwd)":/cdl_training_data/ -p 8888:8888 cdl-data
```
The IP to jupyterlab would be displayed automatically.

*Notice: If running from EC2 you should replace the ip address by the public DNS of the EC2*
<br />

## Requirements
<br />

Docker should be installed in your machine. 

The `workflow.ipynb` notebook requires 4 external files.
- the file `data/2022_30m_cdls_clipped.tif` and this should be generated using the code in `clip.ipynb`. You need to include the raw CDL data for this code. The raw data can be downloaded from [here](https://www.nass.usda.gov/Research_and_Science/Cropland/Release) (the 2022 version).
- the file `data/chip_bbox.geojson` which contains chip boundaries (in CDL crs), and attributes for chip centroids (in long, lat coordinates). The chip centroids are needed to associate each chip to an HLS tile. 
- the file `data/sentinel_tile_grid.kml` for associating chips to HLS tiles.
- the file `data/chip_freq.csv` for reclassifying the original ~200 CDL values to 13 values (e.g. grass, forest, corn, cotton...)
<br />