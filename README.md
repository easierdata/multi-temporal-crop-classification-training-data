# Multi-Temporal Crop Classification: Data generation pipeline

## Introduction

This repository, originally [based](https://github.com/ClarkCGA/multi-temporal-crop-classification-training-data) from [Clark Center for Geospatial Analytics](https://www.clarku.edu/centers/geospatial-analytics/), contains the pipeline to generate data for input into the [multi-temporal crop classification model pipeline](https://github.com/easierdata/multi-temporal-crop-classification-baseline) to fine-tune, test or inference using the baseline supervised CNN model or creating a new baseline model from scratch.

Input model data is derived the USDA Cropland Data Layer (CDL) to curate crop type labels and scenes from [NASAs Harmonized Landsat and Sentinel-2](https://hls.gsfc.nasa.gov/) (HLS) to capture snapshots of time across the growing season. A selection of HLS scenes is available on [IPFS](https://ipfs.io/) and [Filecoin](https://filecoin.io/) for decentralized and accessible open science collaboration.

> The dataset is published as part of the [Prithvi 100M](https://arxiv.org/abs/2310.18660) foundation model release on [HuggingFace](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification) and [Source Cooperative](https://beta.source.coop/repositories/clarkcga/multi-temporal-crop-classification/) with an open-access license.

## Dataset Overview

The primary purpose of the data generation pipeline is for training geospatial segmentation machine learning models. The datasets used in the data generation pipeline consist of the HLS product, a NASA initiative that provides compatible surface reflectance data from Landsat-8 and Sentinel-2 satellites, enabling global land observation every 2-3 days at a 30-meter spatial resolution, and the CDL product,an annual, geo-referenced, crop-specific land cover raster layer produced by the USDA National Agricultural Statistics Service using satellite imagery and extensive agricultural ground reference data.

### Data Access

A selection of scenes from HLS, is available on decentralized networks such as [IPFS](https://ipfs.io/) and [Filecoin](https://filecoin.io/) for resilient and accessible open science collaboration. The python library, ipfs-stac, leverages the [STAC spec](https://stacspec.org/en) to discover content via the [Easier STAC API](https://stac.easierdata.info/) and IPFS for content retrieval.

[HLS data is subdivided](https://sentiwiki.copernicus.eu/web/s2-products) on a predefined set of tiles, which dictates the tile naming convention. The tile grid file is included in the repository as `sentinel_tile_grid.kml` which can also be sourced from the Copernicus Open Access Hub at this [link](https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip?inst-v=d43ccf82-85e7-4081-860f-990e7b6e9407).

The CDL data is available for download from the [USDA NASS website](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/). The data generation pipeline requires the HLS data to be downloaded in HDF format and the CDL data to be downloaded in GeoTIFF format.

## Data Generation Pipeline

The data generation pipeline is a multi-step process that involves identifying, determining, downloading and processing HLS scenes. The final output of the processed HLS scenes is used as the input for running the multi-temporal crop classification model.

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

To get started:

1. create a virtual environment and install the required dependencies:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Install the [IPFS desktop](https://docs.ipfs.tech/how-to/desktop-app/) app or [Kubo CLI client](https://docs.ipfs.tech/install/command-line/) as this will will allow you to run a local IPFS node on your machine.

### Configuration

Optionally, you can override the default pipeline coverage of the CONUS with [Select-AOI](Select-AOI.html) by drawing a square or polygon feature. A geojson file is exported, selecting all the intersecting CLD sample  chips.

A configuration file is included to optionally override default properties, located in the [config](./config) directory named `config.yaml`. The configuration file contains the following settings:

- `data_dir`: The directory where the HLS and CDL data is stored
- `train_dataset_name`: Name of the directory where the training data will be stored
- `chip_payload_filename`: Path to the geojson file containing the AOI selection from `Select-AOI`
- `ipfs_stac_params`: Parameters for the `ipfs-stac` library as to access a STAC endpoint configured with alternate extension with CID support

### Running the Pipeline

The pipeline starts by running the following scripts [found here](./crop_classification/data_prep):

1. [identify_hls_scenes.py](./crop_classification/data_prep/identify_hls_scenes.py) - This script prepares the CDL chips and identifies intersecting HLS scenes that correspond to each chip. The output of this script is a CSV file containing the HLS scenes that intersect with each CDL chip.
2. [retrieve_hls_scenes.py](./crop_classification/data_prep/retrieve_hls_scenes.py) - This script downloads the HLS scenes from IPFS based on the CSV file generated in the previous step.
3. [reproject_hls_scenes.py](./crop_classification/data_prep/reproject_hls_scenes.py) - This script reprojects scene based on the CDL projection.
4. [generate_training_chips.py](./crop_classification/data_prep/generate_training_chips.py) - This script merges scene bands and clips to chip boundaries. It also discards clipped results that do not meet QA and NA criteria.

The final images used for model training are stored in the `chips_filtered` directory.

To split the final images into training and validation datasets, run the following script:

```shell
python crop_classification/data_prep/split_training_data.py
```

which creates two files `train_ids.csv` and `val_ids.csv` in the training dataset directory.
