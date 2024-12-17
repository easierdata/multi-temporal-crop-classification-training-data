# Steps to Reproduce Training Data

## Introduction

The data generation pipeline is a multi-step process that involves identifying, determining, downloading and processing HLS scenes. The final output of the processed HLS scenes is used as the input for running the for the multi-temporal crop classification model.

### Process Outline

The steps for generating the training data are as follows:

1. Specify an area of interest (AOI) representing a region for which training data will be generated*
2. Prepare CDL chips and identify intersecting HLS scenes that correspond to each chip
3. Determine candidate scenes that meet cloud cover and spatial coverage criteria
4. Select and download scenes from IPFS
5. Reproject each tile based on the CDL projection
6. Merge scene bands and clip to chip boundaries
7. Discard clipped results that do not meet QA and NA criteria

### Crop Classification `data_prep` module

The prior [workflow](https://github.com/ClarkCGA/multi-temporal-crop-classification-training-data/blob/main/workflow.ipynb) was self contained within a Jupyter notebook, but has been refactored into a more [modular](../crop_classification/data_prep) and reusable format. The process outlined above is now broken down into a series of scripts that can be run independently or as a whole.

1. [identify_hls_scenes.py](./crop_classification/data_prep/identify_hls_scenes.py) - This script identifies HLS scenes that correspond to each intersecting chip. The output of this script is a CSV file containing the HLS scenes that intersect with each training chip.
2. [retrieve_hls_scenes.py](./crop_classification/data_prep/retrieve_hls_scenes.py) - This script downloads the HLS scenes from IPFS based on the CSV file of selected scene, generated in the previous step.
3. [reproject_hls_scenes.py](./crop_classification/data_prep/reproject_hls_scenes.py) - This script reprojects the HLS scene to match CDL projection.
4. [generate_training_chips.py](./crop_classification/data_prep/generate_training_chips.py) - This script generates the training chip dataset that's sent through the model pipeline.  merges scene bands and clips to chip boundaries. It also discards clipped results that do not meet QA and NA criteria.

#### Scripts

##### grab_cids_from_selected_tiles.py

Handles getting the CIDs for the selected tiles and generates a JSON file containing the Content Identifiers (CIDs) for each tile.

**When to run**: This script should be run after selecting the AOI and downloading the required HLS data to generate the CIDs for each tile, which are necessary for further processing in the pipeline. The resulting output can be used to run `ipfs_cli_download.py` to download the HLS scenes or `check_ipfs_content_retrievability`.

##### create_bb.py

Responsible for generating bounding boxes for chips based on the Cropland Data Layer (CDL) data. It processes the CDL data to create a GeoJSON file containing the bounding boxes and their associated properties. These bounding boxes are used in subsequent steps of the pipeline to define the spatial extent of the chips that will be processed.

**When to run:** This script should be run at the beginning of the pipeline to prepare the spatial extents for the chips.

##### select_aoi.py

Generates an interactive web map tool for users to select an Area of Interest (AOI). Users can draw polygons on a map to define the AOI, which is then saved as a GeoJSON file. The selected AOI is used to filter the chips and tiles that will be processed in the pipeline, ensuring that only data within the specified area is considered.

**When to run**: This script is only necessary to run if the bounding box chips were updated with `create_bb.py`.

##### ipfs_cli_download.py

Handles the downloading of HLS scenes based on the selected tiles, with the [ipfs CLI tool](https://docs.ipfs.tech/reference/kubo/cli/#ipfs). It retrieves the assets associated with each tile from IPFS and saves them to the local directory for further processing. This script ensures that all necessary HLS data is available locally for subsequent steps in the pipeline.

**When to run:** This script should be run after selecting the AOI to download the required HLS data.

> Note: This script is optional and can be used as an alternative to the `retrieve_hls_scenes.py` script. It can also be used as an auxiliary tool to help identify and download HLS scenes from IPFS using the ipfs CLI tool directly.

##### check_ipfs_content_retrievability.py

Checks the retrievability of content from IPFS by attempting to download each Content Identifier (CID). It iterates through a list of CIDs, tries to download the corresponding content, and logs any missing or inaccessible CIDs to a JSON file. This information is used to identify and address any issues with data availability on IPFS.

**When to run**: This script can be run after generating the json from `grab_cids_from_selected_tiles.py`.

##### split_training_data.py

Splits the chip IDs into training and validation sets for the specified training dataset in the configuration file. A unique list of chip IDs are determined from the files in the `chips_filtered` directory. It reads the list of chip IDs, randomly assigns them to either the training or validation set, and saves the resulting lists to CSV files. These CSV files are used to train and validate the crop classification model, ensuring that the model is evaluated on a separate set of data from what it was trained on.

**When to run:** This script should be run after preparing the chip data to create the training and validation datasets.

##### calc_band_mean_sd.py

Calculates statistical metrics (mean, standard deviation, minimum, and maximum values) for each set of bands in the final merged image in the `filtered_chips` directory. The statistics are saved to a text files and also stores all band values in a binary file.

**When to run:** This script should be run after preparing the chip data to create the training and validation datasets. This output can be used to compute the mean and standard deviation for normalization purposes during model training.

##### calc_class_weights.py

Calculates class weights for crop classification based on the frequency of each class of the mask image files found in the `filtered_chips` directory. The data in each image file is flattened to calculate the class weights, normalizing them so their sum equals 1. It inserts a weight of 0 for the background class (no crop) and saves the calculated class weights to a CSV file.

**When to run:** This script should be run after preparing the chip data to create the training and validation datasets. The calculated class weights can then be used for training the model.

##### generate_class_distribution.py.py

Generates and plots the class distribution for training and validation datasets in a crop classification pipeline. Processes each TIFF mask file to calculate the frequency of each class, to plot the class distributions. resulting plots are saved as PNG files, providing a visual representation of the class occurrences in the training and validation datasets.

**When to run:** This script should be run after preparing the chip data to create the training and validation datasets. The plots aid in understanding the data distribution and ensuring balanced class representation for model training.

### Data Prerequisites

The required source files to download are:

* [2022 National Cropland Data Layer (CDL)](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/)
* Harmonized Landsat-Sentinel (HLS) imagery
  * Sourced from Earthdata and prepared with [Singularity](https://data-programs.gitbook.io/singularity) and onboard to Filecoin and IPFS
* [Sentinel-2 Tile Grid](https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip) - Details on the tile grid for Sentinel-2 can be found [here](https://sentiwiki.copernicus.eu/web/s2-products).

## Dataset Summary

Each HLS scene requires the following spectral bands in a GeoTIFF format.

1. Blue
2. Green
3. Red
4. Narrow NIR
5. SWIR 1
6. SWIR 2

HLS scenes are clipped to match the bounding box of the CDL chips, covering a 224 x 224 pixel area at 30m spatial resolution The scenes are then merged into a single GeoTIFF, containing 18 bands for three time steps, where each time step represents three observations throughout the growing season. Each GeoTIFF is accompanied with a mask, containing one band with the target classes for each pixel.

> The processed HLS scenes are saved to the directory `./data/training_datasets/<training dataset name>`, where `<training dataset name>` represents the value passed into `train_dataset_name` property in the configuration file.

### Band Order

In each input GeoTIFF the following bands are repeated three times for three observations throughout the growing season: Channel, Name, HLS S30 Band number
>
    Channel, Name,   HLSS30 Band number
    1,       Blue,   B02
    2,       Green,  B03
    3,       Red,    B04
    4,       NIR,    B8A
    5,       SW 1,   B11
    6,       SW 2,   B12
>

Masks are a single band with values:
> 0 : "No Data" 1 : "Natural Vegetation" 2 : "Forest" 3 : "Corn" 4 : "Soybeans" 5 : "Wetlands" 6 : "Developed/Barren" 7 : "Open Water" 8 : "Winter Wheat" 9 : "Alfalfa" 10 : "Fallow/Idle Cropland" 11 : "Cotton" 12 : "Sorghum" 13 : "Other"

### Data Splits

The 3,854 chips have been randomly split into training (80%) and validation (20%) with corresponding ids recorded in cvs files `train_data.txt` and `validation_data.txt`.

## Detailed Explanation of the Pipeline Process

## Prepare Chip Detail Payloads

This payload contains the details about the chip information that will be used to identify corresponding tiles. This information is derived from the `chips_id.json` file to generate the following dataframe:

* Chip ID
* Chip X centroid coordinate
* Chip Y centroid coordinate
* Tile Name

## Prepare HLS Tile Spatial Context

This payload contains the spatial context about the Sentinel scene tiles in `sentinel_tile_grid.kml`.  The information derived is:

* Tile Name
* Tile X centroid coordinate
* Tile Y centroid coordinate
* Tile bounding box

## Find and Identify the nearest tile to each chip

Loop each chip and identify closest tile by the xy centroid. The chip payload dataframe is modified with tile information and is saved to file for reuse later.

## Query metadata on the overlapping tiles

Here, we need to capture and store metadata details all scenes for a given tile that are above a specific cloud cover threshold. This information comes from a xml metadata file found at [this STAC instance endpoint](https://cmr.earthdata.nasa.gov/stac/LPCLOUD).

1. Create a unique list of tile names that are captured in the chip payload dataframe.
2. Vist STAC endpoint and loop through each tile in the list and check if the scene is above the cloud cover threshold
   1. If true, read in the XML file and capture the following details:
      * Tile ID
      * Cloud Cover
      * Scene Date
      * Spatial Coverage
      * HTTP & S3 links for the bands `B02`, `B03`, `B04`, `B8A`, `B11`, `B12` and `fmask`
3. Save scene results down as tile payload dataframe to file for reuse later.

## Filter the tile metadata based on the spatial coverage threshold

Here we identify scenes that meet our criteria of:

1. Cloud cover of 5% or less total cloud coverage in entire image.
2. Spatial coverage threshold of 50% and above.

The cloud cover criteria is < 5% total cloud in entire image. The spatial coverage criteria first attempts to find 3 candidate images with 100% spatial coverage, and then decreases this threshold to 90%, 80%... 50%.

## Select Candidate images based on timestamps

1. Sort the filtered coverage list by date, grouped by tile ID
2. Select from the list the first, middle and last scene
   * If list contains an even number of scenes, the lower of the two is selected for the middle scene
3. Save list of selected tiles to disk

## Download the selected tiles from IPFS

1. Loop through the selected tiles list and grab the [CIDs](https://docs.ipfs.tech/concepts/content-addressing/) for the required bands.
2. With the CIDs, retrieve the image bands from IPFS and save to disk, to a directory as `tile-id.tile-date.v2.0`

## Reproject each tile based on the CDL projection

Loop through each band and reproject, matching the CDL coordinate system `EPSG:5070`. Files are saved to the directory `tiles_reprojected`.

## Import saved dataframe files and preparing tile chipping process

1. Import the tile chip df files and reinstate as a dataframe
2. Create a unique list of scenes that need to be clipped
3. Set-up the CDL reclass properties in order to reclass chips
4. Run the `Process Chip` function to clip scenes for all tiles.
   * The bands for each scene is processed and clipped.
   * Resultant clip is saved to the chips folder, grouped by Chip ID
5. Chip details are saved to a file to validation

> **chip_xxx_xxx_merged.tif**: Three HLS scenes merged together per chip, with band order R, G, B, NIR for first date, then second date, then third date.
> **chip_xxx_xxx.mask.tif**: used to handle cloud cover and other quality aspects of the images. Specifically, it helps in filtering out bad quality pixels and ensuring only valid data is used for generating the training dataset.

## Filter out resultant chipped scenes

Review the chip details and filter chips based on QA values and NA values. Chips are filtered based on the following logic:

* Exclude chips that have >5% image values for a single bad QA class, in any of the three HLS image scenes.
* Exclude any chips that have 1 or more `NA Pixels` in any HLS image, in any band.

The `merged.tif` and `mask.tif` images for the resultant selection are then copied to the `chips filtered` directory.

---

### *Notes*

When first determining which HLS tiles to use in the pipeline, please check that there are erroneous HLS tiles (see step 0a in workflow.ipynb). In our use case, we found that certain chips in southern CONUS were associated with HLS tile 01SBU, which is wrong.

---

# Diagrams

```mermaid
%%{init: {
    "theme": "dark",
    "themeCSS": ".cardinality text { fill: #ededed }",
    "themeVariables": {
    "primaryTextColor": "#ededed",
        "nodeBorder": "#393939",
        "mainBkg": "#292929",
        "lineColor": "orange"
        },
    "flowchart": {
        "curve": "basis",
        "useMaxWidth": true,
        "htmlLabels": true
    }
}}%%
flowchart TD
    %% Properties for the subgraph notes
    classDef sub opacity:0
    classDef note fill:#ffd, stroke:#ccb

    %% Flowchart steps
    D --> E[Query metadata on the overlapping tiles]
    E --> F[Filter set of scenes for each tile based on the spatial coverage threshold]
    F --> G[Select set of scenes based on timestamps for each tile]
    G --> H[Download the selected tiles from the LPDAAC cloud endpoint]
    H --> I[Reproject each tile based on the CDL projection]
    I --> J[Import saved dataframe files and preparing tile chipping process]
    K[Filter out resultant chipped scenes]

    %% Subgraphs groups that self-contain processes
    subgraph loop1 [Prepare CDL Chip Detail and HLS Tile Payloads]
        B["Extract details from bounding box file"]
        C["Extract details from Sentinel Tile Grid KML"]
        D["Find and Identify the nearest tile to each chip
        Store results down to disk"]
        B --> D
        C --> D
    end
  
    A[Start] --> loop1

    subgraph loop2 [Gather metadata on the overlapping tiles]
        E1[Create a unique list of tile names] --> E2[Vist STAC endpoint]
        E2 --> E3[Check if the scene is above the cloud cover threshold]
        E3 --> |"True"| E4[Read XML file and capture details]
        E4 --> E2
        E3 --> |"False"| E2
    end

    E --> loop2

    subgraph loop3 [Loop through the tiles]
        H1[Download the set of bands for each scene]
        H2[Capture and store process details for tracking] 
        H1 --> H2
        H2 --> H1
    end

    H --> loop3

    subgraph loop4 [Loop through the tiles]
        I1[Loop through scene bands and reproject]
        I2[Save reprojection to disk]
        I1 --> I2
        I2 --> I1

    end

    I --> loop4

    subgraph loop5 [Run the Process Chip function]
        J1[Set-up the CDL reclass properties] --> J2[Run the Process Chip function]
        J2 --> J3[Save resultant clip to the chips folder]
        J3 --> J1
    end

    J --> loop5
    loop5 --> K
    %% Notes for the subgraphs
    subgraph noteLoop1 [" "]
        loop1
        loop1-note("
        Generation of the chip detail payload is dependent on the 
        bounding box file generated using gen_chip_bbbox.ipynb.
        ")
    end

    subgraph noteG [" "]
        G
        g-note("
        Tiles are sorted by group id and date. The first, middle 
        and last scene is selected for each group.
        ")
    end

    subgraph noteK [" "]
        K
        k-note("
        Any tile sets with scenes containing
        'NA' pixels are removed from the
        final training dataset.
        ")
    end

    %% Setting the note properties for the subgraphs
    class noteLoop1,noteG,noteK sub
    class loop1-note,g-note,k-note note
```
