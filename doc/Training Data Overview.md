# Steps to Reproduce Training Data

Below is an outline of the process's that take place in the [workflow notebook](https://github.com/ClarkCGA/multi-temporal-crop-classification-training-data/blob/main/workflow.ipynb) to create the training dataset for crop [classification baseline model](https://github.com/ClarkCGA/multi-temporal-crop-classification-baseline).

The required source files to download are:

* [2022 National Cropland Data Layer (CDL)](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/)
* Harmonized Landsat-Sentinel (HLS) imagery
  * Sourced from Earthdata to be packaged with Singularity and uploaded to IPFS

## Prepare CDL Chip Detail Payloads

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

Here we identify and select scenes that meet our spatial coverage threshold of 50% and above.

## Select tile coverage based on timestamps

1. Sort the filtered coverage list by date, grouped by tile ID
2. Select from the list the first, middle and last scene
   * If list contains an even number of scenes, the lower of the two is selected for the middle scene
3. Save list of selected tiles to disk

## Download the selected tiles from the LPDAAC cloud endpoint

1. Loop through the selected tiles list and download the set of bands for each scene. The image files are stored to a directory named <tile-name_tile-date>.
2. A list of file details is also saved to file for each processed scene that contains the following details:
   * Tilename
   * timestamp download
   * Directory file path
   * Filename

## Reproject each tile based on the CDL projection

Loop through each band to reproject and save file with the `.reproject.tif` suffix

## Import saved dataframe files and preparing tile chipping process

1. Import the tile chip df files and reinstiate as a dataframe
2. Create a unique list of scenes that need to be clipped
3. Set-up the CDL reclass properties in order to reclass chips
4. Run the `Process Chip` function to clip scenes for all tiles.
   * The bands for each scene is processed and clipped.
   * Resultant clip is saved to the chips folder, grouped by Chip ID
5. Chip details are saved to a file to validation

## Filter out resultant chipped scenes

Review the chip details by selecting a set of scenes that do not contain any `NA Pixels`. The resultant selection is then copied to the `chips filtered` directory.

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