# Configuration Properties Summary

## Custom Dataset Parameters

- **data_dir**: Specifies the directory where the data is stored. Default is `./data`.
- **train_dataset_name**: Name of the training dataset. Default is `training_data`.

## User Defined AOI (Area of Interest)

- **chip_payload_filename**: Filename of the GeoJSON file containing the chip payload for the area of interest. Default is `midwest_crops_chip_payload.geojson`.

## ipfs-stac Parameters

- **ipfs_stac_params**: Configuration parameters for using the `ipfs-stac` library to access a STAC endpoint with CID support.
  - **stac_endpoint**: URL of the STAC endpoint. Default is `https://stac.easierdata.info`.
  - **api_port**: Port for the API. Leave blank to use the default value.
  - **gateway_port**: Port for the gateway. Leave blank to use the default value.
  - **local_gateway**: Local gateway address. Leave blank to use the default value.

## Training Dataset Parameters

The following parameters define the criteria for creating the training dataset. It's not recommended to change these values unless you are familiar with the data and the requirements for the training dataset. The default values are based on the defined criteria that's defined by Clark CGA's [crop classification training data](https://github.com/ClarkCGA/multi-temporal-crop-classification-training-data) repository.

- **cloud_cover_threshold**: The cloud cover criteria for % total cloud coverage in the entire image. Default is `5`.

## Coordinate Reference Systems

- **geographic_crs**: Default coordinate reference system for geographic transformations. Default is `"EPSG:4326"`.
- **projected_crs**: Default coordinate reference system for projected transformations. Default is `"EPSG:5070"`, which CDL dataset uses.

## Crop Classification Mapping

- **class_mapping**: Classification values for the crop type classes.
  - `0`: No Data
  - `1`: Natural Vegetation
  - `2`: Forest
  - `3`: Corn
  - `4`: Soybeans
  - `5`: Wetlands
  - `6`: Developed/Barren
  - `7`: Open Water
  - `8`: Winter Wheat
  - `9`: Alfalfa
  - `10`: Fallow/Idle Cropland
  - `11`: Cotton
  - `12`: Sorghum
  - `13`: Other

> Note: The classification values are based on the CDL dataset, containing ~200 values, which were reclassified into 13 classes.

## Selected HLS Bands

- **selected_hls_bands**: Selected bands from the Sentinel-2 satellite imagery that will be used to create the training dataset. Default is `["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]`.
