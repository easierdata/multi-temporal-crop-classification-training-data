import os
import sys
import numpy as np
from typing import Any, Dict, List, Tuple
from datetime import datetime
import json
import aiohttp
import asyncio
from pystac_client import Client
import fiona
import geopandas
import requests
import tqdm
from pathlib import Path
import pandas as pd
from shapely.geometry import Point
import concurrent.futures
import multiprocessing

# The code cell is used to add the src directory to the Python path, making
# it possible to import modules from that directory.

module_path = module_path = os.path.abspath(Path(__file__).parent.parent.resolve())
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass

# Set the authentication selection. Options are 'netrc' or 'token'
AUTH_SELECTION = "token"

# Identify the number of CPUs on the system to set the concurrency limit
NUM_CPUS = multiprocessing.cpu_count()
if NUM_CPUS > 24:
    NUM_CPUS = 24

# Set the concurrency limit for running asynchronous tasks
CONCURRENCY_LIMIT = 50

# global variables for the bands of interest and cloud threshold
BANDS_OF_INTEREST = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]
CLOUD_THRES = 5

# Number of chips to select. Not required to modify but useful for testing
# This variable can be referenced to only select a subset of chips from the BB_CHIP_PAYLOAD file when identifying intersecting tiles.
# If the value is set to 0, all chips will be used.
SELECTION_SUBSET = 0

# Default coordinate reference systems for transformations between geographic and projected
CRS_GEO = "EPSG:4326"
CRS_PROJ = "EPSG:5070"


def load_chips() -> List[Dict[str, Any]]:
    """
    Load chips from a JSON file.  The number of chips to load. If greater than 0, return the number of specified chips from the
    starting index, otherwise return all chips.

    Returns:
        List[Dict[str, Any]]: A list of chips, where each chip is represented as a dictionary.

    """
    with open(BB_CHIP_PAYLOAD, "r") as file:
        chips = json.load(file)
    if SELECTION_SUBSET > 0:
        return chips["features"][:SELECTION_SUBSET]
    else:
        return chips["features"]


def save_chip_ids(chip_ids: List[str]) -> None:
    """Save chip IDs to a JSON file."""
    with open(CHIPS_ID_JSON, "w") as f:
        json.dump(chip_ids, f, indent=2)


def load_sentinel_tile_data() -> geopandas.GeoDataFrame:
    """Read and process tile data from a KML file."""
    fiona.drvsupport.supported_drivers["KML"] = "rw"
    kml_tile_df = geopandas.read_file(HLS_KML_FILE, driver="KML")
    return kml_tile_df


def find_closest_tile(
    chip_coords: geopandas.GeoSeries,
    tile_coords: geopandas.GeoSeries,
    tile_name: List[str],
) -> List[str]:
    """Vectorized operation to find the closest tile for each chip.

    Args:
        chip_coords (geopandas.GeoSeries): The coordinates of the chips.
        tile_coords (geopandas.GeoSeries): The coordinates of the tiles.
        tile_name (List[str]): The names of the tiles.

    Returns:
        List[str]: The names of the closest tile for each chip.
    """
    # Extract the x and y coordinates for the chips and tiles
    chip_x = chip_coords.geometry.x.values
    chip_y = chip_coords.geometry.y.values
    tile_x = tile_coords.geometry.x.values
    tile_y = tile_coords.geometry.y.values

    # Perform element-wise subtraction between the arrays of different shapes.  This allows numpy to broadcast chip coordinates
    # against each tile coordinate.
    distances = (tile_x - chip_x[:, np.newaxis]) ** 2 + (
        tile_y - chip_y[:, np.newaxis]
    ) ** 2
    # Find the index of the tile with the minimum distance for each chip and return the tile name via the index.
    closest_indices = distances.argmin(axis=1)
    return tile_name[closest_indices]


def select_intersecting_features(
    chips: pd.DataFrame, tile_src: geopandas.GeoDataFrame
) -> pd.DataFrame:
    """
    Identify the closest tile to each chip. Prepare a DataFrame containing intersecting tiles for each chip.
    Identifying the closest tile to each chip is done by calculating the Euclidean distance between the chip and tile
    which requires to convert the coordinates to a projected coordinate system for accurate distance calculations.

    Args:
        chips (pd.DataFrame): A pandas DataFrame representing the chips.
        tile_src (geopandas.GeoDataFrame): A GeoDataFrame representing the tile source.

    Returns:
        pd.DataFrame: A DataFrame containing the chip id, chip x/y centroid coordinates and the tile name. Coordinates are in
        a geographic coordinate system (EPSG:4326).
    """

    # Convert the chip coordinates to a GeoSeries of Points. The CRS is assumed to be EPSG:4326 (WGS 84) since the the
    # chip bounding box coordinates are loaded in from `BB_CHIP_PAYLOAD.geojson`
    chip_points = geopandas.GeoSeries(
        [Point(x, y) for x, y in zip(chips["chip_x"].values, chips["chip_y"].values)],
        crs=CRS_GEO,
    )

    # Convert the tile coordinates to a GeoSeries of Points. The CRS is assumed to be EPSG:4326 (WGS 84) since the tile
    # coordinates are loaded in from `HLS_Sentinel2_Tiles.kml`
    tile_points = geopandas.GeoSeries(
        [
            Point(x, y)
            for x, y in zip(
                tile_src.geometry.centroid.x.values, tile_src.geometry.centroid.y.values
            )
        ],
        crs=CRS_GEO,
    )

    # Convert the points to a projected coordinate system for accurate distance calculations
    chip_points_reprojected = chip_points.to_crs(CRS_PROJ)
    tile_points_reprojected = tile_points.to_crs(CRS_PROJ)

    # Save the closest intersecting tile to each chip. The tile name is used to identify the tile.
    chips_df = chips.copy()
    chips_df["tile"] = find_closest_tile(
        chip_points_reprojected, tile_points_reprojected, tile_src["Name"].values
    )
    return chips_df


def select_tiles() -> pd.DataFrame:
    """
    Selects tiles that intersect with the chips.

    Returns:
        pd.DataFrame: A DataFrame containing the selected tiles with columns 'chip_id', 'chip_x', and 'chip_y'.
    """
    sample_chips = load_chips()

    # Save the chip IDs to a JSON file that will be later used in process_chips.py
    save_chip_ids([chip["properties"]["id"] for chip in sample_chips])

    # Convert the list of dictionaries to a DataFrame
    sample_chips_df = pd.DataFrame(
        {
            "chip_id": [item["properties"]["id"] for item in sample_chips],
            "chip_x": [item["properties"]["center"][0] for item in sample_chips],
            "chip_y": [item["properties"]["center"][1] for item in sample_chips],
        }
    )
    tile_grid_df = load_sentinel_tile_data()
    chip_df = select_intersecting_features(sample_chips_df, tile_grid_df)
    chip_df.to_pickle(CHIPS_DF_PKL)
    return chip_df


def get_earthdata_auth(auth_type: str = ["netrc", "token"]) -> requests.Session:
    """
    Get the Earthdata authentication token.

    Args:
        type (str): The type of authentication to
            use. Options are 'netrc' or 'Bearer'.

    Returns:
    """
    # Set default values for different auth types
    netrc_cred = None
    headers = None

    import earthaccess

    auth = earthaccess.login(strategy="netrc")

    if auth_type == "netrc":
        netrc_cred = aiohttp.BasicAuth(auth.username, auth.password)
    elif auth_type == "token":
        headers = {"Authorization": f"Bearer {auth.token['access_token']}"}
    else:
        raise ValueError("Invalid authentication type. Use 'netrc' or Bearer Token")

    return netrc_cred, headers


def parse_content(search_result_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the content of a search result JSON and extract relevant information.

    Args:
        search_result_json (Dict[str, Any]): The search result JSON to parse.

    Returns:
        Dict[str, Any]: A dictionary containing the extracted tile details.

    """
    # Create a dictionary to store the tile details
    tile_details: Dict[str, Any] = {}
    tile_details["title_id"] = search_result_json["GranuleUR"]
    tile_details["tile_id"] = tile_details["title_id"].split(".")[2]

    # Extract additional attributes from the search result
    additional_attributes = search_result_json["AdditionalAttributes"]
    attrs = [
        ("cloud_cover", "CLOUD_COVERAGE"),
        ("spatial_cover", "SPATIAL_COVERAGE"),
        ("date", "SENSING_TIME"),
    ]
    for attr in attrs:
        tile_details[attr[0]] = [
            i for i in additional_attributes if i["Name"] == attr[1]
        ][0]["Values"][0]

    tile_details["date"] = datetime.strptime(
        tile_details["date"].split("T")[0], "%Y-%m-%d"
    )
    # Extract the related URLs from the search result
    https_urls = []
    s3_urls = []

    for item in search_result_json["RelatedUrls"]:
        url = item["URL"]
        suffix = url.split(".")[-2]  # Get the suffix after the second to last period
        if url.startswith("https"):
            https_urls.append({"Suffix": suffix, "URL": url})
        elif url.startswith("s3"):
            s3_urls.append({"Suffix": suffix, "URL": url})

    # Only store the urls for the bands of interest
    tile_details["http_links"] = {
        url["Suffix"]: url["URL"]
        for url in https_urls
        if url["Suffix"] in BANDS_OF_INTEREST
    }
    tile_details["s3_links"] = {
        url["Suffix"]: url["URL"]
        for url in s3_urls
        if url["Suffix"] in BANDS_OF_INTEREST
    }

    return tile_details


def run_stac_search(chips_bbox: Tuple[float, float, float, float]) -> Dict[str, Dict]:
    """
    Perform a parallel search for STAC tiles based on and bounding box.

    Args:
        chips_bbox: The bounding box coordinates (minx, miny, maxx, maxy) for the chips.

    Returns:
        A dictionary containing the search results for each tile.
    """
    print(
        "Running STAC search to identify all tiles that intersect the chip bounding box."
    )

    chip_df = select_tiles()

    tiles = chip_df.tile.unique().tolist()
    chip_payload = create_chip_payload(chip_df, chips_bbox)

    tile_search_results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each tile query as a separate thread
        future_to_tile = {
            executor.submit(
                query_tiles_based_on_bounding_box,
                chip_payload[tile],
            ): tile
            for tile in tiles
        }

        # Retrieve results as they become available
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_tile), total=len(future_to_tile)
        ):
            tile = future_to_tile[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Tile {tile} generated an exception: {exc}")
            else:
                tile_search_results[tile] = result

    return tile_search_results


def create_chip_payload(
    chip_df: pd.DataFrame, chips_bbox: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a payload dictionary containing chip details for each unique tile.

    Args:
        chip_df (pd.DataFrame): DataFrame containing chip details.
        chips_bbox (Dict[str, Any]): Dictionary containing chip bounding box information.

    Returns:
        Dict[str, Any]: Payload dictionary with tile as key and chip bounding box as value.
    """
    payload = {}

    tiles = chip_df.tile.unique().tolist()
    for tile in tiles:
        chip_df_filt = chip_df.loc[chip_df.tile == tile]
        first_chip_id = chip_df_filt.chip_id.iloc[0]
        chip_bbox_feature = [
            feature
            for feature in chips_bbox["features"]
            if feature.get("properties", {}).get("id") == first_chip_id
        ]
        payload[tile] = chip_bbox_feature[0]["geometry"]
    return payload


def query_tiles_based_on_bounding_box(chip_bbox: List[float]) -> List[dict]:
    """
    Queries tiles based on a given bounding box.

    Args:
        chip_bbox (List[float]): The bounding box coordinates in the format [minx, miny, maxx, maxy].

    Returns:
        List[dict]: A list of tiles matching the query.

    Raises:
        None

    Examples:
        >>> bbox = [10.0, 20.0, 30.0, 40.0]
        >>> query_tiles_based_on_bounding_box(bbox)
        [{'id': 'tile1', 'name': 'Tile 1'}, {'id': 'tile2', 'name': 'Tile 2'}]
    """
    # Building STAC url to query tiles
    STAC_URL = "https://cmr.earthdata.nasa.gov/stac"
    catalog = Client.open(f"{STAC_URL}/LPCLOUD/")

    search = catalog.search(
        collections=["HLSS30.v2.0"],
        intersects=chip_bbox,
        datetime="2022-03-01/2022-09-30",
    )

    return search.item_collection()


async def crawl_results(search_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Crawls the search results and retrieves the metadata pages for parsing.

    Args:
        search_results (Dict[str, Dict]): A list of search results.

    Returns:
        List[Result]: A list of all the results.

    """
    all_results = []
    # Get the Earthdata authentication token
    netrc_auth, headers_auth = get_earthdata_auth(auth_type=AUTH_SELECTION)

    async with aiohttp.ClientSession(auth=netrc_auth, headers=headers_auth) as session:
        # Create a semaphore with X as the maximum number of concurrent tasks
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
        tasks = []

        # Loop through each result page and grab the metadata page that will be parsed
        for search_result in search_results:
            metadata_page = [
                link.href
                for link in search_result.links
                if link.href.endswith(".umm_json")
            ][0]
            tasks.append(process_page(sem, session, metadata_page))

        for i in tqdm.tqdm(
            range(0, len(tasks), sem._value),
            desc=f"({len(search_results)} pages)",
        ):
            chunk = tasks[i : i + sem._value]
            chunk_results = await asyncio.gather(*chunk)
            # Filter out None values before extending all_results
            chunk_results = [result for result in chunk_results if result is not None]
            # Flatten the list of lists into a single list of dictionaries
            chunk_results = [item for sublist in chunk_results for item in sublist]
            all_results.extend(chunk_results)

    return all_results


async def process_page(
    sem: asyncio.Semaphore, session: aiohttp.ClientSession, page_url: str
) -> Dict[str, Any]:
    """
    Process a page asynchronously.

    Args:
        sem (asyncio.Semaphore): Semaphore to limit the number of concurrent requests.
        session (aiohttp.ClientSession): HTTP session for making requests.
        page_url (str): URL of the page to process.

    Returns:
        dict: Parsed details of the page.

    """
    async with sem:
        parsed_details = []
        page_response_json = await fetch_page(session, page_url)

        if page_response_json:
            cloud_coverage = [
                i
                for i in page_response_json["AdditionalAttributes"]
                if i["Name"] == "CLOUD_COVERAGE"
            ][0]["Values"][0]
            if int(cloud_coverage) <= CLOUD_THRES:
                parsed_details.append(parse_content(page_response_json))
            else:
                # print(
                #     f"Skipping tile {page_response_json['GranuleUR']} due to high cloud coverage: {cloud_coverage}%"
                # )
                return None

    return parsed_details


async def fetch_page(session: aiohttp.ClientSession, url: str) -> None | str:
    """
    Fetches the content of a web page using an async HTTP GET request.

    Args:
        session (aiohttp.ClientSession): The aiohttp client session to use for the request.
        url (str): The URL of the web page to fetch.

    Returns:
        str: The content of the web page as a string, or None if the request was unsuccessful.
    """
    async with session.get(url) as response:
        if response.status != 200:
            # check if response.real_url is valid if the response was was not 200
            print(
                f"Failed to retrieve content from page: {url}. Status code: {response.status}"
            )
            # Retry session with the real_url value
            return None
        else:
            return await response.json()


### Filter based on spatial coverage
def spatial_filtering(dataframe):
    """
    Using spatial coverage percentage to filter chips

    Args:
        dataframe: A pandas dataframe that generated previously
    """
    cover_list = [100, 90, 80, 70, 60, 50]
    tile_list_ft = []
    tile_list = dataframe.tile_id.unique().tolist()

    for tile in tqdm.tqdm(tile_list):
        temp_df = dataframe[dataframe.tile_id == tile]
        for cover_pct in cover_list:

            temp_df_filtered = temp_df[temp_df.spatial_cover.astype(int) >= cover_pct]
            if len(temp_df_filtered) >= 3:
                for i in range(len(temp_df_filtered)):
                    tile_list_ft.append(temp_df_filtered.iloc[i])
                break

    tile_df_filtered = pd.DataFrame(tile_list_ft)
    return tile_df_filtered


def select_scenes(dataframe):
    """
    Selecting best spatial covered scenes based on timesteps

    Args:
        dataframe: A pandas dataframe that generated previously
    """
    select_tiles = []
    tile_list = dataframe.tile_id.unique().tolist()

    for tile in tqdm.tqdm(tile_list):
        temp_df = (
            dataframe[dataframe.tile_id == tile]
            .sort_values("date")
            .reset_index(drop=True)
        )
        select_tiles.extend(
            [temp_df.iloc[0], temp_df.iloc[len(temp_df) // 2], temp_df.iloc[-1]]
        )

    return pd.DataFrame(select_tiles).reset_index(drop=True)


def main():

    # Query the tiles based on the bounding box of the chips
    with open(BB_CHIP_PAYLOAD, "r") as f:
        chips_bbox = json.load(f)
    search_results = run_stac_search(chips_bbox)

    # Process the search results by looping through each tile that intersects the chips
    try:
        tiles_intersecting_chips = pd.read_pickle(CHIPS_DF_PKL)
        tiles = tiles_intersecting_chips.tile.unique().tolist()
    except FileNotFoundError:
        print(f"File not found: {CHIPS_DF_PKL}")
        return
    print("Processing the search results...")
    crawled_results = []
    try:
        for tile in tiles:

            # Continue the loop if the value for the key, tile is empty
            if not search_results[tile]:
                print(f"No search results for tile: {tile}")
                continue

            crawled_results.extend(asyncio.run(crawl_results(search_results[tile])))
    except Exception as e:
        print(f"Failed to process collection: {CHIPS_DF_PKL}. Reason: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Check if crawled_results is not empty
        if crawled_results:
            # Write the crawled results to disc
            crawled_results_df = pd.DataFrame(crawled_results)
            crawled_results_df.to_pickle(TILES_DF_PKL)
            crawled_results_df.to_csv(TILES_DF_CSV, index=False)

            cover_df = spatial_filtering(crawled_results_df)
            selected_tiles = select_scenes(cover_df)
            # Save for later uses
            selected_tiles.to_pickle(SELECTED_TILES_PKL)
            selected_tiles.to_csv(SELECTED_TILES_CSV, index=False)

        else:
            print("No results to write to disc")


if __name__ == "__main__":
    main()
