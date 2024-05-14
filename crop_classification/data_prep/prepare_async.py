import os
import sys
from typing import Any, Dict, List, Tuple
from datetime import datetime
import json
import aiohttp
import asyncio
from pystac_client import Client
import requests
import tqdm
from pathlib import Path
import pandas as pd
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
    tile_details["http_links"] = [
        url for url in https_urls if url["Suffix"] in BANDS_OF_INTEREST
    ]
    tile_details["s3_links"] = [
        url for url in s3_urls if url["Suffix"] in BANDS_OF_INTEREST
    ]

    return tile_details


def run_stac_search(
    chip_df: pd.DataFrame, chips_bbox: Tuple[float, float, float, float]
) -> Dict[str, Dict]:
    """
    Perform a parallel search for STAC tiles based on the given chip dataframe and bounding box.

    Args:
        chip_df: The chip dataframe containing information about the tiles.
        chips_bbox: The bounding box coordinates (minx, miny, maxx, maxy) for the chips.

    Returns:
        A dictionary containing the search results for each tile.
    """
    print(
        "Running STAC search to identify all tiles that intersect the chip bounding box."
    )

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
        parsed_details = {}
        page_response_json = await fetch_page(session, page_url)

        if page_response_json:
            cloud_coverage = [
                i
                for i in page_response_json["AdditionalAttributes"]
                if i["Name"] == "CLOUD_COVERAGE"
            ][0]["Values"][0]
            if int(cloud_coverage) <= CLOUD_THRES:
                parsed_details = parse_content(page_response_json)

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


def main():
    crawled_results = []
    try:
        # Load the chips_df.pkl file
        chip_df = pd.read_pickle(CHIPS_DF_PKL)
        with open(BB_CHIP_PAYLOAD, "r") as f:
            chips_bbox = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {CHIPS_DF_PKL}")
        return

    # Get the unique tiles in the item collection
    tiles = chip_df.tile.unique().tolist()
    print(f"There are a total of {len(tiles)} tiles that will be processed.")

    # Query the tiles based on the bounding box of the chips
    search_results = run_stac_search(chip_df, chips_bbox)

    print("Processing the search results...")
    for tile in tiles:
        try:
            crawled_results.append(asyncio.run(crawl_results(search_results[tile])))
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
            else:
                print("No results to write to disc")


if __name__ == "__main__":
    main()
