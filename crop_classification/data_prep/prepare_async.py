from datetime import datetime
import json
import aiohttp
import asyncio
from pystac_client import Client
import tqdm
import yaml
from pathlib import Path
import pandas as pd


AUTH_SELECTION = "token"
concurrency_limit = 50
BANDS_OF_INTEREST = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]
CLOUD_THRES = 5


def get_earthdata_auth(auth_type: str = ["netrc", "token"]):  # -> requests.Session:
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


def get_tile_metadata(item_collection, bb_chip_payload):
    ### Querying tile links based on geometry of chips
    STAC_URL = "https://cmr.earthdata.nasa.gov/stac"
    catalog = Client.open(f"{STAC_URL}/LPCLOUD/")
    tiles = item_collection.tile.unique().tolist()
    for tile in tiles:
        chip_df_filt = item_collection.loc[item_collection.tile == tile]
        first_chip_id = chip_df_filt.chip_id.iloc[0]
        chip_feature = [
            feature
            for feature in bb_chip_payload["features"]
            if feature.get("properties", {}).get("id") == first_chip_id
        ]
        aoi = chip_feature[0]["geometry"]

        search = catalog.search(
            collections=["HLSS30.v2.0"],
            intersects=aoi,
            datetime="2022-03-01/2022-09-30",
        )
        print(f"Tile: {search.matched()}")

    num_results = search.matched()
    print(f"Number of results: {num_results}")
    search_results = search.item_collection()
    return search_results


def parse_content(search_result_json):

    # Create a dictionary to store the tile details
    tile_details = {}
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


def query_tiles_based_on_bounding_box(selected_chip_id, chips_bbox):
    # Building STAC url to query tiles
    STAC_URL = "https://cmr.earthdata.nasa.gov/stac"
    catalog = Client.open(f"{STAC_URL}/LPCLOUD/")
    chip_feature = [
        feature
        for feature in chips_bbox["features"]
        if feature.get("properties", {}).get("id") == selected_chip_id
    ]
    aoi = chip_feature[0]["geometry"]

    search = catalog.search(
        collections=["HLSS30.v2.0"],
        intersects=aoi,
        datetime="2022-03-01/2022-09-30",
    )
    print(f"Number of tiles found in query: {search.matched()}")
    return search.item_collection()


async def crawl_results(search_results, tile_id):

    all_results = []
    # Get the Earthdata authentication token
    netrc_auth, headers_auth = get_earthdata_auth(auth_type=AUTH_SELECTION)
    tile_iter = 0
    # tile_name = "T" + tile_id
    async with aiohttp.ClientSession(auth=netrc_auth, headers=headers_auth) as session:
        # Create a semaphore with X as the maximum number of concurrent tasks
        sem = asyncio.Semaphore(concurrency_limit)
        tasks = []

        # Loop through each result page and grab the metadata page that will be parsed
        for search_result in search_results:
            # Assuming `links` is your list of dictionaries
            metadata_page = [
                link.href
                for link in search_result.links
                if link.href.endswith(".umm_json")
            ][0]
            tasks.append(process_page(sem, session, metadata_page))

        print(f"Hold tight! Parsing content from {len(tasks)} pages.")
        for i in tqdm.tqdm(
            range(0, len(tasks), sem._value),
            desc=f"({tile_iter}/{len(search_results)})",
        ):
            chunk = tasks[i : i + sem._value]
            chunk_results = await asyncio.gather(*chunk)
            all_results.extend(chunk_results)

        return all_results

    # print(f"Finished processing {len(all_results)} items.")
    # print(f"First item: {all_results[0]}")


async def process_page(sem, session, page_url):
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
    x = []
    with open(
        r"C:\github\client_projects\umd\multi-temporal-crop-classification-training-data\config\default_config.yaml",
        "r",
    ) as file:
        config = yaml.safe_load(file)

    DATA_DIR = config["data_dir"]
    # TRAINING_DATASET_DIR_NAME = config["train_dataset_name"]

    # MISC_DIR = Path(DATA_DIR) / "misc"
    # REQUIRED_SOURCES = Path(DATA_DIR) / "required_sources"

    CHIPS_DF_PKL = Path(DATA_DIR) / "chips_df.pkl"
    BB_CHIP_PAYLOAD = Path(DATA_DIR) / "required_sources" / "bb_chip_payload.geojson"

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
    print(f"There are a total of {len(tiles)} tiles")

    for tile in tiles:
        # Grab details about the current tile
        chip_df_filt = chip_df.loc[chip_df.tile == tile]
        first_chip_id = chip_df_filt.chip_id.iloc[0]
        query_results = query_tiles_based_on_bounding_box(first_chip_id, chips_bbox)

        try:
            x.append(asyncio.run(crawl_results(query_results, tile)))
        except Exception as e:
            print(f"Failed to process collection: {CHIPS_DF_PKL}. Reason: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
