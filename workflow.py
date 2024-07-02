#!/usr/bin/env python
# coding: utf-8

# # Pipeline to query, download and chip HLS and CDL layers

# In[6]:


import geopandas
import json
import xarray
import rasterio
import rioxarray
import os
import fiona
import urllib.request as urlreq
import pandas as pd
import numpy as np
import requests
import xmltodict
import shutil
import datetime
#import boto3
import pyproj
import multiprocessing as mp
from urllib3 import PoolManager
from urllib3.util import Retry

from pystac_client import Client 
from collections import defaultdict
from glob import glob
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
from tqdm import tqdm
from netrc import netrc
from platform import system
from getpass import getpass
from rasterio.session import AWSSession
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ### Setting folder paths and file paths

# In[5]:


cloud_thres = 5 # percent cloud cover for tile level query

root_path = "/data/"
req_path = "/cdl_training_data/data/"
extra_files = "/data/requirements/"

## file paths
chip_file =  req_path + "chip_bbox.geojson"
chipping_json = req_path + "chip_bbox_5070.geojson"
chip_csv = req_path + "chip_tracker.csv"
kml_file = extra_files + 'sentinel_tile_grid.kml'
# tile_tracker_csv = req_path + "tile_tracker.csv"
cdl_file = extra_files + "2022_30m_cdls.tif"
cdl_reclass_csv = req_path + "cdl_total_dst.csv"

## folder paths
chip_dir = root_path + 'chips/'
tile_dir = root_path + 'tiles/'
chip_dir_filt = root_path + 'chips_filtered/'
chip_fmask_dir = root_path + 'chips_fmask/'

## Create necessary folders
if not os.path.exists(chip_dir):
   os.makedirs(chip_dir)
if not os.path.exists(tile_dir):
   os.makedirs(tile_dir)
if not os.path.exists(chip_dir_filt):
   os.makedirs(chip_dir_filt)
if not os.path.exists(chip_fmask_dir):
   os.makedirs(chip_fmask_dir)


# # ### Data Processing

# # In[33]:


# # Loading chips bounding boxes from geojson
# with open(chip_file, "r") as file:
#     chips = json.load(file)
#     # print(chips)

# # Create lists about chip information to find tiles corresponding to it later
# chip_ids = []
# chip_x = []
# chip_y = []
# for item in chips['features']:
#     chip_ids.append(item['properties']['id'])
#     chip_x.append(item['properties']['center'][0])
#     chip_y.append(item['properties']['center'][1])


# # In[7]:


# with open("/data/chip_ids.json", "w") as f:
#     json.dump(chip_ids, f, indent=2)


# # In[8]:


# # Read in sentinel kml file
# fiona.drvsupport.supported_drivers['KML'] = 'rw'
# tile_src = geopandas.read_file(kml_file, driver='KML')

# # Create table containing information about sentinel tiles
# tile_name = []
# tile_x = []
# tile_y = []
# for tile_ind in range(tile_src.shape[0]):
#     tile_name.append(tile_src.iloc[tile_ind].Name)
#     tile_x.append(tile_src.iloc[tile_ind].geometry.centroid.x)
#     tile_y.append(tile_src.iloc[tile_ind].geometry.centroid.y)
# tile_name = np.array(tile_name)
# tile_x = np.array(tile_x)
# tile_y = np.array(tile_y)
# tile_src = pd.concat([tile_src, tile_src.bounds], axis = 1)


# # In[9]:


# def find_tile(x,y):
#     """
#     Identify closest tile
#     """
    
#     s = (tile_x - x)**2+(tile_y - y)**2
#     tname = tile_name[np.argmin(s)]
#     return(tname)


# # In[10]:


# chip_df = pd.DataFrame({"chip_id" : chip_ids, "chip_x" : chip_x, "chip_y" : chip_y})
# chip_df['tile'] = chip_df.apply(lambda row : find_tile(row['chip_x'], row['chip_y']), axis = 1)
# chip_df.tail(5)


# # In[11]:


# # Save dataframe to csv for later uses
# chip_df.to_csv(req_path + "chip_df.csv", index=False)


# # In[30]:


# chip_df = pd.read_csv(req_path + "chip_df.csv")


# # In[31]:


# tiles = chip_df.tile.unique().tolist()
# tiles[0:5]


# # ### Querying tile links based on geometry of chips

# # In[28]:


# STAC_URL = 'https://cmr.earthdata.nasa.gov/stac'
# catalog = Client.open(f'{STAC_URL}/LPCLOUD/')

# retries = Retry(connect=5, read=2, redirect=5)
# http = PoolManager(retries=retries)
# # In[34]:

# from time import sleep

# def search_item_collection(search):
#     max_retries = 3
#     retries = 0
    
#     while retries < max_retries:
#         try:
#             item_collection = search.item_collection()
#             return item_collection
#         except pystac_client.exceptions.APIError as e:
#             retries += 1
#             print(f"Request failed with error: {e}. Retrying in 3 seconds...")
#             sleep(3)  # Wait for 5 seconds before retrying
    
#     # If all retries fail, raise the last exception
#     raise e


# # Tile-level 5% cloud cover threshold query
# tile_list = []
# print(f"There are a total of {len(tiles)} tiles")
# tile_iter = 0
# for current_tile in tiles:

#     chip_df_filt = chip_df.loc[chip_df.tile == current_tile]#.reset_index()
#     first_chip_id = chip_df_filt.chip_id.iloc[0]
#     first_chip_index_in_json = chip_ids.index(first_chip_id)
#     roi = chips['features'][first_chip_index_in_json]['geometry']

#     search = catalog.search(
#         collections = ['HLSS30.v2.0'],
#         intersects = roi,
#         datetime = '2022-03-01/2022-09-30',
#     ) 
    
#     num_results = search.matched()
#     item_collection = search_item_collection(search)
    
#     tile_name = "T" + current_tile
#     iter_items = 0
#     for i in tqdm(item_collection ,desc=f"({tile_iter}/{len(tiles)})"):
#         if i.id.split('.')[2] == tile_name:
#             if i.properties['eo:cloud_cover'] <= cloud_thres:
#                 #response = requests.get(i.assets['metadata'].href)
#                 response = http.request("GET", i.assets['metadata'].href)
#                 if response.status == 200:
#                     temp_xml = response.data #text
#                     temp_xml = xmltodict.parse(temp_xml)
#                     temp_dict = {"tile_id": tile_name, "cloud_cover": i.properties['eo:cloud_cover'],
#                                  "date": datetime.datetime.strptime(i.properties['datetime'].split('T')[0], "%Y-%m-%d"), 
#                                  "spatial_cover": int(temp_xml['Granule']['AdditionalAttributes']['AdditionalAttribute'][3]['Values']['Value']),
#                                  "http_links": {"B02": i.assets['B02'].href, "B03": i.assets['B03'].href, "B04": i.assets['B04'].href,  "B8A": i.assets['B8A'].href,
#                                                 "B11": i.assets['B11'].href, "B12": i.assets['B12'].href, "Fmask": i.assets['Fmask'].href},
#                                 "s3_links": {"B02": i.assets['B02'].href.replace('https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/', 's3:/'), 
#                                              "B03": i.assets['B03'].href.replace('https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/', 's3:/'), 
#                                              "B04": i.assets['B04'].href.replace('https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/', 's3:/'), 
#                                              "B8A": i.assets['B8A'].href.replace('https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/', 's3:/'),
#                                              "B11": i.assets['B11'].href.replace('https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/', 's3:/'),
#                                              "B12": i.assets['B12'].href.replace('https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/', 's3:/'),
#                                              "Fmask": i.assets['Fmask'].href.replace('https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/', 's3:/')}}
#                     tile_list.append(temp_dict)
#                     iter_items += 1
#                 else: 
#                     assert False, f"Failed to fetch XML from {i.assets['metadata'].href}. Error code: {response.status_code}"
            
#     tile_iter += 1
    
# tile_df = pd.DataFrame(tile_list)


# # In[35]:


# # Save to csv for later uses
# tile_df.to_csv(req_path + "tile_df.csv", index=False)


# # In[36]:


# tile_df = pd.read_csv(req_path + "tile_df.csv")


# # In[37]:


# tile_df.head()


# # ### Filtering based on spatial coverage of the tiles we gathered earlier

# # In[38]:


# def spatial_filtering (dataframe):
#     """
#         Using spatial coverage percentage to filter chips

#         Args:
#             dataframe: A pandas dataframe that generated previously
#     """
#     cover_list = [100, 90, 80, 70, 60, 50]
#     tile_list_ft = []
#     tile_list = dataframe.tile_id.unique().tolist()
    
#     for tile in tqdm(tile_list):
#         temp_df = dataframe[dataframe.tile_id == tile]
#         for cover_pct in cover_list:
            
#             temp_df_filtered = temp_df[temp_df.spatial_cover >= cover_pct]
#             if len(temp_df_filtered) >= 3:
#                 for i in range(len(temp_df_filtered)):
#                     tile_list_ft.append(temp_df_filtered.iloc[i])
#                 break
    
#     tile_df_filtered = pd.DataFrame(tile_list_ft)
#     return tile_df_filtered


# # In[39]:


# cover_df = spatial_filtering(tile_df)


# # In[40]:


# def select_scenes(dataframe):
#     """
#         Selecting best spatial covered scenes based on timesteps

#         Args:
#             dataframe: A pandas dataframe that generated previously
#     """
#     select_tiles = []
#     tile_list = dataframe.tile_id.unique().tolist()

#     for tile in tqdm(tile_list):
#         temp_df = dataframe[dataframe.tile_id == tile].sort_values('date').reset_index(drop=True)
#         select_tiles.extend([temp_df.iloc[0], temp_df.iloc[len(temp_df) // 2], temp_df.iloc[-1]])

#     return pd.DataFrame(select_tiles).reset_index(drop=True)


# # In[41]:


# selected_tiles = select_scenes(cover_df)


# # In[42]:


# selected_tiles.head()


# # In[43]:


# # Save to csv for later uses
# selected_tiles.to_csv(req_path + "selected_tiles.csv", index=False)


# # In[23]:


selected_tiles = pd.read_csv(req_path + "selected_tiles.csv")



# # ### Data downloading
# # 
# # Creating netrc file on root for credentials (Run Once each session)

# # In[6]:


# urs = 'urs.earthdata.nasa.gov'    # Earthdata URL endpoint for authentication
# prompts = ['Enter NASA Earthdata Login Username: ',
#            'Enter NASA Earthdata Login Password: ']

# # Determine the OS (Windows machines usually use an '_netrc' file)
# netrc_name = "_netrc" if system()=="Windows" else ".netrc"

# # Determine if netrc file exists, and if so, if it includes NASA Earthdata Login Credentials
# try:
#     netrcDir = os.path.expanduser(f"~/{netrc_name}")
#     netrc(netrcDir).authenticators(urs)[0]

# # Below, create a netrc file and prompt user for NASA Earthdata Login Username and Password
# except FileNotFoundError:
#     homeDir = os.path.expanduser("~")
#     Popen('touch {0}{2} | echo machine {1} >> {0}{2}'.format(homeDir + os.sep, urs, netrc_name), shell=True)
#     Popen('echo login {} >> {}{}'.format(getpass(prompt=prompts[0]), homeDir + os.sep, netrc_name), shell=True)
#     Popen('echo \'password {} \'>> {}{}'.format(getpass(prompt=prompts[1]), homeDir + os.sep, netrc_name), shell=True)
#     # Set restrictive permissions
#     Popen('chmod 0600 {0}{1}'.format(homeDir + os.sep, netrc_name), shell=True)

#     # Determine OS and edit netrc file if it exists but is not set up for NASA Earthdata Login
# except TypeError:
#     homeDir = os.path.expanduser("~")
#     Popen('echo machine {1} >> {0}{2}'.format(homeDir + os.sep, urs, netrc_name), shell=True)
#     Popen('echo login {} >> {}{}'.format(getpass(prompt=prompts[0]), homeDir + os.sep, netrc_name), shell=True)
#     Popen('echo \'password {} \'>> {}{}'.format(getpass(prompt=prompts[1]), homeDir + os.sep, netrc_name), shell=True)


# # ### Getting temporary credentials for NASA's S3 Bucket(Run once every 1 hrs)

# # # In[32]:


# # s3_cred_endpoint = 'https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials'


# # # In[36]:


# # def get_temp_creds():
# #     temp_creds_url = s3_cred_endpoint
# #     return requests.get(temp_creds_url).json()


# # # In[ ]:


# # temp_creds_req = get_temp_creds()

# # boto3_session = boto3.Session(aws_access_key_id=temp_creds_req['accessKeyId'], 
# #                             aws_secret_access_key=temp_creds_req['secretAccessKey'],
# #                             aws_session_token=temp_creds_req['sessionToken'],
# #                             region_name='us-west-2')


# # # In[9]:


# # rio_env = rasterio.Env(AWSSession(boto3_session),
# #                   GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
# #                   GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
# #                   GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'))
# # rio_env.__enter__()


# # # In[ ]:





# # ### Tile downloading (Run the crendentials chunk if connecting error)

# # In[2]:


import earthaccess

earthaccess.login()


# In[4]:


def tile_download(table, from_csv = True):
    """
        Downloading tiles by reading from the metadata information gathered earlier

        Args:
            table: A pandas dataframe that generated previously
            boto3_session: The session that set earlier when getting credentials
            from_csv: If the tile information is from a csv, then True
    """
    
    
    info_list = []

    save_path = []
    bands = ["B02","B03","B04","B8A","B11","B12","Fmask"]
    accept_tiles = np.unique(table.tile_id)
    for tile in tqdm(accept_tiles):
        temp_tb = table[table.tile_id == tile]
        for i in range(3):
            if from_csv:
                bands_dict = json.loads(temp_tb.iloc[i].http_links.replace("'", '"'))
            else:
                bands_dict = temp_tb.iloc[i].http_links
            temp_sav_path = f"/data/tiles/{bands_dict['B02'].split('/')[-2]}/"
            os.makedirs(temp_sav_path, exist_ok=True)
            queue_keys = []
            for band in bands:
                
                temp_key = bands_dict[band]

                if not Path(temp_sav_path).is_file():
                    #print(bands_dict,temp_key,temp_sav_path)
                    queue_keys.append(temp_key)
                    if len(save_path) < 1:
                        save_path.append(temp_sav_path)
                    # break
                    # boto3_session.resource('s3').Bucket('lp-prod-protected').download_file(Key = temp_key, Filename = temp_sav_path)
            temp_dict = {"tile":tile, "timestep":i, "date":temp_tb.iloc[i].date, "save_path":f"/data/tiles/{bands_dict[band].split('/')[-2]}/", "filename":bands_dict["B02"].split('/')[-1].replace(".B02.tif","")}
            info_list.append(temp_dict)
            earthaccess.download(queue_keys, temp_sav_path)
    return pd.DataFrame(info_list)


# In[22]:


selected_tiles


# In[11]:


track_df = tile_download(selected_tiles, from_csv=True)


# In[12]:


track_df.to_csv(req_path + "track_df.csv", index=False)


# In[ ]:


track_df


# ### Using the CDL tif to reproject each HLS scene to CDL projection

# In[14]:


track_df = pd.read_csv(req_path + "track_df.csv")


# In[7]:


def point_transform(coor, src_crs, target_crs=5070):
    proj = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
    projected_coor = proj.transform(coor[0], coor[1])
    return [projected_coor[0], projected_coor[1]]

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# In[19]:


def reproject_hls(tile_path,
                  cdl_ds,
                  target_crs ="EPSG:5070", 
                  remove_original = True, 
                  resampling_method = Resampling.bilinear):
    
    """
    This function receives the path to a specific HLS tile and reproject it to the targeting crs_ds.
    The option of removing the raw HLS tile is provided
    
    Assumptions:
    - tile_path is a full path that end with .tif
    - cdl_ds is a rioxarray dataset that is opened with `cache=False` setting.
    
    
    Inputs:
    - tile_path: The full path to a specific HLS tile
    - target_crs: The crs that you wish to reproject the tile to, default is EPSG 4326
    - remove_original: The option to remove raw HLS tile after reprojecting, default is True
    - resampling_method: The method that rioxarray use to reproject, default is bilinear
    """

    xds = rioxarray.open_rasterio(tile_path)
    half_scene_len = np.abs(np.round((xds.x.max().data - xds.x.min().data) / 2))
    coor_min = point_transform([xds.x.min().data - half_scene_len, xds.y.min().data - half_scene_len], xds.rio.crs)
    coor_max = point_transform([xds.x.max().data + half_scene_len, xds.y.max().data + half_scene_len], xds.rio.crs)
    
    x0 = find_nearest(cdl_ds.x.data, coor_min[0])
    y0 = find_nearest(cdl_ds.y.data, coor_min[1])
    x1 = find_nearest(cdl_ds.x.data, coor_max[0])
    y1 = find_nearest(cdl_ds.y.data, coor_max[1])
    
    cdl_for_reprojection = cdl_ds.rio.slice_xy(x0, y0, x1, y1)
    
    xds_new = xds.rio.reproject_match(cdl_for_reprojection, resampling = resampling_method)

    if remove_original:
        if Path(tile_path).is_file():
            os.remove(tile_path)
        xds_new.rio.to_raster(raster_path = tile_path.replace(".tif", ".reproject.tif"))
    else:
        xds_new.rio.to_raster(raster_path = tile_path.replace(".tif", ".reproject.tif"))


# In[85]:


# Add a quality control to ensure there are three scenes for each tile.
failed_tiles = []
for tile in list(track_df.tile.unique()):
    if len(track_df[track_df.tile == tile]) != 3:
        failed_tiles.append(tile)
if len(failed_tiles) == 0:
    print("All tiles passed the quality test!")
else:
    print(f"Tile {failed_tiles} does not pass the quality test.")


# In[15]:
print('track_df')

track_df["cdl_file"] = cdl_file
track_df.loc[:, "bands"] = '["B02","B03","B04","B8A","B11","B12","Fmask"]'


# In[16]:


print(track_df.head())


# In[29]:


def hls_process(kwargs):

    remove_original = False
    
    save_path = kwargs["save_path"]
    filename= kwargs["filename"]
    bands = json.loads(kwargs["bands"])
    cdl_file = kwargs["cdl_file"]
    
    cdl_ds = rioxarray.open_rasterio(cdl_file, cache=False)
    
    for band in bands:
        tile_path = f"{save_path}{filename}.{band}.tif"
        rep_path = f"{save_path}{filename}.{band}.reproject.tif"
        if not Path(rep_path).is_file() and Path(tile_path).is_file():
            if band == "Fmask":
                reproject_hls(tile_path, cdl_ds, remove_original = remove_original, resampling_method = Resampling.nearest)
            else :
                reproject_hls(tile_path, cdl_ds, remove_original = remove_original) 


# In[30]:


with mp.Pool(processes=8) as pool:
    pool.map(hls_process, track_df.to_dict('records'))


# ### Chipping

# In[31]:


# Getting all saved dataframes and json
chip_df = pd.read_csv(req_path + "chip_df.csv")
with open("/data/chip_ids.json", 'r') as f:
    chip_ids = json.load(f)
track_df = pd.read_csv(req_path + "track_df.csv")
with open(chip_file, "r") as file:
    chips = json.load(file)


# In[32]:


tiles_to_chip = track_df.tile.unique().tolist()
with open(chipping_json, "r") as file_chip:
    chipping_js = json.load(file_chip)


# In[33]:


## set up CDL reclass
cdl_class_df = pd.read_csv(cdl_reclass_csv)
crop_dict = dict(zip(cdl_class_df.old_class_value, cdl_class_df.new_class_value))

def crop_multi(x):
    return(crop_dict[x])

c_multi = np.vectorize(crop_multi)


# In[34]:


def check_qa(qa_path, shape,  valid_qa = [0, 4, 32, 36, 64, 68, 96, 100, 128, 132, 160, 164, 192, 196, 224, 228]):
    
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
        qa_df = pd.DataFrame({"qa_val" : unique, "counts" : counts})
        qa_df
        qa_df[~ qa_df.qa_val.isin(valid_qa)].sort_values(['counts'], ascending = False)
        qa_df['pct'] = (100 *qa_df['counts'])/(224.0 * 224.0)
        
        bad_qa = qa_df[~ qa_df.qa_val.isin(valid_qa)].sort_values(['counts'], ascending = False)
        if len(bad_qa) > 0:
            highest_invalid_percent = bad_qa.pct.tolist()[0]
        else: 
            highest_invalid_percent = 0
        # ncell = len(vals)
        valid_count = sum(x in valid_qa for x in vals)
        return(valid_count, highest_invalid_percent, out_image[0])


# In[35]:


def process_chip(chip_id, 
                 chip_tile,
                 shape,
                 track_csv,
                 cdl_file,
                 bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]):
    
    """
    This function receives a chip id, HLS tile, chip geometry, and a list of bands to process. 
    
    Assumptions:
    
    Inputs:
    - chip_id: string of chip id, e.g. '000_001'
    - chip_tile: string of HLS tile , e.g. '15ABC'
    - shape: 'geometry' property of single polygon feature read by fiona
    
    The function writes out a multi-date TIF containing the bands for each of the three image dates for an HLS tile. 
    The function writes out a multi-date TIF containing the QA bands of each date.
    The function writes out a chipped version of CDL. 
    The function calls check_qa(), which makes assumptions about what QA pixels are valid.
    The function returns the number of valid QA pixels at each date, as a tuple.
    
    """
    ## get reprojected image paths
    tile_info_df = track_csv[track_csv.tile == chip_tile]
    
    selected_image_folders = tile_info_df.save_path.to_list()

    
    assert len(selected_image_folders) == 3
    
                     
    first_image_date = tile_info_df.iloc[0].date
    second_image_date = tile_info_df.iloc[1].date
    third_image_date = tile_info_df.iloc[2].date
    

    all_date_images = []
    all_date_qa = []
                     
    for i in range(3):
        for band in bands:
            all_date_images.append(tile_info_df.iloc[i].save_path + f"{tile_info_df.iloc[i].filename}.{band}.reproject.tif")
        all_date_qa.append(tile_info_df.iloc[i].save_path + f"{tile_info_df.iloc[i].filename}.Fmask.reproject.tif")
        

    valid_first, bad_pct_first, qa_first = check_qa(all_date_qa[0], shape)
    valid_second, bad_pct_second, qa_second = check_qa(all_date_qa[1], shape)
    valid_third, bad_pct_third, qa_third = check_qa(all_date_qa[2], shape)
    
    qa_bands = []
    qa_bands.append(qa_first)
    qa_bands.append(qa_second)
    qa_bands.append(qa_third)
    qa_bands = np.array(qa_bands).astype(np.uint8)
    

    assert len(all_date_images) == 3 * len(bands)
    
    out_bands = []
    print('out_bands_loop')
    for img in all_date_images:
        with rasterio.open(img) as src:
            print(img)
            out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
            out_meta = src.meta
            out_bands.append(out_image[0])
    
    out_bands = np.array(out_bands)
    # print(out_bands.shape)


    out_meta.update({"driver": "GTiff",
                     "height": out_bands.shape[1],
                     "width": out_bands.shape[2],
                     "count": out_bands.shape[0],
                     "transform": out_transform})
    
    # get NA count for HLS
    na_count = sum(out_bands.flatten() == -1000)
    
    # reclass negative HLS values to 0
    out_bands = np.clip(out_bands, 0, None)
    
    # write HLS chips
    with rasterio.open(chip_dir + str(chip_id) + "_merged.tif", "w", **out_meta) as dest:
        dest.write(out_bands)
      
    
        
    ## write QA bands
    out_meta.update({"driver": "GTiff",
                     "height": qa_bands.shape[1],
                     "width": qa_bands.shape[2],
                     "count": qa_bands.shape[0],
                     "transform": out_transform})
    
    with rasterio.open(chip_fmask_dir + str(chip_id) + "_Fmask.tif", "w", **out_meta) as dest:
        dest.write(qa_bands)  

    
    ## clip cdl to chip
    with rasterio.open(cdl_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
        out_meta = src.meta
        colormap = src.colormap(1)

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    
    # write multiclass reclassed CDL chips
    out_image_multi = c_multi(out_image).astype(np.uint8)
    with rasterio.open(chip_dir + str(chip_id) + ".mask.tif", "w", **out_meta) as dest:
        dest.write(out_image_multi)
        dest.write_colormap(1, colormap)

                     
    return(valid_first,
           valid_second,
           valid_third, 
           bad_pct_first,
           bad_pct_second,
           bad_pct_third,
           na_count,
           first_image_date,
           second_image_date,
           third_image_date)
    


# In[ ]:


## process chips
failed_tiles = []

for tile in tiles_to_chip:
    print(tile)
    chips_to_process = chip_df[chip_df.tile == tile[1:]].reset_index(drop = True)
    for k in range(len(chips_to_process)):
        current_id = chips_to_process.chip_id[k]
        chip_tile = chips_to_process.tile[k]
        chip_index = chip_ids.index(current_id)

        chip_feature = chipping_js['features'][chip_index]

        shape = [chip_feature['geometry']]
        full_tile_name = "T" + chip_tile
        try:
            valid_first, valid_second, valid_third, bad_pct_first, bad_pct_second, bad_pct_third, na_count, first_image_date, second_image_date, third_image_date = process_chip(current_id, full_tile_name, shape, track_df, cdl_file)
        except:
            failed_tiles.append(tile)
            break

        chip_df_index = chip_df.index[chip_df['chip_id'] == current_id].tolist()[0]
        chip_df.at[chip_df_index, 'valid_first'] = valid_first
        chip_df.at[chip_df_index, 'valid_second'] = valid_second
        chip_df.at[chip_df_index, 'valid_third'] = valid_third
        chip_df.at[chip_df_index, 'bad_pct_first'] = bad_pct_first
        chip_df.at[chip_df_index, 'bad_pct_second'] = bad_pct_second
        chip_df.at[chip_df_index, 'bad_pct_third'] = bad_pct_third
        chip_df.at[chip_df_index, 'first_image_date'] = first_image_date
        chip_df.at[chip_df_index, 'second_image_date'] = second_image_date
        chip_df.at[chip_df_index, 'third_image_date'] = third_image_date
        chip_df['bad_pct_max'] = chip_df[['bad_pct_first', 'bad_pct_second', 'bad_pct_third']].max(axis=1)
        chip_df.at[chip_df_index, 'na_count'] = na_count
chip_df.to_csv(chip_csv, index=False)


# In[ ]:


chip_df.head()


# ### Filtering Chips

# In[ ]:


selected_tiles = pd.read_csv(req_path + "selected_tiles.csv")
tiles_to_filter = selected_tiles.tile_id.unique()


# In[ ]:


chip_df = pd.read_csv(chip_csv)

for tile in tiles_to_filter:
    print(tile)
    filtered_chips = chip_df[(chip_df.tile == tile[1:]) & (chip_df.bad_pct_max < 5) & (chip_df.na_count == 0)].chip_id.tolist()
    for chip_id in filtered_chips:
        chip_files = glob('/data/chips/*' + chip_id + '*')
        for file in chip_files:
            name = file.split('/')[-1]
            shutil.copyfile(file, chip_dir_filt + name)


# In[ ]:




