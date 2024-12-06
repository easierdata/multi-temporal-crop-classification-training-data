import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import sys
import earthaccess
from tqdm import tqdm
earthaccess.login(strategy="netrc")

module_path = module_path = os.path.abspath(Path(__file__).parent.parent.resolve())
sys.path.insert(0, module_path)
try:
    from data_prep import *
except ModuleNotFoundError:
    print("Module not found")
    pass

selected_tiles = pd.read_pickle(
    SELECTED_TILES_PKL
)

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
            temp_sav_path = f"{str(TILE_DIR)}/{bands_dict['B02'].split('/')[-2]}/"
            print(temp_sav_path)
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

print(selected_tiles.head())

track_df = tile_download(selected_tiles, from_csv=False)