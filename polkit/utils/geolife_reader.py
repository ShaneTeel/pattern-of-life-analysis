from __future__ import annotations

import os
from pathlib import Path
import glob
import pickle
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class GeoLifeReader:
    
    _COLUMNS = [0, 1, 3, 5, 6]
    
    _NAMES = ["lat", "lon", "alt", "date", "time"]

    def __init__(self, user_id:str, data_path:str, time_zone:str="Asia/Shanghai"):
        self.user_id = user_id
        self.data_dir = Path(data_path)
        self.total_records = None
        
        if self.data_dir.exists():
            self.cache_file = self.data_dir / f"cache/user_{self.user_id}.pkl"
            self.all_files = glob.glob(os.path.join(self.data_dir, "*"))
        else:
            raise FileExistsError(f"No directory found at {data_path}")
        
        self.time_zone = time_zone
        logger.debug(f"DataLoader for GeoLife user {self.user_id} initialized successfully. "
                    f"User {self.user_id} files are located in {self.data_dir}.")

    def read_user(self):
        '''
        Description
        -----------
        Reads all .plt files for a single GeoLife user

        Steps
        -----
        1. Iterate through each .plt file for a single user and append all plt files as DataFrames to a DataFrame list
        2. Concatenate all DataFrames in list into a single DataFrame
        3. Perform date time conversion to Beijing
        4. Drop irrelevant columns
        
        Returns
        -------
        Pandas Dataframe
        '''
        logger.info(f"Loading GeoLife user {self.user_id}'s .plt files.")
        single_users_daily_trajectories = []

        if self.cache_file.exists():
            logger.info(f"Loading pickle file for user {self.user_id} located at {self.cache_file}.")
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)

        for tid, file_path in enumerate(self.all_files, 1):
            df = pd.read_csv(
                file_path, 
                skiprows=6, 
                header=None, 
                usecols=self._COLUMNS,
                names=self._NAMES)
            
            df['tid'] = tid
            df['uid'] = self.user_id
            df['utc'] = df['date'] + ' ' + df['time']

            single_users_daily_trajectories.append(df)

        all_days_df = pd.concat(single_users_daily_trajectories, ignore_index=True)

        all_days_df['utc'] = pd.to_datetime(all_days_df['utc'], utc=True, errors='coerce', format='%Y-%m-%d %H:%M:%S')
        all_days_df['datetime'] = all_days_df['utc'].dt.tz_convert(self.time_zone)
        all_days_df['uid'] = all_days_df['uid'].astype('category')
        all_days_df = all_days_df.sort_values(by="datetime")
        all_days_df.drop(columns=['date', 'time', 'utc', 'alt'], inplace=True)

        self.cache_file.parent.mkdir(exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(all_days_df, f)

        logger.info(f"Finished loading {len(all_days_df)} observed events for user {self.user_id}.")
        return all_days_df
    
    def delete_user(self):

        if self.cache_file.is_file():
            self.cache_file.unlink()
        
            logger.info(f"Deleted .pkl file for User {self.user_id}")
        else:
            raise FileExistsError(f"No file found at {self.cache_file}")

    def move_user(self, dest_path:str):

        if self.cache_file.is_file():
            self.cache_file.rename(dest_path)
        else:
            raise FileExistsError(f"No file found at {self.cache_file}")