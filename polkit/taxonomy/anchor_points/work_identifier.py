from __future__ import annotations

import pandas as pd
from typing import Literal

from polkit.utils import get_logger

logger = get_logger(__name__)

class WorkIdentifier:

    def __init__(self, work_window:tuple[int, int]=(8, 18), min_duration:int=4, work_days:list=[0, 1, 2, 3, 4], coverage:Literal["sparse", "dense"]="sparse"):
        '''
        Parameters
        -            
        work_window : tuple(int, int), default=(8, 18)
            A tuple consisting of a start time (`tuple[0]`) and and end time (`tuple[1]`). The tuple represents the core work window in 24-hour format (i.e., 18 == 18:00 (06:00pm)).  

        min_duration : int, default=4
            If a staypoint overlaps with the core work window and exceeeds the min_duration threshhold, 
            then it is considered a candidate work location 
            
        coverage : str, default="sparse"
            Options to determine the work identification method based on data quality or, more specificially, daytime coverage of user's GPS data.
            
            "sparse" 
            > - indicates that the user's GPS data has significant gaps vis-a-vis the core work window. 
            > - This could be due to the fact that the user's device does not request GPS services during a user's work window. 
            > - "sparse" as an argument will result in a permissive work location identification.
            
            "dense" 
            > - indicates that the user's GPS data is complete during the core work window. 
            > - "dense" as an argument will result in a strict work location identification.

        '''

        self.window_start = work_window[0]
        self.window_end = work_window[1]
        self.min_duration = min_duration
        self.work_days = work_days
        self.coverage = coverage

        logger.debug("WorkIdentifier successfully initialized.")

    def identify(self, df:pd.DataFrame, home_ids:list=None):
        '''
        Description
        -
        The only public method in `WorkIdentifier`.
        Used to catalogue and classify all candidate work locations in a user's preprocessed dataset.
        
        Parameters
        -
        df : pd.DataFrame
            A dataset with the following required columns:
            ["loc_id", "arrived", "departed", "cluster_lat", "cluster_lon", "duration"]
            The value returned from `mobility.preprocess.LocationGenerator()` matches this format.


        Returns
        -
        work_ids : pd.DataFrame
            A dataframe of all candidate work locations identified, sorted by `routine_score`
        '''
        
        if len(df) == 0:
            logger.warning("Empty DataFrame provided, returning `None`")
            return None
        
        if self.coverage == "sparse":
            work_locs = self._permissive_detection(df, home_ids)
        else:
            work_locs = self._strict_detection(df, home_ids)

        if work_locs is None or len(work_locs) == 0:
            logger.warning("No candidate work locations identified. Returning None")
            return None

        work_locs["avg_dwell"] = work_locs["total_dwell"] / work_locs["count"]

        return work_locs

    def _permissive_detection(self, df:pd.DataFrame, home_ids:list=None):
        
        mask = self._create_mask(df, home_ids)

        work_records = df[mask].reset_index(drop=True)
        
        if work_records is None or len(work_records) == 0:
            logger.warning("Warning. Arguments passed at _init_ are too restrictive for work location identification of sparse data. Returning `None`.")
            return None

        candidates = {}
        loc_id = work_records["loc_id"].values 
        sp_arrived = work_records["arrived"]
        duration = work_records["duration"].values
        cluster_lat, cluster_lon = work_records[["cluster_lat", "cluster_lon"]].values.T

        for i in range(len(loc_id)):
            label = loc_id[i]
            s_start = sp_arrived[i]
            dwell_time = pd.Timedelta(hours=duration[i])
            lat = cluster_lat[i]
            lon = cluster_lon[i]

            if label not in candidates:
                candidates[label] = self._create_template(label, lat, lon, s_start)

            self._update_candidate(candidates, label, s_start, dwell_time)

        candidates = pd.DataFrame().from_dict(candidates, orient="index")
        return candidates
        
    def _create_mask(self, df:pd.DataFrame, home_ids:list=None):
        workday_mask = df["arrived"].dt.dayofweek.isin(self.work_days)

        arrival_mask = (
            (df["arrived"].dt.hour < self.window_end) &
            (df["arrived"].dt.hour >= self.window_start)
        )

        departure_mask = (
            (df["departed"].dt.hour > self.window_start) &
            (df["departed"].dt.hour <= self.window_end)
        )
        
        work_window_mask = arrival_mask | departure_mask

        duration_mask = df["duration"] >= self.min_duration

        same_day_mask = df["arrived"].dt.date == df["departed"].dt.date 
        base_mask = workday_mask & work_window_mask & duration_mask & same_day_mask 

        if home_ids is not None:
            home_mask = ~df["loc_id"].isin(home_ids)
            return base_mask & home_mask

        return base_mask
        
    
    def _strict_detection(self, df:pd.DataFrame, home_ids:list=None):
        workday_mask = df["arrived"].dt.dayofweek.isin(self.work_days)

        df = df[workday_mask].reset_index(drop=True)

        if home_ids is not None:
            df = df[~df["loc_id"].isin(home_ids)]

        if df is None or len(df) == 0:
            logger.warning(f"Warning. No work records for work days {self.work_days}. Returning `None`.")
            return None
        
        candidates = {}
        
        loc_id = df["loc_id"].values
        sp_arrived = df["arrived"]
        sp_departed = df["departed"]
        cluster_lat, cluster_lon = df[["cluster_lat", "cluster_lon"]].values.T
        window_start, window_end = self._create_work_window(sp_arrived)
        intersection_thresh = pd.Timedelta(hours=self.min_duration)

        for i in range(len(loc_id)):
            s_start = sp_arrived[i]
            s_end = sp_departed[i]

            w_start = window_start[i]
            w_end = window_end[i]

            if s_end >= w_start and s_start <= w_end:
                overlap_start = max(w_start, s_start)
                overlap_end = min(w_end, s_end)

                intersection = overlap_end - overlap_start
                if intersection > intersection_thresh:
                    label = loc_id[i]
                    lat = cluster_lat[i]
                    lon = cluster_lon[i]
                    if label not in candidates:
                        candidates[label] = self._create_template(label, lat, lon, s_start)

                    self._update_candidate(candidates, label, s_start, intersection)
        return pd.DataFrame().from_dict(candidates, orient="index")

    def _create_work_window(self, stay_start:pd.Series):
        base = stay_start.dt.floor("D")
        window_start = base + pd.Timedelta(hours=self.window_start)
        window_end = base + pd.Timedelta(hours=self.window_end)
        return window_start, window_end
    
    def _create_template(self, label, lat, lon, dwell_date):

        return {
            "loc_id": label,
            "lat": lat,
            "lon": lon,
            "count": 0,
            "total_dwell": 0,
            "avg_dwell": 0,
            "first_dwell": dwell_date,
            "last_dwell": None,
            "last_dwell_duration": None,
            "dwell_dates": []
        }
    
    def _update_candidate(self, candidates:dict, label:int, dwell_date:pd.Timestamp, duration:pd.Timedelta):
        candidates[label]["count"] += 1
        candidates[label]["total_dwell"] += duration.round(freq="s").total_seconds() / 3600
        candidates[label]["last_dwell"] = dwell_date
        candidates[label]["last_dwell_duration"] = duration.round(freq="s")
        candidates[label]["dwell_dates"].append(dwell_date.date())