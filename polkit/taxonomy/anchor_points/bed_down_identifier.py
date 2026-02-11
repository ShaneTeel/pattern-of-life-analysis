import numpy as np
import pandas as pd
from typing import Literal

from polkit.utils import get_logger

logger = get_logger(__name__)

class BedDownIdentifier:

    def __init__(self, sleep_window:tuple[int, int]=(22, 5), min_duration:int=4, coverage:Literal["sparse", "dense"]="sparse"):
        '''
        Parameters
        -
        sleep_window : tuple(int, int), default=(8, 18)
            A tuple consisting of a start time (`tuple[0]`) and and end time (`tuple[1]`). The tuple represents the core work window in 24-hour format (i.e., 18 == 18:00 (06:00pm)).  
            
        min_duration : int, default=4
            If a staypoint overlaps with the core sleep window and exceeeds the min_duration threshhold, 
            then it is considered a candidate sleep location
            
        coverage : str, default="sparse"
            Options to determine the sleep identification method based on data quality or, more specificially, overnight coverage of user's GPS data.
            
            "sparse" 
            > - indicates that the user's GPS data has significant gaps vis-a-vis the sleep window. This could be due to the fact that the user's device does not request GPS services during a user's sleep window. "sparse" as an argument will result in a permissive sleep location detection.
            
            "dense" 
            > - indicates that the user's GPS data is complete during the core sleep window. "dense" as an argument will result in a strict sleep location detection.
        '''
    
        self.window_start = sleep_window[0]
        self.window_end = sleep_window[1]
        self.min_duration = min_duration
        self.coverage = coverage

        logger.debug("BedDownIdentifier successfully initialized.")

    def identify(self, df:pd.DataFrame):
        '''
        Description
        -
        The only public method in `BedDownIdentifier`.
        Used to catalogue and classify all candidate sleep locations in a user's preprocessed dataset.
        
        Parameters
        -
        df : pd.DataFrame
            A dataset with the following required columns:
            ["loc_id", "arrived", "departed", "cluster_lat", "cluster_lon", "duration"]
            The value returned from `mobility.preprocess.LocationGenerator()` matches this format.

        Returns
        -
        routine_locs : pd.DataFrame
            A dataframe of all candidate sleep locations identified, sorted by `routine_score`
        '''
        
        if len(df) == 0:
            logger.warning("Empty DataFrame provided, returning `None`")
            return None
        
        df.sort_values(by="arrived", inplace=True)

        if self.coverage == "sparse":
            bed_down_locs = self._permissive_detection(df)
        else:
            bed_down_locs = self._strict_detection(df)

        if bed_down_locs is None or len(bed_down_locs) == 0:
            logger.warning("No candidate sleep locations detected. Returning `None`.")
            return None

        bed_down_locs["avg_dwell"] = bed_down_locs["total_dwell"] / bed_down_locs["count"]

        return bed_down_locs
    
    def _permissive_detection(self, df:pd.DataFrame):
        candidates = {}

        mask = self._create_mask(df)
        
        overnight = df[mask].reset_index(drop=True)
        
        if overnight is None or len(overnight) == 0:
            logger.warning("Warning. sleep detection using sparse data approach yielded no data. Check sleep window arguments and determine if GPS data has sufficient overnight coverage.")
            return None
        
        loc_id = overnight["loc_id"].values 
        sp_arrived = overnight["arrived"]
        duration = overnight["duration"].values
        cluster_lat, cluster_lon = overnight[["cluster_lat", "cluster_lon"]].values.T

        for i in range(len(loc_id)):
            label = loc_id[i]
            s_start = sp_arrived[i]
            dwell_time = pd.Timedelta(hours=duration[i])
            lat = cluster_lat[i]
            lon = cluster_lon[i]

            if label not in candidates:
                candidates[label] = self._create_template(label, lat, lon, s_start)

            self._update_candidate(candidates, label, s_start, dwell_time)

        return pd.DataFrame().from_dict(candidates, orient="index")
    
    def _strict_detection(self, df:pd.DataFrame):
        candidates = {}
        
        loc_id = df["loc_id"].values
        sp_arrived = df["arrived"].reset_index(drop=True)
        sp_departed = df["departed"].reset_index(drop=True)
        cluster_lat, cluster_lon = df[["cluster_lat", "cluster_lon"]].values.T
        window_start, window_end = self._create_bed_down_window(sp_arrived)
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
    
    def _create_mask(self, df:pd.DataFrame):
        overnight_mask = (df["arrived"].dt.date != df["departed"].dt.date)
        arrival_mask = ((df["arrived"].dt.hour >= self.window_start) & (df["duration"] >= self.min_duration))
        departure_mask = ((df["departed"].dt.hour <= self.window_end) & (df["duration"] >= self.min_duration))
        return overnight_mask | arrival_mask | departure_mask 
    
    def _create_bed_down_window(self, stay_start:pd.Series):
        mask = stay_start.dt.hour <= self.window_end
        base = stay_start.dt.floor("D")
        diff = pd.Timedelta(days=1)

        window_start = np.where(
            mask,
            (base - diff) + pd.Timedelta(hours=self.window_start),
            base + pd.Timedelta(hours=self.window_start)
        )

        window_end = np.where(
            mask,
            base + pd.Timedelta(hours=self.window_end),
            base + diff + pd.Timedelta(hours=self.window_end)
        )

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