import numpy as np
import pandas as pd

from polkit.analyze import great_circle_distance
from polkit.utils import get_logger

logger = get_logger(__name__)

class StayPointDetector:

    def __init__(self, distance_thresh:int=100, time_thresh:int=30, gap_thresh:int=60):
        self.distance_thresh = distance_thresh
        self.time_thresh = time_thresh
        self.gap_thresh = gap_thresh

        logger.debug("Initialized StayPointDetector with "
                     f"distance threshold of {self.distance_thresh} meters "
                     f"time threshold of {self.time_thresh} "
                     f"event gap threshold of {self.gap_thresh}.")

    def detect(self, user_data:pd.DataFrame):
        logger.info("Staypont detection starting.")
        
        lats = user_data["lat"].values
        lons = user_data["lon"].values
        datetime = user_data["datetime"].values

        staypoints = []

        i = 0
        while i < len(user_data):
            
            dt1 = datetime[i]
            j = i + 1

            while j < len(user_data): 
                dt2 = datetime[j]
                
                if j > i + 1:
                    prev_dt = datetime[j-1]
                    gap_minutes = self._time_delta(dt2 - prev_dt)

                    if gap_minutes > self.gap_thresh:
                        time_delta = self._time_delta(prev_dt - dt1)
                        if time_delta >= self.time_thresh:
                            staypoints.append(self._create_staypoint(user_data.iloc[i:j], time_delta))
                        i = j
                        break

                distance = great_circle_distance((lats[i], lons[i]), (lats[j], lons[j]))
                time_delta = self._time_delta(dt2 - dt1)

                if distance >= self.distance_thresh:
                    if time_delta >= self.time_thresh:
                        staypoints.append(self._create_staypoint(user_data.iloc[i:j], time_delta))
                    i = j
                    break
                
                else:
                    j += 1

            if j == len(user_data):
                time_delta = self._time_delta(datetime[-1] - datetime[i])
                if time_delta >= self.time_thresh:
                    staypoints.append(self._create_staypoint(user_data.iloc[i:], time_delta))
                break
            
        if not staypoints:
            logger.warning("Detected 0 staypoints. Returning empty dataframe, and original position fixes with '-1' assigned to all events.")
            sp = pd.DataFrame(columns=["user_id", "started_at", "finished_at", "lat", "lon", "duration", "n_points"])
            return sp

        sp = pd.DataFrame(staypoints)
        logger.info(f"Staypoint detection complete. Found {len(sp)} staypoints.")
        return sp

    def _create_staypoint(self, records:pd.DataFrame, duration:float):
        return {
            "user_id": records.iloc[0]["uid"],
            "arrived": records.iloc[0]["datetime"],
            "departed": records.iloc[-1]["datetime"],
            "lat": records["lat"].mean(),
            "lon": records["lon"].mean(),
            "duration": duration / 60,
            "n_points": len(records)
        }
    
    def _time_delta(self, td: pd.Timedelta):
        return td / np.timedelta64(60, "s")