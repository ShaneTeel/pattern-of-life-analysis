import numpy as np
import pandas as pd
from datetime import timedelta
from shapely.geometry import LineString

from mobility.analyze import great_circle_distance
from mobility.utils import get_logger

logger = get_logger(__name__)

class TripLegs:

    def __init__(self, ):
        
        logger.debug("Initialized TripLegs: "
                     f"")

    def detect_triplegs(self, sp:pd.DataFrame, pfs:pd.DataFrame):
        if len(sp) < 2:
            logger.warning(f"Not enough staypoints to compute triplegs. Staypoints == {len(sp)}. Need at least 3.")
            return pd.DataFrame(
                columns=["origin_id", "destination_id", "started_at", "finished_at", "geometry", "distance", "duration"]
            )
        
        logger.info("Tripleg detection starting.")

        triplegs = []
        pfs = pfs[pfs["staypoint_id"] >= 0]
        sp_ids = pfs["staypoint_id"].unique()
        for i in range(len(sp_ids) - 1):
            if sp.loc[i, "n_points"] < 2:
                continue

            condition = pfs["staypoint_id"] == i
            points = pfs[condition].loc[:, ["lat", "lon"]].values
            trip_start = sp.loc[i, "finished_at"]
            trip_end = sp.loc[i+1, "started_at"]
            timedelta = self._time_delta(trip_end - trip_start)
            triplegs.append(self._create_tripleg(trip_start, trip_end, points, i))

        if not triplegs:
            logger.warning("Detected 0 triplegs. Returning empty dataframe.")
            tps = pd.DataFrame(columns=["origin_id", "destination_id", "started_at", "finished_at", "geometry", "distance", "duration"]) 
            return tps
        
        tps = pd.DataFrame(triplegs)
        logger.info(f"Tripleg detection complete.")
        return tps


    def _create_tripleg(self, trip_start, trip_end, points:pd.DataFrame, index:int):
        return {
            "origin_id": index,
            "destination_id": index + 1,
            "started_at": trip_start,
            "finished_at": trip_end,
            "geometry": LineString([(p[1], p[0]) for p in points]),
            "distance": self._tripleg_distance(points),
            "duration": self._time_delta(trip_end - trip_start)
        }
    
    def _tripleg_distance(self, points):
        total = 0
        for i in range(len(points) - 1):
            total += great_circle_distance(points[i, :], points[i+1, :])
        return total
    
    def _time_delta(self, td: timedelta):
        return td / np.timedelta64(60, "s")