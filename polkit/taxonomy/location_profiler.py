from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.preprocessing import MinMaxScaler
from typing import Literal

from polkit.analyze import radius_of_gyration, normalized_consistency, exponential_decay, exponential_saturation
from .anchor_points import BedDownIdentifier, WorkIdentifier
from polkit.utils import get_logger

logger = get_logger(__name__)

class LocationProfiler:
    '''
    Classifies locations by a routine score that is computed based on visit ratio, dwell ratio, predictability, and regularity.
    
    Classification system assigns:
    - "Routine": Core locations; serve as the backbone of the User's pattern
    - "Staple": Lifestyle habits; more flexible than "Routine", 
    but still lacks the rigid adherence to a pattern to be classified as "Routine"
    - "Periodic": Secondary interests; low-density
    - "Transient": Environmental noise, atypical behavior, one-off visits, or "stops along the way" to more established locations
    '''

    _REQUIRED_COLS = ["loc_id", "arrived", "departed", "cluster_lat", "cluster_lon"]

    _COL_ORDER = ["Location ID", 
                  "Lat", "Lon", "Spatial Focus", 
                  "Total Dwell", "First Seen", "Last Seen", "Total Visits", 
                  "Arrival Consistency", "Dwell Consistency", "Gap Consistency",
                  "Recency", "Depth", "Visit Count",
                  "Loyalty Index", "Predictability Index", "Loyalty Label"]

    _OPTIONS = ["Transient", "Recurring", "Habit", "Anchor"]

    def __init__(
            self, 
            sleep_window:tuple[int, int]=(22, 5), 
            min_sleep:int=4, 
            sleep_coverage:Literal["sparse", "dense"]="sparse", 
            work_window:tuple[int, int]=(9, 18),
            min_work:int=4,
            work_days:list=[0, 1, 2, 3, 4],
            work_coverage:Literal["sparse", "dense"]="sparse"
        ):

        self.profile_df = None


        self.sleep = BedDownIdentifier(sleep_window=sleep_window, min_duration=min_sleep, coverage=sleep_coverage)
        self.sleep_df = None

        self.work = WorkIdentifier(work_window=work_window, min_duration=min_work, work_days=work_days, coverage=work_coverage)
        self.work_df = None
        
        self.scaler = MinMaxScaler()

        logger.debug("LocationProfiler initialized")

    def profile(self, locs:pd.DataFrame):
        '''
        Description
        -
        The only public method in `LocationProfiler`.
        Used to catalogue and classify all locations in a user's preprocessed dataset by frequency and predictability.
        Excludes home and work if provided as arguments.
        
        Parameters
        -
        locs : pd.DataFrame
            A dataset with the following required columns:
            ["loc_id", "arrived", "departed", "cluster_lat", "cluster_lon", "duration"]
            The value returned from `mobility.taxonomy.StayPointClusterer()` matches this format.

        Returns
        -
        profiled_df : pd.DataFrame
            A dataframe of all locations profiled according to frequency / regularity and sorted by `routine_score`
        '''

        if len(locs) == 0:
            logger.warning("Empty DataFrame provided, returning empty results")
            return pd.DataFrame()

        missing = set(self._REQUIRED_COLS) - set(locs.columns)
        
        if missing:
            raise ValueError(f"DataFrame is missing the following required columns: {missing}")

        locs = locs.copy()
        
        locs.sort_values(by="arrived", inplace=True)

        profile = self._build_profile(locs)

        self.sleep_df, self.work_df = self._identify_anchors(locs)

        if self.sleep_df is None:
            profile["Candidate Home"] = False
        else:
            profile["Candidate Home"] = profile["Location ID"].isin(self.sleep_df["loc_id"])

        if self.work_df is None:
            profile["Candidate Work"] = False
        else:
            profile["Candidate Work"] = profile["Location ID"].isin(self.work_df["loc_id"])

        self.profile_df = profile

        return self.profile_df
    
    def _build_profile(self, locs:pd.DataFrame):
        # Pre-Profiling Optimization 
        locs["hours"] = locs["arrived"].dt.hour
        locs["date"] = locs["arrived"].dt.date
        last = locs["arrived"].max()
        
        def profile_group(group:pd.Series):
            return pd.Series({
                "Lat": group["cluster_lat"].iloc[0],
                "Lon": group["cluster_lon"].iloc[0],
                "Spatial Focus": radius_of_gyration(group["sp_lat"].tolist(), group["sp_lon"].tolist()),

                "Total Dwell": group["duration"].sum(),
                "Avg Dwell": group["duration"].mean(), 
                
                "First Seen": group["arrived"].min(),
                "Last Seen": group["arrived"].max(),
                "Total Visits": len(group), 
                
                "Arrival Consistency": normalized_consistency(group["hours"]),
                "Dwell Consistency": normalized_consistency(group["duration"]),
                "Gap Consistency": normalized_consistency(self._find_gaps(group["arrived"])),

                "Recency": exponential_decay((last - group["arrived"].max()).days, 30),
                "Depth": exponential_saturation(group["duration"].sum(), 4),
                "Visit Count": exponential_saturation(len(group["arrived"]), 10),
            })

        profile = locs.groupby("loc_id").apply(profile_group, include_groups=False).reset_index(names="Location ID")
    
        # Take harmonic mean of "Loyalty" metrics; assign Label
        profile["Loyalty Index"] = profile[["Recency", "Depth", "Visit Count"]].apply(hmean, axis=1)
        profile["Loyalty Label"] = self._assign_label(profile["Loyalty Index"])

        # Determine which locations are assessed for "Predictiability"
        profile["Predictability Index"] = profile[["Arrival Consistency", "Dwell Consistency", "Gap Consistency"]].mean(axis=1)

        return profile[self._COL_ORDER].sort_values(by="Location ID")
    
    def _assign_label(self, score:pd.Series):
        
        conditions = [
            score >= 0.50,
            score >= 0.25,
            score >= 0.05
        ]
        choices = self._OPTIONS[-3:][::-1]
        return pd.Series(np.select(
            conditions,
            choices,
            default=self._OPTIONS[0]
        ),
        index=score.index
    )

    def _identify_anchors(self, locs: pd.DataFrame):

        sleep_df = self.sleep.identify(locs)
        work_df = self.work.identify(locs)

        return sleep_df, work_df
    
    def format_profiles_for_charts(self):
        if self.profile_df is None:
            raise ValueError("Warning - Location-specific profiles not created yet; user must call `.profile()` first.")
        
        profile_df = self.profile_df.copy()
        profile_df["Hover"] = profile_df.apply(lambda x: f"""
<b>Location ID</b>: {int(x["Location ID"])}<br>
<b>Spatial Focus</b>: {x["Spatial Focus"]:.2f} meters<br>
<b>Loyalty</b>: {x["Loyalty Index"]:.2%}<br>
<b>Predictability</b>: {x["Predictability Index"]:.2%}<br>
<b>Classification</b>: {x["Loyalty Label"]}<br>
<b>Home / Work Candidacy</b>: {"Home, Work" if x["Candidate Home"] and x["Candidate Work"] else "Home" if x["Candidate Home"] else "Work" if x["Candidate Work"] else ""}<br>
""", axis=1)
                
        profile_df["Spatial Focus"] = 1 - self.scaler.fit_transform(profile_df[["Spatial Focus"]])

        return profile_df
    
    def get_anchor_point_data(self):
        if self.sleep_df is None or self.work_df is None:
            raise ValueError("Warning - Anchor Point Identification not performed yet; user must call `.profile()` first.")

        return self.sleep_df, self.work_df

    def get_likely_home(self):
        if self.sleep_df is None:
            return self.profile_df.loc[self.profile_df["Loyalty Index"].idxmax(), "Location_ID"]
        
        home_subset = self.profile_df[self.profile_df["Candidate Home"] == True]
        return home_subset.loc[home_subset["Loyalty Index"].idxmax(), "Location ID"]
    
    def _find_gaps(self, visit_dates:pd.Series):

        if len(visit_dates) < 5:
            return [0.0]
    
        visits = sorted(visit_dates)

        gaps = [
            (visits[i+1] - visits[i]).days for i in range(len(visits) - 1)
        ]
        return gaps