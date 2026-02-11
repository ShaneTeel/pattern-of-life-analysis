from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

from polkit.analyze import radius_of_gyration, compute_regularity, compute_loyalty
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

    _COL_ORDER = ["Location ID", "Lat", "Lon", "Spatial Focus", 
                  "Total Dwell", "Dwell Ratio", 
                  "First Seen", "Last Seen", "Total Days Seen", 
                  "Total Visits", "Visit Ratio", 
                  "Loyalty", "Regularity",
                  "Routine Index", "Label"]

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

        # Per group variables
        last = locs["arrived"].max()
        n_days = locs["date"].nunique()
        total_duration = locs["duration"].sum()
        total_visits = len(locs)

        
        def profile_group(group:pd.Series):
            return pd.Series({
                "Lat": group["cluster_lat"].iloc[0],
                "Lon": group["cluster_lon"].iloc[0],
                "Total Dwell": group["duration"].sum(),
                "Dwell Ratio": group["duration"].sum() / total_duration,
                "First Seen": group["arrived"].min(),
                "Last Seen": group["arrived"].max(),
                "Total Days Seen": group["arrived"].nunique(),
                "Total Visits": len(group), 
                "Visit Ratio": len(group) / total_visits,
                "Regularity": compute_regularity(group["date"].tolist()),
                "Loyalty": compute_loyalty(group["arrived"], n_days, last),
                "Spatial Focus": radius_of_gyration(group["sp_lat"].tolist(), group["sp_lon"].tolist())
            })

        profile = locs.groupby("loc_id").apply(profile_group, include_groups=False).reset_index(names="Location ID")
    
        # Aggregate metrics to compute "habit" score
        profile["Routine Index"] = profile[["Regularity", "Loyalty", "Visit Ratio", "Dwell Ratio"]].mean(axis=1)

        profile["Label"] = self._assign_label(profile["Routine Index"])

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
<b>Routine Index</b>: {x["Routine Index"]:.2f}<br>
<b>Classification</b>: {x["Label"]}<br>
<b>Candidate Home</b>: {x["Candidate Home"]}<br>
<b>Candidate Work</b>: {x["Candidate Work"]}<br>
""", axis=1)
    
        df_short = profile_df[["Location ID", "Hover", "Visit Ratio", "Loyalty", "Dwell Ratio", "Regularity"]]
        diamond_data = df_short.melt(
            id_vars=["Hover", "Location ID"],
            value_vars=df_short.columns.tolist(),
            var_name="Metric",
            value_name="Score"
        )

        return diamond_data, profile_df
    
    def get_anchor_point_data(self):
        if self.sleep_df is None or self.work_df is None:
            raise ValueError("Warning - Anchor Point Identification not performed yet; user must call `.profile()` first.")

        return self.sleep_df, self.work_df

    def get_likely_home(self):
        if self.sleep_df is None:
            return None
        
        home_subset = self.profile_df[self.profile_df["Candidate Home"] == True]
        return home_subset.loc[home_subset["Routine Index"].idxmax(), "Location ID"]