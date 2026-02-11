from __future__ import annotations

import pandas as pd

from polkit.utils import get_logger

logger = get_logger(__name__)

class TemporalAnalyzer:

    _CONFIDENCE_THRESHOLDS = [0.80, 0.40]

    def __init__(self, user_id:int, gap_thresh:int=24):
        self.user_id = user_id
        self.gap_thresh = gap_thresh
        self.results = None
        
        logger.debug(f"Initialized TemporalAnalyzer for user {self.user_id}'s position fixes (raw GPS traces).")

    def analyze(self, df:pd.DataFrame):
        retval = {}
        day_mask, night_mask = self._create_temporal_masks(df)

        retval["full"] = self._run_analysis(df)
        
        day_df = df[day_mask]
        retval["day"] = self._run_analysis(day_df)

        night_df = df[night_mask]
        retval["night"] = self._run_analysis(night_df)

        self.results = retval
        return self.results
    
    def _create_temporal_masks(self, df:pd.DataFrame):
        day_mask = ((df["datetime"].dt.hour < 18) & 
                      (df["datetime"].dt.hour >= 6))
        return day_mask, ~day_mask

    def _run_analysis(self, df:pd.DataFrame):
        temp_coverage = self._analyze_temporal_coverage(df)
        density = self._analyze_density(df)
        gaps = self._identify_gaps(df, self.gap_thresh)

        return {
            "temporal_coverage": temp_coverage,
            "density": density,
            "gaps": gaps
        }

    def _analyze_temporal_coverage(self, df:pd.DataFrame):
        '''
        How well does the data coverage period account of each day of potential activity
        '''
        latest = df["datetime"].max()
        earliest = df["datetime"].min()
        total_days = (latest - earliest).days

        active_days = df["datetime"].dt.date.nunique()
        coverage_ratio = active_days / total_days if total_days > 0 else 0
        return {
            "total_days": total_days,
            "active_days": active_days,
            "coverage_ratio": coverage_ratio
        }

    def _analyze_density(self, df:pd.DataFrame):
        time_diff = df["datetime"].diff().dt.total_seconds() / 60
        median_gap = time_diff.median()
        return {
            "time_diff_minutes": time_diff,
            "median_gap_minutes": median_gap
        }

    def _identify_gaps(self, df:pd.DataFrame, gap_thresh:int=24):
        gaps = []
        sorted_data = df.sort_values(by="datetime")
        time_diffs = sorted_data["datetime"].diff()
        gap_thresh = pd.Timedelta(hours=gap_thresh)
        large_gaps = time_diffs[time_diffs > gap_thresh]

        if large_gaps is not None or len(large_gaps) != 0:
            for idx in large_gaps.index:
                prev_idx = sorted_data.index.get_loc(idx) - 1
                if prev_idx >= 0:
                    prev_time = sorted_data.iloc[prev_idx]["datetime"]
                    curr_time = sorted_data.loc[idx, "datetime"]

                    gaps.append({
                        "start": prev_time,
                        "end": curr_time,
                        "duration_hours": (curr_time - prev_time).total_seconds() / 3600
                    })

        return gaps