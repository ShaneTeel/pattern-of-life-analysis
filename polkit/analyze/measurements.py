from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint
from geopy.distance import great_circle

from polkit.utils import get_logger

logger = get_logger(__name__)

def radius_of_gyration(lat:list, lon:list, weights:list=None):
    '''
    Description
    -----------
    Computes the weighted radius of gyration for an individual given a list of positions and weights (visit count / duration).
    Computes center of mass and great circle distance as intermediate steps.

    If `weights == None`, then the un-weighted radius of gyration is computed.

    Parameters
    ----------
    positions : list[tuple]
        A list of tuples consisting of (lat, lon)
    weights: list, default=None
        List of visit counts / durations
    
    Returns
    -------
    The Weighted (or un-weighted) Radius of Gyration for a specific individual.
    '''
    if weights is None:
        unique_points = list(set(zip(lat, lon)))
        lat, lon = zip(*unique_points)
        weights = [1] * len(lat)

    cm = center_of_mass(lat, lon, weights)
    total_weight = sum(weights)

    squared_distances = [
        great_circle_distance((lt, ln), cm)**2 * w for lt, ln, w in zip(lat, lon, weights)
    ]
    return np.sqrt(sum(squared_distances) / total_weight)

def great_circle_distance(pt1:tuple, pt2:tuple):
    '''
    Description
    -----------
    Implements the haversine formula to determine the distance between two points in meters.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        scaler values representing the lat, lon values for two distinct points.
    
    Returns
    -------
    The distance between the two points provided in meters
    '''
    lat1, lon1 = pt1
    lat2, lon2 = pt2

    R = 6371 # Radius in kms
    phi_1, phi_2 = np.radians(lat1), np.radians(lat2) # Equitorial distance scalers
    delta_phi = np.radians(lat2 - lat1) # Change in latitutde (in radians)
    delta_lambda = np.radians(lon2 - lon1) # Change in longitude (in radians)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2) ** 2
    c = np.atan2(np.sqrt(a), np.sqrt(1-a))
    return R * c * 1000

def center_of_mass(lat:list, lon:list, weights:list):
    '''
    Description
    -----------
    Computes the center of mass for an individual 
    given a list of positions and frequency / duration of visits to positions.
    
    Parameters
    ----------
    positions : list[tuple]
        A list of tuples consisting of (lat, lon)
    weights: list
        List of visit counts / durations 
    
    Returns
    -------
    center of mass : tuple
        center of mass for an individuals movements as a tuple (lat, lon)
    '''
    if weights is None:
        weights = [1] * len(lat)

    total_weight = sum(weights)

    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Assuming a unit sphere (Radius = 1) for mean calc
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    # Compute Weighted Average
    avg_x = sum(x * weights) / total_weight
    avg_y = sum(y * weights) / total_weight
    avg_z = sum(z * weights) / total_weight

    # Convert back to Lat/Lon
    central_lon = np.arctan2(avg_y, avg_x)
    hyp = np.sqrt(avg_x**2 + avg_y**2)
    central_lat = np.arctan2(avg_z, hyp)

    return (np.degrees(central_lat), np.degrees(central_lon))

def centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

def normalized_entropy(weights, n_bins:int=None):
    '''
    Description
    -----------
    Computes the normalized entropy (measure of unpredictability, scaled) for an individual
    given a list containing the frequency of visits to computed locations.

    Parameters
    ----------
    weights : list
        A list of weights representing the number of times or duration an individual spent at discrete locations.
    
    Returns
    -------
    Entropy normalized
    '''
    if weights is None or len(weights) <= 1:
        return 1.0 # Single location == 100% unpredictability (one or none does not earn a high predictability)
    
    total = sum(weights)

    probas = [count / total for count in weights]

    shannon = -sum(p * np.log2(p) for p in probas if p > 0)
    
    if n_bins is None:
        N = len(weights)
    else:
        N = n_bins

    max_entropy = np.log2(N)
    return shannon / max_entropy

def compute_regularity(dwell_dates:pd.Series):

    if len(dwell_dates) < 5:
        return 0.0
    
    dwells = sorted(dwell_dates)

    gaps = [
        (dwells[i+1] - dwells[i]).days for i in range(len(dwells) - 1)
    ]

    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    if mean_gap > 0:
        cv = std_gap / mean_gap
        return 1 / (cv + 1)
    else:
        return 0.0
    
def compute_loyalty(arrival:pd.Series[pd.Timestamp], active_days:int, collection_end:pd.Timestamp, attenuation_thresh:int=30, saturation_thresh:int=10):
    '''
    Description
    -----------
    Determine a user's loyalty regarding a specific location.
    Computes M (maturity), S (saturation), and A (attenuation) as intermediate steps and returns the geometric mean of all three
    - M (maturity): Characterizes the temporal length of a users relationship with a location.
    If M is high, then the user's visits to the location has persisted across the collection range.
    - S (saturation): Characterizes the number of times a user visited a location. 
    The goal is to reward M if the location consists of a high number of visits.
    - A (attenuation), which characterizes the amount of time since last visit. 
    The goal is to penalize a location if it is an "old haunt" that a user no longer visits.
    
    Parameters
    ----------
    arrival : pd.Series
        The datetime objects for each visit to a location
    collection_end : pd.Timestamp
        A pd.Timestamp object representing the last day of the collection window 
        for the entire dataset.
    active_days : int
        The number of active days in the collection period
    saturation_thresh : int, default=10
        The number of visits a location needs to achieve a doubling affect on M (maturity)

    Returns
    -------
    loyalty : float
        ```
        (M * S * A)**(1/3)
        ```
    '''
    # Compute M (maturity, or how long a users relationship with a location persists)
    dates = arrival.dt.date
    visit_days = dates.nunique()
    M = visit_days / active_days if active_days != 0 else 0.0

    # Compute S (saturation, or how many times a user visited a location) if n_visits is below thresh
    n_visits = len(arrival)
    learn_rate = np.log(0.5) / saturation_thresh
    S = 1 - np.exp(learn_rate * n_visits)
    M *= S

    # Compute A (attenuation, or how long has it been since the last visit) if time delta is below thresh
    T_d_last_visit = (collection_end - arrival.max()).days
    forget_rate = np.log(0.5) / attenuation_thresh
    A = np.exp(forget_rate * T_d_last_visit)
    M *= A

    return M**(1/3)