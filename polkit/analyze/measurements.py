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

    probas = [count / total for count in weights if total > 0]

    shannon = -sum(p * np.log2(p) for p in probas if p > 0)
    
    if n_bins is None:
        N = len(weights)
    else:
        N = n_bins

    max_entropy = np.log2(N)
    return shannon / max_entropy

def normalized_consistency(X:pd.Series | list):
    if X is None or len(X) < 2:
        return 0.0

    mean_X = np.mean(X)
    range_X = np.max(X) - np.min(X)


    if mean_X > 0:
        sensitive_CV = range_X / mean_X
        return 1 / (sensitive_CV + 1)
    else:
        return 0.0
    
def exponential_saturation(X:int, half_life:int):
    return 1 - exponential_decay(X, half_life)

def exponential_decay(X:int, half_life:int):
    decay_rate = (np.log(0.5) / half_life)
    return np.exp(decay_rate * X)