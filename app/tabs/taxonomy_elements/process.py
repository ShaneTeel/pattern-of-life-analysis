import streamlit as st

from polkit.taxonomy import *
from polkit.analyze import radius_of_gyration, normalized_entropy, center_of_mass

def run_process(det_configs, cluster_configs, sleep_configs, work_configs):
    pfs = st.session_state["raw"]
    
    detector = StayPointDetector(**det_configs)
    sps = detector.detect(pfs)
    if len(sps) < 2 or sps is None:
        st.warning("Arguments for Stay-Point Detection are too restrictive. Please adjust before continuing.")
        st.stop()

    st.session_state["stay_points"] = sps
        
    clusterer = StayPointClusterer(**cluster_configs)
    locs = clusterer.cluster(sps)
    if len(locs) == 0 or locs is None:
        st.warning("Arguments for Stay-Point Clustering are too restrictive. Please adjust before continuing.")
        st.stop()
        
    st.session_state["locations"] = locs    

    profiler = LocationProfiler(**sleep_configs, **work_configs)
    profiles = profiler.profile(locs)
    st.session_state["profiles"] = profiles

    chart_data = profiler.format_profiles_for_charts()
    st.session_state["chart_data"] = chart_data

    home_id = profiler.get_likely_home()
    st.session_state["likely_home"] = home_id

    lat, lon = profiles["Lat"].tolist(), profiles["Lon"].tolist()
    dwell = profiles["Total Dwell"].tolist()
    Rg = radius_of_gyration(lat, lon, dwell)
    cm = center_of_mass(lat, lon, dwell)
    H = 1 - normalized_entropy(profiles["Total Visits"])

    locations = len(profiles)
    anchors = len(profiles[profiles["Loyalty Label"] == "Anchor"])
    persistent = len(profiles[profiles["Loyalty Label"] == "Persistent"])
    recurring = len(profiles[profiles["Loyalty Label"] == "Recurring"])
    transient = len(profiles[profiles["Loyalty Label"] == "Transient"])
    homes = len(profiles[profiles["Candidate Home"] == True])
    works = len(profiles[profiles["Candidate Work"] == True])

    st.session_state["profile_metrics"] = {
        "Rg": Rg,
        "cm": cm,
        "H": H,
        "Confidence": "HIGH" if H > 0.66 else "MODERATE" if H > 0.33 else "LOW",
        "Locations": locations,
        "Anchors": anchors,
        "Persistent": persistent,
        "Recurring": recurring,
        "Transient": transient,
        "Homes": homes,
        "Works": works
    }