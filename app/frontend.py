import streamlit as st
import pickle
from pathlib import Path

from polkit.analyze import TemporalAnalyzer
from polkit.visualize import *
from tabs import *
from polkit.utils import get_logger

logger = get_logger(__name__)

# Dashboard Session States
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "raw" not in st.session_state:
    st.session_state["raw"] = None
if "chart_maker" not in st.session_state:
    st.session_state["chart_maker"] = ChartMaker()
if "map_maker" not in st.session_state:
    st.session_state["map_maker"] = None

# Integrity Tab Session States
if "time_analysis" not in st.session_state:
    st.session_state["time_analysis"] = None

# Taxonomy Tab Session States
if "stay_points" not in st.session_state:
    st.session_state["stay_points"] = None
if "locations" not in st.session_state:
    st.session_state["locations"] = None
if "profiles" not in st.session_state:
    st.session_state["profiles"] = None
if "profile_metrics" not in st.session_state:
    st.session_state["profile_metrics"] = None
if "chart_data" not in st.session_state:
    st.session_state["chart_data"] = None
if "likely_home" not in st.session_state:
    st.session_state["likely_home"] = None

# Prediction Session States
if "eval_metrics" not in st.session_state:
    st.session_state["eval_metrics"] = None
if "eval_df" not in st.session_state:
    st.session_state["eval_df"] = None
if "analytic_summary" not in st.session_state:
    st.session_state["analytic_summary"] = None
if "matrix" not in st.session_state:
    st.session_state["matrix"] = None
if "matrix_key" not in st.session_state:
    st.session_state["matrix_key"] = None
if "top_k_pred" not in st.session_state:
    st.session_state["top_k_pred"] = None
if "digraph_fig" not in st.session_state:
    st.session_state["digraph_fig"] = None
if "layout_style" not in st.session_state:
    st.session_state["layout_style"] = None

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title("Pattern-of-Life Analysis Dashboard")

user_ids = ["000", "003", "014"]

user = st.selectbox("Select a User", options=user_ids)

BASE_DIR = Path(__file__).parent

if user != st.session_state["user_id"]:
    st.session_state["user_id"] = user
    file_path = BASE_DIR / "data" / f"user_{user}.pkl"

    with open(file_path, "rb") as f:
        try:
            pfs = pickle.load(f)
            st.session_state["raw"] = pfs
            logger.debug(f"Sucessfully read .pkl file for user {user}.")

        except FileNotFoundError as e:
            logger.debug(f"A FileNotFoundError occurred: {e}")

    # Integrity Tab State Management
    st.session_state["time_analysis"] = TemporalAnalyzer(user).analyze(pfs)
    logger.debug(f"Temporal analysis complete for user {user}.")

    # Discovery Tab State Management
    c_lat, c_lon = pfs[["lat", "lon"]].median().values
    coords = pfs[["lat", "lon"]].values.tolist()
    map_maker = MapMaker(c_lat, c_lon)
    map_maker.generate_heatmap(coords)
    st.session_state["map_maker"] = map_maker

    logger.debug(f"Base-Map w/ HeatMap FeatureGroup successfully created for user {user}.")

    # Taxonomy Tab State Reset
    st.session_state["stay_points"] = None
    st.session_state["locations"] = None
    st.session_state["profiles"] = None
    st.session_state["profile_metrics"] = None
    st.session_state["chart_data"] = None
    st.session_state["likely_home"] = None

    logger.debug(f"Taxonomy tab session-state values cleared / re-assigned `None`.")

    # Prediction State Reset
    st.session_state["eval_metrics"] = None
    st.session_state["eval_df"] = None
    st.session_state["matrix"] = None
    st.session_state["matrix_key"] = None
    st.session_state["top_k_pred"] = None
    st.session_state["digraph"] = None
    st.session_state["analytic_summary"] = None
    st.session_state["layout_style"] = None

    logger.debug(f"Prediction session-state values cleared / re-assigned `None`.")

    st.rerun()

st.header("Tools")
integrity, tax = st.tabs(["Integrity", "Taxonomy"], width="stretch")

if st.session_state["raw"] is not None:

    with integrity:
        show_integrity()

    with tax:
        show_taxonomy()