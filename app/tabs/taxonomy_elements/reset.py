import streamlit as st

def run_reset():

    # Map reset
    map_maker = st.session_state["map_maker"]
    
    for i, fg in enumerate(map_maker.feature_groups):
        if i != 0:
            del fg

    # Taxonomy Tab State Reset
    st.session_state["stay_points"] = None
    st.session_state["locations"] = None
    st.session_state["profiles"] = None
    st.session_state["profile_metrics"] = None
    st.session_state["diamond_data"] = None
    st.session_state["digraph_data"] = None
    st.session_state["likely_home"] = None

    # Prediction State Reset
    st.session_state["eval_metrics"] = None
    st.session_state["eval_df"] = None
    st.session_state["matrix"] = None
    st.session_state["matrix_key"] = None
    st.session_state["top_k_pred"] = None
    st.session_state["digraph"] = None
    st.session_state["analytic_summary"] = None
    st.session_state["layout_style"] = None