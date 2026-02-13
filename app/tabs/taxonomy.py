import streamlit as st

from .taxonomy_elements import *

def show_taxonomy():

    st.caption("""Welcome to Taxonomy!
               
Taxonomy is a multi-step location-based behavioral profiling tool. Taxonomy starts with location mining, a process that transforms raw data into discrete locations.
It then generates a behavioral profile for each discrete location based on visit patterns (duration, visit count, etc.). Lastly, Taxonomy identifies a user's anchor points, specifically, bed-down and work locations. 
To get started, adjust the configuration options and select the corresponding action button below (`Process`). 
""")
    with st.sidebar:
        process = st.button("Process")
        det_configs, cluster_configs = show_location_mining()
        sleep_configs, work_configs = show_location_profiling()

    if process:
        run_reset()
        run_process(det_configs, cluster_configs, sleep_configs, work_configs)

    train_eval_markov_model()

    show_analytic_summary()

    st.divider()

    show_global_view()
    
    st.divider()

    show_local_view()
    
    st.divider()

    show_data()