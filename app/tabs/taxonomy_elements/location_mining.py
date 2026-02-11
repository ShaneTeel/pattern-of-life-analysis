import streamlit as st

def show_location_mining():

    st.subheader("Location Mining")

    with st.container(border=True):
        st.markdown("<b><u>Stay-Point Detection</b></u>", unsafe_allow_html=True)   
        det_param_cols = st.columns(3, width="stretch")
        with det_param_cols[0]:
            dist_help = "Reflects the maximum spatial radius within which consecutive GPS points must be for consideration as a stay-point."
            dist_thresh = st.slider(
                label="Max Radius (meters)", 
                min_value=20, 
                max_value=200, 
                value=100, 
                help=dist_help
            )
            
        with det_param_cols[1]:
            time_help = "Reflects the minimum amount of time consecutive GPS points must be within a spatial radius for consideration as a stay-point."
            time_thresh = st.slider(
                "Min Duration (minutes)", 
                min_value=5, 
                max_value=90, 
                value=30, 
                help=time_help
            )

        with det_param_cols[2]:
            gap_help = """Reflects the maximum amount of time (in minutes) separating two consecutive GPS points. 
            If the time separating the two points is greater than the max gap, than the two points void their consideration as a unique staypoint."""
            gap_thresh = st.slider(
                "Max Gap (minutes)",
                min_value=5, 
                max_value=120, 
                value=60,
                help=gap_help
            )

    det_configs = {
        "distance_thresh": dist_thresh,
        "time_thresh": time_thresh,
        "gap_thresh": gap_thresh
    }

    with st.container(border=True):
        st.markdown("<b><u>Stay-Point Clustering</b></u>", unsafe_allow_html=True)   

        cluster_param_cols = st.columns(2, width="stretch")
            
        with cluster_param_cols[0]:
            eps_help = """Defines a maximum spatial radius for Stay-Points to be within for them to be considered part of the same location.
            The distance provided is in meters; however, the value is converted to a radian-based distance."""
            eps_dist = st.slider(
                label="Min Distance (Meters)", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.2,
                step=0.1,
                help=eps_help
            )
        
        with cluster_param_cols[1]:
            min_k_help = "Defines the minimum number of Stay-Points needed within a specified minimum distance to be considered part of a location."
            min_k = st.slider(
                "Min K (Neighbors)", 
                min_value=2, 
                max_value=10, 
                value=2, 
                help=min_k_help
            )

    cluster_configs = {
        "distance": eps_dist,
        "min_k": min_k
    }

    return det_configs, cluster_configs