import streamlit as st
from streamlit_folium import folium_static

from polkit.visualize import NetworkBuilder

def show_global_view():
    st.subheader("Global View")

    map_view, digraph_view = st.columns(2, width="stretch", border=True)
    
    with map_view:
        st.markdown("<b>Location Map</b>", unsafe_allow_html=True)        
        if st.session_state["map_maker"] is not None:
            map_maker = st.session_state["map_maker"]

            if st.session_state["stay_points"] is not None:
                sp_name = "Stay-Points"
                if sp_name not in map_maker.feature_groups:
                    sps = st.session_state["stay_points"]
                    map_maker.add_staypoints(sps, sp_name)
            
            if st.session_state["profiles"] is not None:
                loc_focus_name = "Spatial Focus (Local)"
                if loc_focus_name not in map_maker.feature_groups:
                    profile_df = st.session_state["profiles"]
                    map_maker.add_location_radius(profile_df, loc_focus_name)

            if st.session_state["profile_metrics"] is not None:
                global_focus_name = "Spatial Focus (Global)"
                if global_focus_name not in map_maker.feature_groups:
                    metrics = st.session_state["profile_metrics"]
                    cm = metrics["cm"]
                    Rg = metrics["Rg"]
                    map_maker.add_profile_metrics(cm, Rg, global_focus_name)

            map_maker.add_layer_control()
            

            folium_static(
                map_maker.m,
                width=1200,
                height=500
            )

    with digraph_view:
        layout_style = st.selectbox("Layout Style", options=["spring", "kamada", "circular"], index=2)
        if st.session_state["matrix"] is None:
            st.info("A users Movement Network will become available after performing Location Profiling (i.e., `Process`)")
        else:
            if st.session_state["digraph_fig"] is None or st.session_state["layout_style"] != layout_style:
                network_builder = NetworkBuilder()
                digraph_data = st.session_state["digraph_data"]
                matrix = st.session_state["matrix"]
                matrix_key = st.session_state["matrix_key"]
                fig = network_builder.build_network(layout_style, matrix, matrix_key, digraph_data)
                st.session_state["digraph_fig"] = fig
                st.session_state["layout_style"] = layout_style

            fig = st.session_state["digraph_fig"]
            st.plotly_chart(fig, width="stretch", key="digraph_chart")