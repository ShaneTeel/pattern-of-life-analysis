import streamlit as st

def show_local_view():
    st.subheader("Local View")
    chart_maker = st.session_state["chart_maker"]
    

    if st.session_state["chart_data"] is None:
        st.info("A Location's Profile will become available after performing Location Profiling (i.e., `Process`)")
    else:
        chart_data = st.session_state["chart_data"]
        profile_radar_figs = chart_maker.create_location_profile_chart(chart_data)
        
        st.plotly_chart(profile_radar_figs, width="stretch", key="profile_radar_charts")