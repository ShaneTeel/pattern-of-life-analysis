import streamlit as st

def show_local_view():
    st.subheader("Local View")
    chart_maker = st.session_state["chart_maker"]
    
    chart_view, time_view = st.columns(2, border=True, width="stretch")

    with chart_view:
        if st.session_state["diamond_data"] is None:
            st.info("A Location's Profile will become available after performing Location Profiling (i.e., `Process`)")
        else:
            diamond_data = st.session_state["diamond_data"]
            diamond_figs = chart_maker.create_location_profile_chart(diamond_data)
            
            st.plotly_chart(diamond_figs, width="stretch", key="routine_diamond_chart")

    with time_view:
        if st.session_state["digraph_data"] is None:
            st.info("A Location's Stability will become available after performing Location Profiling (i.e., `Process`)")
        else:
            digraph_data = st.session_state["digraph_data"]
            stab_gantt = chart_maker.create_stability_gantt(digraph_data)
            st.plotly_chart(stab_gantt, width="stretch", key="stability_gantt")