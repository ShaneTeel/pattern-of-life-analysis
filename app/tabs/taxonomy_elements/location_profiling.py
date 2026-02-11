import streamlit as st
from datetime import time, timedelta

def show_location_profiling():
    st.subheader("Location Profiling Configuration")

    with st.container(border=True):
        st.markdown("<b><u>Sleep Location Options</b></u>", unsafe_allow_html=True)
        sleep1, sleep2 = st.columns(2, width="stretch")
        with sleep1:
            sleep_start_help = f"Represents the start hour of the user's core sleep window (i.e., 20:00 == 08:00pm)."
            sleep_start = st.time_input("Window Start", value=time(hour=20, minute=00), step=timedelta(hours=1), help=sleep_start_help, key="sleep_start").hour
        with sleep2:
            sleep_end_help = f"Represents the end hour of the user's core sleep window (i.e., 05:00 == 05:00am)."
            sleep_end = st.time_input("Window End", value=time(hour=5, minute=00), step=timedelta(hours=1), help=sleep_end_help, key="sleep_end").hour

        sleep_dur, sleep_cov = st.columns(2, width="stretch")

        with sleep_dur:
            min_sleep_help = f"The minimum amount of time (in hours) that a user must be within a core sleep window"
            min_sleep_duration = st.number_input("Sleep Duration", value=4, step=1, help=min_sleep_help, key="sleep_duration")

        with sleep_cov:                            
            sleep_coverage_help = f"""Used to determine the method implemented to identify bed-down (home) locations.
    
`sparse` - indicates that the user's GPS data has significant gaps vis-a-vis the sleep window. `sparse` as an argument will result in a permissive bed down location identification.
    
`dense` - indicates that the user's GPS data is complete during the core sleep window. `dense` as an argument will result in a strict bed-down location identification."""
            sleep_coverage = st.selectbox("GPS Coverage", options=["dense", "sparse"], help=sleep_coverage_help, key="sleep_coverage")

    sleep_configs = {
        "sleep_window": (sleep_start, sleep_end),
        "min_sleep": min_sleep_duration,
        "sleep_coverage": sleep_coverage
    }

    with st.container(border=True):
        st.markdown("<b><u>Work Location Options</b></u>", unsafe_allow_html=True)
        work1, work2 = st.columns(2, width="stretch")
        with work1:
            work_start_help = f"Represents the start hour of the user's core work window (i.e., 09:00 == 09:00am)."
            work_start = st.time_input("Window Start", value=time(hour=9, minute=00), step=timedelta(hours=1), help=work_start_help, key="work_win_start").hour
        with work2:
            work_end_help = f"Represents the end hour of the user's core work window (i.e., 18:00 == 06:00pm)."
            work_end = st.time_input("Window End", value=time(hour=18, minute=00), step=timedelta(hours=1), help=work_end_help, key="work_win_end").hour

        work_dur, work_cov = st.columns(2, width="stretch")

        with work_dur:
            min_work_help = f"The minimum amount of time (in hours) that a user must be within a core work window"
            min_work_duration = st.number_input("Work Duration", value=4, step=1, help=min_work_help, key="work_duration")
            
        with work_cov:
        
            work_coverage_help = f"""Used to determine the method implemented to identify work locations.
    
`sparse` - indicates that the user's GPS data has significant gaps vis-a-vis the core work window. `sparse` as an argument will result in a permissive work location identification.

`dense` - indicates that the user's GPS data is complete during the core work window. `dense` as an argument will result in a strict work location identification."""
    
            work_coverage = st.selectbox("GPS Coverage", options=["dense", "sparse"], help=work_coverage_help, key="work_coverage")


    work_configs = {
        "work_window": (work_start, work_end),
        "min_work": min_work_duration,
        "work_days": [1, 2, 3, 4, 5],
        "work_coverage": work_coverage
    }

    return sleep_configs, work_configs