import streamlit as st
import pandas as pd

def show_integrity():

    st.caption("""Welcome to Integrity!
               
Integrity requires no user action. The system auto-generates a visual report of the temporal completeness of a user's raw GPS fixes.
Think of Integrity as a data health check that provides the viewer with a quick snapshot of 
the density, distrubtion, and resolution of a user's GPS data, with regards to time, to inform a viewers confidence in any subsequent analysis.
""")
    # Get global states
    pfs = st.session_state["raw"]
    user_id = st.session_state["user_id"]

    if st.session_state["chart_maker"] is not None:
        chart_maker = st.session_state["chart_maker"]

    if st.session_state["time_analysis"] is not None:
        analysis = st.session_state["time_analysis"]["full"]
        
        analysis_cols = st.columns([1,2], border=True)
        with analysis_cols[0]:
            with st.container():
                st.markdown("<b><u>Analytic Summary</u></b>", unsafe_allow_html=True)
                st.write(f"""User {user_id}'s data contains {len(pfs):,.0f} recorded GPS fixes and ranges **{analysis["temporal_coverage"]["total_days"]}** total days of collection
                        with **{analysis["temporal_coverage"]["active_days"]}** active days (**{analysis["temporal_coverage"]["coverage_ratio"]:.2%}** coverage). Collection starts on {pfs["datetime"].dt.date.min()} 
                        and continues through {pfs["datetime"].dt.date.max()} with {len(analysis["gaps"])} identified gaps in data collection.
                        On days with active collection, a median gap (resolution) of {analysis["density"]["median_gap_minutes"]:.2f} minutes exists between GPS fixes.""")
            
            with st.container():
                if analysis["gaps"]:
                    st.markdown("<b><u>Significant Gaps</u></b>", unsafe_allow_html=True)
                    gaps_df = pd.DataFrame(analysis["gaps"]).reset_index(names="Gap ID")
                    gaps_df.sort_values(by="duration_hours", ascending=False, inplace=True)
                    st.dataframe(gaps_df[["start", "end", "duration_hours"]].head(), hide_index=True)

        with analysis_cols[1]:
            if gaps_df is not None or len(gaps_df) != 0:   
                gap_fig = chart_maker.create_gaps_gantt(gaps_df)
                st.plotly_chart(gap_fig, width="stretch", key="gap_fig")

        with st.container(border=True):
            st.subheader("Collection Density by Date")
            calplot_fig = chart_maker.create_calendar_heatmap(pfs, user_id)
            st.pyplot(calplot_fig)

        time_cols = st.columns(2, width="stretch", border=True)
        
        with time_cols[0]:
            st.subheader("Collection Density by Hour")
            polar_fig = chart_maker.create_time_wheel(pfs, user_id)    
            st.plotly_chart(polar_fig, width="stretch", key="polar_fig")

        with time_cols[1]:
            st.subheader("Collection Density by Day of Week")
            bar_fig = chart_maker.create_day_of_week_chart(pfs, user_id)
            st.plotly_chart(bar_fig, width="stretch", key="bar_fig")
