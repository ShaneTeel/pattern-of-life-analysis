import streamlit as st

def show_analytic_summary():
    
    with st.container(border=True, width="stretch"):
        st.markdown("<b><u>Analytic Summary</u></b>", unsafe_allow_html=True)
        if st.session_state["eval_metrics"] is None:
            st.info("An Analytic Summary will become available after performing Location Profiling (i.e., `Process`)")
        else:
            if st.session_state["analytic_summary"] is None:
                user_id = st.session_state["user_id"]
                eval_metrics = st.session_state["eval_metrics"]
                profile_metrics = st.session_state["profile_metrics"]

                analytic_summary = f"""User {user_id}'s location history, as defined by the collection period, consists of {profile_metrics["Locations"]} total unique locations with the following distribution by classification:
- {profile_metrics["Anchors"]} Anchor Locations
- {profile_metrics["Habits"]} Habit Locations
- {profile_metrics["Recurring"]} Recurring Locations
- {profile_metrics["Transient"]} Transient Locations

Taxonomy identified {profile_metrics["Homes"]} candidate Home locations and {profile_metrics["Works"]} work locations,
User {user_id}'s overall spatial focus (weighted by dwell-time) is {profile_metrics["Rg"] / 1000:.2f} kilometers from their center of mass (see map below). 

Next-location prediction for User {user_id} achieves {eval_metrics["next_step_accuracy"]:.2%} next-step accuracy and a Top-{eval_metrics["k"]} accuracy of {eval_metrics["top_k"]:.2%},
demonstrating a ~{eval_metrics["improvement"]:.2f}% improvement over a random top-{eval_metrics["k"]} guess ({eval_metrics["rand_top_k"]:.2f}%).
Given that User {user_id}'s movements are characterized by a {profile_metrics["H"]:.2%} level of overall certainty, 
Taxonomy has {profile_metrics["Confidence"]} confidence in it's ability to forecast a User {user_id}'s movements.
"""
                st.session_state["analytic_summary"] = analytic_summary
            if st.session_state["analytic_summary"] is not None:
                analytic_summary = st.session_state["analytic_summary"]
                st.write(analytic_summary)