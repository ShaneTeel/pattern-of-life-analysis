import streamlit as st

def show_data():
    st.subheader("Data View")
    df_options = ["Stay-Point Clusters", "Profiled Locations", "Top-K by State"]
    data_frame = st.selectbox("Select a DataFrame", df_options)

    
    if data_frame == df_options[0]:
        if st.session_state["locations"] is not None:
            locs = st.session_state["locations"]

            st.dataframe(locs.drop("user_id", axis=1), hide_index=True)

        else:
            st.info("Stay-Point Clusters DataFrame does not exist yet. User must detect and cluster stay-points.")
    
    if data_frame == df_options[1]:
        if st.session_state["profiles"] is not None:
            profiled_df = st.session_state["profiles"]
            st.dataframe(profiled_df, hide_index=True)
        else:
            st.info("Profiled Locations DataFrame does not exist yet. User must cluster stay-points and profile clusters.")

    if data_frame == df_options[2]:
        if st.session_state["eval_df"] is not None:            
            eval_df = st.session_state["eval_df"]
            st.dataframe(eval_df, hide_index=False)
        else:
            st.info("Top-K by State DataFrame does not exist yet. User must profile clusters and fit Markov Chain.")