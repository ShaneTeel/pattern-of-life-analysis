import streamlit as st
import pandas as pd

from polkit.strategy import MarkovChain, MarkovEvaluator
from polkit.utils import get_logger, train_test_split

logger = get_logger(__name__)

def train_eval_markov_model():

    if st.session_state["locations"] is not None:
        locs = st.session_state["locations"]

        labels = locs["loc_id"].reset_index(drop=True)
        states = sorted(set(labels))

        ## Hours
        datetime = locs["arrived"].reset_index(drop=True)
        hours = datetime.dt.hour

        if st.session_state["eval_df"] is None:
            # Get Train / Test sets
            S_train, H_train, S_test = train_test_split(labels, hours)

            # Initialize / train model
            eval_model = MarkovChain(states)
            eval_model.fit(S_train, H_train)

            # Evaluate Model
            eval = MarkovEvaluator(eval_model, k=3)
            results = eval.evaluate(S_test)

            n_states = len(set(eval.test_states))
            
            if n_states is None or n_states == 0:
                st.warning("Arguments for Stay-Point Detection and Stay-Point Clustering are too restrictive. Please adjust before continuing.")
                st.stop()

            rand_top_k = min(eval.k, n_states) / n_states

            eval_metrics = {
                "top_k": results["top_k"],
                "k": eval.k,
                "next_step_accuracy": results["next_step"],
                "improvement": (results["top_k"] - rand_top_k) / rand_top_k * 100,
                "rand_top_k": rand_top_k * 100
            }

            # Generate / View Results
            st.session_state["eval_metrics"] = eval_metrics
        
            results_df = pd.DataFrame().from_dict(results["top_k_by_state"], orient="index")
            st.session_state["eval_df"] = results_df

        if st.session_state["matrix"] is None:
            model = MarkovChain(states)

            model.fit(labels, hours)
            matrix, key = model.get_transition_matrix()
            st.session_state["matrix"] = matrix
            st.session_state["matrix_key"] = key

            if st.session_state["likely_home"] is not None:
                start = st.session_state["likely_home"]

            else:
                profiles = st.session_state["profiles"]
                start = profiles.loc[profiles["Routine Index"].idxmax(), "Location ID"]
            
            top_k_pred = model.predict_next_k(start, k=3)
            st.session_state["top_k_pred"] = top_k_pred