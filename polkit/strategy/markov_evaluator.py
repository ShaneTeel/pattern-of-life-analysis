from __future__ import annotations

import numpy as np
import pandas as pd

from .markov_chain import MarkovChain
from polkit.utils import get_logger

logger = get_logger(__name__)

class MarkovEvaluator:

    def __init__(self, markov_model:MarkovChain, k:int=3):

        '''
        Parameters
        ----------
        markov_model : MarkovChain
            An instance of a fitted MarkovChain model that can be used to generate predictions. 
        k : int, default=3
            The number of probabilities (sorted from high --> low) to use when generating a prediction
        '''
        self.model = markov_model
        self.results = {}
        self.k = k
        self.test_states = []

        logger.debug("MarkovEvaluator successfully initialized.")

    def next_step_accuracy(self, test_sequences:list | np.ndarray | pd.Series):
        '''
        Description
        -----------
        For each sequence in test_sequences
            for every (current, next) pair in sequence
                `X_test` == current 
                `y_true` == next
                `y_pred` = model.predict_next_k(`X_test`)

        If `y_true` == `y_pred` then the number of correct predictions is incremented by 1. 
        After every sequential pair in every sequence is iterated through, the number of correct next-step 
        predictions is divided by the total number of next-step predictions that occurred.

        Parameters
        ----------
        test_sequence : list | NDArray | pd.Series
            List or array-like of integer values representing labels/states that the model will use to generate predictions
            and whose values will be used to compare predictions against.
            
        Returns
        -------
        Next-Step Accuracy : float
            The number of correct next-step predictions divided by the total number of next-step predictions. 
        '''        
        correct = 0
        total = 0
        
        for i, sequence in enumerate(test_sequences):
            if len(sequence) < 2:
                logger.info(f"Sequence at index {i} contains less than 2 points. Skipping")
                continue
                
            for j in range(len(sequence) - 1):
                # Get X_test, y_true
                X_test = sequence[j]
                y_true = sequence[j+1]

                # Get y_pred
                y_pred = self.model.predict_next_k(X_test, 1, random=False)

                # Evaluate
                if y_true in y_pred:
                    correct += 1
                total += 1
                
        return correct / total if total != 0 else 0.0
    
    def top_k_accuracy(self, test_sequences):
        '''
        Description
        -----------
        For each sequence in test_sequences
            for every (current, next) pair in sequence
                `X_test` == current 
                `y_true` == next
                `y_pred` = model.predict_next_k(`X_test`, k)

        Top-k Accuracy: Is `y_true` in the top-k number of predictions?
        
        Parameters
        ----------
        test_sequence : list | NDArray | pd.Series
            List or array-like of integer values representing labels/states that the model will use to generate predictions
            and whose values will be used to compare predictions against.
        
        Returns
        -------
        Top-k Accuracy : float
            The number of correct-k predictions divided by the total number of predictions.
        '''
        correct = 0
        total = 0

        for i, sequence in enumerate(test_sequences):
            if len(sequence) < 2:
                logger.info(f"Sequence at index {i} contains less than 2 points. Skipping")
                continue
    
            for j in range(len(sequence) - 1):
                X_test = sequence[j]
                y_true = sequence[j + 1]

                y_pred = self.model.predict_next_k(X_test, self.k, False)

                if y_true in y_pred:
                    correct += 1

                total += 1

        return correct / total if total != 0 else 0.0
    
    def top_k_by_state_accuracy(self, test_sequences: list | np.ndarray | pd.Series):
        '''
        Description
        -----------
        For each sequence in test_sequences
            for every (current, next) pair in sequence
                `X_test` == current 
                `y_true` == next
                `y_pred` = model.predict_next_k(`X_test`, k)

        Top-k Per State Accuracy: Is `y_true` in the top-k number of predictions for a state?
        
        Parameters
        ----------
        test_sequence : list | NDArray | pd.Series
            List or array-like of integer values representing labels/states that the model will use to generate predictions
            and whose values will be used to compare predictions against.

        Returns
        -------
        Top-k Accuracy Per State : dict[dict]
            A dict-like object formatted as outlined in the example below:
            ```
            {
            0: {
                "correct": 5,
                "total": 10,
                "ratio": 0.50
                }
            }
            ```
        '''
        state_stats = {}

        for i, sequence in enumerate(test_sequences):
            if len(sequence) < 2:
                logger.info(f"Sequence at index {i} contains less than 2 points. Skipping")
                continue

            for j in range(len(sequence) - 1):
                X_test = sequence[j]
                y_true = sequence[j+1]

                if X_test not in state_stats:
                    state_stats[X_test] = {"correct": 0, "total": 0}
                
                y_pred = self.model.predict_next_k(X_test, self.k, False)

                if y_true in y_pred:
                    state_stats[X_test]["correct"] += 1
                
                state_stats[X_test]["total"] += 1
                
        state_stats = self._calculate_state_ratio(state_stats)

        return state_stats
    
    def evaluate(self, test_sequences: list | np.ndarray | pd.Series):

        next_step = 0
        top_k = 0
        total = 0

        state_stats = {}

        for i, sequence in enumerate(test_sequences):
            if len(sequence) < 2:
                logger.info(f"Sequence at index {i} contains less than 2 points. Skipping")
                continue
            
            for j in range(0, len(sequence) - 1):
                X_test = sequence[j]
                y_true = sequence[j+1]
                self.test_states.append(X_test)

                if X_test not in state_stats:
                    state_stats[X_test] = {"correct": 0, "total": 0}

                y_pred = self.model.predict_next_k(X_test, self.k, False)

                if y_true in y_pred:
                    top_k += 1
                    state_stats[X_test]["correct"] += 1
                    if y_true == y_pred[0]:
                        next_step += 1
                
                total += 1
                state_stats[X_test]["total"] += 1
        state_stats = self._calculate_state_ratio(state_stats)

        self.results = {
            "top_k_by_state": state_stats,
            "top_k": top_k / total if total != 0 else 0.0,
            "next_step": next_step / total if total != 0 else 0.0
        }
        return self.results

    def generate_summary(self):
        if not self.results:
            raise RuntimeError("Error, user must call `.evaluate()` before calling `.generate_summary()`.")

        summary = "\n"
        summary += "="*50
        summary += "\n"
        summary += "MarkovChain Evaluator Results\n"
        summary += "="*50
        summary += "\n"
        summary += f"\tNext-Step Accuracy: {self.results["next_step"]:.2%}\n"
        summary += f"\tTop-{self.k} Accuracy: {self.results["top_k"]:.2%}\n"
        summary += f"\tTop-{self.k} Accuracy by State\n"
        for state, stat in sorted(self.results["top_k_by_state"].items()):
            summary += f"\t\tState {state}: {stat["ratio"]:.2%} ({stat["correct"]} / {stat["total"]})\n"
        
        n_states = len(set(self.test_states))
        
        rand_top_k = min(self.k, n_states) / n_states

        top_k = self.results["top_k"]
        improvement = (top_k - rand_top_k) / rand_top_k * 100
        rand_top_k *= 100

        summary += "="*50
        summary += "\n\tBaseline Comparison:\n"
        summary += f"\t\tRandom top-{self.k} accuracy: {rand_top_k:.2f}%\n"
        summary += f"\t\tModel Improvement: {improvement:.2f}%\n"

        summary += "="*50

        return summary

    def _calculate_state_ratio(self, state_stats:dict):
        for state in state_stats.keys():
            correct = state_stats[state]["correct"]
            total = state_stats[state]["total"]
            state_stats[state]["ratio"] = correct / total if total != 0 else 0.0
        return state_stats