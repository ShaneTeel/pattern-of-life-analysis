from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal

from polkit.utils import get_logger

logger = get_logger(__name__)

class MarkovChain:
    '''
    Description
    -----------
    Class dedicated to computing transition probabilities between states and generating predictions based on the computed probabilities.
    Developed specifically for location prediction and movement simulation.
    '''

    def __init__(self, labels:pd.Series | np.ndarray, time_gap:int=24, length:int=5, n_sims:int=5):
        '''
        Description
        -----------
        Class dedicated to computing transition probabilities between states and generating predictions based on the computed probabilities.
        Developed specifically for location prediction and movement simulation.

        Parameters
        ----------
        labels : np.ndarray
            An array consisting of the unique discrete states within the dataset. 
            Assumes that states are a contigous range of integer values.

        time_gap : int, default=24
            The maximimum amount of time, in hours, that separate one state from the state immediately following it. 
            This is used to prevent computing the transition probability of one state to the next if their is a likelhood that the transition is the result of data quality issues.
            For example, if data collection is sparse on a given day and results in only a single detected staypoint, 
            computing the transition probability of any state to or from that observed event could result inaccurate results

        length : int, default=5
            The number of state transitions to predict when calling `.predict()` or `.fit_predict()`

        n_sims : int, default=5
            The value must be an odd number or a `RuntimeError()` is raised. This number represents the number of simulations that are run when calling `.predict()` or `.fit_predict()`.
            After all simulations complete, the median value for all simulations at a given index are returned as the final prediction. 

        Raises
        ------
        `RuntimeError()`: If argument passed to `n_sims` is NOT an odd integer
        '''
        if n_sims % 2 == 0:
            raise RuntimeError(f"Error: Argument passed for `n_sims` must be odd.")
        
        # State management
        self.states = labels
        self.n_learned_states = 0
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
        self.population = list(self.state_to_idx.values())

        self.time_gap = time_gap # Filtering mechanism to prevent gaps in data from affecting probas
        self.length = length # Sequence length used for prediction
        self.n_sims = n_sims # Number of sims to run through before generating a final prediction
        self.matrix = None # Placeholder for the transition probability matrix

        self._is_fitted = False

        logger.debug("MarkovChain successfully initialized.")

    def fit_predict(self, states:pd.Series, hours:pd.Series, start:int, method:Literal["median", "mode"]="mode"):
        '''
        Description
        -----------
        Public method chaining the `fit()` and `.predict()` methods together.
        Calling this method will first compute the probabilities of a known state transition from one state to a subsequent state
        and then will generate a prediction based on the computed probabilities.

        Parameters
        ----------
        states : pd.Series
            The semantic labels, ordered by datetime, that will be used to generate the transition probability matrix by 
            determining the likelihood of transitioning from label at index `i` to label at index `i+1`

        hours : pd.Series
            The hours that correspond to each semantic label included in the locations argument. The time delta resulting from the absolute difference between the hour that corresponds to the location at index `i`
            and the hour that corresponds to the location and index `i+1` is used against the argument for `time_gap` passed at initialization. If the time delta is > `time_gap`,
            the that transition is not inlcuded in the probability matrix calculation.

        start : int, default=None
            An integer representing the start of the sequence the user wishes to generate. 
        
        method : Literal["median", "mode"], default="mode"
            Aggregation method used to determine which prediction is returned from a sequence for a given index.
        
        Returns
        ------- 
        sequence : list
            A list of predicted values with the argument passsed for `start` beginning the sequence
        '''
        return self.fit(states, hours).predict(start, method)
    
    def fit(self, states:pd.Series, hours:pd.Series):
        '''
        Description
        -----------
        Public method for computing the probabilities of a known state transition from one state to a subsequent state.

        Parameters
        ----------
        states : pd.Series
            The semantic labels, ordered by datetime, that will be used to generate the transition probability matrix by 
            determining the likelihood of transitioning from label at index `i` to label at index `i+1`

        hours : pd.Series
            The hours that correspond to each semantic label included in the locations argument. The time delta resulting from the absolute difference between the hour that corresponds to the location at index `i`
            and the hour that corresponds to the location and index `i+1` is used against the argument for `time_gap` passed at initialization. If the time delta is > `time_gap`,
            the that transition is not inlcuded in the probability matrix calculation. 
        
        Returns
        ------- 
        self : MarkovChain
            The model fitted
        '''
        if len(states) < 2:
            logger.debug("Error, argument for states does not include enough states to calculate probability matrix.")
            raise ValueError("Error, argument for states does not include enough states to calculate probability matrix.")
        
        trans_mat = np.zeros((len(self.population), len(self.population)), dtype=np.float32)

        for i in range(0, len(states) - 1):
            time_delta = abs(hours[i+1] - hours[i])
            if time_delta <= self.time_gap:
                origin = states[i]
                dest = states[i+1]

                if pd.notna(origin) and pd.notna(dest):
                    origin_idx = self.state_to_idx[origin]
                    dest_idx = self.state_to_idx[dest]
                    trans_mat[origin_idx, dest_idx] += 1

        for i in self.state_to_idx.values():

            row_sum = trans_mat[i, :].sum()
            if row_sum > 0:
                self.n_learned_states += 1
                trans_mat[i, :] /= row_sum
            else:
                trans_mat[i, :] = 1.0 / len(self.population)

        self.matrix = trans_mat
        self._is_fitted = True
        return self

    def predict(self, start:str, method:Literal["median", "mode"]="mode"):
        '''
        Description
        -----------
        Public method for simulating a series of sequences based on the argument passed for `n_sims` at object initialization.
        The final prediction is based on the median of all simulations for a given index.

        Parameters
        ----------
        start : int, default=None
            An integer representing the start of the sequence the user wishes to generate. 
        
        method : Literal["median", "mode"], default="mode"
            Aggregation method used to determine which prediction is returned from a sequence for a given index.
        
        Returns
        ------- 
        sequence : list
            A list of predicted values with the argument passsed for `start` beginning the sequence
        '''
        self._fit_check()

        predictions = np.zeros((self.length, self.n_sims))

        start_idx = self.state_to_idx[start]

        for i in range(self.n_sims):
            predictions[:, i] = self._generate_sequence(start_idx)
        
        if method == "median":
            aggregated = np.median(predictions, axis=1)

        aggregated, _ = stats.mode(predictions, axis=1)

        final = [self.idx_to_state[idx] for idx in aggregated]

        return final
    
    def get_transition_matrix(self):
        '''
        Description
        -----------
        Public method for access the calculated transition matrix (probabilities of transition from one state to a subsequent state).

        Returns
        -------
        Matrix : 2-d NumPy Array
            Transition matrix
        state_to_idx : dict
            Key to map matrix indices to original state labels
        '''
        self._fit_check()
        return self.matrix, self.state_to_idx

    def _generate_sequence(self, start_idx:int):
        '''
        Description
        -----------
        Private method for generating a sequence based on probability matrix
        calculated during `.fit()`
        
        Parameters
        ----------
        start : int
            An integer representing the start of the sequence the user wishes to generate. 
        
        Returns
        ------- 
        sequence : list
            A list of predicted values ordered beginning with the argument passsed for `start`
        '''

        sequence = [start_idx]

        current_state_idx = start_idx

        for _ in range(self.length - 1):
            probas = self.matrix[current_state_idx, :]
            pred_idx = np.random.choice(
                a=self.population,
                p=probas
            )
            current_state_idx = pred_idx
            sequence.append(pred_idx)

        return np.array(sequence)
    
    def predict_next_k(self, state:str, k:int, random:bool=False):
        '''
        Description
        -----------
        Used to return k-predictions for a single state using the state's specific probability matrix generated during `.fit()`.
        
        Parameters
        ----------
        state: str
            Label representing a specific state that the user wants generate predictions for.
        k : int
            The number of probabilities (sorted from high --> low) to use when generating a prediction
        random : bool, default=False
            Boolean argument to determine whether `np.random.choice()` or `np.argsort()` (with `[-k:]` slice) is used for prediction.            

        Returns
        -------
        k-predictions : list
            List of predictions length k.

        Raises
        ------
        RuntimeError
            If user calls method prior to `.fit()`.

        '''
        self._fit_check()

        state_idx = self.state_to_idx[state]

        probas = self.matrix[state_idx]

        if not random:
            k_preds = np.argsort(probas)[-k:][::-1]
        else:
            k_preds = np.random.choice(
                a=self.population,
                p=probas,
                replace=False,
                size=k
            )

        return [self.idx_to_state[idx] for idx in k_preds]

    def _fit_check(self):
        '''
        Description
        -----------
        Utility function used to raise a `RuntimeError()` whenever a public method is called before `.fit()` or `.fit_predict()`.
        '''
        if not self._is_fitted:
            raise RuntimeError("MarkovChain must be fitted before performing this operation.")