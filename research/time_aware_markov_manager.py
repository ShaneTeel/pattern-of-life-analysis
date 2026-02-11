from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

from polkit.strategy import MarkovChain, MarkovEvaluator
from polkit.utils import get_logger

logger = get_logger(__name__)

class TimeAwareMarkovManager:
    '''
    Description
    -----------
    Class dedicated to managing Markov models generated per time step.
    Developed specifically for location prediction and movement simulation.
    '''

    def __init__(self, time_step:Literal["month", "day_of_week", "time_of_day"]="day_of_week", time_gap:int=8, length:int=5, n_sims:int=5):
        '''
        Description
        -----------
        Class dedicated to computing transition probabilities between states and generating predictions based on the computed probabilities.
        Developed specifically for location prediction and movement simulation.

        Parameters
        ----------
        time_step : Literal["all", "month", "day_of_week", "time_of_day"], default="day_of_week"
            Mechanism for temporally discretizing the data into specific time periods. 
            This allows the user to generate probabilities and predictions for each distinct time period contained within the time step.
        
        time_gap : int, default=8
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
        self.time_step = time_step
        self.time_gap = time_gap
        self.length = length
        self.n_sims = n_sims
        self.models = {}
        self.predictions = {}
        self.evaluations = {}

        self._is_fitted = False

        logger.debug("TimeAwareMarkov successfully initialized.")

    def fit_predict(self, states:pd.Series, datetime:pd.Series, start:int, method:Literal["median", "mode"]="mode"):
        '''
        Description
        -----------
        Public method chaining the `fit()` and `.predict()` methods together.
        Calling this method will first compute the probabilities of a known state transition from one state to a subsequent state for step in time_step
        and then will generate a prediction for each step in time_step based on the computed probabilities for each time_step.

        Parameters
        ----------
        states : pd.Series
            The semantic labels, ordered by datetime, that will be used to generate the transition probability matrix by 
            determining the likelihood of transitioning from label at index `i` to label at index `i+1`

        datetime : pd.Series
            The datetime objects that correspond to each semantic label included in the locations argument. 
            The method will parse the datetime series into time-series info that will be used in conjuction with the `time_step` argument to generate separate models for each step in time_step.

        start : int, default=None
            An integer representing the start of the sequence the user wishes to generate. 
            
        method : Literal["median", "mode"], default="mode"
            Aggregation method used to determine which prediction is returned from a sequence for a given index.
        
        Returns
        ------- 
        predictions : dict[list]
            A dict of lists containing predicted values with the argument passsed for `start` beginning the sequence
        '''
        return self.fit(states, datetime).predict(start, method)

    def fit(self, states:pd.Series, datetime:pd.Series):
        '''
        Description
        -----------
        Public method for computing the probabilities of known state transitions across each time step specified at initialization.

        Parameters
        ----------
        states : pd.Series
            The semantic labels, ordered by datetime, that will be used to generate the transition probability matrix by 
            determining the likelihood of transitioning from label at index `i` to label at index `i+1`

        datetime : pd.Series
            The datetime objects that correspond to each semantic label included in the locations argument. 
            The method will parse the datetime series into time-series info that will be used in conjuction with the `time_step` argument to generate separate models for each step in time_step.
        
        Returns
        ------- 
        self : MarkovChain
            The model fitted
        '''
        time_series = self._parse_datetime(datetime)
        states = states.unique()
        
        self.time_states = time_series[self.time_step].unique()
        masks = [time_series[self.time_step] == state for state in self.time_states]

        for state, mask  in zip(self.time_states, masks):
            locs = states[mask].reset_index(drop=True)
            hours = time_series.loc[mask, "hour"].reset_index(drop=True)
            if len(locs) >= 2:

                S_train, H_train, S_test = self._train_test_split(locs, hours) 
                chain = MarkovChain(states.unique(), self.time_gap, self.length, self.n_sims)
                self.models[state] = chain.fit(S_train, H_train)
                eval = MarkovEvaluator(chain, k=3)
                _ = eval.evaluate(S_test)
                self.evaluations[state] = eval.generate_summary()
            else:
                self.models[state] = None

        self._is_fitted = True

        return self
    
    def predict(self, start:int, method:Literal["median", "mode"]="mode"):
        '''
        Description
        -----------
        Public method for simulating a series of sequences based on the argument passed for `n_sims` at object initialization and for each step in time step.
        The final prediction is based on the median of all simulations for a given index.

        Parameters
        ----------
        start : int, default=None
            An integer representing the start of the sequence the user wishes to generate.

        method : Literal["median", "mode"], default="mode"
            Aggregation method used to determine which prediction is returned from a sequence for a given index.
        
        Returns
        ------- 
        predictions : dict[list]
            A dict of lists containing predicted values with the argument passsed for `start` beginning the sequence
        '''
        self._fit_check()

        for state, model in self.models.items():
            model = self.models[state]
            if model is not None:
                self.predictions[state.item()] = model.predict(start, method)
            else:
                self.predictions[state.item()] = np.array([])
        
        return self.predictions

    def _parse_datetime(self, datetime:pd.Series):
        hour = datetime.dt.hour
        conditions = [
        (hour >= 5) & (hour < 12),
        (hour >= 12) & (hour < 17),
        (hour >= 17) & (hour < 22)
        ]
        options = ['morning', 'afternoon', 'evening']
        time_of_day = np.select(conditions, options, default='night')

        month = datetime.dt.month
        day_of_week = datetime.dt.day_of_week

        return pd.DataFrame({
            "datetime": datetime, 
            "month": month, 
            "day_of_week": day_of_week, 
            "hour": hour, 
            "time_of_day": time_of_day})
    
    def _train_test_split(self, states:pd.Series | np.ndarray | list, hours:pd.Series | np.ndarray | list):
        # Get train indices; train model
        split = int(len(states) * .8)

        S_train = states[:split]
        H_train = hours[:split]

        # Get test indices 
        stop = len(states[split:])

        S_test = []

        for step in range(5, stop, 5):
            X_test_labels = states[split:split+step]
            X_test_labels.reset_index(drop=True, inplace=True)
            S_test.append(X_test_labels)
            split += 5

        return S_train, H_train, S_test
    
    def _fit_check(self):
        '''
        Description
        -----------
        Utility function used to raise a `RuntimeError()` whenever a method is called before `.fit()` or `.fit_predict()`.
        '''
        if not self._is_fitted:
            raise RuntimeError("MarkovChainGenerator must be fitted before performing this operation.")