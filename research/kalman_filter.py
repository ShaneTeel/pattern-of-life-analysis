import numpy as np
import pandas as pd
from mobility.utils import get_logger

logger = get_logger(__name__)
    
class KalmanFilter():
    '''
    Kalman filter for tracking location behavioral patterns.
    
    Each location has a state vector describing temporal / tendency characteristics:
    - Temporal: arrival hour, duration
    - Tendency: visit rate, recency (last visit)

    Intent is for state to evolve as behavior changes
    '''
        
    def __init__(self, base_visits:pd.DataFrame, location_id:int):
        '''
        Description
        -----------
        Initialize filter for single location.

        Parameters
        ----------
        base_visits : pd.DataFrame
            Historical visits to this location
        location_id : int
            Semantic label or identifier assigned to the location
        '''

        self.loc_id = location_id
        self.state_dims = 3 # Hour, duration, frequency, d_hour, d_duration, d_frequency

        # Initial state from base records
        self.x = self._initialize_state(base_visits) 
        self.I = np.eye(len(self.x)) # identity
        
        # Covariance (uncertainty in state estimate)
        self.P = self._initialize_P()

        # Initial observation, state transition, and process noise
        self.H, self.F, self.Q = self._initialize_dynamics()
        
        # Cache transposes
        self.F_T = self.F.T
        self.H_T = self.H.T

        # Last update time (for computing dt between visits)
        self.last_visit_time = base_visits["arrived"].max()

        logger.debug("Initialized KalmanFilter")

    def update_with_visit(self, visit_time, duration, context):
        '''
        Description
        -----------
        Update lcoation state estimate with new visit observation.
        
        Parameters
        ----------
        visit_time : datetime
            When user arrived at location
        duration : float
            How long user stayed (minutes)
        context : dict
            Additional details (day_of_week, previous_locations, etc.)
            
        Returns
        -------
        state_estimate : dict
            Updated location profile
            
        '''
        # Calculate time since last visit, in hours (affect prediction uncerainty)
        dt = (visit_time - self.last_visit_time).total_seconds() / 3600
        
        # Predict step: Extrapolate state forward
        self._predict(dt)
        
        # Measurement: [hour, duration, frequency]
        observation = self._create_observation(visit_time, duration, dt)

        # Adaptive measurement noise based on context quality
        R = self._compute_R(context)
        
        # Update step: Incorporate new observation
        self._update(observation, R)

        # Update last visit time
        self.last_visit_time = visit_time

        # Return current state estimate
        return self.get_state_estimate()

    def _predict(self, dt):
        '''
        Prediciton step: Extrapolate state forward by dt (time between visits)
        '''
        F_scaled = self._scale_F_by_dt(dt)
        
        # Predict state
        self.x = F_scaled @ self.x
        
        # Predict covariance
        # Scale Q by dt (more time between visits == more uncertainty accumulation)
        Q_scaled = self.Q * dt
        self.P = F_scaled @ self.P @ F_scaled.T + Q_scaled

        if np.trace(self.P) > 1000.0:
            logger.warning(f"Location {self.loc_id} filter divering, resetting state.")
            self.P = self._initialize_P()

    def _scale_F_by_dt(self, dt):
        '''
        Scale state transition matrix by time interval.
        new = old + velocity * dt
        '''
        I = np.eye(self.state_dims)
        zeros = np.zeros((self.state_dims, self.state_dims))

        # Position update scaled by dt
        # Velocity remains constant
        return np.block([
            [I, dt * I],
            [zeros, I]
        ])

    def _create_observation(self, visit_time:pd.Timestamp, duration:float, dt:float):
        '''
        Description
        -----------
        Create observation vector from visit.

        Parameters
        ----------
        visit_time : datetime
        duration : float (minutes)
        dt : float (timedelta (in hours) since last visit)
        '''
        hour = visit_time.hour
        
        # Frequency observation: compute instantaneous frequency
        # If dt is small (indicates frequent visits), frequency is high
        # If dt is large (indicates infrequent visits), frequency is low

        # visits per week
        instant_freq = 1.0 / max(dt / 24.0, 0.1)
        return np.array([hour, duration, instant_freq]).reshape(-1, 1)
    
    def _update(self, observation, R):
        '''Update step: Incorporate new measurement'''

        # Innovation == difference between observation and prediction
        z = observation
        innovation = z - self.H @ self.x

        # 
        innovation_magnitude = np.linalg.norm(innovation)

        if innovation_magnitude > 5.0:
            logger.warning(f'Large innovation {innovation_magnitude:.2f} computed for location {self.loc_id}, measurement may be outlier.')

        # Innovation covariance
        S = self.H @ self.P @ self.H_T + R

        # Kalman gain
        try:
            K = self.P @ self.H_T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning(f"Singular S for location {self.loc_id}, using pseudo-inverse")
            K = self.P @ self.H_T @ np.linalg.pinv(S)

        # State update
        self.x = self.x + K @ innovation

        # Covariance Update (Joseph form for numerical stability)
        I_KH = self.I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
    
    def get_state_estimate(self):
        '''Return current location profile'''
        # Extract position component (first half of state vector)
        pos = self.x[:self.state_dims].flatten()

        # Extract velocity component (second half of state vectory)
        vel = self.x[self.state_dims:].flatten()
        
        # Also return uncertainty (P diag)
        uncertainty = np.sqrt(np.diag(self.P)[:self.state_dims])

        return {
            "typical_hour": pos[0],
            "hour_trend": vel[0],
            "hour_uncertainty": uncertainty[0],
            "typical_duration": pos[1],
            "duration_trend": vel[1],
            "duration_uncertainty": uncertainty[1],
            "visit_freq": pos[2],
            "freq_trend": vel[2],
            "freq_uncertainty": uncertainty[2],
        }
    
    def predict_likelihood(self, query_time, query_duration):
        '''
        Likelihood of being at the state at query time.
        Users current state estimate + uncertainty.
        '''
        state = self.get_state_estimate()

        # Guassian likelihood based on typical hour
        hour_diff = min(
            abs(query_time.hour - state["typical_hour"]),
            24 - abs(query_time.hour - state["typical_hour"])
        )

        # Use uncertainty from kalman filter
        hour_likelihood = np.exp(-0.5 * (hour_diff / state["hour_uncertainty"])**2)
        return hour_likelihood

    def _initialize_state(self, base_visits:pd.DataFrame):
        '''Initialize from historic visits'''
        if len(base_visits) < 2:
            logger.error(f"Invalid base_visits: {len(base_visits)}. Need at least 2.")
            raise ValueError(f"Need at least 2 records; got {len(base_visits)}.")
        
        # Position variables
        typical_hour = base_visits["arrived"].dt.hour.median()
        typical_duration = base_visits["duration"].median()
        visit_freq = len(base_visits) / ((base_visits["arrived"].max() - base_visits["arrived"].min()).days / 7)
        
        # Position compononent (observables)
        position = np.array([typical_hour, typical_duration, visit_freq])

        # Velocity component (unobserved; initially zero as there is not known trend)
        velocity = np.zeros(self.state_dims)
        
        return np.concatenate([position, velocity]).reshape(-1, 1)
    
    def _initialize_P(self):
        '''Covariance matrix'''
        # [Hour: +/- 2 hours; Duration: +/- 30 mins; freq: +/- 1 visit per week]
        pos_uncertainty = np.array([2.0, 30.0, 1.0])

        # Initially high (trends unknown at initialization)
        vel_uncertainty = pos_uncertainty * 5.0
        
        # Diag covariance
        diag = np.concatenate([pos_uncertainty, vel_uncertainty])
        return np.diag(diag)

    def _initialize_dynamics(self):
        '''Initializes the following:
        - State observation matrix (H): An observed position == [Hour, Duration, Freq.], not velocity
        - State transition matrix (F): How does state evolve between observations
        - Process noise (Q): What is the likely rate of change 
        '''
        I = np.eye(self.state_dims)
        zeros = np.zeros((self.state_dims, self.state_dims))

        # z = H @ x extracts position component
        H = np.block([I, zeros])

        # State transition: x(t + dt) = F @ x(t)
        # Position updates: new = old + velocity * dt
        # Velocity updates: new = old (constant velocity assumed)
        # Note: dt will be variable (time between visits), applied in `_predict()`

        F = np.block([
            [I, I],
            [zeros, I]
        ])

        # Process noise (i.e., behavior likely changes slowly 
        # [hour:0.5 drift per week; duration: 10 mins drift per week; freq: 0.2 visit drift per week])
        Q_pos = np.array([0.5, 10.0, 0.2])
        Q_vel = Q_pos * 0.5 # Even slower velocity change

        Q_diag = np.concatenate([Q_pos, Q_vel])
        Q = np.diag(Q_diag)

        return H, F, Q

    def _compute_R(self, context:dict):
        """
        Compute adaptive measurement noise covariance matrix.
        
        Trust measurement more when:
        - Visit pattern is typical (weekday vs weekend matches history)
        - Previous location makes sense (common transition probability)
        - Duration is typical

        Trust measurement less when:
        - Anomalous context (unexpected time / prior location)
        - Very brief visit (possible noise)
            
        Returns
        -------
        R : NDArray
            Diagonal measurement noise covariance matrix
        """
        # Base noise for location
        base_noise = 1.0

        # Adjust based on context quality
        # Anomalous visits == high noise / less trustworthy
        if context.get("is_anomalous", False):
            base_noise *= 5.0
    
        # Brief visits == less trustworthy
        if context.get("duration", 30) < 10:
            base_noise *= 3.0

        # Different noise for different state components
        # Hour: Less certain (varies more) == 2.0
        # Duration: More certain (more stable) == 1.0
        # Frequency: less certain (computed, not directly observed) == 3.0
        noise_scales = np.array([
            2.0 * base_noise,
            1.0 * base_noise,
            3.0 * base_noise
        ])
        return np.diag(noise_scales)