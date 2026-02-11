import numpy as np
import pandas as pd

def parse_datetime(datetime:pd.Series):
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

def train_test_split(states:pd.Series | np.ndarray | list, hours:pd.Series | np.ndarray | list, test_size:float=0.2):
    # Get train indices; train model
    train_size = 1-test_size
    split = int(len(states) * train_size)

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