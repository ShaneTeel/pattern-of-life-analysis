import pickle
import numpy as np

from polkit.taxonomy import *
from polkit.strategy import *
from polkit.utils import get_logger, setup_logging, train_test_split, GeoLifeReader

setup_logging(
    log_dir="../logs/polkit"
)

logger = get_logger(__name__)


if __name__=="__main__":

    # Declare source info
    user_id = "003"
    data_path = f"./app/data/user_{user_id}.pkl"
    user_id = data_path.split("/")[2]
    
    # Initialize Reader / Preprocessor Objects
    detector = StayPointDetector()
    clusterer = StayPointClusterer()
    profiler = LocationProfiler()

    # Load / Preprocess Data
    with open(data_path, "rb") as f:
        try:
            pfs = pickle.load(f)
            logger.debug(f"Sucessfully read .pkl file for user {user_id}.")    

        except FileNotFoundError as e:
            logger.debug(f"A FileNotFoundError occurred: {e}")
    
    sps = detector.detect(pfs)
    locs = clusterer.cluster(sps)
    weights = np.array(locs["n_points"].values)

    logger.info(f"Total Clusters: {locs["loc_id"].nunique()}")
    logger.info(f"Top 3 clusters account for: {np.sort(weights)[-3:].sum() / weights.sum():.2%}")

    # Profile User
    profiled_df = profiler.profile(locs)

    # Get model inputs
    ## States
    counts = locs["loc_id"].value_counts()
    outlier_ids = counts[counts <= 1].index
    locs.loc[locs["loc_id"].isin(outlier_ids), "loc_id"] = -1
    labels = locs["loc_id"].reset_index(drop=True)
    states = sorted(set(labels))

    ## Hours
    datetime = locs["arrived"].reset_index(drop=True)
    hours = datetime.dt.hour

    # Get Train / Test sets
    S_train, H_train, S_test = train_test_split(labels, hours)

    # Initialize / train model
    model = MarkovChain(states)
    model.fit(S_train, H_train)

    # Evaluate Model
    eval = MarkovEvaluator(model, k=3)
    _ = eval.evaluate(S_test)

    # Generate / View Results
    summary = eval.generate_summary()
    logger.info(summary)