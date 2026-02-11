import pickle

from polkit.taxonomy import *
from polkit.strategy import MarkovChain
from polkit.visualize import *
from polkit.utils import get_logger, setup_logging

setup_logging(
    log_dir="../logs/polkit"
)

logger = get_logger(__name__)


if __name__=="__main__":

    # Declare source info
    data_path = "./app/data/"
    user_id = data_path.split("/")[2]
    
    # Initialize Reader / Preprocessor Objects
    detector = StayPointDetector()
    clusterer = StayPointClusterer()
    profiler = LocationProfiler()

    # Load / Preprocess Data
    file_path = f"./app/data/user_014.pkl"

    with open(file_path, "rb") as f:
        pfs = pickle.load(f)

    sps = detector.detect(pfs)
    locs = clusterer.cluster(sps)

    # Profile User
    profiled_df = profiler.profile(locs)
    diamon_data, digraph_data = profiler.format_profiles_for_charts()
    print(profiled_df)

    chart_maker = ChartMaker()

    fig = chart_maker.create_location_profile_chart(diamon_data)
    fig.show()

    # Get model inputs
    ## States
    counts = locs["loc_id"].value_counts()
    outlier_ids = counts[counts <= 1].index
    locs.loc[locs["loc_id"].isin(outlier_ids), "loc_id"] = -1
    labels = locs["loc_id"].reset_index(drop=True)
    states = labels.unique()
    
    ## Hours
    datetime = locs["arrived"].reset_index(drop=True)
    hours = datetime.dt.hour

    # Initialize / train model / get predidction
    model = MarkovChain(states, length=5)
    y_pred = model.fit_predict(labels, hours, start=states[0], method="median")
    k_pred = model.predict_next_k(states[0], k=3)

    # View Results
    logger.info(f"\nRaw Predictions for User {user_id}\n{y_pred}")
    logger.info(f"\nNext K-Predictions for User {user_id}\n{k_pred}")