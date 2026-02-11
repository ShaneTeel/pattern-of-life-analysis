import pickle
from pathlib import Path

from polkit.taxonomy import StayPointDetector
from polkit.utils import GeoLifeReader, get_logger, setup_logging

logger = get_logger(__name__)

setup_logging(
    log_dir="../logs/polkit"
)

def screen():
        
    for i in range(17, 182):
        if i < 10:
            user_id = f"00{i}"
        elif 10 <= i < 100:
            user_id = f"0{i}"
        else:
            user_id = i

        file_in = f"./GeoLife/{user_id}/Trajectory/"
        file_out = f"./app/data/user_{user_id}.pkl"

        reader = GeoLifeReader(user_id, file_in)
        detector = StayPointDetector()

        pfs = reader.read_user()
        sps = detector.detect(pfs)
        
        if len(sps) >= 150:
            logger.info(f"User {user_id} has {len(sps)} stay-points. Moving .pkl file.")
            

            new_pkl = Path(file_out)
            if new_pkl.exists():
                logger.info(f".pkl file already exists. Skipping user {user_id}")
                continue

            reader.move_user(new_pkl)

            if new_pkl.exists():
                logger.info("Successfully moved .pkl file")
            else:
                logger.warning(f"Failed to move .pkl file for user {user_id}")

        reader.delete_user()
        logger.info("Deleting old .pkl file.")

if __name__=="__main__":

    screen()