import os
import joblib
from eye_track import Eyetracker

if __name__ == "__main__":
    tracker = Eyetracker()
    if os.path.exists('model.pkl'):
        print("Loading data....")
        data = joblib.load('model.pkl')
        tracker.eye_mod = data['eye_mod']
        tracker.comb_mod = data['comb_mod']
        tracker.eye_scl = data['eye_scl']
        tracker.comb_scl = data['comb_scl']
        tracker.eye_poly = data['eye_poly']
        tracker.comb_poly = data['comb_poly']
        print("data loaded.")
    else:
        tracker.calib()
    tracker.start()
