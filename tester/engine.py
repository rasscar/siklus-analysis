import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.ridge_cycle_predictor import RidgeCyclePredictor
from models.logistic_peak_time import LogisticPeakTime
from models.gwg_spline_tracker import GWGSplineTracker
from models.explainable_risk_analyzer import ExplainableRiskAnalyzer
from models.bbt_cusum_detector import BBTCusumDetector
from models.cycle_regularity_classifier import CycleRegularityClassifier
from utils.scaler import ZScoreScaler

def main():
    tokens = sys.stdin.read().split()
    if not tokens:
        return
    idx = 0
    mode = int(tokens[idx])
    idx += 1

    if mode == 1:
        prev_dur = float(tokens[idx])
        sleep = float(tokens[idx + 1])
        stress = float(tokens[idx + 2])
        delta_bmi = float(tokens[idx + 3])
        junk = float(tokens[idx + 4])

        means = [22.5, 6.585, 48.175, 0.135, 1.231]
        stds = [1.624, 1.382, 23.131, 1.022, 0.992]
        weights = [19.2178, 0.3243, -0.0072, 1.0142, 0.1542, -0.099]

        scaler = ZScoreScaler()
        scaler.set_params(means, stds)
        scaled = scaler.transform([[prev_dur, sleep, stress, delta_bmi, junk]])[0]

        model = RidgeCyclePredictor()
        model.set_weights(weights)
        res = model.estimate_fertility(scaled)

        dur = res["predicted_duration_days"]
        ovul = res["ovulation_day_from_start"]
        f_start, f_end = res["fertile_window_days"]
        print(f"{dur:.2f} {ovul} {f_start} {f_end}")

    elif mode == 2:
        cycle_dur = float(tokens[idx])
        day_offset = int(tokens[idx + 1])
        stress = float(tokens[idx + 2])
        sleep = float(tokens[idx + 3])

        model = LogisticPeakTime()
        features = [day_offset ** 2, stress, sleep]
        prob = model.predict_probability(features)
        is_peak = 1 if model.predict_is_peak(features) else 0
        print(f"{prob:.4f} {is_peak}")

    elif mode == 3:
        bmi_pra = float(tokens[idx])
        berat_pra = float(tokens[idx + 1])
        week = int(tokens[idx + 2])
        berat_aktual = float(tokens[idx + 3])

        tracker = GWGSplineTracker(bmi_pra, berat_pra)
        eval_res = tracker.evaluate(week, berat_aktual)

        ideal = eval_res["ideal_weight_kg"]
        dev = eval_res["deviation_kg"]
        status_code = 0
        if "ATAS" in eval_res["status"]:
            status_code = 1
        elif "BAWAH" in eval_res["status"]:
            status_code = 2
        print(f"{ideal:.2f} {dev:.2f} {status_code}")

    elif mode == 4:
        pred_dur = float(tokens[idx])
        prev_dur = float(tokens[idx + 1])
        sleep = float(tokens[idx + 2])
        stress = float(tokens[idx + 3])
        delta_bmi = float(tokens[idx + 4])
        junk = float(tokens[idx + 5])

        weights = [19.2178, 0.3243, -0.0072, 1.0142, 0.1542, -0.099]
        raw_feats = [prev_dur, sleep, stress, delta_bmi, junk]
        means = [22.5, 6.585, 48.175, 0.135, 1.231]
        stds = [1.624, 1.382, 23.131, 1.022, 0.992]

        analyzer = ExplainableRiskAnalyzer()
        res = analyzer.analyze(pred_dur, weights, raw_feats, means, stds)

        risk_code = 0
        if "TINGGI" in res["risk_level"]:
            risk_code = 2
        elif "SEDANG" in res["risk_level"]:
            risk_code = 1

        top = res["attribution_scores"][0]
        top_name = top["feature_name"]
        top_delta = top["delta_y_days"]
        print(f"{risk_code} {top_name} {top_delta:.2f}")

    elif mode == 5:
        num_days = int(tokens[idx])
        idx += 1
        temps = [float(tokens[idx + d]) for d in range(num_days)]
        detector = BBTCusumDetector()
        res = detector.detect_ovulation(temps)
        conf = 1 if res["confirmed"] else 0
        ovul = res["ovulation_day"]
        delta = res["temperature_delta"]
        print(f"{conf} {ovul} {delta:.2f}")

    elif mode == 6:
        diet = float(tokens[idx])
        stress = float(tokens[idx + 1])
        sleep = float(tokens[idx + 2])
        sugar = float(tokens[idx + 3])
        classifier = CycleRegularityClassifier()
        res = classifier.evaluate_risk(diet, stress, sleep, sugar)
        prob = res["probability"]
        code = res["risk_code"]
        print(f"{prob:.4f} {code}")

if __name__ == "__main__":
    main()
