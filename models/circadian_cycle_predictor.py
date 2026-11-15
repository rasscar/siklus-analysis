from models.ridge_cycle_predictor import RidgeCyclePredictor
from utils.scaler import ZScoreScaler
from typing import List, Dict


class CircadianCyclePredictor(RidgeCyclePredictor):

    NUM_FEATURES = 7

    def __init__(self, alpha: float = 0.1):
        super().__init__(alpha)
        self.feature_names = [
            "prev_duration", "sleep_hours", "stress_score",
            "delta_bmi", "junk_freq", "sjl_hours", "sigma_tmid"
        ]
        self.weights = [0.0] * (self.NUM_FEATURES + 1)

    def set_weights(self, weights: list) -> None:
        if len(weights) != self.NUM_FEATURES + 1:
            raise ValueError(
                f"Circadian model butuh {self.NUM_FEATURES + 1} bobot (bias + 7 fitur), diterima {len(weights)}."
            )
        self.weights = list(weights)

    def predict(self, features: list) -> float:
        if len(features) != self.NUM_FEATURES:
            raise ValueError(
                f"Dibutuhkan {self.NUM_FEATURES} fitur, diterima {len(features)}."
            )
        y_hat = self.weights[0]
        for j in range(self.NUM_FEATURES):
            y_hat += self.weights[j + 1] * features[j]
        return y_hat

    def train_circadian(
        self,
        X: List[List[float]],
        y: List[float],
        epochs: int = 2000,
        lr: float = 0.005,
    ) -> Dict:
        return self.train(X, y, epochs, lr)

    def predict_circadian(self, features_7: List[float]) -> Dict:
        if len(features_7) != 7:
            return {
                "error": "Butuh tepat 7 fitur: prev_dur, sleep, stress, delta_bmi, junk, sjl, sigma_tmid"
            }

        pred_dur = self.predict(features_7)

        ovulation_day = max(1, round(pred_dur * 0.45))
        fertile_start = max(1, ovulation_day - 2)
        fertile_end = ovulation_day + 1

        sjl = features_7[5]
        sigma_tmid = features_7[6]

        desync_risk = "RENDAH"
        if sjl >= 2.0 or sigma_tmid >= 1.5:
            desync_risk = "TINGGI"
        elif sjl >= 1.5 or sigma_tmid >= 1.0:
            desync_risk = "SEDANG"

        return {
            "predicted_duration_days": round(pred_dur, 2),
            "ovulation_day_from_start": ovulation_day,
            "fertile_window_days": (fertile_start, fertile_end),
            "sjl_hours": round(sjl, 2),
            "sigma_tmid": round(sigma_tmid, 2),
            "desync_risk": desync_risk,
        }
