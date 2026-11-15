"""
=============================================================================
MODEL 2a: Logistic Peak Time (Program Kehamilan / Promil)
=============================================================================
Estimasi probabilitas kesuburan harian menggunakan Logistic Regression.

Formulasi:
  P(Fertil|t) = σ(z(t)) = 1 / (1 + e^(-z(t)))
  z(t) = w₀ - w₁(t - D_ovulasi)² - w₂·S_stres(t) + w₃·Q_tidur(t)

Ambang: P >= 0.70 → Peak Time Berhubungan
"""

import math
from typing import List, Dict, Optional

from config import (
    LEARNING_RATE, PEAK_FERTILITY_THRESHOLD, MAX_EPOCHS_LOCAL,
    DEFAULT_LUTEAL_PHASE
)


class LogisticPeakTime:
    """
    Model Logistic Regression untuk estimasi Peak Time kesuburan.
    Tanpa dependensi library eksternal (pure Python).
    """

    NUM_FEATURES = 3  # offset^2, stress, sleep

    def __init__(self, alpha: float = LEARNING_RATE):
        self.alpha = alpha
        # Bobot: [w0(bias), w1(offset²), w2(stress), w3(sleep)]
        self.weights: List[float] = [2.0, -0.15, -0.02, 0.1]
        self.training_loss_history: List[float] = []

    # ─── Sigmoid ─────────────────────────────────────────────────────────

    @staticmethod
    def sigmoid(z: float) -> float:
        """
        Fungsi aktivasi Sigmoid: σ(z) = 1 / (1 + e^(-z))
        Dilengkapi clipping untuk mencegah overflow.
        """
        z = max(-500.0, min(500.0, z))  # clamp
        return 1.0 / (1.0 + math.exp(-z))

    # ─── Forward Pass ────────────────────────────────────────────────────

    def compute_logit(self, features: List[float]) -> float:
        """
        Hitung logit z(t):
        z = w₀ + w₁·(offset²) + w₂·stress + w₃·sleep

        Args:
            features: [offset_squared, stress_score, sleep_quality]
        """
        z = self.weights[0]
        for j in range(self.NUM_FEATURES):
            z += self.weights[j + 1] * features[j]
        return z

    def predict_probability(self, features: List[float]) -> float:
        """Hitung P(Fertil|t) = σ(z(t))."""
        z = self.compute_logit(features)
        return self.sigmoid(z)

    def predict_is_peak(self, features: List[float]) -> bool:
        """Apakah hari ini termasuk Peak Time? (P >= 0.70)"""
        return self.predict_probability(features) >= PEAK_FERTILITY_THRESHOLD

    # ─── Full Window Prediction ──────────────────────────────────────────

    def predict_fertility_window(
        self,
        predicted_cycle_dur: float,
        daily_stress: List[float],
        daily_sleep: List[float],
    ) -> List[Dict]:
        """
        Prediksi probabilitas kesuburan untuk seluruh hari dalam siklus.

        Args:
            predicted_cycle_dur: Prediksi durasi siklus (dari Model 1).
            daily_stress: Skor stres harian [hari-1, hari-2, ...].
            daily_sleep: Durasi tidur harian [hari-1, hari-2, ...].

        Returns:
            List of dict per hari: {day, offset, probability, is_peak}.
        """
        ovulation_day = round(predicted_cycle_dur - DEFAULT_LUTEAL_PHASE)
        results = []

        num_days = min(len(daily_stress), len(daily_sleep),
                       int(predicted_cycle_dur))

        for day in range(1, num_days + 1):
            offset = day - ovulation_day
            features = [
                offset ** 2,                   # Jarak kuadrat dari ovulasi
                daily_stress[day - 1],         # Stres hari ini
                daily_sleep[day - 1],          # Tidur hari ini
            ]
            prob = self.predict_probability(features)
            results.append({
                "day": day,
                "offset_from_ovulation": offset,
                "probability": round(prob, 4),
                "is_peak": prob >= PEAK_FERTILITY_THRESHOLD,
                "label": self._label(prob),
            })

        return results

    # ─── Training (Binary Cross-Entropy + SGD) ──────────────────────────

    def train(self, X: List[List[float]], y: List[float],
              epochs: int = MAX_EPOCHS_LOCAL,
              verbose: bool = False) -> Dict:
        """
        Latih model dengan Binary Cross-Entropy Loss + SGD.

        Cost: J = -(1/m) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
        Gradient: ∂J/∂wⱼ = (1/m) Σ (ŷᵢ - yᵢ) Xⱼ

        Args:
            X: Matriks fitur [[offset², stress, sleep], ...].
            y: Label biner [1.0=fertil, 0.0=tidak].
            epochs: Jumlah iterasi.

        Returns:
            Dict info pelatihan.
        """
        m = len(X)
        self.training_loss_history = []

        for epoch in range(epochs):
            grad = [0.0] * len(self.weights)
            total_loss = 0.0

            for i in range(m):
                y_hat = self.predict_probability(X[i])
                y_hat = max(1e-7, min(1 - 1e-7, y_hat))  # clip
                error = y_hat - y[i]

                # Binary Cross-Entropy
                total_loss += -(
                    y[i] * math.log(y_hat) +
                    (1 - y[i]) * math.log(1 - y_hat)
                )

                grad[0] += error
                for j in range(self.NUM_FEATURES):
                    grad[j + 1] += error * X[i][j]

            # Update weights
            for j in range(len(self.weights)):
                self.weights[j] -= self.alpha * (grad[j] / m)

            avg_loss = total_loss / m
            self.training_loss_history.append(avg_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:>4d}/{epochs}  |  "
                      f"BCE Loss: {avg_loss:.6f}")

        return {
            "final_loss": self.training_loss_history[-1],
            "epochs_run": epochs,
            "weights": self.weights.copy(),
        }

    # ─── Helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _label(prob: float) -> str:
        if prob >= 0.85:
            return "🔴 SANGAT SUBUR"
        elif prob >= PEAK_FERTILITY_THRESHOLD:
            return "🟠 SUBUR TINGGI (Peak)"
        elif prob >= 0.40:
            return "🟡 SUBUR SEDANG"
        elif prob >= 0.15:
            return "🟢 SUBUR RENDAH"
        else:
            return "⚪ TIDAK SUBUR"

    def get_weights(self) -> List[float]:
        return self.weights.copy()

    def set_weights(self, weights: List[float]):
        self.weights = weights.copy()

    def summary(self) -> str:
        names = ["bias(w0)", "offset²(w1)", "stress(w2)", "sleep(w3)"]
        lines = ["═══ Logistic Peak Time ═══"]
        for i, name in enumerate(names):
            lines.append(f"  {name:>15s} = {self.weights[i]:+.6f}")
        return "\n".join(lines)

