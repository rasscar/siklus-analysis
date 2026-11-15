"""
=============================================================================
MODEL 1: Ridge Cycle Predictor (Wanita Menstruasi)
=============================================================================
Prediksi durasi siklus haid berikutnya menggunakan Ridge Regression (L2)
dengan Exponential Moving Average (EMA) sebagai fitur tambahan.

Formulasi:
  Ŷ = β₀ + Σ βⱼXⱼ
  J(β) = (1/2m) Σ(ŷ - y)² + (λ/2) Σβⱼ²

Pembaruan On-Device SGD:
  β₀ := β₀ - α × (1/m) Σ(ŷ - y)
  βⱼ := βⱼ(1 - αλ) - α × (1/m) Σ(ŷ - y)Xⱼ

Batasan SKPL: α=0.01, λ=0.05, minDataPoints=30
"""

import math
from typing import List, Tuple, Optional, Dict

from config import (
    LEARNING_RATE, REGULARIZATION_LAMBDA, MIN_DATA_POINTS,
    MAX_EPOCHS_LOCAL, DEFAULT_CYCLE_DURATION, DEFAULT_LUTEAL_PHASE,
    FERTILE_WINDOW_BEFORE, FERTILE_WINDOW_AFTER,
)


class RidgeCyclePredictor:
    """
    Model Ridge Regression untuk prediksi durasi siklus haid.
    Dirancang 100% on-device tanpa dependensi NumPy/SciPy.
    """

    NUM_FEATURES = 5  # prev_cycle, sleep, stress, delta_bmi, junkfood

    def __init__(self, alpha: float = LEARNING_RATE,
                 lam: float = REGULARIZATION_LAMBDA):
        """
        Inisialisasi model.

        Args:
            alpha: Laju belajar (α = 0.01 sesuai NFR-005).
            lam: Koefisien regularisasi L2 (λ = 0.05).
        """
        self.alpha = alpha
        self.lam = lam
        # Inisialisasi bobot: [bias, w1, w2, w3, w4, w5]
        self.weights: List[float] = [0.0] * (self.NUM_FEATURES + 1)
        self.training_loss_history: List[float] = []
        self.data_count: int = 0

    # ─── Cold Start Heuristic ────────────────────────────────────────────

    def predict_cold_start(self, prev_duration: Optional[float] = None) -> Dict:
        """
        Prediksi saat data < 30 poin menggunakan aturan klinis standar.

        Returns:
            Dict berisi estimasi durasi, tanggal ovulasi relatif,
            dan rentang fertile window.
        """
        dur = prev_duration if prev_duration else DEFAULT_CYCLE_DURATION
        ovulation_day = dur - DEFAULT_LUTEAL_PHASE
        return {
            "predicted_duration": dur,
            "method": "cold_start_heuristic (Rumus Naegele)",
            "ovulation_day": ovulation_day,
            "fertile_window": (
                max(1, ovulation_day - FERTILE_WINDOW_BEFORE),
                ovulation_day + FERTILE_WINDOW_AFTER,
            ),
            "confidence": "rendah (data < 30 poin)",
        }

    # ─── Forward Pass ────────────────────────────────────────────────────

    def predict(self, features: List[float]) -> float:
        """
        Hitung prediksi: ŷ = β₀ + Σ βⱼXⱼ

        Args:
            features: Vektor fitur [X1, X2, X3, X4, X5].

        Returns:
            Prediksi durasi siklus (hari).
        """
        if len(features) != self.NUM_FEATURES:
            raise ValueError(
                f"Dibutuhkan {self.NUM_FEATURES} fitur, "
                f"diterima {len(features)}."
            )
        y_hat = self.weights[0]  # bias
        for j in range(self.NUM_FEATURES):
            y_hat += self.weights[j + 1] * features[j]
        return y_hat

    # ─── Cost Function ───────────────────────────────────────────────────

    def compute_cost(self, X: List[List[float]],
                     y: List[float]) -> float:
        """
        Hitung fungsi kerugian Ridge:
        J(β) = (1/2m) Σ(ŷ-y)² + (λ/2) Σβⱼ²

        Args:
            X: Matriks fitur [[x1,x2,...], ...].
            y: Vektor target.

        Returns:
            Nilai cost J(β).
        """
        m = len(X)
        if m == 0:
            return 0.0

        # Data loss
        data_loss = 0.0
        for i in range(m):
            y_hat = self.predict(X[i])
            data_loss += (y_hat - y[i]) ** 2
        data_loss /= (2 * m)

        # Regularization loss (tanpa bias)
        reg_loss = 0.0
        for j in range(1, len(self.weights)):
            reg_loss += self.weights[j] ** 2
        reg_loss *= self.lam / 2

        return data_loss + reg_loss

    # ─── Training (Batch Gradient Descent) ───────────────────────────────

    def train(self, X: List[List[float]], y: List[float],
              epochs: int = MAX_EPOCHS_LOCAL,
              verbose: bool = False) -> Dict:
        """
        Latih model menggunakan Batch Gradient Descent dengan Ridge.

        Aturan Pembaruan:
          β₀ := β₀ - α × (1/m) Σ(ŷ - y)
          βⱼ := βⱼ(1 - αλ) - α × (1/m) Σ(ŷ - y)Xⱼ

        Args:
            X: Matriks fitur pelatihan.
            y: Vektor target pelatihan.
            epochs: Jumlah iterasi pelatihan.
            verbose: Cetak progress setiap 10 epoch.

        Returns:
            Dict berisi info pelatihan (final_cost, mae, epochs).
        """
        m = len(X)
        if m < 3:
            raise ValueError(
                f"Minimal 3 sampel untuk pelatihan, diterima {m}."
            )

        self.data_count = m
        self.training_loss_history = []

        for epoch in range(epochs):
            # --- Hitung gradien ---
            grad = [0.0] * len(self.weights)

            for i in range(m):
                y_hat = self.predict(X[i])
                error = y_hat - y[i]

                grad[0] += error  # gradien bias
                for j in range(self.NUM_FEATURES):
                    grad[j + 1] += error * X[i][j]

            # --- Update bobot ---
            # Bias (tanpa regularisasi)
            self.weights[0] -= self.alpha * (grad[0] / m)

            # Koefisien fitur (dengan regularisasi L2)
            for j in range(1, len(self.weights)):
                self.weights[j] = (
                    self.weights[j] * (1 - self.alpha * self.lam)
                    - self.alpha * (grad[j] / m)
                )

            # --- Catat loss ---
            cost = self.compute_cost(X, y)
            self.training_loss_history.append(cost)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:>4d}/{epochs}  |  Cost: {cost:.6f}")

        # Hitung MAE final
        mae = self._compute_mae(X, y)

        return {
            "final_cost": self.training_loss_history[-1],
            "mae": mae,
            "epochs_run": epochs,
            "data_points": m,
            "weights": self.weights.copy(),
        }

    # ─── Incremental Update (On-Device SGD) ──────────────────────────────

    def incremental_update(self, features: List[float],
                           actual_duration: float):
        """
        Pembaruan inkremental satu sampel (SGD) saat siklus baru tercatat.
        Cocok untuk pelatihan on-device real-time tanpa membebani CPU.

        Args:
            features: Vektor fitur siklus baru.
            actual_duration: Durasi siklus aktual (label baru).
        """
        y_hat = self.predict(features)
        error = y_hat - actual_duration

        # Update bias
        self.weights[0] -= self.alpha * error

        # Update koefisien fitur + regularisasi
        for j in range(self.NUM_FEATURES):
            self.weights[j + 1] = (
                self.weights[j + 1] * (1 - self.alpha * self.lam)
                - self.alpha * error * features[j]
            )

        self.data_count += 1

    # ─── Estimasi Ovulasi & Fertile Window ───────────────────────────────

    def estimate_fertility(self, features: List[float],
                           cycle_start_date: str = "") -> Dict:
        """
        Hitung estimasi tanggal ovulasi dan masa subur.

        Ovulasi: D_ovulasi = Ŷ_siklus - 14 (hari dari mulai siklus)
        Fertile Window: [D_ovulasi - 5, D_ovulasi + 1]
        """
        predicted_dur = self.predict(features)
        ovulation_day = max(1, round(predicted_dur - DEFAULT_LUTEAL_PHASE))

        fertile_start = max(1, ovulation_day - FERTILE_WINDOW_BEFORE)
        fertile_end = ovulation_day + FERTILE_WINDOW_AFTER

        return {
            "predicted_duration_days": round(predicted_dur, 1),
            "ovulation_day_from_start": ovulation_day,
            "fertile_window_days": (fertile_start, fertile_end),
            "data_count": self.data_count,
            "model_weights": self.weights.copy(),
        }

    # ─── Helper ──────────────────────────────────────────────────────────

    def _compute_mae(self, X: List[List[float]], y: List[float]) -> float:
        """Mean Absolute Error."""
        total = 0.0
        for i in range(len(X)):
            total += abs(self.predict(X[i]) - y[i])
        return total / len(X) if X else 0.0

    def get_weights(self) -> List[float]:
        return self.weights.copy()

    def set_weights(self, weights: List[float]):
        if len(weights) != self.NUM_FEATURES + 1:
            raise ValueError(
                f"Dibutuhkan {self.NUM_FEATURES + 1} bobot, "
                f"diterima {len(weights)}."
            )
        self.weights = weights.copy()

    def summary(self) -> str:
        """Ringkasan model dalam format teks."""
        names = ["bias", "prev_cycle", "sleep", "stress",
                 "delta_bmi", "junkfood"]
        lines = ["═══ Ridge Cycle Predictor ═══"]
        for i, name in enumerate(names):
            lines.append(f"  {name:>15s} = {self.weights[i]:+.6f}")
        lines.append(f"  {'data_count':>15s} = {self.data_count}")
        if self.training_loss_history:
            lines.append(
                f"  {'last_cost':>15s} = "
                f"{self.training_loss_history[-1]:.6f}"
            )
        return "\n".join(lines)

