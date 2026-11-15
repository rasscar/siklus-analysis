"""
=============================================================================
MODEL B: Multi-Factor Cycle Regularity Classifier
=============================================================================
Klasifikasi risiko gangguan/ketidakteraturan siklus menstruasi berdasarkan
sintesis empiris literatur gizi dan stres:
  - Anggoro & Azizah (2023): Stres berat p=0.004, Pola makan p=0.014
  - Dawali & Widiatrilupi (2025): Diet tidak seimbang OR=5.3, p=0.035
  - Güzeldere et al. (BMC 2024): Asupan gula tinggi p=0.047

Formulasi:
  z = w0 + w_diet·S_diet + w_stres·S_stres + w_tidur·(7.0 - T_tidur) + w_gula·S_gula
  P(Anomali) = 1 / (1 + e^(-z))
"""

import math
from typing import Dict


class CycleRegularityClassifier:
    """Klasifikasi probabilitas risiko ketidakteraturan siklus haid."""

    def __init__(self):
        # Bobot empiris yang dikalibrasi dari odds ratio literatur
        self.w0 = -1.40          # Baseline intercept (~20% populasi normal)
        self.w_diet = 0.85       # Pola makan tidak teratur/seimbang (skor 0.0 - 1.0)
        self.w_stres = 0.024      # Skor stres skala 0 - 100 (HPA axis)
        self.w_tidur = 0.35      # Defisit jam tidur (< 7.0 jam)
        self.w_gula = 0.28       # Frekuensi konsumsi minuman/makanan manis per minggu

    @staticmethod
    def _sigmoid(z: float) -> float:
        z_clamped = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z_clamped))

    def evaluate_risk(
        self,
        diet_unbalanced: float,
        stress_score: float,
        sleep_hours: float,
        sugar_weekly_freq: float = 0.0,
    ) -> Dict:
        """
        Hitung probabilitas anomali dan tentukan kategori risiko klinis.

        Args:
            diet_unbalanced: 1.0 jika tidak teratur/seimbang, 0.0 jika teratur.
            stress_score: Skor stres dari detak jantung / kuesioner (0 - 100).
            sleep_hours: Rata-rata durasi tidur harian (jam).
            sugar_weekly_freq: Frekuensi konsumsi minuman tinggi gula per minggu.

        Returns:
            Dict: {probability, risk_code, risk_label, recommendation}
        """
        sleep_deficit = max(0.0, 7.0 - sleep_hours)

        z = (
            self.w0
            + (self.w_diet * diet_unbalanced)
            + (self.w_stres * stress_score)
            + (self.w_tidur * sleep_deficit)
            + (self.w_gula * sugar_weekly_freq)
        )

        prob = self._sigmoid(z)

        # Kategori risiko
        if prob >= 0.65:
            risk_code = 2
            risk_label = "TINGGI (Beresiko Oligomenorea/Anovulasi)"
            rec = "Pola makan dan stres berpotensi menunda ovulasi. Disarankan perbaiki jam tidur dan kurangi makanan cepat saji/gula."
        elif prob >= 0.40:
            risk_code = 1
            risk_label = "SEDANG (Waspada Fluktuasi)"
            rec = "Terdapat indikasi kelelahan atau pola makan kurang teratur. Jaga asupan gizi seimbang."
        else:
            risk_code = 0
            risk_label = "RENDAH (Normal)"
            rec = "Faktor gaya hidup stabil. Siklus diprediksi berlangsung reguler."

        return {
            "probability": round(prob, 4),
            "risk_code": risk_code,
            "risk_label": risk_label,
            "recommendation": rec,
            "components": {
                "diet_factor": round(self.w_diet * diet_unbalanced, 3),
                "stress_factor": round(self.w_stres * stress_score, 3),
                "sleep_factor": round(self.w_tidur * sleep_deficit, 3),
                "sugar_factor": round(self.w_gula * sugar_weekly_freq, 3),
            },
        }
