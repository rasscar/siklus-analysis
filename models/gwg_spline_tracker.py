"""
=============================================================================
MODEL 2b: GWG Spline Tracker (Pemantauan Kehamilan - Ibu Hamil)
=============================================================================
Pemantauan kenaikan berat badan gestasional (Gestational Weight Gain)
menggunakan Piecewise Quadratic Regression sesuai standar IOM.

Formulasi per trimester k:
  Ŵ(t) = W_pra_hamil + aₖ(t - t_start_k) + bₖ(t - t_start_k)²

Kategori BMI IOM:
  - Underweight (BMI < 18.5)   → Target GWG: 12.5 - 18.0 kg
  - Normal     (18.5 ≤ BMI < 25) → Target GWG: 11.5 - 16.0 kg
  - Overweight (25 ≤ BMI < 30) → Target GWG:  7.0 - 11.5 kg
  - Obese      (BMI ≥ 30)     → Target GWG:  5.0 -  9.0 kg
"""

from typing import Dict, List, Tuple

from config import GWG_TRIMESTER_CONFIG


class GWGSplineTracker:
    """
    Model pelacak kenaikan berat badan ibu hamil berbasis kurva
    referensi IOM dengan koefisien polinomial adaptif.
    """

    # Koefisien kurva per kategori BMI (a, b per trimester)
    # Format: {bmi_category: {trimester: (a_linear, b_quadratic)}}
    IOM_COEFFICIENTS = {
        "underweight": {
            1: (0.06, 0.0015),  # ~2 kg total T1
            2: (0.42, 0.0005),  # ~5.5 kg T2
            3: (0.45, 0.0003),  # ~6.0 kg T3
        },
        "normal": {
            1: (0.04, 0.0010),  # ~1.5 kg total T1
            2: (0.38, 0.0004),  # ~5.0 kg T2
            3: (0.42, 0.0003),  # ~5.5 kg T3
        },
        "overweight": {
            1: (0.03, 0.0008),  # ~1.0 kg total T1
            2: (0.28, 0.0003),  # ~3.5 kg T2
            3: (0.30, 0.0002),  # ~4.0 kg T3
        },
        "obese": {
            1: (0.02, 0.0005),  # ~0.7 kg total T1
            2: (0.20, 0.0002),  # ~2.5 kg T2
            3: (0.22, 0.0001),  # ~3.0 kg T3
        },
    }

    def __init__(self, bmi_pra_hamil: float, berat_pra_hamil: float):
        """
        Args:
            bmi_pra_hamil: BMI sebelum hamil.
            berat_pra_hamil: Berat badan sebelum hamil (kg).
        """
        self.bmi_pra_hamil = bmi_pra_hamil
        self.berat_pra_hamil = berat_pra_hamil
        self.bmi_category = self._classify_bmi(bmi_pra_hamil)
        self.coefficients = self.IOM_COEFFICIENTS[self.bmi_category]

    # ─── Klasifikasi BMI ─────────────────────────────────────────────────

    @staticmethod
    def _classify_bmi(bmi: float) -> str:
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25.0:
            return "normal"
        elif bmi < 30.0:
            return "overweight"
        else:
            return "obese"

    @staticmethod
    def get_gwg_target_range(bmi_category: str) -> Tuple[float, float]:
        """Dapatkan rentang target GWG total (kg) sesuai IOM."""
        targets = {
            "underweight": (12.5, 18.0),
            "normal":      (11.5, 16.0),
            "overweight":  (7.0, 11.5),
            "obese":       (5.0, 9.0),
        }
        return targets.get(bmi_category, (11.5, 16.0))

    # ─── Prediksi Berat Ideal ────────────────────────────────────────────

    def predict_ideal_weight(self, week: int) -> Dict:
        """
        Hitung berat badan ideal pada minggu kehamilan tertentu.

        Ŵ(t) = W_pra + aₖ(t - t_start) + bₖ(t - t_start)²

        Args:
            week: Minggu kehamilan (0-40).

        Returns:
            Dict: {week, trimester, ideal_weight, gain_from_start,
                   target_range, status}
        """
        week = max(0, min(40, week))
        trimester = self._get_trimester(week)
        t_start = GWG_TRIMESTER_CONFIG[trimester]["start_week"]

        a, b = self.coefficients[trimester]
        dt = week - t_start

        # Kumulatif dari trimester sebelumnya
        cumulative_gain = 0.0
        for t in range(1, trimester):
            t_s = GWG_TRIMESTER_CONFIG[t]["start_week"]
            t_e = GWG_TRIMESTER_CONFIG[t]["end_week"]
            a_prev, b_prev = self.coefficients[t]
            dt_prev = t_e - t_s
            cumulative_gain += a_prev * dt_prev + b_prev * (dt_prev ** 2)

        # Gain minggu ini dalam trimester berjalan
        current_gain = a * dt + b * (dt ** 2)
        total_gain = cumulative_gain + current_gain
        ideal_weight = self.berat_pra_hamil + total_gain

        # Target range
        target_min, target_max = self.get_gwg_target_range(self.bmi_category)
        progress_pct = (week / 40.0) * 100

        return {
            "week": week,
            "trimester": trimester,
            "ideal_weight_kg": round(ideal_weight, 2),
            "gain_from_start_kg": round(total_gain, 2),
            "expected_total_gain_range": (target_min, target_max),
            "progress_pct": round(progress_pct, 1),
            "bmi_category": self.bmi_category,
        }

    # ─── Evaluasi Aktual vs Ideal ────────────────────────────────────────

    def evaluate(self, week: int, actual_weight: float) -> Dict:
        """
        Bandingkan berat aktual dengan kurva ideal.

        Returns:
            Dict: {deviation_kg, status, recommendation}
        """
        ideal = self.predict_ideal_weight(week)
        deviation = actual_weight - ideal["ideal_weight_kg"]

        if abs(deviation) <= 1.0:
            status = "✅ NORMAL"
            rec = "Pertambahan berat badan sesuai kurva ideal IOM."
        elif deviation > 1.0:
            status = "⚠️ DI ATAS IDEAL"
            rec = (f"Kelebihan {deviation:.1f} kg dari ideal. "
                   "Konsultasikan pola makan dengan ahli gizi.")
        else:
            status = "⚠️ DI BAWAH IDEAL"
            rec = (f"Kekurangan {abs(deviation):.1f} kg dari ideal. "
                   "Tingkatkan asupan nutrisi dan konsultasi dokter.")

        return {
            **ideal,
            "actual_weight_kg": actual_weight,
            "deviation_kg": round(deviation, 2),
            "status": status,
            "recommendation": rec,
        }

    # ─── Full Pregnancy Curve ────────────────────────────────────────────

    def generate_full_curve(self) -> List[Dict]:
        """Hasilkan kurva prediksi berat ideal minggu 0-40."""
        return [self.predict_ideal_weight(w) for w in range(0, 41)]

    # ─── Helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_trimester(week: int) -> int:
        if week <= 13:
            return 1
        elif week <= 27:
            return 2
        else:
            return 3

    def summary(self) -> str:
        lines = [
            "═══ GWG Spline Tracker ═══",
            f"  BMI Pra-Hamil     : {self.bmi_pra_hamil:.1f} "
            f"({self.bmi_category})",
            f"  Berat Pra-Hamil   : {self.berat_pra_hamil:.1f} kg",
        ]
        target = self.get_gwg_target_range(self.bmi_category)
        lines.append(f"  Target GWG Total  : {target[0]} - {target[1]} kg")
        for t in range(1, 4):
            a, b = self.coefficients[t]
            lines.append(f"  Trimester {t} koef  : a={a:.4f}, b={b:.6f}")
        return "\n".join(lines)

