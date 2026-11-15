"""
=============================================================================
SIKLUS ML - Feature Engineering
=============================================================================
Modul rekayasa fitur: mengubah data mentah harian dari SQLite menjadi
matriks fitur tabular siap-latih untuk model prediksi.
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta


class FeatureEngineer:
    """Ekstraksi dan transformasi fitur dari data mentah ke vektor input."""

    # Nama fitur sesuai dokumentasi SKPL
    FEATURE_NAMES = [
        "prev_cycle_dur",     # X5: Durasi siklus sebelumnya
        "avg_sleep_hours",    # X1: Rata-rata jam tidur 30 hari
        "avg_stress_score",   # X2: Rata-rata skor stres 30 hari
        "delta_bmi",          # X3: Perubahan BMI dari baseline
        "junkfood_freq",      # X4: Frekuensi makanan berisiko/hari
    ]

    @staticmethod
    def build_cycle_features(
        siklus_history: List[Dict],
        daily_aggregates: List[Dict],
        bmi_baseline: float = 22.0,
    ) -> List[Tuple[List[float], float]]:
        """
        Bangun pasangan (fitur, target) dari riwayat siklus dan data harian.

        Args:
            siklus_history: Daftar dict siklus terurut kronologis.
            daily_aggregates: Daftar dict agregat harian.
            bmi_baseline: BMI awal pengguna.

        Returns:
            List of (feature_vector, target_duration) tuples.
        """
        if len(siklus_history) < 2:
            return []

        dataset = []
        # Buat lookup harian
        daily_map = {d["tanggal"]: d for d in daily_aggregates}

        for i in range(1, len(siklus_history)):
            current = siklus_history[i]
            prev = siklus_history[i - 1]

            target = current.get("total_durasi_hari")
            prev_dur = prev.get("total_durasi_hari")
            if target is None or prev_dur is None:
                continue

            # Kumpulkan rata-rata harian dalam rentang siklus saat ini
            start_date = current.get("tanggal_mulai", "")
            end_date = current.get("tanggal_selesai", "")

            # Agregasi dari data harian
            sleep_vals = []
            stress_vals = []
            bmi_vals = []
            junk_vals = []

            for d in daily_aggregates:
                dt = d.get("tanggal", "")
                if start_date and end_date and start_date <= dt <= end_date:
                    if d.get("avg_sleep_hours") is not None:
                        sleep_vals.append(d["avg_sleep_hours"])
                    if d.get("avg_stress_score") is not None:
                        stress_vals.append(d["avg_stress_score"])
                    if d.get("bmi") is not None:
                        bmi_vals.append(d["bmi"])
                    if d.get("junkfood_freq") is not None:
                        junk_vals.append(d["junkfood_freq"])

            avg_sleep = sum(sleep_vals) / len(sleep_vals) if sleep_vals else 7.0
            avg_stress = (sum(stress_vals) / len(stress_vals)
                          if stress_vals else 40.0)
            avg_bmi = sum(bmi_vals) / len(bmi_vals) if bmi_vals else bmi_baseline
            delta_bmi = avg_bmi - bmi_baseline
            avg_junk = sum(junk_vals) / len(junk_vals) if junk_vals else 0.5

            feature_vector = [
                prev_dur,       # X5: prev_cycle_dur
                avg_sleep,      # X1: avg_sleep_hours
                avg_stress,     # X2: avg_stress_score
                delta_bmi,      # X3: delta_bmi
                avg_junk,       # X4: junkfood_freq
            ]

            dataset.append((feature_vector, target))

        return dataset

    @staticmethod
    def build_daily_features_for_peak(
        day_offset_from_ovulation: int,
        stress_score: float,
        sleep_quality: float,
    ) -> List[float]:
        """
        Bangun vektor fitur harian untuk model Peak Time Logistic.

        Args:
            day_offset_from_ovulation: Selisih hari dari estimasi ovulasi.
            stress_score: Skor stres hari ini (0-100).
            sleep_quality: Kualitas/durasi tidur hari ini (jam).

        Returns:
            Vektor fitur [offset^2, stress, sleep].
        """
        return [
            day_offset_from_ovulation ** 2,  # Jarak kuadrat dari ovulasi
            stress_score,
            sleep_quality,
        ]

    @staticmethod
    def extract_attribution_scores(
        weights: List[float],
        feature_values: List[float],
        means: List[float],
        stds: List[float],
    ) -> List[Dict]:
        """
        Hitung skor atribusi (ΔY_j) tiap fitur untuk Explainable AI.

        Formula: ΔY_j = β_j × (X_j - μ_j) / σ_j

        Returns:
            List of dicts [{name, value, delta_y, contribution_pct}, ...]
        """
        if len(weights) < 2:
            return []

        # weights[0] = bias, weights[1:] = koefisien fitur
        betas = weights[1:]  # tanpa bias
        attributions = []
        total_abs = 0.0

        for j, name in enumerate(FeatureEngineer.FEATURE_NAMES):
            if j >= len(betas) or j >= len(feature_values):
                break
            z_score = ((feature_values[j] - means[j]) / stds[j]
                       if stds[j] > 1e-8 else 0.0)
            delta_y = betas[j] * z_score
            total_abs += abs(delta_y)
            attributions.append({
                "feature_name": name,
                "raw_value": feature_values[j],
                "z_score": round(z_score, 4),
                "delta_y_days": round(delta_y, 4),
            })

        # Hitung persentase kontribusi
        for attr in attributions:
            pct = ((abs(attr["delta_y_days"]) / total_abs * 100)
                   if total_abs > 0 else 0)
            attr["contribution_pct"] = round(pct, 2)

        # Urutkan berdasarkan dampak terbesar
        attributions.sort(key=lambda x: abs(x["delta_y_days"]), reverse=True)
        return attributions

