"""
=============================================================================
MODEL A: BBT CUSUM Detector (Termoregulasi Ovulasi)
=============================================================================
Deteksi titik balik fase luteal dan konfirmasi hari ovulasi menggunakan
algoritma Cumulative Sum (CUSUM) adaptif berbasis pergeseran suhu basal
tubuh (Basal Body Temperature - BBT).

Formulasi:
  S_t = max(0, S_{t-1} + (T_t - μ_folikuler - K))
  Kondisi konfirmasi: S_t >= H selama minimal 3 hari berturut-turut.
  Ambang standar: H = 0.30°C, K = 0.15°C.
"""

from typing import List, Dict, Optional


class BBTCusumDetector:
    """Detektor titik ovulasi berbasis suhu basal tubuh (BBT)."""

    def __init__(
        self,
        shift_threshold: float = 0.30,
        min_sustained_days: int = 3,
        slack_k: float = 0.15,
    ):
        self.shift_threshold = shift_threshold
        self.min_sustained_days = min_sustained_days
        self.slack_k = slack_k

    def detect_ovulation(
        self, bbt_series: List[float], baseline_days: int = 6
    ) -> Dict:
        """
        Deteksi hari ovulasi dari deret waktu suhu harian.

        Args:
            bbt_series: Daftar pembacaan suhu pagi (°C) terurut hari ke-1, 2, ...
            baseline_days: Jumlah hari awal fase folikuler untuk menghitung μ.

        Returns:
            Dict: status konfirmasi, hari ovulasi, delta suhu, nilai CUSUM.
        """
        n = len(bbt_series)
        if n < baseline_days + self.min_sustained_days:
            return {
                "confirmed": False,
                "ovulation_day": -1,
                "temperature_delta": 0.0,
                "message": "Data suhu belum mencukupi (minimal 9 hari)",
            }

        # Hitung rata-rata baseline fase folikuler awal
        mu_folikuler = sum(bbt_series[:baseline_days]) / baseline_days

        s_t = 0.0
        consecutive_high = 0
        ovulation_day = -1
        max_delta = 0.0

        for t in range(baseline_days, n):
            temp = bbt_series[t]
            delta = temp - mu_folikuler

            # CUSUM step
            s_t = max(0.0, s_t + (delta - self.slack_k))

            if delta >= self.shift_threshold:
                consecutive_high += 1
                if delta > max_delta:
                    max_delta = delta
            else:
                consecutive_high = 0
                s_t = 0.0

            if consecutive_high >= self.min_sustained_days and ovulation_day == -1:
                # Hari ovulasi terjadi tepat sebelum lonjakan suhu termogenik pertama
                ovulation_day = (t + 1) - self.min_sustained_days

        confirmed = ovulation_day != -1

        return {
            "confirmed": confirmed,
            "ovulation_day": ovulation_day,
            "baseline_temperature": round(mu_folikuler, 2),
            "temperature_delta": round(max_delta, 2),
            "message": (
                f"Ovulasi terkonfirmasi pada hari ke-{ovulation_day} (lonjakan +{max_delta:.2f}°C)"
                if confirmed
                else "Belum terdeteksi lonjakan suhu signifikan (fase anovulatoar atau belum ovulasi)"
            ),
        }

