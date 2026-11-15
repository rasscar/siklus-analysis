"""
=============================================================================
MODEL 3: Explainable Risk Analyzer (Tenaga Kesehatan - XAI)
=============================================================================
Analisis risiko transparan untuk dokter/bidan menggunakan dekomposisi
koefisien terbobot (Attribution Score) dan Shallow Decision Rules.

Formulasi Atribusi:
  ΔYⱼ = βⱼ × (Xⱼ - μⱼ) / σⱼ

Deteksi Anomali:
  - Oligomenorea: Ŷ > 35 hari
  - Polimenorea:  Ŷ < 21 hari
  - Stres signifikan: ΔY_stres >= +3.0 hari
  - Defisit tidur:    ΔY_tidur >= +2.5 hari
"""

from typing import List, Dict, Tuple, Optional

from config import (
    ANOMALY_LONG_CYCLE, ANOMALY_SHORT_CYCLE,
    STRESS_IMPACT_THRESHOLD, SLEEP_IMPACT_THRESHOLD,
)
from utils.feature_engineering import FeatureEngineer


class ExplainableRiskAnalyzer:
    """
    Modul Explainable AI (XAI) yang menghitung kontribusi setiap
    faktor gaya hidup terhadap anomali siklus. Dirancang agar
    output-nya langsung dapat dibaca oleh tenaga kesehatan.
    """

    RISK_LABELS = {
        "oligomenorea": {
            "threshold": ANOMALY_LONG_CYCLE,
            "direction": "above",
            "severity": "⚠️ SEDANG",
            "description": "Siklus panjang > 35 hari. Dugaan anovulasi, "
                           "PCOS, atau gangguan hormonal.",
        },
        "polimenorea": {
            "threshold": ANOMALY_SHORT_CYCLE,
            "direction": "below",
            "severity": "⚠️ SEDANG",
            "description": "Siklus pendek < 21 hari. Dugaan disfungsi "
                           "fase luteal.",
        },
        "amenorea_sekunder": {
            "threshold": 90,
            "direction": "above",
            "severity": "🔴 TINGGI",
            "description": "Tidak haid > 90 hari (bukan hamil). "
                           "Segera konsultasi spesialis.",
        },
    }

    def __init__(self):
        self.last_analysis: Optional[Dict] = None

    # ─── Analisis Utama ──────────────────────────────────────────────────

    def analyze(
        self,
        predicted_duration: float,
        weights: List[float],
        feature_values: List[float],
        means: List[float],
        stds: List[float],
        patient_id: str = "anonim",
    ) -> Dict:
        """
        Lakukan analisis risiko lengkap untuk satu pasien.

        Args:
            predicted_duration: Prediksi durasi siklus (hari).
            weights: Bobot model [bias, w1, w2, w3, w4, w5].
            feature_values: Nilai fitur aktual [X1, X2, X3, X4, X5].
            means: Rata-rata fitur dari scaler.
            stds: Standar deviasi fitur dari scaler.
            patient_id: ID pasien (anonim).

        Returns:
            Dict: Laporan analisis risiko lengkap.
        """
        # 1. Hitung attribution scores
        attributions = FeatureEngineer.extract_attribution_scores(
            weights, feature_values, means, stds
        )

        # 2. Deteksi anomali siklus
        anomalies = self._detect_anomalies(predicted_duration)

        # 3. Analisis dampak faktor spesifik
        factor_alerts = self._analyze_factor_impacts(attributions)

        # 4. Buat ringkasan klinis
        clinical_summary = self._generate_clinical_summary(
            predicted_duration, attributions, anomalies, factor_alerts
        )

        result = {
            "patient_id": patient_id,
            "predicted_duration_days": round(predicted_duration, 1),
            "attribution_scores": attributions,
            "anomalies_detected": anomalies,
            "factor_alerts": factor_alerts,
            "clinical_summary": clinical_summary,
            "risk_level": self._overall_risk_level(anomalies, factor_alerts),
        }

        self.last_analysis = result
        return result

    # ─── Deteksi Anomali Siklus ──────────────────────────────────────────

    def _detect_anomalies(self, predicted_dur: float) -> List[Dict]:
        """Deteksi anomali berdasarkan ambang batas klinis."""
        anomalies = []
        for name, rule in self.RISK_LABELS.items():
            triggered = False
            if rule["direction"] == "above" and predicted_dur > rule["threshold"]:
                triggered = True
            elif rule["direction"] == "below" and predicted_dur < rule["threshold"]:
                triggered = True

            if triggered:
                anomalies.append({
                    "type": name,
                    "severity": rule["severity"],
                    "description": rule["description"],
                    "predicted_value": round(predicted_dur, 1),
                    "threshold": rule["threshold"],
                })
        return anomalies

    # ─── Analisis Dampak Faktor ──────────────────────────────────────────

    def _analyze_factor_impacts(
        self, attributions: List[Dict]
    ) -> List[Dict]:
        """Identifikasi faktor gaya hidup dengan dampak signifikan."""
        alerts = []
        for attr in attributions:
            name = attr["feature_name"]
            delta = attr["delta_y_days"]

            if name == "avg_stress_score" and delta >= STRESS_IMPACT_THRESHOLD:
                alerts.append({
                    "factor": "Tingkat Stres",
                    "impact_days": round(delta, 2),
                    "severity": "🟠 SIGNIFIKAN",
                    "detail": (
                        f"Stres memperlambat siklus sebesar "
                        f"+{delta:.1f} hari. Skor stres rata-rata "
                        f"di atas normal."
                    ),
                    "recommendation": (
                        "Sarankan teknik relaksasi, kurangi beban "
                        "kerja, atau rujuk ke psikolog."
                    ),
                })

            elif name == "avg_sleep_hours" and delta >= SLEEP_IMPACT_THRESHOLD:
                alerts.append({
                    "factor": "Durasi Tidur",
                    "impact_days": round(delta, 2),
                    "severity": "🟡 MODERAT",
                    "detail": (
                        f"Kurang tidur memperlambat siklus sebesar "
                        f"+{delta:.1f} hari. Rata-rata tidur di "
                        f"bawah 6 jam."
                    ),
                    "recommendation": (
                        "Anjurkan tidur minimal 7-8 jam per malam "
                        "dan evaluasi higiene tidur."
                    ),
                })

            elif name == "delta_bmi" and abs(delta) >= 2.0:
                direction = "kenaikan" if delta > 0 else "penurunan"
                alerts.append({
                    "factor": "Perubahan BMI",
                    "impact_days": round(delta, 2),
                    "severity": "🟡 MODERAT",
                    "detail": (
                        f"Fluktuasi BMI ({direction}) mempengaruhi "
                        f"siklus sebesar {abs(delta):.1f} hari."
                    ),
                    "recommendation": (
                        "Evaluasi pola makan dan aktivitas fisik. "
                        "Pertimbangkan konsultasi ahli gizi."
                    ),
                })

        return alerts

    # ─── Ringkasan Klinis ────────────────────────────────────────────────

    def _generate_clinical_summary(
        self,
        predicted_dur: float,
        attributions: List[Dict],
        anomalies: List[Dict],
        alerts: List[Dict],
    ) -> str:
        """Buat teks ringkasan untuk laporan PDF tenaga kesehatan."""
        lines = []
        lines.append("═══ RINGKASAN ANALISIS KLINIS ═══")
        lines.append(f"Prediksi durasi siklus: {predicted_dur:.1f} hari")
        lines.append("")

        if not anomalies:
            lines.append("Status: ✅ Siklus dalam rentang normal "
                         "(21-35 hari).")
        else:
            for a in anomalies:
                lines.append(f"[{a['severity']}] {a['type'].upper()}: "
                             f"{a['description']}")

        lines.append("")
        lines.append("─── Kontribusi Faktor Utama ───")
        for attr in attributions[:3]:  # Top 3
            sign = "+" if attr["delta_y_days"] > 0 else ""
            lines.append(
                f"  {attr['feature_name']:>20s}: "
                f"{sign}{attr['delta_y_days']:.2f} hari "
                f"({attr['contribution_pct']:.1f}%)"
            )

        if alerts:
            lines.append("")
            lines.append("─── Peringatan Faktor Gaya Hidup ───")
            for alert in alerts:
                lines.append(f"  [{alert['severity']}] {alert['factor']}: "
                             f"{alert['detail']}")
                lines.append(f"    Rekomendasi: {alert['recommendation']}")

        return "\n".join(lines)

    # ─── Risk Level ──────────────────────────────────────────────────────

    def _overall_risk_level(self, anomalies: List[Dict],
                            alerts: List[Dict]) -> str:
        """Tentukan level risiko keseluruhan."""
        has_high = any("TINGGI" in a["severity"] for a in anomalies)
        has_medium = len(anomalies) > 0 or len(alerts) >= 2
        has_low = len(alerts) == 1

        if has_high:
            return "🔴 TINGGI - Rujuk Segera"
        elif has_medium:
            return "🟠 SEDANG - Perlu Perhatian"
        elif has_low:
            return "🟡 RENDAH - Monitor Berkala"
        else:
            return "🟢 NORMAL"

