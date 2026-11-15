"""
=============================================================================
SIKLUS ML - Synthetic Data Generator & Seeder
=============================================================================
Menghasilkan data sintetis realistis untuk pengujian model dan
mengisi database SQLite lokal.
"""

import random
import math
import csv
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Seed untuk reproduksibilitas
random.seed(42)


class SyntheticDataGenerator:
    """Pembuat data sintetis fisiologis untuk SIKLUS."""

    def __init__(self, num_cycles: int = 12, base_cycle_dur: float = 28.0):
        """
        Args:
            num_cycles: Jumlah siklus yang akan digenerate.
            base_cycle_dur: Durasi siklus dasar (hari).
        """
        self.num_cycles = num_cycles
        self.base_cycle_dur = base_cycle_dur

    def generate_cycles(self) -> Tuple[List[Dict], List[List[float]], List[float]]:
        """
        Generate data siklus haid sintetis beserta fitur dan target.

        Returns:
            Tuple: (cycle_records, feature_matrix_X, target_vector_y)
        """
        cycles = []
        X = []
        y = []

        start_date = datetime(2026, 1, 1)
        prev_dur = self.base_cycle_dur

        for i in range(self.num_cycles):
            # Simulasi variasi fisiologis
            stress = random.uniform(15, 85)
            sleep = random.uniform(4.5, 9.5)
            delta_bmi = random.uniform(-1.5, 2.0)
            junkfood = random.uniform(0.0, 3.0)

            # Model generatif: durasi dipengaruhi faktor gaya hidup
            noise = random.gauss(0, 1.2)
            duration = (
                0.3 * prev_dur
                + 0.5 * self.base_cycle_dur
                - 0.4 * (sleep - 7.0)       # kurang tidur → siklus panjang
                + 0.08 * (stress - 40.0)     # stres tinggi → siklus panjang
                + 0.5 * delta_bmi            # BMI naik → siklus panjang
                + 0.3 * (junkfood - 1.0)     # junk food → siklus panjang
                + noise
            )
            duration = max(21.0, min(45.0, round(duration, 1)))

            end_date = start_date + timedelta(days=int(duration))

            cycle = {
                "tanggal_mulai": start_date.strftime("%Y-%m-%d"),
                "tanggal_selesai": end_date.strftime("%Y-%m-%d"),
                "total_durasi_hari": duration,
                "catatan": f"Siklus sintetis ke-{i+1}",
            }
            cycles.append(cycle)

            if i > 0:  # Fitur memerlukan siklus sebelumnya
                features = [prev_dur, sleep, stress, delta_bmi, junkfood]
                X.append(features)
                y.append(duration)

            prev_dur = duration
            start_date = end_date + timedelta(days=1)

        return cycles, X, y

    def generate_daily_data(
        self, cycles: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Generate data harian (tidur, mood/fisik, asupan) untuk setiap siklus.

        Returns:
            Tuple: (sleep_records, mood_records, food_records)
        """
        sleep_records = []
        mood_records = []
        food_records = []

        bmi_baseline = round(random.uniform(19.0, 27.0), 1)

        for cycle in cycles:
            start = datetime.strptime(cycle["tanggal_mulai"], "%Y-%m-%d")
            end = datetime.strptime(cycle["tanggal_selesai"], "%Y-%m-%d")
            current = start

            while current <= end:
                date_str = current.strftime("%Y-%m-%d")

                # Data tidur
                sleep_dur = round(random.gauss(7.0, 1.2), 1)
                sleep_dur = max(3.5, min(10.0, sleep_dur))
                sleep_records.append({
                    "tanggal": date_str,
                    "durasi_jam": sleep_dur,
                    "kualitas_tidur": random.randint(1, 5),
                })

                # Data mood/fisik
                stress = round(random.gauss(45, 18), 1)
                stress = max(0, min(100, stress))
                bmi_fluctuation = round(
                    bmi_baseline + random.gauss(0, 0.5), 1
                )
                mood_records.append({
                    "tanggal": date_str,
                    "tingkat_stres": stress,
                    "nilai_bmi": bmi_fluctuation,
                    "berat_badan_kg": round(bmi_fluctuation * 1.65 ** 2, 1),
                })

                # Data asupan
                categories = ["sehat", "pedas", "berlemak", "manis"]
                cat = random.choice(categories)
                food_records.append({
                    "tanggal": date_str,
                    "kategori": cat,
                    "jumlah_porsi": round(random.uniform(0.5, 2.0), 1),
                })

                current += timedelta(days=1)

        return sleep_records, mood_records, food_records

    def generate_pregnancy_data(self) -> List[Dict]:
        """Generate data pemantauan kehamilan sintetis (40 minggu)."""
        bmi_pra = round(random.uniform(19.5, 26.0), 1)
        berat_pra = round(bmi_pra * 1.60 ** 2, 1)
        records = []

        for week in range(0, 41):
            # Simulasi kenaikan berat realistis
            if week <= 13:
                gain = 0.04 * week + 0.001 * (week ** 2)
            elif week <= 27:
                gain = 1.5 + 0.38 * (week - 14) + 0.0004 * ((week - 14) ** 2)
            else:
                gain = 1.5 + 5.2 + 0.42 * (week - 28) + 0.0003 * ((week - 28) ** 2)

            noise = random.gauss(0, 0.3)
            actual = round(berat_pra + gain + noise, 1)

            records.append({
                "minggu_ke": week,
                "bmi_pra_hamil": bmi_pra,
                "berat_pra_hamil": berat_pra,
                "berat_aktual_kg": actual,
                "gain_kg": round(actual - berat_pra, 1),
            })

        return records

    def export_to_csv(self, output_dir: str):
        """Export seluruh data sintetis ke file CSV."""
        os.makedirs(output_dir, exist_ok=True)

        # Siklus + fitur
        cycles, X, y = self.generate_cycles()
        sleep_data, mood_data, food_data = self.generate_daily_data(cycles)
        pregnancy_data = self.generate_pregnancy_data()

        # CSV: Dataset Fitur Pelatihan
        feature_path = os.path.join(output_dir, "training_features.csv")
        with open(feature_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cycle_no", "prev_cycle_dur", "avg_sleep_hours",
                "avg_stress_score", "delta_bmi", "junkfood_freq",
                "target_duration"
            ])
            for i, (features, target) in enumerate(zip(X, y)):
                writer.writerow([i + 1] + [round(v, 2) for v in features] +
                                [round(target, 1)])

        # CSV: Data Siklus
        cycle_path = os.path.join(output_dir, "siklus_haid.csv")
        with open(cycle_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cycles[0].keys())
            writer.writeheader()
            writer.writerows(cycles)

        # CSV: Data Tidur
        sleep_path = os.path.join(output_dir, "waktu_tidur.csv")
        with open(sleep_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sleep_data[0].keys())
            writer.writeheader()
            writer.writerows(sleep_data)

        # CSV: Data Fisik/Mood
        mood_path = os.path.join(output_dir, "catatan_fisik_mood.csv")
        with open(mood_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=mood_data[0].keys())
            writer.writeheader()
            writer.writerows(mood_data)

        # CSV: Data Asupan
        food_path = os.path.join(output_dir, "asupan_makan.csv")
        with open(food_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=food_data[0].keys())
            writer.writeheader()
            writer.writerows(food_data)

        # CSV: Data Kehamilan
        preg_path = os.path.join(output_dir, "data_kehamilan.csv")
        with open(preg_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pregnancy_data[0].keys())
            writer.writeheader()
            writer.writerows(pregnancy_data)

        print(f"[Dataset] Exported {len(X)} training samples ke {output_dir}")
        print(f"  - {feature_path}")
        print(f"  - {cycle_path}")
        print(f"  - {sleep_path}")
        print(f"  - {mood_path}")
        print(f"  - {food_path}")
        print(f"  - {preg_path}")

        return {
            "training_features": feature_path,
            "siklus_haid": cycle_path,
            "waktu_tidur": sleep_path,
            "catatan_fisik_mood": mood_path,
            "asupan_makan": food_path,
            "data_kehamilan": preg_path,
        }


if __name__ == "__main__":
    gen = SyntheticDataGenerator(num_cycles=12)
    gen.export_to_csv(os.path.join(os.path.dirname(__file__)))

