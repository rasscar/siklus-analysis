"""
=============================================================================
SIKLUS ML - Database Seeder
=============================================================================
Mengisi database SQLite dengan data sintetis dari SyntheticDataGenerator.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from datasets.synthetic_generator import SyntheticDataGenerator
from config import DB_PATH


def seed_database():
    """Inisialisasi dan isi database dengan data sintetis."""
    db = DatabaseManager(DB_PATH)
    db.initialize_schema()

    gen = SyntheticDataGenerator(num_cycles=12, base_cycle_dur=28.0)
    cycles, X, y = gen.generate_cycles()
    sleep_data, mood_data, food_data = gen.generate_daily_data(cycles)

    print(f"\n[Seeder] Menyimpan {len(cycles)} siklus ke database...")
    for c in cycles:
        db.insert_siklus(
            c["tanggal_mulai"], c["tanggal_selesai"],
            c["total_durasi_hari"], c["catatan"]
        )

    print(f"[Seeder] Menyimpan {len(sleep_data)} catatan tidur...")
    for s in sleep_data:
        db.insert_tidur(s["tanggal"], s["durasi_jam"], s["kualitas_tidur"])

    print(f"[Seeder] Menyimpan {len(mood_data)} catatan fisik/mood...")
    for m in mood_data:
        db.insert_fisik_mood(
            m["tanggal"], m["tingkat_stres"],
            m["nilai_bmi"], m["berat_badan_kg"]
        )

    print(f"[Seeder] Menyimpan {len(food_data)} catatan asupan makan...")
    for f_item in food_data:
        db.insert_asupan(
            f_item["tanggal"], f_item["kategori"], f_item["jumlah_porsi"]
        )

    print("[Seeder] Database berhasil di-seed!")
    db.close()
    return db


if __name__ == "__main__":
    seed_database()

