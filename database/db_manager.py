"""
=============================================================================
SIKLUS ML - Database Manager
=============================================================================
Mengelola koneksi SQLite, inisialisasi skema, dan operasi CRUD lokal.
"""

import sqlite3
import os
from typing import Optional, List, Dict, Any


class DatabaseManager:
    """Manajer database SQLite lokal untuk SIKLUS."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_dir()
        self.conn: Optional[sqlite3.Connection] = None

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """Buka koneksi ke database SQLite."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
        return self.conn

    def close(self):
        """Tutup koneksi database."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def initialize_schema(self):
        """Jalankan schema.sql untuk membuat seluruh tabel."""
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        conn = self.connect()
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
        print(f"[DB] Skema berhasil diinisialisasi: {self.db_path}")

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        conn = self.connect()
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor

    def fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        conn = self.connect()
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        cursor = conn.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def insert_siklus(self, tanggal_mulai: str, tanggal_selesai: str,
                      total_durasi: float, catatan: str = ""):
        """Masukkan data siklus haid baru."""
        self.execute(
            """INSERT INTO siklus_haid (tanggal_mulai, tanggal_selesai,
               total_durasi_hari, catatan) VALUES (?, ?, ?, ?)""",
            (tanggal_mulai, tanggal_selesai, total_durasi, catatan)
        )

    def insert_tidur(self, tanggal: str, durasi_jam: float,
                     kualitas: int = 3):
        """Masukkan data tidur harian."""
        self.execute(
            """INSERT INTO waktu_tidur (tanggal, jam_tidur, durasi_jam,
               kualitas_tidur) VALUES (?, 0, ?, ?)""",
            (tanggal, durasi_jam, kualitas)
        )

    def insert_fisik_mood(self, tanggal: str, tingkat_stres: float,
                          nilai_bmi: float, berat_kg: float = 0,
                          suhu_basal: float = None):
        """Masukkan data fisik dan mood harian."""
        self.execute(
            """INSERT INTO catatan_fisik_mood (tanggal, tingkat_stres,
               nilai_bmi, berat_badan_kg, suhu_basal)
               VALUES (?, ?, ?, ?, ?)""",
            (tanggal, tingkat_stres, nilai_bmi, berat_kg, suhu_basal)
        )

    def insert_asupan(self, tanggal: str, kategori: str,
                      jumlah_porsi: float = 1.0):
        """Masukkan data asupan makan harian."""
        self.execute(
            """INSERT INTO asupan_makan (tanggal, kategori, jumlah_porsi)
               VALUES (?, ?, ?)""",
            (tanggal, kategori, jumlah_porsi)
        )

    def get_siklus_history(self, limit: int = 20) -> List[Dict]:
        """Ambil riwayat siklus terurut dari terbaru."""
        return self.fetch_all(
            """SELECT * FROM siklus_haid
               ORDER BY tanggal_mulai DESC LIMIT ?""", (limit,)
        )

    def get_bbt_history(self, limit: int = 30) -> List[float]:
        """Ambil riwayat suhu basal tubuh (BBT) harian untuk analisis CUSUM."""
        rows = self.fetch_all(
            """SELECT suhu_basal FROM catatan_fisik_mood
               WHERE suhu_basal IS NOT NULL
               ORDER BY tanggal ASC LIMIT ?""", (limit,)
        )
        return [float(r["suhu_basal"]) for r in rows]

    def get_daily_aggregates(self, start_date: str,
                             end_date: str) -> List[Dict]:
        """Ambil agregat harian gabungan tidur, stres, dan asupan."""
        return self.fetch_all("""
            SELECT
                t.tanggal,
                AVG(t.durasi_jam) AS avg_sleep_hours,
                COALESCE(m.tingkat_stres, 50.0) AS avg_stress_score,
                COALESCE(m.nilai_bmi, 22.0) AS bmi,
                COALESCE(
                    (SELECT COUNT(*) FROM asupan_makan a
                     WHERE a.tanggal = t.tanggal
                       AND a.kategori IN ('pedas','berlemak','manis')),
                    0
                ) AS junkfood_freq
            FROM waktu_tidur t
            LEFT JOIN catatan_fisik_mood m
                ON t.tanggal = m.tanggal
                AND t.id_pengguna = m.id_pengguna
            WHERE t.tanggal BETWEEN ? AND ?
            GROUP BY t.tanggal
            ORDER BY t.tanggal
        """, (start_date, end_date))

    def save_model_blob(self, tipe_model: str, blob_data: bytes,
                        mean_blob: bytes = None, std_blob: bytes = None,
                        jumlah_data: int = 0, mae: float = None):
        """Simpan atau perbarui bobot model ke tabel model_prediksi_lokal."""
        existing = self.fetch_one(
            "SELECT id_model FROM model_prediksi_lokal WHERE tipe_model = ?",
            (tipe_model,)
        )
        if existing:
            self.execute(
                """UPDATE model_prediksi_lokal SET
                   bobot_regresi = ?, mean_features = ?, std_features = ?,
                   jumlah_data = ?, mae_terakhir = ?,
                   updated_at = CURRENT_TIMESTAMP
                   WHERE tipe_model = ?""",
                (blob_data, mean_blob, std_blob, jumlah_data, mae, tipe_model)
            )
        else:
            self.execute(
                """INSERT INTO model_prediksi_lokal
                   (tipe_model, bobot_regresi, mean_features, std_features,
                    jumlah_data, mae_terakhir)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (tipe_model, blob_data, mean_blob, std_blob,
                 jumlah_data, mae)
            )

    def load_model_blob(self, tipe_model: str) -> Optional[Dict]:
        """Muat bobot model dari database."""
        return self.fetch_one(
            "SELECT * FROM model_prediksi_lokal WHERE tipe_model = ?",
            (tipe_model,)
        )

