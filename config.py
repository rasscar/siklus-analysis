"""
=============================================================================
SIKLUS ML Models - Konfigurasi Global
=============================================================================
Sistem Informasi Kesehatan Siklus Haid & Program Kehamilan Cerdas
Seluruh parameter sesuai dokumen SKPL (NFR-005).
"""

import os

# ─── Path & Database ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "siklus_local.db")

# ─── Batasan Pelatihan Model (SKPL NFR-005) ─────────────────────────────────
MIN_DATA_POINTS = 30          # Minimal titik data sebelum model aktif
LEARNING_RATE = 0.01          # Alpha (α) laju belajar SGD
REGULARIZATION_LAMBDA = 0.05  # Lambda (λ) penalti Ridge / L2
MAX_EPOCHS_LOCAL = 50         # Iterasi maksimum pelatihan on-device per sesi

# ─── Heuristik Cold-Start (< 30 data) ───────────────────────────────────────
DEFAULT_CYCLE_DURATION = 28   # Rata-rata siklus standar (hari) - Rumus Naegele
DEFAULT_LUTEAL_PHASE = 14     # Fase luteal standar (hari)
FERTILE_WINDOW_BEFORE = 5     # Hari sebelum ovulasi
FERTILE_WINDOW_AFTER = 1      # Hari setelah ovulasi

# ─── Ambang Keputusan ───────────────────────────────────────────────────────
PEAK_FERTILITY_THRESHOLD = 0.70  # P(Fertil) >= 0.70 = Peak Time
ANOMALY_LONG_CYCLE = 35          # Oligomenorea jika > 35 hari
ANOMALY_SHORT_CYCLE = 21         # Polimenorea jika < 21 hari
STRESS_IMPACT_THRESHOLD = 3.0    # Delta Y stres >= 3.0 hari = signifikan
SLEEP_IMPACT_THRESHOLD = 2.5     # Delta Y tidur >= 2.5 hari = signifikan

# ─── GWG (Gestational Weight Gain) per Trimester ────────────────────────────
GWG_TRIMESTER_CONFIG = {
    1: {"start_week": 0,  "end_week": 13, "rate_kg_per_week": 0.10},
    2: {"start_week": 14, "end_week": 27, "rate_kg_per_week": 0.42},
    3: {"start_week": 28, "end_week": 40, "rate_kg_per_week": 0.45},
}

# ─── BLOB Serialisasi ───────────────────────────────────────────────────────
BLOB_FLOAT_FORMAT = "<f"   # IEEE 754 float32 Little-Endian
BLOB_UINT_FORMAT = "<I"    # uint32 Little-Endian (checksum / versi model)
MODEL_VERSION = 1

# ─── Federated Learning ─────────────────────────────────────────────────────
FEDAVG_MIN_CLIENTS = 5    # Minimum partisipan sebelum agregasi
DELTA_OUTLIER_ZSCORE = 2.0 # Filter delta bobot outlier (z > 2σ)
