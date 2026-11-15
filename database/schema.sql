-- ==========================================================================
-- SIKLUS - Skema Database SQLite Lokal (On-Device)
-- ==========================================================================

-- Tabel Data Siklus Haid
CREATE TABLE IF NOT EXISTS siklus_haid (
    id_siklus       INTEGER PRIMARY KEY AUTOINCREMENT,
    id_pengguna     TEXT    NOT NULL DEFAULT 'local_user',
    tanggal_mulai   DATE    NOT NULL,
    tanggal_selesai DATE,
    total_durasi_hari REAL,
    catatan         TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabel Catatan Tidur Harian
CREATE TABLE IF NOT EXISTS waktu_tidur (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    id_pengguna     TEXT    NOT NULL DEFAULT 'local_user',
    tanggal         DATE    NOT NULL,
    jam_tidur       REAL    NOT NULL,  -- dalam jam (misal: 7.5)
    jam_bangun      REAL,
    durasi_jam      REAL    NOT NULL,  -- total durasi tidur (jam)
    kualitas_tidur  INTEGER DEFAULT 3, -- skala 1-5
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabel Catatan Fisik & Mood Harian
CREATE TABLE IF NOT EXISTS catatan_fisik_mood (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    id_pengguna     TEXT    NOT NULL DEFAULT 'local_user',
    tanggal         DATE    NOT NULL,
    tingkat_stres   REAL    DEFAULT 0,  -- skor 0-100 (dari Heart Rate)
    nilai_bmi       REAL,
    berat_badan_kg  REAL,
    tinggi_badan_cm REAL,
    mood_score      INTEGER DEFAULT 3,  -- skala 1-5
    suhu_basal      REAL,               -- Celsius (opsional)
    catatan         TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabel Asupan Makan Harian
CREATE TABLE IF NOT EXISTS asupan_makan (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    id_pengguna     TEXT    NOT NULL DEFAULT 'local_user',
    tanggal         DATE    NOT NULL,
    kategori        TEXT    NOT NULL,    -- 'sehat', 'pedas', 'berlemak', 'manis'
    jumlah_porsi    REAL    DEFAULT 1.0,
    kalori_estimasi REAL,
    catatan         TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabel Penyimpanan Model Prediksi Lokal (BLOB)
CREATE TABLE IF NOT EXISTS model_prediksi_lokal (
    id_model        INTEGER PRIMARY KEY AUTOINCREMENT,
    id_pengguna     TEXT    NOT NULL DEFAULT 'local_user',
    tipe_model      TEXT    NOT NULL,    -- 'ridge_cycle', 'logistic_peak', 'gwg_spline'
    versi_model     INTEGER DEFAULT 1,
    bobot_regresi   BLOB    NOT NULL,    -- IEEE 754 float32 array
    mean_features   BLOB,                -- rata-rata fitur (untuk normalisasi)
    std_features    BLOB,                -- standar deviasi fitur
    jumlah_data     INTEGER DEFAULT 0,
    mae_terakhir    REAL,                -- Mean Absolute Error terakhir
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabel Data Kehamilan
CREATE TABLE IF NOT EXISTS data_kehamilan (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    id_pengguna     TEXT    NOT NULL DEFAULT 'local_user',
    tanggal_hpht    DATE    NOT NULL,    -- Hari Pertama Haid Terakhir
    bmi_pra_hamil   REAL    NOT NULL,
    berat_pra_hamil REAL    NOT NULL,
    minggu_ke       INTEGER DEFAULT 0,
    berat_saat_ini  REAL,
    status          TEXT    DEFAULT 'aktif', -- 'aktif', 'selesai'
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indeks untuk performa query
CREATE INDEX IF NOT EXISTS idx_siklus_pengguna ON siklus_haid(id_pengguna, tanggal_mulai);
CREATE INDEX IF NOT EXISTS idx_tidur_tanggal ON waktu_tidur(id_pengguna, tanggal);
CREATE INDEX IF NOT EXISTS idx_mood_tanggal ON catatan_fisik_mood(id_pengguna, tanggal);
CREATE INDEX IF NOT EXISTS idx_makan_tanggal ON asupan_makan(id_pengguna, tanggal);
CREATE INDEX IF NOT EXISTS idx_model_tipe ON model_prediksi_lokal(id_pengguna, tipe_model);

