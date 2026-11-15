# SIKLUS Analysis — On-Device ML Models & Test Harness

Sistem Informasi Kesehatan Siklus Haid & Program Kehamilan Cerdas berbasis **Edge AI / On-Device Machine Learning (100% Offline)**.

---

## 📁 Struktur Repositori

```text
siklus-analysis/
├── config.py                      # Parameter global model (NFR-005 SKPL)
├── main.py                        # Demo pipeline inferensi & pelatihan
├── database/
│   ├── schema.sql                 # Skema DDL tabel lokal SQLite
│   ├── db_manager.py              # Pengelola koneksi & transaksi ACID
│   └── seed_data.py               # Seeder data pengujian awal
├── models/                        # Modul Model Prediksi Inti (Python Pure)
│   ├── ridge_cycle_predictor.py   # Model 1: Siklus haid & fertile window (Ridge L2)
│   ├── logistic_peak_time.py      # Model 2a: Peluang kesuburan harian (Sigmoid)
│   ├── gwg_spline_tracker.py      # Model 2b: Trajektori berat badan hamil (IOM)
│   ├── explainable_risk_analyzer.py # Model 3: Analisis risiko klinis (XAI)
│   └── federated_aggregator.py    # Model 4: Agregasi bobot anonim (FedAvg)
├── utils/                         # Pustaka Utilitas Matematika On-Device
│   ├── scaler.py                  # Z-Score Feature Scaling
│   ├── blob_serializer.py         # IEEE 754 float32 biner ↔ SQLite BLOB
│   └── feature_engineering.py     # Ekstraksi fitur harian & atribusi
├── datasets/                      # Generator & CSV Data Sintetis
│   └── synthetic_generator.py     # Pembuat dataset sintetis 12 siklus
├── tests/                         # Unit Test Python
│   └── test_models.py             # 7/7 test suite untuk seluruh model
├── tester/                        # [FOLDER KHUSUS TESTER] Harness Go CP-Style
│   ├── main.go                    # Test runner berbasis Go (fast I/O)
│   ├── engine.py                  # Jembatan eksekusi model Python
│   ├── input.txt                  # Berkas kasus uji kompetitif
│   └── README.md                  # Dokumentasi modul tester
└── .gitignore                     # Proteksi file rahasia, cache, & DB
```

---

## 🚀 Ringkasan Model Prediksi

| # | Model | Peran Pengguna | Algoritma | Karakteristik Edge AI |
|---|---|---|---|---|
| 1 | **Ridge Cycle Predictor** | Wanita Menstruasi | Ridge Regression ($L_2$) + EMA | Ukuran < 10 KB, inferensi < 1 ms, update on-device SGD |
| 2a | **Logistic Peak Time** | Promil | Logistic Regression (Sigmoid) | Ambang kesuburan puncak $P \ge 0.70$ |
| 2b | **GWG Spline Tracker** | Ibu Hamil | Piecewise Quadratic (IOM) | Kurva kenaikan berat trimester I-III |
| 3 | **Explainable Risk** | Tenaga Kesehatan | Attribution Score (XAI) | Menjelaskan faktor pemicu anomali siklus |
| 4 | **Federated Aggregator** | Developer | FedAvg + Outlier Sanitizer | Pembaruan bobot anonim via BLOB SQLite |

---

## 💻 Cara Menjalankan

### 1. Menjalankan Demo Pipeline Lengkap (Python)
```bash
python main.py
```

### 2. Menjalankan Unit Test (Python)
```bash
python tests/test_models.py
```

### 3. Menjalankan Test Harness (Go CP-Style)
```bash
cd tester
Get-Content input.txt | go run main.go
```
