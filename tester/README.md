# SIKLUS Go Tester (Competitive Programming Style)

Harness pengujian Input/Output berbasis Go yang mengeksekusi dan memvalidasi hasil kalkulasi model Machine Learning Python secara real-time.

## Struktur File
- `main.go` : Test runner berbasis Go dengan fast I/O (`bufio`), deklarasi terpusat, tanpa komentar, tanpa underscore, dan logika manual.
- `engine.py`: Jembatan komputasi Python yang memanggil model terlatih dari `models/`.
- `input.txt` : Berkas kasus uji (test cases) kompetitif.

## Mode Pengujian
1. **Mode 1**: Ridge Cycle & Fertile Window (`models/ridge_cycle_predictor.py`)
2. **Mode 2**: Logistic Peak Time (`models/logistic_peak_time.py`)
3. **Mode 3**: Gestational Weight Gain (GWG) Spline (`models/gwg_spline_tracker.py`)
4. **Mode 4**: Explainable AI (XAI) Risk Assessment (`models/explainable_risk_analyzer.py`)
5. **Mode 5**: BBT CUSUM Detector Ovulation Confirmation (`models/bbt_cusum_detector.py`)
6. **Mode 6**: Multi-Factor Cycle Regularity Classifier (`models/cycle_regularity_classifier.py`)
7. **Mode 7**: Circadian-Disruption Cycle Predictor with SJL & $\sigma(T_{mid})$ (`models/circadian_cycle_predictor.py`)

## Cara Menjalankan
```powershell
Set-Location tester
Get-Content input.txt | go run main.go
```
