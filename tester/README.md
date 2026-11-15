# SIKLUS Go Tester (Competitive Programming Style)

Harness pengujian Input/Output berbasis Go yang mengeksekusi dan memvalidasi hasil kalkulasi model Machine Learning Python secara real-time.

## Struktur File
- `main.go` : Test runner berbasis Go dengan fast I/O (`bufio`), deklarasi terpusat, tanpa komentar, tanpa underscore, dan logika manual.
- `engine.py`: Jembatan komputasi Python yang memanggil model terlatih dari `siklus-ml-models`.
- `input.txt` : Berkas kasus uji (test cases) kompetitif.

## Mode Pengujian
1. **Mode 1**: Ridge Cycle & Fertile Window
2. **Mode 2**: Logistic Peak Time
3. **Mode 3**: Gestational Weight Gain (GWG) Spline
4. **Mode 4**: Explainable AI (XAI) Risk Assessment

## Cara Menjalankan
```bash
cd siklus-go-tester
Get-Content input.txt | go run main.go
```

