"""
=============================================================================
SIKLUS ML - Main Demo Runner
=============================================================================
Menjalankan demonstrasi lengkap seluruh pipeline:
  1. Generate data sintetis → CSV + SQLite
  2. Pelatihan Model 1 (Ridge Cycle Predictor)
  3. Prediksi Model 2a (Logistic Peak Time)
  4. Pemantauan Model 2b (GWG Spline Tracker)
  5. Analisis Model 3 (Explainable Risk Analyzer)
  6. Simulasi Model 4 (Federated Aggregator)
  7. Serialisasi BLOB SQLite
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_PATH
from database.db_manager import DatabaseManager
from datasets.synthetic_generator import SyntheticDataGenerator
from models.ridge_cycle_predictor import RidgeCyclePredictor
from models.logistic_peak_time import LogisticPeakTime
from models.gwg_spline_tracker import GWGSplineTracker
from models.explainable_risk_analyzer import ExplainableRiskAnalyzer
from models.federated_aggregator import FederatedAggregator
from utils.scaler import ZScoreScaler
from utils.blob_serializer import BlobSerializer


def print_header(title):
    print(f"\n{'━'*70}")
    print(f"  {title}")
    print(f"{'━'*70}")


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   SIKLUS ML Models — Full Pipeline Demo                        ║")
    print("║   Sistem Informasi Kesehatan Siklus Haid & Kehamilan Cerdas    ║")
    print("║   100% Offline / Edge AI / On-Device Machine Learning          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Generate Data Sintetis
    # ──────────────────────────────────────────────────────────────────────
    print_header("STEP 1: Generate Data Sintetis & Inisialisasi Database")

    gen = SyntheticDataGenerator(num_cycles=12, base_cycle_dur=28.0)
    cycles, X_raw, y_raw = gen.generate_cycles()
    sleep_data, mood_data, food_data = gen.generate_daily_data(cycles)

    # Export CSV
    csv_dir = os.path.join(os.path.dirname(__file__), "datasets")
    csv_paths = gen.export_to_csv(csv_dir)

    # Setup database
    db = DatabaseManager(DB_PATH)
    db.initialize_schema()

    for c in cycles:
        db.insert_siklus(c["tanggal_mulai"], c["tanggal_selesai"],
                         c["total_durasi_hari"], c["catatan"])
    print(f"  ✅ {len(cycles)} siklus disimpan ke SQLite")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Pelatihan Ridge Cycle Predictor
    # ──────────────────────────────────────────────────────────────────────
    print_header("STEP 2: Pelatihan Model 1 — Ridge Cycle Predictor")

    scaler = ZScoreScaler()
    X_scaled = scaler.fit_transform(X_raw)
    means, stds = scaler.get_params()

    ridge = RidgeCyclePredictor(alpha=0.01, lam=0.05)
    result = ridge.train(X_scaled, y_raw, epochs=200, verbose=True)

    print(f"\n  ───── Hasil Pelatihan ─────")
    print(f"  Final Cost : {result['final_cost']:.6f}")
    print(f"  MAE        : {result['mae']:.2f} hari")
    print(f"  Data Points: {result['data_points']}")
    print(f"  Weights    : {[round(w, 4) for w in result['weights']]}")

    # Simpan ke BLOB SQLite
    blob = BlobSerializer.weights_to_blob(ridge.get_weights())
    mean_blob = BlobSerializer.array_to_blob(means)
    std_blob = BlobSerializer.array_to_blob(stds)
    db.save_model_blob("ridge_cycle", blob, mean_blob, std_blob,
                       len(X_raw), result['mae'])
    print(f"  ✅ Model disimpan ke SQLite ({len(blob)} bytes BLOB)")

    # Prediksi siklus berikutnya
    last_features = X_raw[-1]
    last_scaled = scaler.transform([last_features])[0]
    fertility = ridge.estimate_fertility(last_scaled)
    print(f"\n  ───── Prediksi Siklus Berikutnya ─────")
    print(f"  Durasi prediksi    : {fertility['predicted_duration_days']} hari")
    print(f"  Hari ovulasi ke-   : {fertility['ovulation_day_from_start']}")
    print(f"  Fertile Window     : hari ke {fertility['fertile_window_days']}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Prediksi Peak Time (Logistic)
    # ──────────────────────────────────────────────────────────────────────
    print_header("STEP 3: Model 2a — Logistic Peak Time (Promil)")

    logistic = LogisticPeakTime(alpha=0.005)

    # Buat data training sintetis untuk logistic
    X_log = []
    y_log = []
    for offset in range(-12, 13):
        stress = 30 + abs(offset) * 4.5
        sleep = 8.0 - abs(offset) * 0.12
        X_log.append([offset ** 2, stress, sleep])
        y_log.append(1.0 if abs(offset) <= 2 else 0.0)

    log_result = logistic.train(X_log, y_log, epochs=150, verbose=True)
    print(f"\n  Final BCE Loss: {log_result['final_loss']:.6f}")

    # Prediksi window lengkap
    daily_stress = [35 + i * 1.5 for i in range(28)]
    daily_sleep = [7.8 - i * 0.04 for i in range(28)]
    window = logistic.predict_fertility_window(28.0, daily_stress, daily_sleep)

    print(f"\n  ───── Fertility Window (28 hari) ─────")
    print(f"  {'Hari':>4s} | {'Offset':>6s} | {'P(Fertil)':>9s} | Label")
    print(f"  {'─'*4:>4s}─┼─{'─'*6:>6s}─┼─{'─'*9:>9s}─┼─{'─'*20}")
    for d in window:
        if abs(d["offset_from_ovulation"]) <= 6:
            marker = " ◀" if d["is_peak"] else ""
            print(f"  {d['day']:>4d} | {d['offset_from_ovulation']:>+6d} | "
                  f"{d['probability']:>9.4f} | {d['label']}{marker}")

    peak_days = [d for d in window if d["is_peak"]]
    print(f"\n  Total Peak Days: {len(peak_days)}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: GWG Spline Tracker (Kehamilan)
    # ──────────────────────────────────────────────────────────────────────
    print_header("STEP 4: Model 2b — GWG Spline Tracker (Bumil)")

    gwg = GWGSplineTracker(bmi_pra_hamil=22.0, berat_pra_hamil=56.5)
    print(f"  Kategori BMI : {gwg.bmi_category}")
    target_range = gwg.get_gwg_target_range(gwg.bmi_category)
    print(f"  Target GWG   : {target_range[0]} - {target_range[1]} kg\n")

    print(f"  {'Minggu':>6s} | {'T':>1s} | {'Ideal (kg)':>10s} | "
          f"{'Gain (kg)':>9s} | {'Progress':>8s}")
    print(f"  {'─'*6}─┼─{'─'*1}─┼─{'─'*10}─┼─{'─'*9}─┼─{'─'*8}")
    for week in [0, 4, 8, 13, 16, 20, 24, 28, 32, 36, 40]:
        r = gwg.predict_ideal_weight(week)
        print(f"  {week:>6d} | {r['trimester']:>1d} | "
              f"{r['ideal_weight_kg']:>10.1f} | "
              f"+{r['gain_from_start_kg']:>8.1f} | "
              f"{r['progress_pct']:>7.1f}%")

    # Evaluasi aktual
    print(f"\n  ───── Evaluasi Berat Aktual ─────")
    eval_r = gwg.evaluate(week=24, actual_weight=62.0)
    print(f"  Minggu 24: Aktual={eval_r['actual_weight_kg']} kg, "
          f"Ideal={eval_r['ideal_weight_kg']} kg")
    print(f"  Deviasi: {eval_r['deviation_kg']:+.1f} kg → {eval_r['status']}")
    print(f"  {eval_r['recommendation']}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5: Explainable Risk Analyzer (Tenaga Kesehatan)
    # ──────────────────────────────────────────────────────────────────────
    print_header("STEP 5: Model 3 — Explainable Risk Analyzer (Nakes)")

    analyzer = ExplainableRiskAnalyzer()

    # Kasus: pasien stres tinggi + kurang tidur
    test_features_raw = [32.0, 5.2, 78.0, 1.2, 2.8]
    test_features_scaled = scaler.transform([test_features_raw])[0]
    pred_dur = ridge.predict(test_features_scaled)

    analysis = analyzer.analyze(
        predicted_duration=pred_dur,
        weights=ridge.get_weights(),
        feature_values=test_features_raw,
        means=means,
        stds=stds,
        patient_id="Pasien-001",
    )

    print(f"  Prediksi Durasi : {analysis['predicted_duration_days']} hari")
    print(f"  Risk Level      : {analysis['risk_level']}")
    print(f"\n{analysis['clinical_summary']}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6: Federated Averaging (Developer)
    # ──────────────────────────────────────────────────────────────────────
    print_header("STEP 6: Model 4 — Federated Aggregator (Developer)")

    base_weights = ridge.get_weights()
    aggregator = FederatedAggregator(
        global_weights=base_weights, num_features=5
    )

    random.seed(2026)
    print(f"  Base Model: {[round(w, 4) for w in base_weights]}\n")

    for k in range(8):
        local_w = [
            base_weights[j] + random.gauss(0, 0.25)
            for j in range(len(base_weights))
        ]
        n_data = random.randint(30, 200)
        delta = FederatedAggregator.compute_client_delta(
            local_w, base_weights, n_data
        )
        aggregator.receive_client_update(delta)
        print(f"  Klien {k+1}: nₖ={n_data:>3d}, "
              f"|ΔW|_max={max(abs(d) for d in delta['delta_weights']):.3f}")

    fed_result = aggregator.aggregate(sanitize=True)
    print(f"\n  Status           : {fed_result['status']}")
    print(f"  Klien digunakan  : {fed_result['num_clients_used']}")
    print(f"  Klien dibuang    : {fed_result['num_clients_removed']}")
    print(f"  Total data points: {fed_result['total_data_points']}")
    print(f"  Global baru      : "
          f"{[round(w, 4) for w in fed_result['new_global_weights']]}")

    export = aggregator.export_base_model()
    print(f"\n  Base Model untuk rilis v2:")
    print(f"  {[round(w, 4) for w in export['weights']]}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 7: Verifikasi BLOB Serialisasi
    # ──────────────────────────────────────────────────────────────────────
    print_header("STEP 7: Verifikasi BLOB Serialisasi SQLite")

    saved = db.load_model_blob("ridge_cycle")
    if saved:
        recovered_weights, version = BlobSerializer.blob_to_weights(
            saved["bobot_regresi"]
        )
        recovered_means = BlobSerializer.blob_to_array(saved["mean_features"])
        recovered_stds = BlobSerializer.blob_to_array(saved["std_features"])

        print(f"  Model dimuat dari SQLite:")
        print(f"    Tipe     : {saved['tipe_model']}")
        print(f"    Versi    : {version}")
        print(f"    MAE      : {saved['mae_terakhir']:.2f} hari")
        print(f"    Weights  : {[round(w, 4) for w in recovered_weights]}")
        print(f"    Means    : {[round(m, 3) for m in recovered_means]}")
        print(f"    Stds     : {[round(s, 3) for s in recovered_stds]}")
        print(f"    BLOB size: {len(saved['bobot_regresi'])} bytes")
        print("  ✅ Model berhasil di-load dari BLOB SQLite")
    else:
        print("  ⚠️ Model belum tersimpan di database")

    # ──────────────────────────────────────────────────────────────────────
    # RINGKASAN
    # ──────────────────────────────────────────────────────────────────────
    print_header("RINGKASAN PIPELINE")

    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    SIKLUS ML Pipeline OK                        │
  ├─────────────────────────────────────────────────────────────────┤
  │  ✅ Data sintetis    : 12 siklus + data harian → CSV + SQLite  │
  │  ✅ Ridge Predictor  : Durasi siklus + Fertile Window          │
  │  ✅ Logistic Peak    : Probabilitas kesuburan harian           │
  │  ✅ GWG Tracker      : Kurva berat badan kehamilan IOM         │
  │  ✅ XAI Analyzer     : Atribusi faktor risiko transparan       │
  │  ✅ FedAvg Aggregator: Agregasi bobot anonim multi-klien       │
  │  ✅ BLOB Serializer  : IEEE 754 float32 ↔ SQLite BLOB         │
  └─────────────────────────────────────────────────────────────────┘
    """)

    db.close()
    print("  Selesai. Seluruh model berjalan 100% offline (Edge AI).\n")


if __name__ == "__main__":
    main()

