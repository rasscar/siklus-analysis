"""
=============================================================================
MODEL 4: Federated Aggregator (Developer - FedAvg)
=============================================================================
Implementasi Federated Averaging untuk agregasi bobot model anonim
dari banyak perangkat pengguna. Developer tidak pernah menerima data
mentah, hanya vektor delta parameter model (ΔW).

Formulasi FedAvg:
  W_global^(t+1) = W_global^(t) + Σ (nₖ / N_total) × ΔWₖ

Sanitasi:
  Filter delta outlier menggunakan Z-Score > 3σ.
"""

import math
from typing import List, Dict, Tuple, Optional

from config import FEDAVG_MIN_CLIENTS, DELTA_OUTLIER_ZSCORE


class FederatedAggregator:
    """
    Implementasi server-side Federated Averaging (FedAvg).

    Alur:
    1. Developer mendistribusikan Base Model (W_global) ke aplikasi.
    2. Setiap perangkat melatih model lokal dan menghitung ΔW = W_lokal - W_global.
    3. Perangkat mengirim ΔW teranonimkan + jumlah data lokal (n_k).
    4. Developer menjalankan FedAvg untuk menghasilkan W_global baru.
    5. W_global baru didistribusikan pada rilis aplikasi berikutnya.
    """

    def __init__(self, global_weights: Optional[List[float]] = None,
                 num_features: int = 5):
        """
        Args:
            global_weights: Bobot global saat ini [bias, w1, ..., wN].
            num_features: Jumlah fitur model.
        """
        self.num_params = num_features + 1  # +1 untuk bias
        self.global_weights = (
            global_weights if global_weights
            else [0.0] * self.num_params
        )
        self.client_deltas: List[Dict] = []
        self.aggregation_history: List[Dict] = []

    # ─── Client-Side: Hitung Delta ───────────────────────────────────────

    @staticmethod
    def compute_client_delta(
        local_weights: List[float],
        global_weights: List[float],
        num_local_data: int,
    ) -> Dict:
        """
        Hitung delta bobot pada sisi klien (perangkat pengguna).
        Fungsi ini dijalankan di dalam aplikasi mobile.

        ΔWₖ = W_lokal - W_global

        Args:
            local_weights: Bobot model lokal setelah pelatihan.
            global_weights: Bobot global dari rilis terakhir.
            num_local_data: Jumlah titik data lokal (nₖ).

        Returns:
            Dict: {delta_weights, num_data} siap dikirim anonim.
        """
        delta = [
            local_weights[j] - global_weights[j]
            for j in range(len(local_weights))
        ]
        return {
            "delta_weights": delta,
            "num_data": num_local_data,
            # Tidak ada ID pengguna, tanggal, atau data pribadi
        }

    # ─── Server-Side: Terima Update Klien ────────────────────────────────

    def receive_client_update(self, client_delta: Dict):
        """
        Terima pembaruan delta dari satu klien.

        Args:
            client_delta: Dict dari compute_client_delta().
        """
        if len(client_delta["delta_weights"]) != self.num_params:
            raise ValueError(
                f"Delta harus memiliki {self.num_params} parameter, "
                f"diterima {len(client_delta['delta_weights'])}."
            )
        self.client_deltas.append(client_delta)

    # ─── Server-Side: Sanitasi Outlier ───────────────────────────────────

    def sanitize_deltas(self) -> Tuple[List[Dict], List[int]]:
        """
        Filter delta outlier menggunakan Z-Score per parameter.
        Delta dengan z > 3σ pada parameter manapun akan dibuang.

        Returns:
            Tuple: (clean_deltas, removed_indices)
        """
        if len(self.client_deltas) < 3:
            return self.client_deltas, []

        num_clients = len(self.client_deltas)
        removed_indices = []

        # Hitung mean dan std per parameter
        means = [0.0] * self.num_params
        for cd in self.client_deltas:
            for j in range(self.num_params):
                means[j] += cd["delta_weights"][j]
        means = [m / num_clients for m in means]

        stds = [0.0] * self.num_params
        for cd in self.client_deltas:
            for j in range(self.num_params):
                stds[j] += (cd["delta_weights"][j] - means[j]) ** 2
        stds = [math.sqrt(s / max(num_clients - 1, 1)) for s in stds]
        stds = [s if s > 1e-8 else 1.0 for s in stds]

        # Filter outlier
        clean = []
        for idx, cd in enumerate(self.client_deltas):
            is_outlier = False
            for j in range(self.num_params):
                z = abs(cd["delta_weights"][j] - means[j]) / stds[j]
                if z > DELTA_OUTLIER_ZSCORE:
                    is_outlier = True
                    break
            if is_outlier:
                removed_indices.append(idx)
            else:
                clean.append(cd)

        return clean, removed_indices

    # ─── Server-Side: Federated Averaging ────────────────────────────────

    def aggregate(self, sanitize: bool = True) -> Dict:
        """
        Jalankan algoritma Federated Averaging (FedAvg).

        W_global^(t+1) = W_global^(t) + Σ (nₖ / N) × ΔWₖ

        Args:
            sanitize: Apakah filter outlier sebelum agregasi.

        Returns:
            Dict: {new_global_weights, num_clients_used,
                   num_clients_removed, total_data_points}
        """
        if len(self.client_deltas) < FEDAVG_MIN_CLIENTS:
            return {
                "status": "insufficient_clients",
                "message": (
                    f"Minimal {FEDAVG_MIN_CLIENTS} klien dibutuhkan, "
                    f"baru {len(self.client_deltas)} terkumpul."
                ),
                "num_clients_received": len(self.client_deltas),
            }

        # Sanitasi
        if sanitize:
            clean_deltas, removed = self.sanitize_deltas()
        else:
            clean_deltas = self.client_deltas
            removed = []

        if not clean_deltas:
            return {"status": "all_filtered", "message": "Semua delta outlier."}

        # Hitung total data points
        total_n = sum(cd["num_data"] for cd in clean_deltas)

        # FedAvg: weighted average delta
        weighted_delta = [0.0] * self.num_params
        for cd in clean_deltas:
            weight = cd["num_data"] / total_n  # nₖ / N
            for j in range(self.num_params):
                weighted_delta[j] += weight * cd["delta_weights"][j]

        # Update global weights
        new_global = [
            self.global_weights[j] + weighted_delta[j]
            for j in range(self.num_params)
        ]

        result = {
            "status": "success",
            "previous_global_weights": self.global_weights.copy(),
            "new_global_weights": new_global,
            "weighted_delta": weighted_delta,
            "num_clients_used": len(clean_deltas),
            "num_clients_removed": len(removed),
            "total_data_points": total_n,
        }

        # Simpan ke history dan update global
        self.aggregation_history.append(result)
        self.global_weights = new_global
        self.client_deltas = []  # reset buffer

        return result

    # ─── Export untuk Rilis ───────────────────────────────────────────────

    def export_base_model(self) -> Dict:
        """
        Export bobot global untuk didistribusikan sebagai Base Model
        pada rilis aplikasi berikutnya.
        """
        return {
            "model_type": "ridge_cycle",
            "weights": self.global_weights.copy(),
            "num_params": self.num_params,
            "aggregation_rounds": len(self.aggregation_history),
            "total_clients_contributed": sum(
                h.get("num_clients_used", 0)
                for h in self.aggregation_history
            ),
        }

    # ─── Helper ──────────────────────────────────────────────────────────

    def get_global_weights(self) -> List[float]:
        return self.global_weights.copy()

    def set_global_weights(self, weights: List[float]):
        self.global_weights = weights.copy()

    def summary(self) -> str:
        lines = [
            "═══ Federated Aggregator ═══",
            f"  Global Weights    : {[round(w, 6) for w in self.global_weights]}",
            f"  Pending Clients   : {len(self.client_deltas)}",
            f"  Aggregation Rounds: {len(self.aggregation_history)}",
        ]
        if self.aggregation_history:
            last = self.aggregation_history[-1]
            lines.append(f"  Last Round Clients: {last.get('num_clients_used', 0)}")
            lines.append(f"  Last Total Data   : {last.get('total_data_points', 0)}")
        return "\n".join(lines)

