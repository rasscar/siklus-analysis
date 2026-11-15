"""
=============================================================================
SIKLUS ML - Z-Score Scaler (Normalisasi Lokal On-Device)
=============================================================================
Penskalaan fitur menggunakan Z-Score: X_scaled = (X - μ) / σ
μ dan σ dihitung dari data lokal pengguna dan disimpan ke BLOB SQLite.
"""

import math
from typing import List, Tuple


class ZScoreScaler:
    """Implementasi Z-Score Normalization tanpa dependensi NumPy."""

    def __init__(self):
        self.means: List[float] = []
        self.stds: List[float] = []
        self.is_fitted: bool = False

    def fit(self, data: List[List[float]]):
        """
        Hitung rata-rata (μ) dan standar deviasi (σ) dari data pelatihan.

        Args:
            data: Matriks fitur [[x1, x2, ...], [x1, x2, ...], ...]
        """
        if not data:
            raise ValueError("Data kosong, tidak dapat melakukan fit.")

        num_features = len(data[0])
        n = len(data)
        self.means = [0.0] * num_features
        self.stds = [0.0] * num_features

        # Hitung mean
        for row in data:
            for j in range(num_features):
                self.means[j] += row[j]
        self.means = [m / n for m in self.means]

        # Hitung std
        for row in data:
            for j in range(num_features):
                self.stds[j] += (row[j] - self.means[j]) ** 2
        self.stds = [math.sqrt(s / max(n - 1, 1)) for s in self.stds]

        # Cegah pembagian nol
        self.stds = [s if s > 1e-8 else 1.0 for s in self.stds]
        self.is_fitted = True

    def transform(self, data: List[List[float]]) -> List[List[float]]:
        """
        Transformasi data menggunakan μ dan σ yang telah dihitung.

        Args:
            data: Matriks fitur untuk dinormalisasi.

        Returns:
            Matriks fitur ternormalisasi.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler belum di-fit. Panggil fit() dulu.")

        scaled = []
        for row in data:
            scaled_row = [
                (row[j] - self.means[j]) / self.stds[j]
                for j in range(len(row))
            ]
            scaled.append(scaled_row)
        return scaled

    def fit_transform(self, data: List[List[float]]) -> List[List[float]]:
        """Fit dan transform dalam satu langkah."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform_single(self, scaled_val: float,
                                  feature_idx: int) -> float:
        """Kembalikan nilai ternormalisasi ke skala asli."""
        return scaled_val * self.stds[feature_idx] + self.means[feature_idx]

    def get_params(self) -> Tuple[List[float], List[float]]:
        """Dapatkan parameter μ dan σ untuk disimpan ke BLOB."""
        return self.means.copy(), self.stds.copy()

    def set_params(self, means: List[float], stds: List[float]):
        """Muat parameter μ dan σ dari BLOB."""
        self.means = means
        self.stds = stds
        self.is_fitted = True

