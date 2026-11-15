"""
=============================================================================
SIKLUS ML - BLOB Serializer (IEEE 754 Float32 Little-Endian)
=============================================================================
Mengkonversi vektor bobot model (Python list/numpy) ke format biner BLOB
untuk disimpan di kolom `bobot_regresi` tabel SQLite.

Format Biner (28 Bytes total untuk Ridge Cycle):
  Offset  Ukuran  Tipe       Nama
  0x00    4       float32    beta_0 (Bias)
  0x04    4       float32    beta_1 (Durasi Rata-rata Siklus)
  0x08    4       float32    beta_2 (Durasi Tidur)
  0x0C    4       float32    beta_3 (Skor Stres)
  0x10    4       float32    beta_4 (Delta BMI)
  0x14    4       float32    beta_5 (Frekuensi Makanan Berisiko)
  0x18    4       uint32     checksum / versi model
"""

import struct
import hashlib
from typing import List, Tuple

from config import BLOB_FLOAT_FORMAT, BLOB_UINT_FORMAT, MODEL_VERSION


class BlobSerializer:
    """Serialisasi dan deserialisasi bobot model ke/dari format BLOB biner."""

    @staticmethod
    def weights_to_blob(weights: List[float],
                        version: int = MODEL_VERSION) -> bytes:
        """
        Konversi list bobot float → bytes BLOB.

        Args:
            weights: Daftar koefisien regresi [beta_0, beta_1, ..., beta_n].
            version: Versi model (uint32).

        Returns:
            bytes: Data biner siap disimpan ke kolom BLOB SQLite.
        """
        blob = b""
        for w in weights:
            blob += struct.pack(BLOB_FLOAT_FORMAT, w)
        # Tambahkan versi model sebagai uint32
        blob += struct.pack(BLOB_UINT_FORMAT, version)
        return blob

    @staticmethod
    def blob_to_weights(blob: bytes) -> Tuple[List[float], int]:
        """
        Konversi bytes BLOB → list bobot + versi model.

        Args:
            blob: Data biner dari kolom BLOB SQLite.

        Returns:
            Tuple: (list_bobot, versi_model)
        """
        float_size = struct.calcsize(BLOB_FLOAT_FORMAT)
        uint_size = struct.calcsize(BLOB_UINT_FORMAT)

        # Jumlah float = (total_bytes - uint_size) / float_size
        num_weights = (len(blob) - uint_size) // float_size
        weights = []
        offset = 0
        for _ in range(num_weights):
            val = struct.unpack_from(BLOB_FLOAT_FORMAT, blob, offset)[0]
            weights.append(val)
            offset += float_size

        version = struct.unpack_from(BLOB_UINT_FORMAT, blob, offset)[0]
        return weights, version

    @staticmethod
    def array_to_blob(arr: List[float]) -> bytes:
        """Konversi array float biasa ke BLOB (tanpa checksum)."""
        blob = b""
        for val in arr:
            blob += struct.pack(BLOB_FLOAT_FORMAT, val)
        return blob

    @staticmethod
    def blob_to_array(blob: bytes) -> List[float]:
        """Konversi BLOB ke array float biasa."""
        float_size = struct.calcsize(BLOB_FLOAT_FORMAT)
        num = len(blob) // float_size
        arr = []
        for i in range(num):
            val = struct.unpack_from(BLOB_FLOAT_FORMAT, blob, i * float_size)[0]
            arr.append(val)
        return arr

    @staticmethod
    def compute_checksum(weights: List[float]) -> int:
        """Hitung checksum CRC32 dari vektor bobot."""
        import binascii
        data = struct.pack(f"<{len(weights)}f", *weights)
        return binascii.crc32(data) & 0xFFFFFFFF

