"""
helpers.py
Fungsi bantu: logging pelanggaran ke dalam session, ekspor CSV,
dan perhitungan statistik untuk dashboard.
"""

import pandas as pd
import datetime
import io


def new_log_entry(source_name: str, summary: dict) -> dict:
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sumber": source_name,
        "jumlah_orang": summary["persons"],
        "item_patuh": summary["compliant_items"],
        "item_melanggar": summary["violation_items"],
        "jenis_pelanggaran": ", ".join(summary["violation_labels"]) if summary["violation_labels"] else "-",
        "status": "MELANGGAR" if summary["violation_items"] > 0 else "PATUH",
    }


def log_to_dataframe(log_list: list) -> pd.DataFrame:
    if not log_list:
        return pd.DataFrame(
            columns=[
                "timestamp", "sumber", "jumlah_orang",
                "item_patuh", "item_melanggar", "jenis_pelanggaran", "status",
            ]
        )
    return pd.DataFrame(log_list)


def compute_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_frame": 0,
            "total_pelanggaran": 0,
            "tingkat_kepatuhan": 100.0,
            "total_orang_terdeteksi": 0,
        }
    total_frame = len(df)
    total_pelanggaran = int((df["status"] == "MELANGGAR").sum())
    tingkat_kepatuhan = round(100 * (1 - total_pelanggaran / total_frame), 2) if total_frame else 100.0
    total_orang = int(df["jumlah_orang"].sum())
    return {
        "total_frame": total_frame,
        "total_pelanggaran": total_pelanggaran,
        "tingkat_kepatuhan": tingkat_kepatuhan,
        "total_orang_terdeteksi": total_orang,
    }


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
