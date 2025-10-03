from typing import Any, Dict

import pandas as pd


def format_entry(row: pd.Series) -> Dict[str, Any]:
    return {
        "Language": row["lang"],
        "Forward (ms)": f"{row['f_mean'] * 1_000:.6f} ± {row['f_std'] * 1_000:.4f}",
        "Backward (ms)": f"{row['b_mean'] * 1_000:.6f} ± {row['b_std'] * 1_000:.4f}",
        "Epoch (s)": f"{row['e_mean']:.6f} ± {row['e_std']:.4f}",
        "MAE": f"{row['mae']:.4f}",
        "MSE": f"{row['mse']:.4f}",
        "R²": f"{row['r2']:.4f}",
    }


def format_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    data = df.apply(format_entry, axis=1).tolist()
    return pd.DataFrame(data)
