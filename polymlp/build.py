import math
from pathlib import Path
import re
from typing import Any, Dict, Generator, Tuple

import pandas as pd

from .read import read_csv


def yield_results_dirs(cfg: Dict[str, Any]) -> Generator[Path, None, None]:
    ignore = set(cfg["ignore"])
    for lang in (Path(cfg["root"]) / "implementations").iterdir():
        if not lang.is_dir():
            continue
        lang_path = lang / "results"
        if not lang_path.is_dir() or lang.name in ignore:
            continue
        yield lang_path


def parse_lang(path: Path) -> str:
    lang = re.sub(r"([*_`[\]])", r"\\\1", path.parent.name).lower().capitalize()
    lang = f"[{lang}](JOURNAL.md#{lang})"

    return lang


def extract_batch_timing(
    cfg: Dict[str, Any],
    path: Path,
) -> Tuple[float, float, float, float]:
    df = read_csv(
        path=path / "batch_times.csv",
        columns=["forward", "backward"],
        shape_0=math.ceil(cfg["samples"][0] / cfg["batch_size"])
        * cfg["epochs"],
    )

    f_mean = df["forward"].mean()
    f_std = df["forward"].std(ddof=1)
    b_mean = df["backward"].mean()
    b_std = df["backward"].std(ddof=1)

    return f_mean, f_std, b_mean, b_std


def extract_epoch_timing(
    cfg: Dict[str, Any],
    path: Path,
) -> Tuple[float, float]:
    df = read_csv(
        path=path / "epoch_times.csv",
        columns=["epoch"],
        shape_0=cfg["epochs"],
    )

    e_mean = df["epoch"].mean()
    e_std = df["epoch"].std(ddof=1)

    return e_mean, e_std


def extract_test_metrics(
    cfg: Dict[str, Any],
    path: Path,
    y: pd.Series,
) -> Tuple[float, float, float]:
    y_ = read_csv(
        path=path / "test_predictions.csv",
        columns=["prediction"],
        shape_0=cfg["samples"][1],
    )["prediction"]

    sse = ((y - y_) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()

    mae = (y - y_).abs().mean()
    mse = sse / cfg["samples"][1]
    r2 = 1 - sse / sst

    return mae, mse, r2


def build_leaderboard(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = read_csv(
        path=Path(cfg["root"]) / "data" / "test.csv",
        columns=["y"],
        shape_0=cfg["samples"][1],
    )["y"]

    rows = []
    for path in yield_results_dirs(cfg):

        try:
            f_mean, f_std, b_mean, b_std = extract_batch_timing(cfg, path)
            e_mean, e_std = extract_epoch_timing(cfg, path)
            mae, mse, r2 = extract_test_metrics(cfg, path, y)

            for t, col in zip(
                [f_mean, b_mean, e_mean], ["forward", "backward", "epoch"]
            ):
                if t <= 0:
                    raise ValueError(
                        f"Mean {col} time is invalid: {t:.8f} seconds."
                    )

            rows.append(
                {
                    "score": (f_mean + b_mean) / 2,
                    "lang": parse_lang(path),
                    "f_mean": f_mean,
                    "f_std": f_std,
                    "b_mean": b_mean,
                    "b_std": b_std,
                    "e_mean": e_mean,
                    "e_std": e_std,
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                }
            )
        except Exception as e:
            print(f"Skipping implementation '{path.parent.name}': {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid leaderboard entries found.")

    df = df.sort_values(by="score", ascending=True)

    mask = pd.Series(True, index=df.index)
    for key in cfg["thresholds"]:
        if key == "r2":
            mask &= df[key] > cfg["thresholds"][key]
        else:
            mask &= df[key] < cfg["thresholds"][key]

    passed = df[mask].copy().reset_index(drop=True)
    failed = df[~mask].copy().reset_index(drop=True)

    return passed, failed
