from pathlib import Path
import re
from typing import Any, Callable, Dict, List
import yaml

import numpy as np
import pandas as pd


def is_dir(x: str):
    return Path(x).is_dir()


def is_list_of_str(x: List[Any]) -> bool:
    return all(isinstance(i, str) for i in x)


def is_pos_int(x: int):
    return x > 0


def is_list_of_pos_int(x: List[Any]) -> bool:
    return all(isinstance(i, int) and is_pos_int(i) for i in x)


def is_list_of_two_pos_int(x: List[Any]) -> bool:
    return len(x) == 2 and is_list_of_pos_int(x)


def is_list_of_str_and_has_at_most_one_formatting_placeholder(
    x: List[Any],
) -> bool:
    return is_list_of_str(x) and len(re.findall(r"{[^}]*}", "".join(x))) <= 1


def is_dict_of_float_or_int(x: Dict[str, Any]) -> bool:
    return all(isinstance(v, (float, int)) for v in x.values())


def keys_in_mae_mse_r2(x: Dict[str, Any]) -> bool:
    return all(k in ["mae", "mse", "r2"] for k in x.keys())


def is_dict_of_float_or_int_and_keys_in_mae_mse_r2(x: Dict[str, Any]) -> bool:
    return is_dict_of_float_or_int(x) and keys_in_mae_mse_r2(x)


CFG_SCHEMA: dict[str, tuple[type, Callable[[Any], bool] | None]] = {
    "root": (str, is_dir),
    "ignore": (list, is_list_of_str),
    "batch_size": (int, is_pos_int),
    "epochs": (int, (is_pos_int)),
    "samples": (list, is_list_of_two_pos_int),
    "insert_md": (bool, None),
    "badge_marker": (str, None),
    "leaderboard_marker": (str, None),
    "badges": (list, is_list_of_str_and_has_at_most_one_formatting_placeholder),
    "thresholds": (dict, is_dict_of_float_or_int_and_keys_in_mae_mse_r2),
}


def validate_path_is_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"File '{path}' does not exist.")


def validate_cfg(cfg: dict[str, Any]) -> None:
    if cfg is None:
        raise ValueError("Config is empty or invalid.")
    for key, (expected_type, validator) in CFG_SCHEMA.items():
        if key not in cfg:
            raise ValueError(f"Missing required config entry: '{key}'.")
        if not isinstance(cfg[key], expected_type):
            raise TypeError(
                f"Config entry '{key}' must be of type {expected_type.__name__}."
            )
        if validator and not validator(cfg[key]):
            raise ValueError(
                f"Config entry '{key}' failed validation: {validator.__name__.replace('_', ' ')}."
            )


def validate_df(
    path: Path,
    df: pd.DataFrame,
    columns: List[str],
    shape_0: int,
) -> None:
    path_str = f"{path.parent.name}/{path.name}"
    if not all(col in df.columns for col in columns):
        raise ValueError(f"Expected columns {columns} in '{path_str}'.")
    df = df[columns]
    if df.shape[0] != shape_0:
        raise ValueError(f"Expected {shape_0} rows in '{path_str}'.")
    if not all(dtype.kind == "f" for dtype in df.dtypes):
        raise ValueError(
            f"Expected all wanted columns to have float dtype in '{path_str}'."
        )
    if not np.isfinite(df.values).all():
        raise ValueError(f"Found NaN or infinite values in '{path_str}'.")


def read_cfg(path: Path) -> dict[str, Any]:
    validate_path_is_file(path)
    cfg = yaml.safe_load(path.read_text())
    validate_cfg(cfg)
    return cfg


def read_csv(path: Path, columns: List[str], shape_0: int) -> pd.DataFrame:
    validate_path_is_file(path)
    df = pd.read_csv(path)
    validate_df(path, df, columns, shape_0)
    return df
