from datetime import datetime
from pathlib import Path
from tabulate import tabulate
from typing import Any, Dict

import pandas as pd

from .format import format_leaderboard


def df_to_table(df: pd.DataFrame) -> str:
    table = tabulate(
        df,  # pyright: ignore[reportArgumentType]
        headers="keys",
        tablefmt="github",
        showindex=False,
        colalign=["left"] * len(df.columns),
        floatfmt=".4f",
    )

    return table


def print_entries(df: pd.DataFrame, passed: bool) -> None:
    print(f"{'Passed' if passed else 'Failed'} entries:\n")
    print(f"{df_to_table(format_leaderboard(df))}\n")


def insert_into_readme(cfg: Dict[str, Any], df: pd.DataFrame) -> None:
    readme_path = Path(cfg["root"]) / "README.md"
    backup_path = readme_path.with_name("README.backup.md")

    if not readme_path.exists():
        raise FileNotFoundError(f"README.md not found.")

    content = readme_path.read_text().replace("\r\n", "\n")
    backup_path.write_text(content)

    if cfg["badge_marker"] not in content:
        print("Skipping badge insertion: No markers found.")
    elif content.count(cfg["badge_marker"]) != 2:
        print("Skipping badge insertion: Expected exactly two markers.")
    else:
        content = (
            content.split(cfg["badge_marker"])[0]
            + cfg["badge_marker"]
            + "\n\n"
            + "\n".join(cfg["badges"]).format(len(df))
            + "\n\n"
            + cfg["badge_marker"]
            + content.split(cfg["badge_marker"])[-1]
        )

    if cfg["leaderboard_marker"] not in content:
        print("Skipping leaderboard insertion: No markers found.\n")
        print_entries(df, True)
    elif content.count(cfg["leaderboard_marker"]) != 2:
        print("Skipping leaderboard insertion: Expected exactly two markers.\n")
        print_entries(df, True)
    else:
        requirements = []
        for key in cfg["thresholds"]:
            prefix = "`R²` >" if key == "r2" else f"`{key.upper()}` <"
            requirements.append(f"{prefix} {float(cfg['thresholds'][key]):.2f}")

        content = (
            content.split(cfg["leaderboard_marker"])[0]
            + cfg["leaderboard_marker"]
            + "\n\n"
            + f"<!-- LAST UPDATED {datetime.now()} -->"
            + "\n\n"
            + f"Entry requirements: {', '.join(requirements)}"
            + "\n\n"
            + "Ranked by arithmetic mean of `Forward` and `Backward`"
            + "\n\n"
            + df_to_table(format_leaderboard(df))
            + "\n\n"
            + cfg["leaderboard_marker"]
            + content.split(cfg["leaderboard_marker"])[-1]
        )
        print(
            f"\n{len(df)} leaderboard {'entry' if len(df) == 1 else 'entries'} inserted into README.\n"
        )

    readme_path.write_text(content)
