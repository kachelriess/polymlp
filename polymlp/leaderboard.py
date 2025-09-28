from pathlib import Path

from .build import build_leaderboard
from .markdown import insert_into_readme, print_entries
from .read import read_cfg


def leaderboard(config: str) -> None:
    """
    Runs the leaderboard pipeline and updates the README.

    YAML Config Requirements:
        - root (str): Root path. Expects the following to exist:
            - `implementations/` containing `language/results/` subdirectories.
            - `data/test.csv` containing column `y`.
            - `README.md` containing markers (optional).
            - `JOURNAL.md` containing `language` headers (optional).
        - ignore (List[str]): List of directories in `implementations/` to ignore.
        - batch_size (int): Batch size used during training.
        - epochs (int): Number of epochs used during training.
        - samples (List[int]): Defines train and test set sizes.
        - insert_md (bool): Whether to insert results into `README.md`.
        - badge_marker (str): Marker for badges in `README.md`.
        - leaderboard_marker (str): Marker for leaderboard in `README.md`.
        - badges (List[str, ...]): Badge strings to insert. Use `{}` to format language count.
        - thresholds (Dict[str, float]): Thresholds for filtering (< `mae`, < `mse`, > `r2`).

    Args:
        config (str): Path to the YAML config file.
    """

    cfg = read_cfg(Path(config))
    passed, failed = build_leaderboard(cfg)

    if cfg["insert_md"]:
        if not passed.empty:
            insert_into_readme(cfg, passed)
        if not failed.empty:
            if passed.empty:
                print()
            print_entries(failed, False)
    else:
        if not passed.empty or not failed.empty:
            print()
        if not passed.empty:
            print_entries(passed, True)
        if not failed.empty:
            print_entries(failed, False)

    print(f"Summary: {len(passed)} passed, {len(failed)} failed.\n")
