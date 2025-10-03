from pathlib import Path
import random

from layers import MLP
from train import train_and_eval


def main():
    random.seed(42)

    model = MLP(topology=[6, 16, 16, 1])
    print(f"\n{model}\n")

    train_and_eval(
        model=model,
        lr=5e-3,
        epochs=100,
        batch_size=32,
        train_path=Path("../../data/train.csv"),
        test_path=Path("../../data/test.csv"),
    )


if __name__ == "__main__":
    main()
