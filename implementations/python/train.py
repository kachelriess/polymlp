import csv
from pathlib import Path
import random
import time
from typing import Dict, Generator, List, Tuple

from layers import MLP
from matrix import Matrix


def load_data(path: Path) -> Tuple[List[List[float]], List[List[float]]]:
    assert path.is_file()

    features, targets = [], []
    with path.open(newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)

        for row in reader:
            row = [float(cell) for cell in row]
            features.append(row[:-1])
            targets.append([row[-1]])

    return features, targets


def batch_loader(
    features: List[List[float]],
    targets: List[List[float]],
    batch_size: int,
) -> Generator[Tuple[Matrix, Matrix], None, None]:
    assert len(features) == len(targets)

    num_samples = len(features)
    indices = list(range(num_samples))
    random.shuffle(indices)

    for i in range(0, num_samples, batch_size):

        feature_batch, target_batch = [], []
        for j in range(i, min(i + batch_size, num_samples)):
            k = indices[j]
            feature_batch.append(features[k])
            target_batch.append(targets[k])

        B, F = len(feature_batch), len(feature_batch[0])
        feature_batch = Matrix(B, F, feature_batch)
        target_batch = Matrix(B, 1, target_batch)

        yield feature_batch, target_batch


def write_results(results: Dict[str, List]) -> None:
    assert all(
        k in ["forward", "backward", "epoch", "prediction"]
        for k in results.keys()
    )

    results_path = Path("results")
    results_path.mkdir(exist_ok=False)

    with (results_path / "batch_times.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["forward", "backward"])
        for data in zip(results["forward"], results["backward"]):
            writer.writerow(data)

    with (results_path / "epoch_times.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"])
        for data in results["epoch"]:
            writer.writerow([data])

    with (results_path / "test_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prediction"])
        for data in results["prediction"]:
            writer.writerow([data])


def train_and_eval(
    model: MLP,
    lr: float,
    epochs: int,
    batch_size: int,
    train_path: Path,
    test_path: Path,
) -> None:
    assert all(0 < arg for arg in [lr, epochs, batch_size])

    train_data = load_data(train_path)
    benchmark = {
        "forward": [],
        "backward": [],
        "epoch": [],
        "prediction": None,
    }

    for epoch in range(epochs):

        running_loss, batch_count = 0, 0
        start_epoch = time.perf_counter()

        for features, targets in batch_loader(*train_data, batch_size):

            start_forward = time.perf_counter()
            prediction = model(features)
            benchmark["forward"].append(time.perf_counter() - start_forward)

            running_loss += model.criterion(prediction, targets).item()
            batch_count += 1

            start_backward = time.perf_counter()
            model.backward()
            benchmark["backward"].append(time.perf_counter() - start_backward)

            model.step(lr)

        benchmark["epoch"].append(time.perf_counter() - start_epoch)

        print(f"epoch {epoch + 1:03d}: {running_loss / batch_count:.6f}")

    test_features = load_data(test_path)[0]
    B, F = len(test_features), len(test_features[0])
    test_features = Matrix(B, F, test_features)
    benchmark["prediction"] = model(test_features).T.data[0]

    write_results(benchmark)
