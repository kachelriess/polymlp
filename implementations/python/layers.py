from __future__ import annotations
import math
import random
from typing import Any, List

from matrix import Matrix


class Module:

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def step(self, *args: Any, **kwargs: Any) -> Any:
        pass


class Linear(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        assert 0 < in_features and 0 < out_features

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Matrix(in_features, out_features)
        self.bias = Matrix(1, out_features)
        self.reset_parameters()

        self.input: Matrix | None = None
        self.weight_grad: Matrix | None = None
        self.bias_grad: Matrix | None = None

    def reset_parameters(self) -> None:
        bound = math.sqrt(1 / self.in_features)
        self.weight = self.weight.fill(lambda: random.uniform(-bound, bound))
        self.bias = self.bias.fill(0)

    def forward(self, input: Matrix) -> Matrix:
        self.input = input
        output = input @ self.weight + self.bias
        return output

    def backward(self, output_grad: Matrix) -> Matrix:
        assert self.input is not None

        self.weight_grad = self.input.T @ output_grad
        self.bias_grad = output_grad.sum(dim=0)
        input_grad = output_grad @ self.weight.T

        self.input = None

        return input_grad

    def step(self, lr: float) -> None:
        assert self.weight_grad is not None
        assert self.bias_grad is not None

        self.weight = self.weight - lr * self.weight_grad
        self.bias = self.bias - lr * self.bias_grad

        self.weight_grad = None
        self.bias_grad = None

    def __repr__(self) -> str:
        return f"Linear({self.in_features}, {self.out_features})"


class ReLU(Module):

    def __init__(self) -> None:
        self.mask: Matrix | None = None

    def forward(self, input: Matrix) -> Matrix:
        self.mask = input > 0
        output = input * self.mask
        return output

    def backward(self, output_grad: Matrix) -> Matrix:
        assert self.mask is not None

        input_grad = output_grad * self.mask

        self.mask = None

        return input_grad

    def __repr__(self) -> str:
        return "ReLU()"


class MSE(Module):

    def __init__(self) -> None:
        self.residual: Matrix | None = None

    def forward(self, prediction: Matrix, response: Matrix) -> Matrix:
        assert prediction.shape == response.shape
        assert prediction.shape[1] == 1

        self.residual = prediction - response

        output = (self.residual**2).mean()

        return output

    def backward(self) -> Matrix:
        assert self.residual is not None

        input_grad = (2 / self.residual.shape[0]) * self.residual

        self.residual = None

        return input_grad

    def __repr__(self) -> str:
        return "MSE()"


class MLP(Module):

    def __init__(self, topology: List[int]) -> None:
        self.layers = []
        self.criterion = MSE()

        in_features = topology[0]
        for out_features in topology[1:-1]:
            self.layers.append(Linear(in_features, out_features))
            self.layers.append(ReLU())
            in_features = out_features
        self.layers.append(Linear(in_features, topology[-1]))

    def forward(self, input: Matrix) -> Matrix:
        for layer in self.layers:
            input = layer(input)
        return input

    def backward(self) -> None:
        assert self.criterion.residual is not None

        input_grad = self.criterion.backward()
        for layer in self.layers[::-1]:
            input_grad = layer.backward(input_grad)

    def step(self, lr: float) -> None:
        for layer in self.layers:
            layer.step(lr)

    def __repr__(self) -> str:
        layers_repr = [
            f"({i + 1}): {layer.__repr__()}"
            for i, layer in enumerate(self.layers + [self.criterion])
        ]
        layers_repr = ",\n  ".join(layers_repr)

        return f"MLP(\n  {layers_repr}\n)"
