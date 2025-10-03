from __future__ import annotations
from typing import Callable, List, Tuple


class Matrix:

    def __init__(
        self,
        rows: int,
        cols: int,
        data: List[List[float]] | None = None,
    ) -> None:
        assert 0 < rows and 0 < cols

        self.rows = rows
        self.cols = cols

        if data is None:
            self.data = self.fill(0).data
        else:
            assert len(data) == rows
            assert all(len(row) == cols for row in data)
            self.data = data

    def fill(self, value: Callable[[], float | int] | float | int) -> Matrix:
        if isinstance(value, (float, int)):
            fn = lambda: value
        else:
            fn = value

        data = []
        for _ in range(self.rows):
            row = []
            for _ in range(self.cols):
                row.append(float(fn()))
            data.append(row)

        return Matrix(self.rows, self.cols, data)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.rows, self.cols

    @property
    def T(self) -> Matrix:
        assert self.data is not None

        data = []
        for i in range(self.cols):
            row = []
            for j in range(self.rows):
                row.append(self.data[j][i])
            data.append(row)

        return Matrix(self.cols, self.rows, data)

    def item(self) -> float:
        assert self.data is not None
        assert self.rows == self.cols == 1
        return self.data[0][0]

    def equals(self, other: Matrix | float | int) -> bool:
        elements_eq = self == other
        for i in range(self.rows):
            if any(x == 0.0 for x in elements_eq.data[i]):
                return False
        return True

    def sum(self, dim: int | None = None) -> Matrix:
        assert self.data is not None

        if dim is None:
            total = sum(sum(row) for row in self.data)
            return Matrix(1, 1, [[total]])

        assert dim in [-1, 0, 1]

        if dim == 0:
            row = []
            for j in range(self.cols):
                col_sum = sum(self.data[i][j] for i in range(self.rows))
                row.append(col_sum)

            return Matrix(1, self.cols, [row])

        col = []
        for i in range(self.rows):
            row_sum = sum(self.data[i])
            col.append([row_sum])

        return Matrix(self.rows, 1, col)

    def mean(self, dim: int | None = None) -> Matrix:
        if dim is None:
            return self.sum() / (self.rows * self.cols)

        assert dim in [-1, 0, 1]

        if dim == 0:
            return self.sum(dim=dim) / self.rows

        return self.sum(dim=dim) / self.cols

    @staticmethod
    def broadcastable(a: Matrix, b: Matrix) -> bool:
        if a.rows == b.rows and a.cols == b.cols:
            return True
        if (a.rows == 1 and a.cols == 1) or (b.rows == 1 and b.cols == 1):
            return True
        if a.rows == b.rows and (a.cols == 1 or b.cols == 1):
            return True
        if a.cols == b.cols and (a.rows == 1 or b.rows == 1):
            return True
        return False

    def __matmul__(self, other: Matrix) -> Matrix:
        assert self.data is not None
        assert isinstance(other, Matrix)
        assert other.data is not None
        assert self.cols == other.rows

        data = []
        for i in range(self.rows):
            row = []
            for j in range(other.cols):
                prod = []
                for k in range(self.cols):
                    prod.append(self.data[i][k] * other.data[k][j])
                row.append(sum(prod))
            data.append(row)

        return Matrix(self.rows, other.cols, data)

    def __add__(self, other: Matrix | float | int) -> Matrix:
        op = lambda x, y: x + y
        return self._element_wise(other, op)

    def __iadd__(self, other: Matrix | float | int) -> Matrix:
        result = self + other
        self.data = result.data
        return self

    def __radd__(self, other: float | int) -> Matrix:
        return self + other

    def __sub__(self, other: Matrix | float | int) -> Matrix:
        op = lambda x, y: x - y
        return self._element_wise(other, op)

    def __isub__(self, other: Matrix | float | int) -> Matrix:
        result = self - other
        self.data = result.data
        return self

    def __rsub__(self, other: float | int) -> Matrix:
        return Matrix(1, 1, [[float(other)]]) - self

    def __mul__(self, other: Matrix | float | int) -> Matrix:
        op = lambda x, y: x * y
        return self._element_wise(other, op)

    def __imul__(self, other: Matrix | float | int) -> Matrix:
        result = self * other
        self.data = result.data
        return self

    def __rmul__(self, other: float | int) -> Matrix:
        return self * other

    def __truediv__(self, other: Matrix | float | int) -> Matrix:
        op = lambda x, y: x / y
        return self._element_wise(other, op)

    def __itruediv__(self, other: Matrix | float | int) -> Matrix:
        result = self / other
        self.data = result.data
        return self

    def __rtruediv__(self, other: float | int) -> Matrix:
        return Matrix(1, 1, [[float(other)]]) / self

    def __pow__(self, other: Matrix | float | int) -> Matrix:
        op = lambda x, y: x**y
        return self._element_wise(other, op)

    def __ipow__(self, other: Matrix | float | int) -> Matrix:
        result = self**other
        self.data = result.data
        return self

    def __rpow__(self, other: float | int) -> Matrix:
        return Matrix(1, 1, [[float(other)]]) ** self

    def __gt__(self, other: Matrix | float | int) -> Matrix:
        op = lambda x, y: 1.0 if x > y else 0.0
        return self._element_wise(other, op)

    def __ge__(self, other: Matrix | float | int) -> Matrix:
        op = lambda x, y: 1.0 if x >= y else 0.0
        return self._element_wise(other, op)

    def __lt__(self, other: Matrix | float | int) -> Matrix:
        op = lambda x, y: 1.0 if x < y else 0.0
        return self._element_wise(other, op)

    def __le__(self, other: Matrix | float | int) -> Matrix:
        op = lambda x, y: 1.0 if x <= y else 0.0
        return self._element_wise(other, op)

    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Matrix | float | int
    ) -> Matrix:
        op = lambda x, y: 1.0 if x == y else 0.0
        return self._element_wise(other, op)

    def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Matrix | float | int
    ) -> Matrix:
        op = lambda x, y: 1.0 if x != y else 0.0
        return self._element_wise(other, op)

    def __neg__(self) -> Matrix:
        return self * -1

    def _element_wise(
        self, other: Matrix | float | int, op: Callable[[float, float], float]
    ) -> Matrix:
        assert self.data is not None

        if isinstance(other, (float, int)):
            other = Matrix(1, 1, [[float(other)]])

        assert other.data is not None
        assert Matrix.broadcastable(self, other)

        if self.rows < other.rows or self.cols < other.cols:
            self, other, op_ = other, self, op
            op = lambda x, y: op_(y, x)

        other_data = other.data
        if other.rows == 1 and other.cols == 1:
            get_other_data = lambda i, j: other_data[0][0]
        elif other.rows == 1 and other.cols == self.cols:
            get_other_data = lambda i, j: other_data[0][j]
        elif other.cols == 1 and other.rows == self.rows:
            get_other_data = lambda i, j: other_data[i][0]
        else:
            get_other_data = lambda i, j: other_data[i][j]

        data = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(op(self.data[i][j], get_other_data(i, j)))
            data.append(row)

        return Matrix(self.rows, self.cols, data)

    def __repr__(self) -> str:
        rows = []
        for row in self.data:
            formatted = []
            for x in row:
                formatted.append(f"{x:4.4e}")
            rows.append(f"[{', '.join(formatted)}]")
        formatted_rows = ",\n\t".join(rows)
        return f"Matrix([{formatted_rows}])"
