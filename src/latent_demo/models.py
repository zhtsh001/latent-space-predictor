"""Tiny pure-Python models for latent and pixel prediction."""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


def zeros(length: int) -> List[float]:
    return [0.0 for _ in range(length)]


class TinyMLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.w1 = [[rng.uniform(-0.2, 0.2) / math.sqrt(input_dim) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = zeros(hidden_dim)
        self.w2 = [[rng.uniform(-0.2, 0.2) / math.sqrt(hidden_dim) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = zeros(output_dim)

    def forward(self, x: Sequence[float]) -> Tuple[List[float], List[float]]:
        hidden_pre = []
        hidden = []
        for j in range(self.hidden_dim):
            total = self.b1[j]
            for i in range(self.input_dim):
                total += x[i] * self.w1[i][j]
            hidden_pre.append(total)
            hidden.append(math.tanh(total))

        output = []
        for k in range(self.output_dim):
            total = self.b2[k]
            for j in range(self.hidden_dim):
                total += hidden[j] * self.w2[j][k]
            output.append(total)
        return hidden, output

    def predict(self, x: Sequence[float]) -> List[float]:
        _, output = self.forward(x)
        return output

    def parameter_count(self) -> int:
        return (self.input_dim * self.hidden_dim) + self.hidden_dim + (self.hidden_dim * self.output_dim) + self.output_dim

    def train_step(self, x: Sequence[float], y: Sequence[float], learning_rate: float) -> float:
        hidden, pred = self.forward(x)
        diff = [pred[k] - y[k] for k in range(self.output_dim)]
        loss = sum(d * d for d in diff) / self.output_dim

        grad_out = [2.0 * d / self.output_dim for d in diff]

        grad_w2 = [[0.0 for _ in range(self.output_dim)] for _ in range(self.hidden_dim)]
        grad_b2 = grad_out[:]
        grad_hidden = [0.0 for _ in range(self.hidden_dim)]

        for j in range(self.hidden_dim):
            for k in range(self.output_dim):
                grad_w2[j][k] = hidden[j] * grad_out[k]
                grad_hidden[j] += self.w2[j][k] * grad_out[k]

        grad_hidden_pre = [grad_hidden[j] * (1.0 - hidden[j] * hidden[j]) for j in range(self.hidden_dim)]
        grad_w1 = [[0.0 for _ in range(self.hidden_dim)] for _ in range(self.input_dim)]
        grad_b1 = grad_hidden_pre[:]

        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                grad_w1[i][j] = x[i] * grad_hidden_pre[j]

        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                self.w1[i][j] -= learning_rate * grad_w1[i][j]
        for j in range(self.hidden_dim):
            self.b1[j] -= learning_rate * grad_b1[j]
        for j in range(self.hidden_dim):
            for k in range(self.output_dim):
                self.w2[j][k] -= learning_rate * grad_w2[j][k]
        for k in range(self.output_dim):
            self.b2[k] -= learning_rate * grad_b2[k]

        return loss
