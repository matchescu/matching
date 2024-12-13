import itertools
import math
from typing import Iterable

import numpy as np

from matchescu.matching.similarity._string import StringSimilarity


class LevenshteinLearner(StringSimilarity):
    def __init__(self):
        super().__init__()
        self._underlying_alphabet = dict()
        self._surface_alphabet = dict()
        self._deltas = np.zeros((0, 0))
        self._delta_sharp = 1
        self._gamma = np.zeros((0, 0))
        self._gamma_sharp = 0
        self._inserts = set()
        self._updates = set()
        self._deletes = set()
        self._transitions = set()

    def _init_transitions(self):
        self._updates = set(
            itertools.product(self._underlying_alphabet, self._surface_alphabet)
        )
        self._inserts = set(itertools.product([None], self._surface_alphabet))
        self._deletes = set(itertools.product(self._underlying_alphabet, [None]))
        self._transitions = self._inserts | self._updates | self._deletes

    def _init_alphabets(self, pairs: Iterable[tuple[str, str]]) -> None:
        surface = set()
        underlying = set()
        for pair in pairs:
            for surface_char in pair[0]:
                surface.add(surface_char)
            for underlying_char in pair[1]:
                underlying.add(underlying_char)
        self._surface_alphabet = {c: idx + 1 for idx, c in enumerate(sorted(surface))}
        self._underlying_alphabet = {
            c: idx + 1 for idx, c in enumerate(sorted(underlying))
        }
        self._init_transitions()

    def _init_probabilities(self) -> None:
        rows = len(self._underlying_alphabet) + 1
        cols = len(self._surface_alphabet) + 1
        inserts = len(self._inserts)
        updates = len(self._updates)
        deletes = len(self._deletes)
        self._deltas = np.zeros((rows, cols))
        for i in range(1, rows):
            self._deltas[i, 0] = 1 / deletes
        for j in range(1, cols):
            self._deltas[0, j] = 1 / inserts

        for i in range(1, rows):
            for j in range(1, cols):
                self._deltas[i, j] = 1 / updates
        self._delta_sharp = 1 / (inserts + updates + deletes)

    def _pos(self, x_t: str, y_v) -> tuple[int, int]:
        return (
            self._underlying_alphabet.get(x_t, 0),
            self._surface_alphabet.get(y_v, 0),
        )

    def _underlying_index(self, char: str) -> int:
        return self._underlying_alphabet.get(char, 0)

    def _surface_index(self, char: str) -> int:
        return self._surface_alphabet.get(char, 0)

    def _delta(self, x_t: str | None, y_v: str | None) -> float:
        if self._deltas.ndim != 2:
            raise ValueError("probability matrix not initialized properly")
        if x_t is None and y_v is None:
            raise ValueError("x and y can't both be 'None'")
        i, j = self._pos(x_t, y_v)
        if x_t is None or x_t not in self._underlying_alphabet:
            return self._deltas[0, j]
        if y_v is None or y_v not in self._surface_alphabet:
            return self._deltas[i, 0]
        return self._deltas[i, j]

    def forward_eval(self, x: str, y: str) -> np.ndarray:
        T = len(x)
        V = len(y)
        alpha = np.zeros((T + 1, V + 1))
        alpha[0, 0] = 1
        for t in range(T + 1):
            for v in range(V + 1):
                if v > 0:
                    alpha[t, v] += self._delta(None, y[v - 1]) * alpha[t, v - 1]
                if t > 0:
                    alpha[t, v] += self._delta(x[t - 1], None) * alpha[t - 1, v]
                if t > 0 and v > 0:
                    alpha[t, v] += self._delta(x[t - 1], y[v - 1]) * alpha[t - 1, v - 1]
        alpha *= self._delta_sharp
        return alpha

    def backward_eval(self, x: str, y: str) -> np.ndarray:
        T = len(x)
        V = len(y)
        beta = np.zeros((T + 1, V + 1))
        beta[T, V] = 1

        for t in range(T, -1, -1):
            for v in range(V, -1, -1):
                if v < V:
                    beta[t, v] += self._delta(None, y[v]) * beta[t, v + 1]
                if t < T:
                    beta[t, v] += self._delta(x[t], None) * beta[t + 1, v]
                if t < T and v < V:
                    beta[t, v] += self._delta(x[t], y[v]) * beta[t + 1, v + 1]

        beta *= self._delta_sharp
        return beta

    def _compute_expectations(
        self,
        x: str,
        y: str,
        lambda_: float = 0.1,
    ) -> None:
        alpha = self.forward_eval(x, y)
        beta = self.backward_eval(x, y)
        T = len(x)
        V = len(y)

        if alpha[T, V] == 0:
            return

        norm_factor = alpha[T, V]

        self._gamma_sharp += lambda_
        for t in range(T + 1):
            for v in range(V + 1):
                if t > 0:
                    delta = self._delta(x[t - 1], None)
                    gamma_val = (
                        lambda_ * alpha[t - 1, v] * delta * beta[t, v]
                    ) / norm_factor
                    x_t = self._underlying_index(x[t - 1])
                    if x_t > 0:  # if it's a symbol from the underlying alphabet
                        self._gamma[x_t, 0] += gamma_val
                if v > 0:
                    delta = self._delta(None, y[v - 1])
                    y_v = self._surface_index(y[v - 1])
                    if y_v > 0:  # if it's a symbol from the surface alphabet
                        gamma_val = (
                            lambda_ * alpha[t, v - 1] * delta * beta[t, v]
                        ) / norm_factor
                        self._gamma[0, y_v] += gamma_val
                if t > 0 and v > 0:
                    delta = self._delta(x[t - 1], y[v - 1])
                    x_t, y_v = self._pos(x[t - 1], y[v - 1])
                    if x_t > 0 and y_v > 0:
                        gamma_val = (
                            lambda_ * alpha[t - 1, v - 1] * delta * beta[t, v]
                        ) / norm_factor
                        self._gamma[x_t, y_v] += gamma_val

    def _maximize_expectations(self) -> None:
        n = self._gamma_sharp + np.sum(self._gamma)
        for x_t, y_v in self._transitions:
            row, col = self._pos(x_t, y_v)
            self._deltas[row, col] = self._gamma[row, col] / n
        self._delta_sharp = self._gamma_sharp / n

    def fit(
        self, corpus: list[tuple[str, str]], epochs: int = 10
    ) -> "LevenshteinLearner":
        self._init_alphabets(corpus)
        self._init_probabilities()
        epoch = 0
        prev_likelihood = 0
        while epoch < epochs:
            self._gamma = np.zeros(
                (len(self._underlying_alphabet) + 1, len(self._surface_alphabet) + 1)
            )
            for x, y in corpus:
                self._compute_expectations(x, y)
            self._maximize_expectations()
            if abs(self._delta_sharp - prev_likelihood) < 0.1:
                print("converged after", epoch, "epochs")
                break
            prev_likelihood = self._delta_sharp
            epoch += 1
        return self

    def compute_distance(self, x: str, y: str) -> float:
        alpha = self.forward_eval(x, y)
        dist = alpha[len(x), len(y)]
        if dist == 0:
            return 0
        return -np.log(dist)  # return value in log space

    def _compute_string_similarity(self, x: str, y: str) -> float:
        d = self.compute_distance(x, y)
        c = 100
        exponent = -(math.pow(d, 2) / (2 * math.pow(c, 2)))
        return math.exp(exponent)
