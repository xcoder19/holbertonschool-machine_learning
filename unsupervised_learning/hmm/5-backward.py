#!/usr/bin/env python3

"""backward"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """backward"""
    if not isinstance(
            Observation,
            np.ndarray) or not isinstance(
            Emission,
            np.ndarray) or not isinstance(
                Transition,
                np.ndarray) or not isinstance(
                    Initial,
            np.ndarray):
        return None, None

    if len(
        Observation.shape) != 1 or len(
        Emission.shape) != 2 or len(
            Transition.shape) != 2 or len(
                Initial.shape) != 2:
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None

    if not np.allclose(
            np.sum(
                Emission,
                axis=1),
            1.0) or not np.allclose(
                np.sum(
                    Transition,
                    axis=1),
            1.0) or not np.allclose(
            np.sum(Initial),
            1.0):
        return None, None

    B = np.zeros((N, T))

    B[:, -1] = 1.0

    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                B[:, t + 1] *
                Emission[:, Observation[t + 1]] *
                Transition[i, :])

    P = np.sum(Initial.flatten() * Emission[:, Observation[0]] * B[:, 0])

    return P, B
