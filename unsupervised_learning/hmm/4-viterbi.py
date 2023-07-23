#!/usr/bin/env python3

"""viterbi"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """viterbi"""
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

    viterbi_probs = np.zeros((N, T))
    backpointers = np.zeros((N, T), dtype=int)

    viterbi_probs[:, 0] = Initial.flatten() * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            max_prob = viterbi_probs[:, t - 1] * \
                Transition[:, j] * Emission[j, Observation[t]]
            viterbi_probs[j, t] = np.max(max_prob)
            backpointers[j, t] = np.argmax(max_prob)

    path = []
    last_state = np.argmax(viterbi_probs[:, -1])
    path.append(last_state)

    for t in range(T - 1, 0, -1):
        last_state = backpointers[last_state, t]
        path.insert(0, last_state)

    P = np.max(viterbi_probs[:, -1])

    return path, P
