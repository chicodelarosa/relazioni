"""Module for checking consistency across data strucutres and values."""

import numpy as np

def check_variables(v1: np.ndarray, v2: np.ndarray) -> None:
    """
    Check that variables are consistent in terms of their data structure type and dimensions.

    Parameters
    ----------
    v1 : array_like
        A 1-D array containing multiple variables and observations.
    v2 : array_like
        A 1-D array containing multiple variables and observations.
    """
    if type(v1) != np.ndarray:
        raise ValueError(f"v1's type = {type(v1)} must be of type np.ndarray.")

    if type(v2) != np.ndarray:
        raise ValueError(f"v2's type = {type(v2)} must be of type np.ndarray.")

    if len(v1) != len(v2):
        raise ValueError(f"Length of v1 = {len(v1)} and v2 = {len(v2)} must be equal.")

    if v1.ndim != 1:
        raise ValueError(
            f"Number of array dimensions of v1 = {v1.ndim} mus be equal to 1"
        )

    if v2.ndim != 1:
        raise ValueError(
            f"Number of array dimensions of v2 = {v2.ndim} mus be equal to 1"
        )


def check_binary_categorical(v1: np.ndarray, v2: np.ndarray) -> None:
    """
    Check that variables are consistent in terms of their data structure type, dimensions, and type.

    Parameters
    ----------
    v1 : array_like
        A 1-D array containing multiple variables and observations.
    v2 : array_like
        A 1-D array containing multiple variables and observations.
    """
    check_variables(v1, v2)

    v1_unique = np.unique(v1)

    if len(v1_unique) != 2:
        raise ValueError(
            f"v1 contains more than one or two unique values ({v1_unique})."
        )

    v2_unique = np.unique(v2)

    if len(v2_unique) != 2:
        raise ValueError(f"v2 contains more than one two unique values ({v2_unique}).")
