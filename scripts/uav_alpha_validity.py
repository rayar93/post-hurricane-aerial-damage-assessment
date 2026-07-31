#!/usr/bin/env python3

"""Utilities for alpha-based validity handling in UAV imagery."""

from typing import Optional, Sequence, Tuple

import numpy as np


IGNORE_INDEX = 255
DEFAULT_REJECTION_THRESHOLD = 0.50


def alpha_valid_mask(alpha: np.ndarray) -> np.ndarray:
    """
    Return True where imagery exists.

    Alpha values greater than zero are valid. RGB intensity is deliberately
    ignored because valid roofs, shadows, roads, and backgrounds can be dark.
    """
    alpha_array = np.asarray(alpha)

    if alpha_array.ndim != 2:
        raise ValueError(
            f"Alpha must be a 2D array, got shape {alpha_array.shape}"
        )

    return alpha_array > 0


def apply_alpha_validity(
    rgb: np.ndarray,
    alpha: np.ndarray,
    target_mask: Optional[np.ndarray] = None,
    fill_value: Sequence[int] = (0, 0, 0),
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Fill invalid RGB pixels and optionally mark them IGNORE in a target mask.

    The RGB fill is only a deterministic placeholder. The alpha-derived
    validity mask remains the authoritative source of missing-data status.
    """
    rgb_array = np.asarray(rgb)
    alpha_array = np.asarray(alpha)

    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(
            f"RGB must have shape (H, W, 3), got {rgb_array.shape}"
        )

    if alpha_array.shape != rgb_array.shape[:2]:
        raise ValueError(
            "Alpha and RGB spatial shapes differ: "
            f"{alpha_array.shape} versus {rgb_array.shape[:2]}"
        )

    valid_mask = alpha_valid_mask(alpha_array)

    filled_rgb = rgb_array.copy()
    fill_array = np.asarray(fill_value, dtype=filled_rgb.dtype)

    if fill_array.shape != (3,):
        raise ValueError(
            f"fill_value must contain three channels, got {fill_value}"
        )

    filled_rgb[~valid_mask] = fill_array

    updated_target = None

    if target_mask is not None:
        target_array = np.asarray(target_mask)

        if target_array.shape != valid_mask.shape:
            raise ValueError(
                "Target and alpha spatial shapes differ: "
                f"{target_array.shape} versus {valid_mask.shape}"
            )

        updated_target = target_array.copy()
        updated_target[~valid_mask] = ignore_index

    return filled_rgb, updated_target, valid_mask


def polygon_invalid_fraction(
    valid_mask: np.ndarray,
    polygon_mask: np.ndarray,
) -> float:
    """Calculate the invalid fraction inside a building polygon."""
    valid_array = np.asarray(valid_mask, dtype=bool)
    polygon_array = np.asarray(polygon_mask, dtype=bool)

    if valid_array.shape != polygon_array.shape:
        raise ValueError(
            "Validity and polygon masks must have the same shape."
        )

    polygon_pixels = int(polygon_array.sum())

    if polygon_pixels == 0:
        raise ValueError("Building polygon contains no pixels.")

    invalid_inside_polygon = int(
        np.logical_and(~valid_array, polygon_array).sum()
    )

    return invalid_inside_polygon / polygon_pixels


def should_reject_crop(
    building_polygon_invalid_fraction: float,
    threshold: float = DEFAULT_REJECTION_THRESHOLD,
) -> bool:
    """
    Apply the conservative future-data safeguard.

    No sample in the validated development corpus exceeded 50%.
    """
    if not 0.0 <= building_polygon_invalid_fraction <= 1.0:
        raise ValueError(
            "building_polygon_invalid_fraction must be between 0 and 1."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    return building_polygon_invalid_fraction > threshold
