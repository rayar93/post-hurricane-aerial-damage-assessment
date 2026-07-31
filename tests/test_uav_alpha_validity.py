import unittest

import numpy as np

from scripts.uav_alpha_validity import (
    IGNORE_INDEX,
    alpha_valid_mask,
    apply_alpha_validity,
    polygon_invalid_fraction,
    should_reject_crop,
)


class AlphaValidityTests(unittest.TestCase):
    def test_dark_rgb_with_valid_alpha_remains_valid(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        alpha = np.full((2, 2), 255, dtype=np.uint8)

        filled, target, valid = apply_alpha_validity(
            rgb=rgb,
            alpha=alpha,
        )

        self.assertTrue(valid.all())
        self.assertTrue(np.array_equal(filled, rgb))
        self.assertIsNone(target)

    def test_alpha_zero_is_black_filled_and_ignored(self):
        rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
        alpha = np.array(
            [
                [255, 0],
                [255, 255],
            ],
            dtype=np.uint8,
        )
        target = np.ones((2, 2), dtype=np.uint8)

        filled, updated_target, valid = apply_alpha_validity(
            rgb=rgb,
            alpha=alpha,
            target_mask=target,
        )

        self.assertFalse(valid[0, 1])
        self.assertTrue(
            np.array_equal(
                filled[0, 1],
                np.array([0, 0, 0], dtype=np.uint8),
            )
        )
        self.assertEqual(
            int(updated_target[0, 1]),
            IGNORE_INDEX,
        )
        self.assertEqual(
            int(updated_target[1, 1]),
            1,
        )

    def test_polygon_invalid_fraction(self):
        valid = np.array(
            [
                [True, False],
                [True, True],
            ]
        )
        polygon = np.ones((2, 2), dtype=bool)

        fraction = polygon_invalid_fraction(
            valid_mask=valid,
            polygon_mask=polygon,
        )

        self.assertEqual(fraction, 0.25)

    def test_conservative_rejection_rule(self):
        self.assertFalse(should_reject_crop(0.472647))
        self.assertFalse(should_reject_crop(0.50))
        self.assertTrue(should_reject_crop(0.500001))

    def test_alpha_valid_mask_does_not_use_rgb(self):
        alpha = np.array(
            [
                [255, 0],
                [1, 255],
            ],
            dtype=np.uint8,
        )

        expected = np.array(
            [
                [True, False],
                [True, True],
            ]
        )

        self.assertTrue(
            np.array_equal(
                alpha_valid_mask(alpha),
                expected,
            )
        )


if __name__ == "__main__":
    unittest.main()
