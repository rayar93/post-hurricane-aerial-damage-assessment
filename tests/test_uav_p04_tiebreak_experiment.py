import unittest

from scripts import run_uav_p04_classifier_experiment as base
from scripts.run_uav_p04_tiebreak_experiment import (
    EVALUATION_ORTHOMOSAICS,
    aggregate_runs,
    choose_result,
    rounded_fraction_count,
    select_stratified_evaluation,
)


def sample(index, orthomosaic, event, label):
    return base.Sample(
        sample_id=f"sample-{orthomosaic}-{label}-{index}",
        building_id=f"building-{orthomosaic}-{label}-{index}",
        record_index=index,
        event=event,
        orthomosaic=orthomosaic,
        label=label,
        class_id=base.LABEL_TO_ID[label],
        legacy_split="validation",
        subset_names=(),
        points=((0, 0), (1, 0), (1, 1)),
        base_bounds=base.Bounds(0, 0, 1, 1),
    )


class UAVP04TiebreakTests(unittest.TestCase):
    def test_rounding_uses_half_up(self):
        self.assertEqual(rounded_fraction_count(5, 0.30), 2)
        self.assertEqual(rounded_fraction_count(57, 0.30), 17)

    def test_stratified_evaluation_is_deterministic_and_complete(self):
        candidates = []
        for orthomosaic, event in EVALUATION_ORTHOMOSAICS.items():
            for label in base.LABELS:
                candidates.extend(
                    sample(index, orthomosaic, event, label) for index in range(10)
                )
        first, first_metadata = select_stratified_evaluation(candidates, 0.30)
        second, second_metadata = select_stratified_evaluation(
            list(reversed(candidates)), 0.30
        )
        self.assertEqual(
            [item.sample_id for item in first], [item.sample_id for item in second],
        )
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(len(first), 24)
        for orthomosaic in EVALUATION_ORTHOMOSAICS:
            for label in base.LABELS:
                self.assertEqual(
                    sum(
                        item.orthomosaic == orthomosaic and item.label == label
                        for item in first
                    ),
                    3,
                )

    def test_aggregate_uses_equal_event_primary_metric_and_paired_seeds(self):
        rows = []
        offsets = {
            "masked_building": -0.01,
            "margin_0": 0.0,
            "margin_12p5": 0.02,
        }
        for variant, offset in offsets.items():
            for seed in (17, 29, 43, 59, 71):
                row = {
                    "variant": variant,
                    "seed": seed,
                }
                for metric in base.METRIC_FIELDS:
                    row[metric] = 0.4
                    row[f"equal_event_{metric}"] = 0.4
                row["equal_event_macro_f1"] = 0.4 + offset
                rows.append(row)
        aggregated = {row["variant"]: row for row in aggregate_runs(rows)}
        self.assertAlmostEqual(
            aggregated["margin_12p5"]["equal_event_macro_f1_mean"], 0.42
        )
        self.assertAlmostEqual(
            aggregated["margin_12p5"]["equal_event_macro_f1_diff_vs_margin_0_mean"],
            0.02,
        )
        self.assertTrue(aggregated["margin_12p5"]["minority_recall_guardrail_passed"])

    def test_selection_retains_zero_when_added_treatments_are_uncertain(self):
        aggregates = [
            {
                "variant": variant,
                "equal_event_macro_f1_mean": value,
                "minority_recall_guardrail_passed": True,
            }
            for variant, value in (
                ("margin_0", 0.40),
                ("masked_building", 0.39),
                ("margin_12p5", 0.42),
            )
        ]
        paired = [
            {
                "left_variant": variant,
                "right_variant": "margin_0",
                "metric": "equal_event_macro_f1",
                "ci95_low": lower,
            }
            for variant, lower in (("masked_building", -0.03), ("margin_12p5", -0.01),)
        ]
        event_rows = [
            {"variant": variant, "event": event, "macro_f1_mean": value}
            for event in (base.EVENT_IAN, base.EVENT_IDA)
            for variant, value in (
                ("margin_0", 0.40),
                ("masked_building", 0.39),
                ("margin_12p5", 0.42),
            )
        ]
        run_rows = [
            {"variant": variant, "seed": seed, "equal_event_macro_f1": value}
            for seed in (17, 29, 43, 59, 71)
            for variant, value in (
                ("margin_0", 0.40),
                ("masked_building", 0.39),
                ("margin_12p5", 0.42),
            )
        ]
        result = choose_result(aggregates, paired, event_rows, run_rows)
        self.assertEqual(result["selected_variant"], "margin_0")
        self.assertEqual(result["decision"], "KEEP")

    def test_selection_accepts_only_a_stable_significant_added_treatment(self):
        aggregates = [
            {
                "variant": variant,
                "equal_event_macro_f1_mean": value,
                "minority_recall_guardrail_passed": True,
            }
            for variant, value in (
                ("margin_0", 0.40),
                ("masked_building", 0.39),
                ("margin_12p5", 0.46),
            )
        ]
        paired = [
            {
                "left_variant": variant,
                "right_variant": "margin_0",
                "metric": "equal_event_macro_f1",
                "ci95_low": lower,
            }
            for variant, lower in (("masked_building", -0.03), ("margin_12p5", 0.02),)
        ]
        event_rows = [
            {"variant": variant, "event": event, "macro_f1_mean": value}
            for event in (base.EVENT_IAN, base.EVENT_IDA)
            for variant, value in (
                ("margin_0", 0.40),
                ("masked_building", 0.39),
                ("margin_12p5", 0.46),
            )
        ]
        run_rows = [
            {"variant": variant, "seed": seed, "equal_event_macro_f1": value}
            for seed in (17, 29, 43, 59, 71)
            for variant, value in (
                ("margin_0", 0.40),
                ("masked_building", 0.39),
                ("margin_12p5", 0.46),
            )
        ]
        result = choose_result(aggregates, paired, event_rows, run_rows)
        self.assertEqual(result["selected_variant"], "margin_12p5")
        self.assertEqual(result["decision"], "KEEP")


if __name__ == "__main__":
    unittest.main()
