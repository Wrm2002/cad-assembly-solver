import unittest

from multimodal_calibration import _auc, _split_metrics


class MultimodalCalibrationTests(unittest.TestCase):
    def test_auc_handles_ordering_and_ties(self):
        self.assertEqual(_auc([1, 0], [0.9, 0.1]), 1.0)
        self.assertEqual(_auc([1, 0], [0.5, 0.5]), 0.5)
        self.assertEqual(_auc([1, 0], [0.1, 0.9]), 0.0)

    def test_all_abstain_cannot_establish_auto_accept_precision(self):
        rows = [
            {
                "label": 1,
                "semantic_score": 0.5,
                "confidence": 0.0,
                "verdict": "abstain",
                "geometry_baseline_score": 0.8,
            },
            {
                "label": 0,
                "semantic_score": 0.5,
                "confidence": 0.0,
                "verdict": "abstain",
                "geometry_baseline_score": 0.8,
            },
        ]
        metrics = _split_metrics(rows)
        self.assertEqual(metrics["semantic_auc"], 0.5)
        self.assertEqual(metrics["semantic_brier"], 0.25)
        self.assertEqual(metrics["false_positive_count"], 0)
        self.assertIsNone(metrics["auto_accept_precision"])

    def test_confident_negative_accept_is_counted_as_false_positive(self):
        rows = [
            {
                "label": 0,
                "semantic_score": 0.95,
                "confidence": 0.9,
                "verdict": "accept",
                "geometry_baseline_score": 0.9,
            }
        ]
        metrics = _split_metrics(rows)
        self.assertEqual(metrics["auto_accept_count"], 1)
        self.assertEqual(metrics["false_positive_count"], 1)
        self.assertEqual(metrics["auto_accept_precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
