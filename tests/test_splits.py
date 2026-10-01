import unittest

from adaptive_reranking.splits import train_calibration_split


class SplitTests(unittest.TestCase):
    def test_identical_queries_do_not_cross_splits(self):
        queries = {
            "1": "identical scientific claim",
            "2": "identical scientific claim",
            "3": "a completely different statement",
            "4": "another unrelated biomedical query",
            "5": "gene expression changes",
        }
        train, calibration, groups = train_calibration_split(queries, 7, 0.4, 0.85)
        self.assertFalse(set(train) & set(calibration))
        self.assertEqual(groups["1"], groups["2"])
        self.assertFalse({groups[item] for item in train} & {groups[item] for item in calibration})


if __name__ == "__main__":
    unittest.main()
