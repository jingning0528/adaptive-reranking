import unittest

from adaptive_reranking.reranking.features import FEATURE_NAMES, candidate_features


class CandidateFeatureTests(unittest.TestCase):
    def test_feature_vector_is_fixed_and_matches_numbers_and_acronyms(self):
        values = candidate_features(
            "AMPK reduces fibrosis by 20 percent",
            {"title": "AMPK study", "text": "Fibrosis was reduced by 20 percent."},
            retrieval_score=0.7,
            rank=2,
            top_score=0.9,
        )
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertEqual(values[12], 1.0)
        self.assertEqual(values[13], 1.0)
        self.assertAlmostEqual(values[2], 0.5)
        self.assertAlmostEqual(values[3], 0.2)


if __name__ == "__main__":
    unittest.main()
