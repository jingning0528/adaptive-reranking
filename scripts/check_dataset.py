"""Validate a real dataset without loading models or running experiments."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from adaptive_reranking.data import load_dataset
from scripts._dataset_options import parse_dataset_args


def main(argv=None):
    args = parse_dataset_args("dataset validation", argv)
    dataset = load_dataset(args.dataset, split=args.split, data_dir=args.data_dir, download=not args.no_download)
    print(
        json.dumps(
            {
                "dataset": dataset.name,
                "split": dataset.split,
                "documents": len(dataset.corpus),
                "queries": len(dataset.queries),
                "judged_queries": len(dataset.qrels),
                "judgments": sum(len(judgments) for judgments in dataset.qrels.values()),
                "schema_valid": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
