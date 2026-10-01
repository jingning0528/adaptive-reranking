from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit


def near_duplicate_groups(queries: dict[str, str], threshold: float) -> dict[str, int]:
    query_ids = sorted(queries)
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(
        [queries[query_id] for query_id in query_ids]
    )
    similarities = cosine_similarity(matrix)
    parents = list(range(len(query_ids)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(query_ids)):
        for right in range(left + 1, len(query_ids)):
            if similarities[left, right] >= threshold:
                union(left, right)
    roots = {}
    return {
        query_id: roots.setdefault(find(index), len(roots))
        for index, query_id in enumerate(query_ids)
    }


def train_calibration_split(
    queries: dict[str, str],
    seed: int,
    calibration_fraction: float,
    threshold: float,
) -> tuple[list[str], list[str], dict[str, int]]:
    groups_by_id = near_duplicate_groups(queries, threshold)
    query_ids = sorted(queries)
    groups = [groups_by_id[query_id] for query_id in query_ids]
    splitter = GroupShuffleSplit(n_splits=1, test_size=calibration_fraction, random_state=seed)
    train_indexes, calibration_indexes = next(splitter.split(query_ids, groups=groups))
    train_ids = sorted(query_ids[index] for index in train_indexes)
    calibration_ids = sorted(query_ids[index] for index in calibration_indexes)
    return train_ids, calibration_ids, groups_by_id


def group_sizes(groups_by_id: dict[str, int]) -> dict[int, int]:
    sizes = defaultdict(int)
    for group in groups_by_id.values():
        sizes[group] += 1
    return dict(sizes)
