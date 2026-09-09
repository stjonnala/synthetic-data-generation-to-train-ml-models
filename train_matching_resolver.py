from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from matching_resolver_model import MatchingResolverModel, MODEL_PATH, build_pairs
from synthetic_data_generation import LINKED_PEOPLE, N_PEOPLE, load_tables

HERE = Path(__file__).resolve().parent
CHART_PATH = HERE / "overmatches.png"


def split_pairs(people, pairs):
    look_src = {f"P{N_PEOPLE - 20 + i:04d}": f"P{i:04d}" for i in range(20)}
    for i in range(390, 400):
        look_src.pop(f"P{i:04d}", None)
    look_src.update(LINKED_PEOPLE)

    def split_key(pid):
        return look_src.get(pid, pid)

    keys = np.array(sorted({split_key(pid) for pid in people["true_person_id"]}))
    split_rng = np.random.default_rng(42)
    order = split_rng.permutation(keys)
    cut = int(0.70 * len(order))
    train_keys = set(order[:cut])
    test_keys = set(order[cut:])
    train_people = {pid for pid in people["true_person_id"] if split_key(pid) in train_keys}
    test_people = {pid for pid in people["true_person_id"] if split_key(pid) in test_keys}

    train = pairs[pairs["id_a"].isin(train_people) & pairs["id_b"].isin(train_people)].copy()
    test = pairs[pairs["id_a"].isin(test_people) & pairs["id_b"].isin(test_people)].copy()
    return train, test


def train_and_score(train, test):
    exact = ((test["name_jaro"] >= 0.99) & (test["dob_exact"] == 1.0)).astype(int)
    resolver = MatchingResolverModel()
    resolver.fit(train)
    pred = resolver.predict(test)

    test = test.copy()
    test["pred"] = pred
    test["match_prob"] = resolver.predict_proba_match(test)
    counts = {
        "true matches caught": int(((test["label"] == 1) & (test["pred"] == 1)).sum()),
        "true matches missed": int(((test["label"] == 1) & (test["pred"] == 0)).sum()),
        "overmatches": int(((test["label"] == 0) & (test["pred"] == 1)).sum()),
    }
    metrics = {
        "exact_precision": precision_score(test["label"], exact, zero_division=0),
        "exact_recall": recall_score(test["label"], exact, zero_division=0),
        "tree_precision": precision_score(test["label"], pred, zero_division=0),
        "tree_recall": recall_score(test["label"], pred, zero_division=0),
        "tree_f1": f1_score(test["label"], pred, zero_division=0),
        "tree_accuracy": accuracy_score(test["label"], pred),
        "n_test": len(test),
        "counts": counts,
    }
    return resolver, test, metrics


def print_scores(metrics):
    print("exact name+dob")
    print("  precision", f"{metrics['exact_precision']:.2f}")
    print("  recall   ", f"{metrics['exact_recall']:.2f}")
    print("boosted trees")
    print("  test pairs", metrics["n_test"])
    print("  precision", f"{metrics['tree_precision']:.2f}")
    print("  recall   ", f"{metrics['tree_recall']:.2f}")
    print("  f1       ", f"{metrics['tree_f1']:.2f}")
    print("  accuracy ", f"{metrics['tree_accuracy']:.2f}")
    counts = metrics["counts"]
    print("caught", counts["true matches caught"])
    print("missed", counts["true matches missed"])
    print("overmatches", counts["overmatches"])


def save_chart(metrics):
    counts = metrics["counts"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(list(counts.keys()), list(counts.values()), color=["#3DA356", "#D96B3A", "#7A5CC4"])
    ax.set_ylabel("test pairs")
    ax.set_title("What the matching resolver did (synthetic labels)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(CHART_PATH, dpi=120)
    plt.close(fig)


def train_from_saved_tables():
    people, records = load_tables()
    pairs = build_pairs(records)
    train, test = split_pairs(people, pairs)
    resolver, test, metrics = train_and_score(train, test)
    path = resolver.save(MODEL_PATH)
    save_chart(metrics)
    return people, records, pairs, test, metrics, path


if __name__ == "__main__":
    people, records, pairs, test, metrics, path = train_from_saved_tables()
    print_scores(metrics)
    print("saved", path.name)
