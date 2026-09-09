from synthetic_data_generation import generate_and_save, print_generation
from train_matching_resolver import print_scores, train_from_saved_tables
from try_match import run_canned


def banner(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print()


def main():
    banner("1. Synthetic data generation")
    people, records, issues = generate_and_save()
    print_generation(people, records, issues)

    banner("2. Train the matching resolver")
    people, records, pairs, test, metrics, path = train_from_saved_tables()
    print_scores(metrics)
    print()
    print(f"Trained model saved: {path}")

    banner("3. Try the trained model")
    run_canned()
    print("For live typing, run: python try_match.py")


if __name__ == "__main__":
    main()
