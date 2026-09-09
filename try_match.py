from matching_resolver_model import MatchingResolverModel, MODEL_PATH
from synthetic_data_generation import RECORDS_CSV, load_tables


def print_resolve(row_a, row_b, result):
    decision = "MATCH" if result["match"] else "NO MATCH"
    print("INPUT A:", row_a.get("first", ""), row_a.get("last", ""), row_a.get("dob", ""))
    print("INPUT B:", row_b.get("first", ""), row_b.get("last", ""), row_b.get("dob", ""))
    print("OUTPUT:", decision, round(result["probability"], 2))


def row_from_series(s):
    return {
        "first": str(s.get("first", "")),
        "last": str(s.get("last", "")),
        "dob": str(s.get("dob", "")),
        "address": str(s.get("address", "")),
        "middle": str(s.get("middle", "")),
    }


def read_record(label):
    print(f"--- {label} ---")
    first = input("First name: ").strip()
    last = input("Last name: ").strip()
    dob = input("Date of birth (YYYY-MM-DD): ").strip()
    address = input("Address (Enter to skip): ").strip()
    return {"first": first, "last": last, "dob": dob, "address": address}


def first_clean(records, person_id):
    rows = records[
        (records["true_person_id"] == person_id) & (records["error_kind"] == "clean")
    ]
    if len(rows) == 0:
        rows = records[records["true_person_id"] == person_id]
    if len(rows) == 0:
        return None
    return rows.iloc[0]


def prepared_pairs(records):
    p0000 = records[records["true_person_id"] == "P0000"].reset_index(drop=True)
    pairs = []
    if len(p0000) >= 2:
        pairs.append(("Same person, extra letter (Angel / Anggel)", p0000.iloc[0], p0000.iloc[1]))
    if len(p0000) >= 3:
        pairs.append(("Same person, transposed name and date (Angel / aWlker)", p0000.iloc[0], p0000.iloc[2]))
    others = records[records["true_person_id"] != "P0000"]
    if len(p0000) and len(others):
        pairs.append(("Two different people", p0000.iloc[0], others.iloc[0]))
    elena = first_clean(records, "P0390")
    elisa = first_clean(records, "P0391")
    if elena is not None and elisa is not None:
        pairs.append(("Close names, same date (Elena / Elisa)", elena, elisa))
    hale_sr = first_clean(records, "P0396")
    hale_jr = first_clean(records, "P0397")
    if hale_sr is not None and hale_jr is not None:
        pairs.append(("Same name, different year (Robert Hale)", hale_sr, hale_jr))
    return pairs


def run_menu():
    if not MODEL_PATH.exists():
        print("No trained model yet. Run this first:")
        print("  python synthetic_data_generation.py")
        print("  python train_matching_resolver.py")
        return

    resolver = MatchingResolverModel.load(MODEL_PATH)
    records = None
    if RECORDS_CSV.exists():
        _, records = load_tables()

    print("Trained matching resolver is loaded.")
    print("Two records in. MATCH or NO MATCH out.")
    print()
    print("  1  Same person, extra letter (Angel / Anggel)")
    print("  2  Same person, transposed name and date (Angel / aWlker)")
    print("  3  Two different people")
    print("  4  Close names, same date (Elena / Elisa)")
    print("  5  Same name, different year (Robert Hale)")
    print("  6  Type two records yourself")
    print("  q  Quit")
    print()

    presets = prepared_pairs(records) if records is not None else []

    while True:
        choice = input("Pick 1, 2, 3, 4, 5, 6, or q: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            print("Done.")
            return
        if choice == "6":
            row_a = read_record("Record A")
            row_b = read_record("Record B")
        elif choice in {"1", "2", "3", "4", "5"}:
            idx = int(choice) - 1
            if idx >= len(presets):
                print("That pair is not in generated_records.csv. Run synthetic_data_generation.py.")
                continue
            title, ser_a, ser_b = presets[idx]
            print(title)
            row_a = row_from_series(ser_a)
            row_b = row_from_series(ser_b)
        else:
            print("Use 1, 2, 3, 4, 5, 6, or q.")
            continue

        print()
        result = resolver.resolve(row_a, row_b)
        print_resolve(row_a, row_b, result)
        print()


def run_canned():
    if not MODEL_PATH.exists() or not RECORDS_CSV.exists():
        print("Need generated records and a trained model first.")
        print("  python synthetic_data_generation.py")
        print("  python train_matching_resolver.py")
        print("  python try_match.py")
        return
    resolver = MatchingResolverModel.load(MODEL_PATH)
    _, records = load_tables()
    print("checks")
    print()
    for title, ser_a, ser_b in prepared_pairs(records):
        print(title)
        row_a = row_from_series(ser_a)
        row_b = row_from_series(ser_b)
        print_resolve(row_a, row_b, resolver.resolve(row_a, row_b))
        print()


if __name__ == "__main__":
    try:
        run_menu()
    except EOFError:
        print()
        run_canned()
