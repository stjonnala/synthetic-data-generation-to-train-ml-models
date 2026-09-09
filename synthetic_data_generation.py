from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

HERE = Path(__file__).resolve().parent
PEOPLE_CSV = HERE / "generated_people.csv"
RECORDS_CSV = HERE / "generated_records.csv"

N_PEOPLE = 400
PHONETIC_SUBS = [("ph", "f"), ("f", "ph"), ("y", "i"), ("i", "y")]
NAME_VARIANTS = {
    "Stephen": "Steven",
    "Steven": "Stephen",
    "Catherine": "Katherine",
    "Katherine": "Catherine",
    "Jonson": "Johnson",
    "Johnson": "Jonson",
}

HARD_CASES = [
    {
        "idx": 390,
        "first": "Elena",
        "last": "Brooks",
        "dob": "1990-03-14",
        "address": "22 Pine Road",
        "city": "Oakland",
        "zip": "94601",
        "sex": "F",
    },
    {
        "idx": 391,
        "first": "Elisa",
        "last": "Brooks",
        "dob": "1990-03-14",
        "address": "22 Pine Road",
        "city": "Oakland",
        "zip": "94601",
        "sex": "F",
    },
    {
        "idx": 392,
        "first": "Luca",
        "last": "Demir",
        "dob": "1988-11-07",
        "address": "9 Cedar Lane",
        "city": "Berkeley",
        "zip": "94702",
        "sex": "M",
    },
    {
        "idx": 393,
        "first": "Luka",
        "last": "Demir",
        "dob": "1988-11-07",
        "address": "9 Cedar Lane",
        "city": "Berkeley",
        "zip": "94702",
        "sex": "M",
    },
    {
        "idx": 394,
        "first": "Anika",
        "last": "Bose",
        "dob": "1994-06-21",
        "address": "3 Willow Court",
        "city": "Alameda",
        "zip": "94501",
        "sex": "F",
    },
    {
        "idx": 395,
        "first": "Annika",
        "last": "Bose",
        "dob": "1994-06-21",
        "address": "3 Willow Court",
        "city": "Alameda",
        "zip": "94501",
        "sex": "F",
    },
    {
        "idx": 396,
        "first": "Robert",
        "last": "Hale",
        "dob": "1960-04-02",
        "address": "14 Oak Street",
        "city": "San Mateo",
        "zip": "94401",
        "sex": "M",
    },
    {
        "idx": 397,
        "first": "Robert",
        "last": "Hale",
        "dob": "1992-04-02",
        "address": "14 Oak Street",
        "city": "San Mateo",
        "zip": "94401",
        "sex": "M",
    },
    {
        "idx": 398,
        "first": "David",
        "last": "Cole",
        "dob": "1958-09-15",
        "address": "81 Maple Avenue",
        "city": "Redwood City",
        "zip": "94061",
        "sex": "M",
    },
    {
        "idx": 399,
        "first": "David",
        "last": "Cole",
        "dob": "1990-09-15",
        "address": "81 Maple Avenue",
        "city": "Redwood City",
        "zip": "94061",
        "sex": "M",
    },
]

LINKED_PEOPLE = {
    "P0391": "P0390",
    "P0393": "P0392",
    "P0395": "P0394",
    "P0397": "P0396",
    "P0399": "P0398",
}

SHOW_COLS = ["record_id", "first", "middle", "last", "dob", "city", "error_kind", "true_person_id"]


def phonetic_sub(text, local_rng):
    lower = text.lower()
    hits = [pair for pair in PHONETIC_SUBS if pair[0] in lower]
    if not hits:
        return text
    src, dst = hits[int(local_rng.integers(0, len(hits)))]
    return text.replace(src, dst, 1).replace(src.title(), dst.title(), 1)


def transpose_chars(text, local_rng):
    chars = list(text)
    if len(chars) < 4:
        return text
    i = int(local_rng.integers(1, len(chars) - 1))
    chars[i], chars[i - 1] = chars[i - 1], chars[i]
    return "".join(chars)


def delete_or_insert(text, local_rng):
    chars = list(text)
    if len(chars) < 4:
        return text
    i = int(local_rng.integers(1, len(chars) - 1))
    if int(local_rng.integers(0, 2)) == 0:
        chars.pop(i)
    else:
        chars.insert(i, chars[i])
    return "".join(chars)


def transpose_iso(iso, local_rng):
    y, m, d = iso.split("-")
    digits = list(y + m + d)
    i = int(local_rng.integers(4, 8))
    digits[i], digits[i - 1] = digits[i - 1], digits[i]
    raw = "".join(digits)
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def perturb(rec, local_rng, issues):
    rec = dict(rec)
    n_times = int(local_rng.integers(1, 4))
    for i in range(n_times):
        kind = int(local_rng.integers(0, 5))
        if kind == 0:
            rec["last"] = phonetic_sub(rec.get("last"), local_rng)
            issues.append("phonetic substitution")
        elif kind == 1:
            rec["last"] = transpose_chars(rec.get("last"), local_rng)
            rec["dob"] = transpose_iso(rec.get("dob"), local_rng)
            issues.append("transposition")
        elif kind == 2:
            rec["first"] = delete_or_insert(rec.get("first"), local_rng)
            issues.append("delete or insert")
        elif kind == 3:
            if rec.get("first") in NAME_VARIANTS:
                rec["first"] = NAME_VARIANTS.get(rec.get("first"))
            else:
                rec["last"] = phonetic_sub(rec.get("last"), local_rng)
            issues.append("phonetic name variant")
        else:
            rec["middle"] = ""
            issues.append("missing field")
    return rec


def make_people_and_records(rng, fake):
    people = []
    for i in range(N_PEOPLE):
        dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
        people.append(
            {
                "true_person_id": f"P{i:04d}",
                "first": fake.first_name(),
                "middle": fake.first_name(),
                "last": fake.last_name(),
                "sex": fake.random_element(["F", "M"]),
                "dob": dob.isoformat(),
                "address": fake.street_address(),
                "city": fake.city(),
                "zip": fake.postcode(),
                "phone": fake.numerify("###-###-####"),
                "record_key": fake.numerify("########"),
            }
        )
    people = pd.DataFrame(people)

    for i in range(20):
        src = people.iloc[i]
        tgt = N_PEOPLE - 20 + i
        people.loc[tgt, "last"] = src["last"]
        people.loc[tgt, "first"] = src["first"]
        people.loc[tgt, "city"] = src["city"]
        people.loc[tgt, "dob"] = src["dob"]
        if i < 12:
            people.loc[tgt, "phone"] = src["phone"]

    for case in HARD_CASES:
        i = case["idx"]
        people.loc[i, "first"] = case["first"]
        people.loc[i, "last"] = case["last"]
        people.loc[i, "dob"] = case["dob"]
        people.loc[i, "address"] = case["address"]
        people.loc[i, "city"] = case["city"]
        people.loc[i, "zip"] = case["zip"]
        people.loc[i, "sex"] = case["sex"]

    rows = []
    issues = []
    rec_id = 0
    for _, person in people.iterrows():
        n_copies = 2 if rng.random() < 0.75 else 3
        for copy_i in range(n_copies):
            rec = person.to_dict()
            rec["record_id"] = f"R{rec_id:05d}"
            rec_id += 1
            if copy_i == 0:
                rec["error_kind"] = "clean"
                rows.append(rec)
                continue
            before = len(issues)
            messy = perturb(rec, rng, issues)
            messy["error_kind"] = "; ".join(issues[before:])
            rows.append(messy)

    records = pd.DataFrame(rows)
    return people, records, issues


def save_tables(people, records):
    people.to_csv(PEOPLE_CSV, index=False)
    records.to_csv(RECORDS_CSV, index=False)


def load_tables():
    if not PEOPLE_CSV.exists() or not RECORDS_CSV.exists():
        raise FileNotFoundError(
            "No generated tables yet. Run: python synthetic_data_generation.py"
        )
    people = pd.read_csv(PEOPLE_CSV, dtype=str, keep_default_na=False)
    records = pd.read_csv(RECORDS_CSV, dtype=str, keep_default_na=False)
    return people, records


def print_kind_samples(records, kind, n=4):
    if kind == "clean":
        sample = records[records["error_kind"] == "clean"]
    else:
        sample = records[records["error_kind"].fillna("").str.contains(kind, regex=False)]
    print()
    print(kind, len(sample))
    if len(sample) == 0:
        return
    print(sample[SHOW_COLS].head(n).to_string(index=False))


def print_generation(people, records, issues):
    print(len(people), "people")
    print(len(records), "records")
    print(RECORDS_CSV)
    print()
    print(pd.Series(issues).value_counts().to_string())
    print()
    print(records.loc[records["true_person_id"] == "P0000", SHOW_COLS].to_string(index=False))
    print()
    print(records.loc[records["true_person_id"].isin(["P0390", "P0391"]), SHOW_COLS].to_string(index=False))
    print()
    print(records.loc[records["true_person_id"].isin(["P0396", "P0397"]), SHOW_COLS].to_string(index=False))
    print_kind_samples(records, "clean")
    print_kind_samples(records, "delete or insert")
    print_kind_samples(records, "transposition")
    print_kind_samples(records, "phonetic substitution")
    print_kind_samples(records, "phonetic name variant")
    print_kind_samples(records, "missing field")
    print()
    print(records[SHOW_COLS].head(12).to_string(index=False))


def generate_and_save():
    rng = np.random.default_rng(42)
    fake = Faker("en_US")
    Faker.seed(42)
    people, records, issues = make_people_and_records(rng, fake)
    save_tables(people, records)
    return people, records, issues


if __name__ == "__main__":
    people, records, issues = generate_and_save()
    print_generation(people, records, issues)
