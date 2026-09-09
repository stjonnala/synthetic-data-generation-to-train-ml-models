from pathlib import Path

import pandas as pd
from joblib import dump, load
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.ensemble import GradientBoostingClassifier

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "match_resolver.joblib"

FEATURE_COLS = ["name_jaro", "addr_lev", "dob_exact", "phonetic_match", "year_gap"]


def soundex(name):
    letters = "".join(ch for ch in str(name).upper() if ch.isalpha())
    if not letters:
        return "0000"
    codes = {
        "BFPV": "1",
        "CGJKQSXZ": "2",
        "DT": "3",
        "L": "4",
        "MN": "5",
        "R": "6",
    }

    def digit(ch):
        for group, d in codes.items():
            if ch in group:
                return d
        return "0"

    out = [letters[0]]
    prev = digit(letters[0])
    for ch in letters[1:]:
        d = digit(ch)
        if d != "0" and d != prev:
            out.append(d)
        if d != "0":
            prev = d
        if len(out) == 4:
            break
    return ("".join(out) + "0000")[:4]


def pair_features(row_a, row_b):
    name_a = str(row_a.get("first")) + " " + str(row_a.get("last"))
    name_b = str(row_b.get("first")) + " " + str(row_b.get("last"))
    addr_a = str(row_a.get("address", ""))
    addr_b = str(row_b.get("address", ""))
    longer = max(len(addr_a), len(addr_b), 1)
    edits = Levenshtein.distance(addr_a, addr_b)

    dob_a = str(row_a.get("dob"))
    dob_b = str(row_b.get("dob"))
    year_a = dob_a[0:4]
    year_b = dob_b[0:4]
    if year_a.isdigit() and year_b.isdigit():
        year_gap = abs(int(year_a) - int(year_b)) / 100.0
    else:
        year_gap = 1.0

    if dob_a == dob_b:
        same_dob = 1.0
    else:
        same_dob = 0.0

    first_ok = soundex(row_a.get("first")) == soundex(row_b.get("first"))
    last_ok = soundex(row_a.get("last")) == soundex(row_b.get("last"))
    if first_ok and last_ok:
        same_sound = 1.0
    else:
        same_sound = 0.0

    out = {}
    out["name_jaro"] = JaroWinkler.similarity(name_a, name_b)
    out["addr_lev"] = 1.0 - edits / longer
    out["dob_exact"] = same_dob
    out["phonetic_match"] = same_sound
    out["year_gap"] = year_gap
    return out


def build_pairs(records):
    work = records.copy()
    work["block"] = work["dob"].astype(str).str[:4] + "|" + work["last"].str[:1].str.upper()
    work["house"] = work["last"].str.lower() + "|" + work["address"].astype(str).str.lower()

    pair_rows = []
    seen = set()

    def add_group(idx):
        if len(idx) < 2:
            return
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a = work.loc[idx[i]]
                b = work.loc[idx[j]]
                key = tuple(sorted([a["record_id"], b["record_id"]]))
                if key in seen:
                    continue
                seen.add(key)
                feats = pair_features(a, b)
                feats["label"] = int(a["true_person_id"] == b["true_person_id"])
                feats["id_a"] = a["true_person_id"]
                feats["id_b"] = b["true_person_id"]
                feats["record_a"] = a["record_id"]
                feats["record_b"] = b["record_id"]
                feats["name_a"] = f"{a['first']} {a['last']}"
                feats["name_b"] = f"{b['first']} {b['last']}"
                pair_rows.append(feats)

    for _, g in work.groupby("block"):
        add_group(g.index.to_list())
    for _, g in work.groupby("house"):
        add_group(g.index.to_list())

    return pd.DataFrame(pair_rows)


class MatchingResolverModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(random_state=42)

    def fit(self, train_pairs):
        x_train = train_pairs[FEATURE_COLS]
        y_train = train_pairs["label"]
        self.model.fit(x_train, y_train)
        return self

    def predict(self, pairs):
        return self.model.predict(pairs[FEATURE_COLS])

    def predict_proba_match(self, pairs):
        return self.model.predict_proba(pairs[FEATURE_COLS])[:, 1]

    def resolve(self, row_a, row_b):
        feats = pair_features(row_a, row_b)
        x = pd.DataFrame([feats])[FEATURE_COLS]
        pred = int(self.model.predict(x)[0])
        proba = float(self.model.predict_proba(x)[0, 1])
        return {
            "match": pred == 1,
            "probability": proba,
            "features": feats,
        }

    def save(self, path=MODEL_PATH):
        dump(self, path)
        return path

    @staticmethod
    def load(path=MODEL_PATH):
        return load(path)
