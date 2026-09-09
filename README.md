# Synthetic data to train ML models

Generate a labeled identity table, train a small matcher, and try two records. No real patient data. Everything is invented.

https://github.com/stjonnala/synthetic-data-generation-to-train-ml-models

## Files

| File | What it does |
|---|---|
| `matching_resolver_model.py` | The matcher. Two rows in. Five numbers. Trees say MATCH or NO MATCH. Do not run this file by itself. |
| `synthetic_data_generation.py` | Invent people with Faker, inject mess, keep `true_person_id`. Writes `generated_records.csv`. |
| `train_matching_resolver.py` | Train the matcher and score it against exact name plus date of birth. |
| `try_match.py` | Try two records. Prints MATCH or NO MATCH. |

## Run

```bash
git clone https://github.com/stjonnala/synthetic-data-generation-to-train-ml-models.git
cd synthetic-data-generation-to-train-ml-models
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python synthetic_data_generation.py
python train_matching_resolver.py
python try_match.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Walk `matching_resolver_model.py` first. Do not run it. Then generate, train, and try.

`synthetic_data_generation.py` writes 400 people and 910 records, seed 42.  
`train_matching_resolver.py` takes about a minute and a half.

Expected scores on this seed:

- Exact name plus date of birth: precision 0.85, recall 0.33
- Boosted trees: precision 1.00, recall 1.00, F1 1.00, overmatches 0

Those numbers are for this generated table. They are not a published paper table.

## try_match.py menu

1. Same person, extra letter (Angel / Anggel). This seed: MATCH
2. Same person, transposed name and date (Angel / aWlker). This seed: MATCH
3. Two different people. This seed: NO MATCH
4. Close names, same date (Elena / Elisa). This seed: NO MATCH. Close names can still overmatch on another seed.
5. Same name, different year (Robert Hale, father and son). This seed: NO MATCH
6. Type two records yourself

Option 6 is not a lookup. You can type names that are not in the CSV.

The five numbers are Jaro-Winkler (names), Levenshtein (address), exact date of birth, Soundex, and year gap. The classifier is `GradientBoostingClassifier` in scikit-learn.

Saiteja Jonnalagadda  
https://www.linkedin.com/in/saitejajo
