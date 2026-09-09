# Synthetic data generation to train machine learning models

These files generate a labeled identity table, train a small matcher, and let you try two records. No real patient data. Everything is invented.

Repository: https://github.com/stjonnala/synthetic-data-generation-to-train-ml-models

## Run it

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

`synthetic_data_generation.py` writes `generated_records.csv` (400 people, 910 records, seed 42).  
`train_matching_resolver.py` fits boosted trees and prints scores. Training takes about a minute and a half.  
`try_match.py` prints MATCH or NO MATCH for two records.

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

One-shot run of generate, train, and the prepared pairs:

```bash
python run_demo.py
```

## What the files are

| File | What it does |
|---|---|
| `matching_resolver_model.py` | Two rows in. Five numbers. Trees say MATCH or NO MATCH. Do not run this file by itself. |
| `synthetic_data_generation.py` | Invent people, inject mess, keep `true_person_id` |
| `train_matching_resolver.py` | Train and score against exact name plus date of birth |
| `try_match.py` | Interactive MATCH / NO MATCH |
| `run_demo.py` | Generate, train, print the prepared pairs |

The five numbers are Jaro-Winkler (names), Levenshtein (address), exact date of birth, Soundex, and year gap. The classifier is `GradientBoostingClassifier` in scikit-learn.

## Talk

Saiteja Jonnalagadda  
linkedin.com/in/saitejajo  
saiteja.jonnalagadda@ieee.org

## If you are presenting this talk

Open `EXPLAIN.md` on a second screen. The `.py` files stay code-only.

1. Walk `matching_resolver_model.py`. Do not run it.
2. `python synthetic_data_generation.py` then open `generated_records.csv`
3. `python train_matching_resolver.py`
4. `python try_match.py`
