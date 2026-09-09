# Synthetic data to train ML models

## 1. Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## 2. Open the model

Open `matching_resolver_model.py`. Do not run it.

## 3. Generate the table

```bash
python synthetic_data_generation.py
```

## 4. Train

```bash
python train_matching_resolver.py
```

## 5. Try two records

```bash
python try_match.py
```
