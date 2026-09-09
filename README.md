# Synthetic data to train ML models

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

Open `matching_resolver_model.py` first. Do not run it. Then run generate, train, and try, in that order.
