# Wholesale Customer ML Experimentation

A reproducible multi-algorithm classification project built around the same Wholesale Customers dataset used in the KNN learning notebook.

## Dataset contract

Place `multiclass.csv` in `data/`.

- Features: `Region`, `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen`
- Target: `class`

The raw dataset is excluded from Git by default. Confirm its license and redistribution permissions before publishing it.

## Models

Dummy baseline, Logistic Regression, KNN, Decision Tree, Bagging, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, plus optional XGBoost and LightGBM.

## Evaluation protocol

- Stratified 75/25 train/test split with fixed random seed.
- Stratified cross-validation on the training partition.
- Imputation and encoding occur inside sklearn pipelines to avoid leakage.
- Numeric features are scaled for distance/linear models; tree models use unscaled numeric values.
- `Region` is one-hot encoded because its integer code should not automatically be interpreted as a continuous distance.
- Model selection uses mean cross-validation macro-F1. The held-out test set is used only for final reporting.

Macro-F1 is the primary metric because it weights each class equally. Also inspect balanced accuracy, per-class recall, confusion matrix, runtime and model complexity. No algorithm is assumed to be the winner.

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional boosting libraries:

```bash
pip install -r requirements-boosting.txt
```

Run the benchmark:

```bash
python -m src.benchmark --data data/multiclass.csv --output artifacts
```

Outputs include model ranking, selected model pipeline, final summary and confusion matrix CSV.

## Run the API after training

```bash
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Quick start (Docker)

Anyone can run the trained model without installing Python:

```bash
docker run -p 8000:8000 ghcr.io/expe-rience/wholesale-customer-ml-experimentation:v0.1.0
```

Open http://localhost:8000/docs and try `POST /predict`:

```json
{"Region": 3, "Fresh": 12669, "Milk": 9656, "Grocery": 7561,
 "Frozen": 214, "Detergents_Paper": 2674, "Delicassen": 1338}
```

Response: `{"predicted_class": "...", "class_probabilities": {...}}`

Build it yourself:

```bash
docker build -t wholesale-ml .
docker run -p 8000:8000 wholesale-ml
```

The image serves the model stored in `models/` (trained with scikit-learn 1.9.1, pinned in `requirements-serve.txt` because pickled models must be loaded with the same version). Override the model with the `MODEL_PATH` environment variable.

## How this maps to production

- Stateless container: the model loads at startup, so you scale by running more copies and roll back by redeploying an older image tag.
- CI/CD: pull request -> tests -> merge -> tag `vX.Y.Z` -> GitHub Actions builds and publishes the image to GHCR.
- A production system would add a model registry (for example MLflow), authentication, request logging, drift and accuracy monitoring, and canary releases.

This is a demonstration service on a small public dataset (440 rows), not a production system.

## Git workflow

Use feature branches and pull requests. Run tests before pushing:

```bash
pytest -q
git status
git add README.md requirements.txt requirements-boosting.txt src tests .github .gitignore data/README.md artifacts/README.md
git commit -m "feat: add wholesale customer model benchmark"
git push -u origin feature/model-benchmark
```

Do not commit raw/private data, credentials, `.env`, virtual environments or generated model files by default.

## Next experiments

1. Tune top candidates using RandomizedSearchCV with the same CV strategy.
2. Add repeated/nested CV if performance estimates need greater confidence.
3. Add class-wise error analysis and SHAP for appropriate models.
4. Track runs and parameters with MLflow.
5. Add API tests, Docker and CI image build.
6. Add data/prediction drift checks before any production claim.
