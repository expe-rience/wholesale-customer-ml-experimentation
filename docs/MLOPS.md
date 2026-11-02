# Production notes: monitoring and retraining

This repository ships a trained model as a Docker image. This page describes what a production
system would add around it, and which parts are implemented here.

| Concern | Production practice | In this repo |
|---|---|---|
| Reproducibility | Pinned dependencies, versioned image, fixed seeds | `requirements-serve.txt`, image tags, `random_state=42` |
| Release safety | Tests, smoke test, staged rollout, rollback to previous tag | CI tests + `/health` smoke test; rollback by redeploying an older tag |
| Input drift | Compare live feature distributions to training data (PSI / KS) | `python -m src.drift` |
| Prediction monitoring | Log requests, latency, errors, predicted-class mix | Not implemented (future work) |
| Performance monitoring | Join predictions with true labels when known; track macro-F1 | Not implemented (needs labels) |
| Retraining | Scheduled or drift-triggered pipeline; challenger vs champion | `python -m src.benchmark`; automation is future work |
| Model registry | Versioned models with metrics and approval stage | Model file in `models/` (MLflow is future work) |

## Drift check
```bash
python -m src.drift --reference data/multiclass.csv --current new_customers.csv
```
PSI below 0.10 is stable, 0.10-0.25 deserves a look, above 0.25 is a major shift.

## Retraining policy (proposed)
1. Trigger: PSI alert on several features, or macro-F1 on recent labelled data falls below an agreed threshold.
2. Train a challenger with `src.benchmark` using the same CV protocol.
3. Promote only if it beats the current model on cross-validation macro-F1 by more than the fold standard deviation.
4. Release as a new version tag; keep the previous image for rollback.

## Limits
The dataset has 440 rows and the top models are statistically tied, so the selected model should not be
treated as definitively best. Drift checks flag change, not harm; they need human judgement.
