# B5 verifier evaluation

Samples: 45,000; candidates: 135,000; binary decisions: 405,000.

Postprocessing of existing B5 probabilities. No new inference or training.

| Head | Precision | Recall | F1 | AUROC | Accuracy |
|---|---:|---:|---:|---:|---:|
| preserve | 0.920029 | 0.987733 | 0.952680 | 0.973329 | 0.934585 |
| edit | 0.851234 | 0.863000 | 0.857077 | 0.891426 | 0.808119 |
| violation | 0.834174 | 0.879933 | 0.856443 | 0.884455 | 0.803341 |
| micro | 0.869012 | 0.910222 | 0.889140 | 0.924962 | 0.848681 |
| macro | 0.868479 | 0.910222 | 0.888733 | 0.916403 | 0.848681 |

## Definitions and provenance

Predicted positive when p >= 0.5. Undefined precision/recall/F1 use zero. AUROC uses probabilities with half credit for ties.
Labels (preserve, edit, violation): positive=(1,1,0), CF-A=(1,0,1), CF-B=(0,1,1).
Each head evaluates 135,000 candidates. Micro pools 405,000 binary decisions; macro averages three heads.
Confusion matrices: rows = actual 0/1; columns = predicted 0/1.

Risk–Coverage: confidence=max(p,1-p); risk=classification errors/retained decisions. Whole confidence ties are retained together. Zero coverage has undefined risk and is omitted. This is verifier classification risk, not gallery retrieval risk.

## Limitations

Labels come from the existing pseudo-edit protocol. Existing label noise and train/validation overlap are not repaired by this export.
Evaluation provenance, image paths and checkpoint/manifest hashes are recorded in verifier_metrics.json. This is TRAIN-set evaluation, not held-out validation.
The historical per_query_predictions.json placeholder is not overwritten. This export covers the stored three-candidate counterfactual evaluation, not CIRR/Fashion-IQ gallery rankings.
