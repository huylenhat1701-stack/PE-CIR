# B5 verifier evaluation

Samples: 5,000; candidates: 15,000; binary decisions: 45,000.

Postprocessing of existing B5 probabilities. No new inference or training.

| Head | Precision | Recall | F1 | AUROC | Accuracy |
|---|---:|---:|---:|---:|---:|
| preserve | 0.916755 | 0.950400 | 0.933274 | 0.956352 | 0.909400 |
| edit | 0.763859 | 0.843300 | 0.801616 | 0.791563 | 0.721733 |
| violation | 0.731312 | 0.902000 | 0.807737 | 0.781804 | 0.713733 |
| micro | 0.798939 | 0.898567 | 0.845829 | 0.863286 | 0.781622 |
| macro | 0.803975 | 0.898567 | 0.847542 | 0.843240 | 0.781622 |

## Definitions and provenance

Predicted positive when p >= 0.5. Undefined precision/recall/F1 use zero. AUROC uses probabilities with half credit for ties.
Labels (preserve, edit, violation): positive=(1,1,0), CF-A=(1,0,1), CF-B=(0,1,1).
Each head evaluates 15,000 candidates. Micro pools 45,000 binary decisions; macro averages three heads.
Confusion matrices: rows = actual 0/1; columns = predicted 0/1.

Risk–Coverage: confidence=max(p,1-p); risk=classification errors/retained decisions. Whole confidence ties are retained together. Zero coverage has undefined risk and is omitted. This is verifier classification risk, not gallery retrieval risk.

## Limitations

Labels come from the existing pseudo-edit protocol. Existing label noise and train/validation overlap are not repaired by this export.
The original scores omit image IDs, evaluation manifest identity, and inference-time checkpoint hash. Row indices identify source positions only. Current checkpoint/source hashes are recorded for traceability, not proof of historical linkage.
The historical per_query_predictions.json placeholder is not overwritten. This export covers the stored three-candidate counterfactual evaluation, not CIRR/Fashion-IQ gallery rankings.
