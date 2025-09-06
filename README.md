# Placeholder main branch

This branch is intentionally minimal; code lives on feature branches.

## Dependency Parsing EN -> BN (branch `dependency-parsing-en-to-bn`)

Artifacts:
- models/parser.pt - trained SuPar (CRF2o) with XLM-R-base (may be in Releases if LFS unavailable)
- results/metrics.json - EN dev & BN test metrics
- results/learning_curve.csv - parsed learning curve
- results/logs/train_stdout.txt - training log
- results/preds/*.conllu - predictions (pretrain, posttrain, manual)

Scripts:
- scripts/bn_manual_parse.py - parse Bengali sentences with the saved model
- scripts/check_dep_scores.py - DEPREL confusion & correlation plots
