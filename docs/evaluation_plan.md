# Evaluation plan (draft: edit to match your report)

| # | Evaluation | Question | Method | Metric |
|---|---|---|---|---|
| 1 | Accuracy vs baselines | Does ML beat mean and linear baselines? | Spatial group CV | MAE, RMSE, R² |
| 2 | Spatial generalization | Does it work on unseen neighborhoods? | Hold out whole units | MAE, RMSE, R² |
| 3 | Temporal generalization | Does it work on unseen dates/years? | Hold out a year | MAE, RMSE, R² |
| 4 | Stakeholder / usability | Is it useful beyond Tree Equity Score and Cal-Heat? | Interviews + task walkthrough | Qualitative notes, task success |

Also log: leakage checks (src/coolblocks/evaluation/leakage.py), failure analysis (worst units/dates), feature importance.
