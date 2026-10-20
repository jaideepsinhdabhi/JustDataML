# Feature selection

Set `features: auto` (or leave `features` out) to let JDML choose the columns.

1. All columns except the target are candidates.
2. Constant columns and ID-like columns (text or whole-number columns where every row has its own value) are removed.
3. The steps in `feature_selection` run in order on the **training rows only**.

| Step | What it does |
|---|---|
| `variance` | Drops columns where one value fills 99% of the rows |
| `correlation` | Drops a numeric column correlated above `correlation_threshold` with one already kept |
| `importance` | Random forest importance; keeps the top `top_k`, or the columns covering 95% of the importance |
| `mutual_info` | Same, using mutual information with the target |
| `rfe` | Recursive feature elimination down to `top_k` (default: half) |

Results: `Output/selected_features.csv` (score, kept, reason) and `artifact/features.json`. Prediction reads `features.json`, so your new data only needs the selected columns.

Giving an explicit `features` list keeps it as is unless you also set `feature_selection`.
