# Python API

```python
import jdml

model = jdml.fit(df, target="price", plots=True)
model.score            # test score of the best model
model.metric           # name of the score
model.best_model_name
model.features         # columns used
model.validation_scores
model.report()         # Model_Summary as a DataFrame
model.predict(new_df)  # Series
```

`jdml.fit(df, target, problem=None, features="auto", models="ALL", scaler="StandardScaler", tuning=False, metric=None, feature_selection=None, top_k=0, cv_folds=0, class_weight=None, plots=False, report=False, explain="none", test_size=0.2, random_state=14, datetime_columns=None, artifact_dir="artifact", output_dir="Output")`

- `problem=None` guesses Classification for text, boolean or whole-number targets with at most 15 values, otherwise Regression.
- Your DataFrame is never modified.
- `jdml.Model.load("Classification", artifact_dir="artifact")` reopens a model trained earlier.
