"""Train the churn pipeline used by apps/04_model_app.py, and save it in models/.

    python train_churn_model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent
CATEGORICAL = ["Int'l Plan", "VMail Plan"]

churn = pd.read_csv(HERE / "data" / "churn.csv")
y = (churn["Churn?"] == "True.").astype(int)
X = churn.drop(columns=["Churn?", "Phone", "Area Code", "State"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline = Pipeline([
    ("preprocessing", ColumnTransformer([("categorical", OneHotEncoder(drop="if_binary"), CATEGORICAL)], remainder="passthrough")),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
]).fit(X_train, y_train)

(HERE / "models").mkdir(exist_ok=True)
joblib.dump(pipeline, HERE / "models" / "churn_pipeline.joblib")
defaults = {column: (X_train[column].mode()[0] if column in CATEGORICAL else float(X_train[column].median())) for column in X.columns}
(HERE / "models" / "churn_defaults.json").write_text(json.dumps(defaults, indent=2))
print(f"Saved models/churn_pipeline.joblib, test ROC AUC {roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]):.3f}")
