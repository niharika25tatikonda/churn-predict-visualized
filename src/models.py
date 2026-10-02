"""Model training, evaluation and comparison."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import NUMERIC_COLS

CATEGORICAL_COLS = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                    "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS
TARGET = "ChurnFlag"


def _preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


def _models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        max_depth=3, random_state=42),
    }


def train_and_compare(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    results, pipes, curves, cms, importances, cv_scores = [], {}, {}, {}, {}, {}
    for name, clf in _models().items():
        pipe = Pipeline([("prep", _preprocessor()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        # Gradient Boosting has no class_weight -> use a lower threshold so recall is comparable
        if name == "Gradient Boosting":
            pred = (proba >= 0.35).astype(int)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred),
            "Recall": recall_score(y_te, pred),
            "F1-Score": f1_score(y_te, pred),
            "ROC-AUC": roc_auc_score(y_te, proba),
        })
        cv_scores[name] = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
        fpr, tpr, _ = roc_curve(y_te, proba)
        curves[name] = (fpr, tpr)
        cms[name] = confusion_matrix(y_te, pred)
        names = pipe.named_steps["prep"].get_feature_names_out()
        names = [n.split("__", 1)[1] for n in names]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        else:
            imp = np.abs(clf.coef_[0])
        importances[name] = pd.Series(imp, index=names).sort_values(ascending=False)
        pipes[name] = pipe

    res = pd.DataFrame(results).set_index("Model")
    return {"results": res, "pipes": pipes, "curves": curves, "cms": cms,
            "importances": importances, "cv": cv_scores,
            "thresholds": {"Logistic Regression": 0.5, "Random Forest": 0.5, "Gradient Boosting": 0.35},
            "test_size": len(y_te), "train_size": len(y_tr)}


def predict_one(pipe, row: dict, threshold: float = 0.5):
    proba = float(pipe.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
    return proba, int(proba >= threshold)
