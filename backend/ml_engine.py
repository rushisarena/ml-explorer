"""
============================================================
ML Engine — Core machine learning logic
============================================================
Handles data processing, model training, evaluation,
and SHAP-based explainability.
============================================================
"""

import io
import json
import warnings
import numpy as np
import pandas as pd
from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error,
)

warnings.filterwarnings("ignore")


# ============================================================
# DATA PROCESSOR
# ============================================================
class DataProcessor:
    """Handles data loading, profiling, and preprocessing."""

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.original_df: pd.DataFrame | None = None
        self.target_col: str | None = None
        self.task_type: str | None = None  # 'classification' or 'regression'
        self.feature_names: list[str] = []
        self.preprocessor: ColumnTransformer | None = None

    def load_csv(self, file_content: bytes) -> dict:
        """Load CSV and return schema + preview."""
        self.df = pd.read_csv(io.BytesIO(file_content))
        self.original_df = self.df.copy()

        schema = {
            "columns": [],
            "shape": {"rows": len(self.df), "cols": len(self.df.columns)},
            "preview": json.loads(self.df.head(10).to_json(orient="records")),
        }

        for col in self.df.columns:
            col_info = {
                "name": col,
                "dtype": str(self.df[col].dtype),
                "missing": int(self.df[col].isnull().sum()),
                "missing_pct": round(self.df[col].isnull().mean() * 100, 1),
                "unique": int(self.df[col].nunique()),
            }

            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_info["type"] = "numeric"
                col_info["min"] = float(self.df[col].min()) if not self.df[col].isnull().all() else None
                col_info["max"] = float(self.df[col].max()) if not self.df[col].isnull().all() else None
                col_info["mean"] = float(self.df[col].mean()) if not self.df[col].isnull().all() else None
                col_info["std"] = float(self.df[col].std()) if not self.df[col].isnull().all() else None
            else:
                col_info["type"] = "categorical"
                top_vals = self.df[col].value_counts().head(5).to_dict()
                col_info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

            schema["columns"].append(col_info)

        return schema

    def generate_eda(self) -> dict:
        """Auto-generate exploratory data analysis."""
        if self.df is None:
            raise ValueError("No data loaded")

        eda = {
            "summary_stats": {},
            "distributions": {},
            "correlations": None,
            "missing_summary": {},
        }

        # Summary statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats_df = self.df[numeric_cols].describe().round(3)
            eda["summary_stats"] = json.loads(stats_df.to_json())

        # Distribution data for charts
        for col in numeric_cols[:12]:  # Limit to 12 charts
            values = self.df[col].dropna().values.tolist()
            if len(values) > 0:
                hist_values, bin_edges = np.histogram(values, bins=min(30, len(set(values))))
                eda["distributions"][col] = {
                    "values": hist_values.tolist(),
                    "bins": [round(b, 4) for b in bin_edges.tolist()],
                    "type": "numeric",
                }

        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols[:6]:
            counts = self.df[col].value_counts().head(10)
            eda["distributions"][col] = {
                "labels": counts.index.tolist(),
                "values": counts.values.tolist(),
                "type": "categorical",
            }

        # Correlation matrix
        if len(numeric_cols) >= 2:
            corr = self.df[numeric_cols].corr().round(3)
            eda["correlations"] = {
                "columns": corr.columns.tolist(),
                "values": corr.values.tolist(),
            }

        # Missing data summary
        missing = self.df.isnull().sum()
        eda["missing_summary"] = {
            col: {"count": int(missing[col]), "pct": round(missing[col] / len(self.df) * 100, 1)}
            for col in self.df.columns if missing[col] > 0
        }

        return eda

    def prepare_data(self, target_col: str, test_size: float = 0.2):
        """Prepare features and target for model training."""
        if self.df is None:
            raise ValueError("No data loaded")

        self.target_col = target_col
        df = self.df.copy()

        # Drop columns with too many missing values (>50%)
        threshold = 0.5
        valid_cols = [c for c in df.columns if df[c].isnull().mean() < threshold]
        df = df[valid_cols]

        # Drop rows with missing target
        df = df.dropna(subset=[target_col])

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Detect task type
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            self.task_type = "regression"
        else:
            self.task_type = "classification"
            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), name=target_col)
                self._label_encoder = le
            else:
                self._label_encoder = None

        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Drop high-cardinality categoricals (>20 unique)
        categorical_features = [c for c in categorical_features if X[c].nunique() <= 20]

        # Drop non-useful columns
        drop_cols = [c for c in X.columns if c not in numeric_features and c not in categorical_features]
        X = X.drop(columns=drop_cols, errors="ignore")

        self.feature_names = numeric_features + categorical_features

        # Build preprocessor
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X[self.feature_names], y, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test


# ============================================================
# MODEL TRAINER
# ============================================================
class ModelTrainer:
    """Train and evaluate multiple ML models."""

    CLASSIFICATION_MODELS = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
            verbosity=0,
        ),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "SVM": lambda: SVC(probability=True, random_state=42, kernel="rbf"),
    }

    REGRESSION_MODELS = {
        "Linear Regression": lambda: LinearRegression(),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0,
        ),
        "KNN": lambda: KNeighborsRegressor(n_neighbors=5),
        "SVM": lambda: SVR(kernel="rbf"),
    }

    def __init__(self, preprocessor, task_type: str):
        self.preprocessor = preprocessor
        self.task_type = task_type
        self.trained_models: dict[str, Pipeline] = {}
        self.results: dict[str, dict] = {}
        self.best_model_name: str | None = None

    def train_all(self, X_train, X_test, y_train, y_test, selected_models: list[str] | None = None):
        """Train all selected models and return results."""
        model_registry = (
            self.CLASSIFICATION_MODELS if self.task_type == "classification"
            else self.REGRESSION_MODELS
        )

        if selected_models:
            model_registry = {k: v for k, v in model_registry.items() if k in selected_models}

        best_score = -np.inf

        for name, model_fn in model_registry.items():
            try:
                model = model_fn()
                pipe = Pipeline(steps=[
                    ("preprocessor", self.preprocessor),
                    ("model", model),
                ])

                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                self.trained_models[name] = pipe

                if self.task_type == "classification":
                    metrics = self._classification_metrics(y_test, y_pred, pipe, X_test)
                    score = metrics.get("f1", 0)
                else:
                    metrics = self._regression_metrics(y_test, y_pred)
                    score = metrics.get("r2", 0)

                self.results[name] = metrics

                if score > best_score:
                    best_score = score
                    self.best_model_name = name

            except Exception as e:
                self.results[name] = {"error": str(e)}

        return {
            "task_type": self.task_type,
            "models": self.results,
            "best_model": self.best_model_name,
        }

    def _classification_metrics(self, y_true, y_pred, pipe, X_test) -> dict:
        """Compute classification metrics."""
        n_classes = len(set(y_true))
        avg = "binary" if n_classes == 2 else "weighted"

        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        }

        # AUC-ROC
        try:
            if hasattr(pipe, "predict_proba"):
                y_proba = pipe.predict_proba(X_test)
                if n_classes == 2:
                    metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
                else:
                    metrics["auc_roc"] = round(float(roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )), 4)
        except Exception:
            metrics["auc_roc"] = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics

    def _regression_metrics(self, y_true, y_pred) -> dict:
        """Compute regression metrics."""
        return {
            "r2": round(float(r2_score(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "mape": round(float(np.mean(np.abs(
                (y_true - y_pred) / np.where(y_true == 0, 1, y_true)
            )) * 100), 2),
        }

    def predict(self, X) -> np.ndarray:
        """Generate predictions using the best model."""
        if self.best_model_name is None:
            raise ValueError("No models trained yet")
        pipe = self.trained_models[self.best_model_name]
        return pipe.predict(X)


# ============================================================
# MODEL EXPLAINER
# ============================================================
class ModelExplainer:
    """SHAP-based model explainability."""

    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer

    def explain(self, X_test, max_samples: int = 100) -> dict:
        """Generate SHAP explanations for the best model."""
        import shap

        if self.trainer.best_model_name is None:
            raise ValueError("No models trained")

        pipe = self.trainer.trained_models[self.trainer.best_model_name]
        preprocessor = pipe.named_steps["preprocessor"]
        model = pipe.named_steps["model"]

        # Get transformed feature names
        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feature_names = None

        # Transform test data
        X_transformed = preprocessor.transform(X_test)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # Subsample for speed
        n_samples = min(max_samples, X_transformed.shape[0])
        indices = np.random.RandomState(42).choice(
            X_transformed.shape[0], n_samples, replace=False
        )
        X_sample = X_transformed[indices]

        # Choose SHAP explainer based on model type
        tree_models = (
            RandomForestClassifier, RandomForestRegressor,
            XGBClassifier, XGBRegressor,
        )

        try:
            if isinstance(model, tree_models):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            else:
                # KernelExplainer for non-tree models — use background sample
                bg = shap.kmeans(X_transformed, min(10, X_transformed.shape[0]))
                explainer = shap.KernelExplainer(model.predict, bg)
                shap_values = explainer.shap_values(X_sample, nsamples=50)
        except Exception as e:
            return {"error": f"SHAP computation failed: {str(e)}"}

        # Handle multi-output SHAP for classification
        if isinstance(shap_values, list):
            # For binary classification, take class 1
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        # Feature importance (mean absolute SHAP)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        if feature_names and len(feature_names) == len(mean_abs_shap):
            names = feature_names
        else:
            names = [f"feature_{i}" for i in range(len(mean_abs_shap))]

        # Sort by importance
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        top_n = min(15, len(sorted_idx))

        feature_importance = [
            {"feature": str(names[i]), "importance": round(float(mean_abs_shap[i]), 5)}
            for i in sorted_idx[:top_n]
        ]

        # SHAP values for individual predictions (first 5)
        individual_explanations = []
        for idx in range(min(5, shap_values.shape[0])):
            row_shap = shap_values[idx]
            sorted_row = np.argsort(np.abs(row_shap))[::-1][:10]
            individual_explanations.append({
                "sample_index": int(indices[idx]),
                "contributions": [
                    {
                        "feature": str(names[j]) if j < len(names) else f"feature_{j}",
                        "shap_value": round(float(row_shap[j]), 5),
                    }
                    for j in sorted_row
                ],
            })

        return {
            "model_name": self.trainer.best_model_name,
            "feature_importance": feature_importance,
            "individual_explanations": individual_explanations,
            "n_samples_explained": n_samples,
        }
