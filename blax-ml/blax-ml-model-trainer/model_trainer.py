import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PredictionErrorDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import (
    SVC,
    SVR,
)

matplotlib.use("Agg")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["regression", "classification"],
        required=True,
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--features",
        default="",
        type=str,
    )
    parser.add_argument(
        "--model_type",
        default="auto",
        type=str,
        choices=[
            "auto",
            "random_forest",
            "svm",
            "linear",
        ],
    )
    parser.add_argument(
        "--n_estimators",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--svm_kernel",
        default="rbf",
        type=str,
        choices=["linear", "poly", "rbf", "sigmoid"],
    )
    parser.add_argument(
        "--missing_strategy",
        default="median",
        type=str,
        choices=["drop", "mean", "median", "most_frequent"],
    )
    parser.add_argument(
        "--remove_outliers",
        action="store_true",
    )
    parser.add_argument(
        "--scaler_type",
        default="standard",
        type=str,
        choices=["none", "standard", "minmax", "robust"],
    )
    parser.add_argument(
        "--test_size",
        default=0.2,
        type=float,
    )
    parser.add_argument(
        "--random_state",
        default=42,
        type=int,
    )
    return parser.parse_args()


def _select_x_y(df, target, features):
    cols = {col.lower(): col for col in df.columns}

    target_col = cols.get(target.strip().lower())
    if not target_col:
        raise ValueError(f"Target column '{target}' not found in dataset")

    if not features:
        feature_cols = [col for col in df.columns if col != target_col]
    else:
        feature_cols = []
        missing = []
        for feature in features.split(","):
            feature = feature.strip()
            if not feature:
                continue
            col = cols.get(feature.lower())
            if col:
                feature_cols.append(col)
            else:
                missing.append(feature)
        if missing:
            raise ValueError(f"Feature column(s) '{missing}' not found in dataset")

    return df[feature_cols].copy(), df[target_col].copy()


def _drop_y_missing(X, y):
    mask = y.notna()
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def _drop_x_missing(X, y):
    mask = X.notna().all(axis=1)
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def _handle_outliers(X, y):
    num_cols = X.select_dtypes(include=["number"]).columns

    if len(num_cols) == 0:
        return X, y

    X_numeric = X[num_cols]
    q1 = X_numeric.quantile(0.25)
    q3 = X_numeric.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = ~((X_numeric < lower_bound) | (X_numeric > upper_bound)).any(axis=1)
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def _identify_column_types(X):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


def _build_preprocessor(num_cols, cat_cols, missing_strategy, scaler_type):
    transformers = []

    if num_cols:
        numeric_pipeline = _build_numeric_pipeline(missing_strategy, scaler_type)
        transformers.append(("num", numeric_pipeline, num_cols))

    if cat_cols:
        categorical_pipeline = _build_categorical_pipeline()
        transformers.append(("cat", categorical_pipeline, cat_cols))

    return ColumnTransformer(transformers)


def _build_numeric_pipeline(missing_strategy, scaler_type):
    steps = []

    strategy = missing_strategy if missing_strategy != "drop" else "median"
    imputer = SimpleImputer(strategy=strategy)
    steps.append(("imputer", imputer))

    if scaler_type != "none":
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        scaler = scalers[scaler_type]
        steps.append(("scaler", scaler))

    return Pipeline(steps)


def _build_categorical_pipeline():
    steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return Pipeline(steps)


def _select_model(model_type, task_type, n_estimators, svm_kernel, random_state):
    models = {
        "auto": {
            "regression": RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
            ),
            "classification": RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
            ),
        },
        "random_forest": {
            "regression": RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
            ),
            "classification": RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
            ),
        },
        "svm": {
            "regression": SVR(kernel=svm_kernel),
            "classification": SVC(
                kernel=svm_kernel,
                probability=True,
                random_state=random_state,
            ),
        },
        "linear": {
            "regression": LinearRegression(),
            "classification": LogisticRegression(
                random_state=random_state,
                max_iter=1000,
            ),
        },
    }
    return models[model_type][task_type]


def _compute_eval_metrics(y_test, y_pred, task_type):
    if task_type == "regression":
        return {
            "r2_score": round(
                r2_score(y_test, y_pred),
                4,
            ),
            "mean_absolute_error": round(
                mean_absolute_error(y_test, y_pred),
                4,
            ),
            "mean_squared_error": round(
                mean_squared_error(y_test, y_pred),
                4,
            ),
            "root_mean_squared_error": round(
                root_mean_squared_error(y_test, y_pred),
                4,
            ),
        }
    elif task_type == "classification":
        return {
            "accuracy": round(
                accuracy_score(y_test, y_pred) * 100,
                2,
            ),
            "precision": round(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
                * 100,
                2,
            ),
            "recall": round(
                recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100,
                2,
            ),
            "f1_score": round(
                f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100,
                2,
            ),
        }


def _save_plots(y_true, y_pred, task_type, output_dir):
    if task_type == "regression":
        PredictionErrorDisplay.from_predictions(y_true, y_pred)
        plt.tight_layout()
        plt.savefig(output_dir / "prediction_error.png", dpi=300)
        plt.close()
    elif task_type == "classification":
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
        plt.close()


def _save_pipeline_metadata(args, df, X_train, metrics):
    features = []
    for col in X_train.columns:
        feature = {"name": col}
        if pd.api.types.is_numeric_dtype(X_train[col]):
            feature["type"] = "numeric"
            feature["min_value"] = float(X_train[col].min())
            feature["max_value"] = float(X_train[col].max())
        else:
            feature["type"] = "categorical"
            if col in df.columns:
                possible_values = df[col].dropna().unique().tolist()
                feature["possible_values"] = possible_values
                feature["num_possible_values"] = len(possible_values)
            else:
                feature["possible_values"] = []
                feature["num_possible_values"] = 0
        features.append(feature)

    metadata = {
        "task_type": args.task_type,
        "target_column": args.target,
        "features": features,
        "metrics": metrics,
        "args": {
            "dataset_path": args.dataset_path,
            "output_dir": args.output_dir,
            "features": args.features,
            "target": args.target,
            "task_type": args.task_type,
            "model_type": args.model_type,
            "n_estimators": args.n_estimators,
            "svm_kernel": args.svm_kernel,
            "missing_strategy": args.missing_strategy,
            "remove_outliers": args.remove_outliers,
            "scaler_type": args.scaler_type,
            "test_size": args.test_size,
            "random_state": args.random_state,
        },
    }
    with open(Path(args.output_dir) / "pipeline_meta.json", "w") as file:
        json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    args = _parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)

    X, y = _select_x_y(df, args.target, args.features)

    X, y = _drop_y_missing(X, y)

    if args.missing_strategy == "drop":
        X, y = _drop_x_missing(X, y)

    if args.remove_outliers:
        X, y = _handle_outliers(X, y)

    if args.task_type == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Regression task requires the target column to be numeric")

    if args.task_type == "classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least 2 unique classes")

    num_cols, cat_cols = _identify_column_types(X)

    preprocessor = _build_preprocessor(
        num_cols,
        cat_cols,
        args.missing_strategy,
        args.scaler_type,
    )
    model = _select_model(
        args.model_type,
        args.task_type,
        args.n_estimators,
        args.svm_kernel,
        args.random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
        if args.task_type == "classification" and y.value_counts().min() >= 2
        else None,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = _compute_eval_metrics(y_test, y_pred, args.task_type)
    _save_plots(y_test, y_pred, args.task_type, output_dir)

    joblib.dump(pipeline, output_dir / "pipeline.joblib")
    _save_pipeline_metadata(args, df, X_train, metrics)
