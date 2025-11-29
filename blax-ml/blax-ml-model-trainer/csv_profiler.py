import argparse
import json
from pathlib import Path

import pandas as pd


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
    return parser.parse_args()


def detect_outliers_iqr(col):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return int(((col < lower_bound) | (col > upper_bound)).sum())


def detect_distribution(col):
    skewness = col.skew()

    if abs(skewness) < 0.5:
        return "normal"
    elif skewness > 0.5:
        return "right_skewed"
    else:
        return "left_skewed"


def profile_column(df, col_name):
    col = df[col_name]
    result = {
        "dtype": str(col.dtype),
        "valid_count": int(col.count()),
        "missing_count": int(col.isna().sum()),
        "missing_percentage": round(col.isna().mean() * 100, 2),
    }

    if pd.api.types.is_numeric_dtype(col):
        result["dtype"] = "numeric"
        valid_data = col.dropna()
        if len(valid_data) > 0:
            result["stats"] = {
                "min": float(valid_data.min()),
                "max": float(valid_data.max()),
                "mean": round(float(valid_data.mean()), 2),
                "median": float(valid_data.median()),
                "std": round(float(valid_data.std()), 2),
            }
            result["outliers_count"] = detect_outliers_iqr(valid_data)
            result["distribution"] = detect_distribution(valid_data)
            result["unique_count"] = int(valid_data.nunique())
            result["is_continuous"] = bool(
                valid_data.nunique() > 20
                and valid_data.nunique() / len(valid_data) > 0.05
                and (valid_data.value_counts() == 1).mean() < 0.98
            )
    elif pd.api.types.is_object_dtype(col):
        result["dtype"] = "categorical"
        result["unique_count"] = int(col.nunique())
        result["top_10_values"] = {
            str(k): int(v) for k, v in col.value_counts().head(10).items()
        }

    return result


def profile_correlations(df):
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) < 2:
        return {}

    correlations = {}
    corr_matrix = df[num_cols].corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.4:
                key = f"{col1}_vs_{col2}"
                correlations[key] = round(float(corr_value), 2)

    return correlations


def profile_csv(dataset_path):
    df = pd.read_csv(dataset_path)

    result = {
        "dataset_info": {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
        },
        "columns": {col: profile_column(df, col) for col in df.columns},
        "feature_correlations": profile_correlations(df),
    }

    total_missing_count = int(df.isna().sum().sum())
    columns_with_missing = [col for col in df.columns if df[col].isna().sum() > 0]

    total_outliers_count = 0
    columns_with_outliers = []
    for col in df.select_dtypes(include=["number"]).columns:
        outliers = detect_outliers_iqr(df[col].dropna())
        if outliers > 0:
            total_outliers_count += outliers
            columns_with_outliers.append(col)

    result["dataset_info"]["total_missing_count"] = total_missing_count
    result["dataset_info"]["columns_with_missing"] = columns_with_missing
    result["dataset_info"]["total_outliers_count"] = total_outliers_count
    result["dataset_info"]["columns_with_outliers"] = columns_with_outliers
    return result


if __name__ == "__main__":
    args = _parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_result = profile_csv(dataset_path)
    with open(output_dir / "csv_profile.json", "w") as file:
        json.dump(profile_result, file, indent=2)
