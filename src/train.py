import os
import sys
import argparse
from pathlib import Path

# Ensure Project root is importable whether invoked as `python src/train.py` or `python -m src.train`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import get_feature_transformer, build_preprocessor


def resolve_dataset_path(data_path: str) -> Path:
    """Auto-discovers used_cars.csv across common project directory layouts."""
    candidates = [
        Path(data_path),
        Path("data/raw/used_cars.csv"),
        Path("data/used_cars.csv"),
        Path("notebooks/used_cars.csv"),
        Path("../used_cars.csv"),
    ]
    resolved = next((p for p in candidates if p.exists()), None)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not locate dataset. Searched: {[str(p.resolve()) for p in candidates]}"
        )
    return resolved


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage outlier filtering:
    1. Global 99th-percentile cap on price and mileage.
    2. Within-cohort (brand, age_bucket) 1.5 * IQR filter in log-price space.
    """
    price_cap = df['price'].quantile(0.99)
    milage_cap = df['milage'].quantile(0.99)
    df_filtered = df[(df['price'] <= price_cap) & (df['milage'] <= milage_cap)].copy()

    df_filtered['log_price'] = np.log1p(df_filtered['price'])
    df_filtered['age_bucket'] = pd.cut(
        2024 - df_filtered['model_year'],
        bins=[-1, 2, 5, 10, 15, 25, 100]
    )

    q1 = df_filtered.groupby(['brand', 'age_bucket'], observed=False)['log_price'].transform(
        lambda x: x.quantile(0.25)
    )
    q3 = df_filtered.groupby(['brand', 'age_bucket'], observed=False)['log_price'].transform(
        lambda x: x.quantile(0.75)
    )
    iqr = q3 - q1

    is_outlier = (df_filtered['log_price'] < (q1 - 1.5 * iqr)) | (
        df_filtered['log_price'] > (q3 + 1.5 * iqr)
    )
    df_clean = df_filtered[~is_outlier].drop(columns=['log_price', 'age_bucket']).copy()
    print(f"Raw rows: {len(df):,} | After 99% cap: {len(df_filtered):,} | After IQR clean: {len(df_clean):,}")
    return df_clean


def build_vehicle_catalog(df_clean: pd.DataFrame) -> dict:
    """Extracts cascading Brand -> Model lists and default specs for the UI/API."""
    brands = df_clean['brand'].value_counts().index.tolist()
    brand_to_models = (
        df_clean.groupby('brand')['model']
        .apply(lambda s: s.value_counts().head(50).index.tolist())
        .to_dict()
    )
    model_defaults_raw = (
        df_clean.groupby(['brand', 'model'])[['engine', 'transmission', 'fuel_type']]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else 'Gasoline')
        .to_dict(orient='index')
    )
    # Flatten tuple keys to "Brand||Model" for clean serialization
    model_defaults = {
        f"{brand}||{model}": specs for (brand, model), specs in model_defaults_raw.items()
    }
    return {
        'brands': brands,
        'brand_to_models': brand_to_models,
        'model_defaults': model_defaults
    }


def train_and_evaluate(data_path: str, output_path: str) -> None:
    resolved_csv = resolve_dataset_path(data_path)
    print(f"1. Loading dataset from '{resolved_csv}'...")
    df = pd.read_csv(resolved_csv)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    df_clean = filter_outliers(df)
    catalog = build_vehicle_catalog(df_clean)

    X = df_clean.drop(columns=['price'])
    y = np.log1p(df_clean['price'])

    # 80% Train+Cal / 20% Test, then split 10% of Train+Cal for Conformal Calibration
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.10, random_state=42
    )

    # 2. Fit Point Estimation Pipeline on Full Training Split
    print("2. Training point-estimate HistGradientBoostingRegressor pipeline...")
    point_pipeline = Pipeline(steps=[
        ('feature_extraction', get_feature_transformer()),
        ('preprocessor', build_preprocessor()),
        ('regressor', HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.03,
            max_leaf_nodes=63,
            min_samples_leaf=50,
            l2_regularization=5.0,
            random_state=42
        ))
    ])
    point_pipeline.fit(X_train_full, y_train_full)

    train_pred_log = point_pipeline.predict(X_train_full)
    smearing_factor = float(np.mean(np.exp(y_train_full - train_pred_log)))

    y_pred_log = point_pipeline.predict(X_test)
    y_test_actual = np.expm1(y_test)
    y_pred_median = np.expm1(y_pred_log)
    y_pred_mean = np.exp(y_pred_log) * smearing_factor - 1

    print("\n--- Held-Out Test Set Point Performance ---")
    print(f"Log Space R2:                {r2_score(y_test, y_pred_log):.4f}")
    print(f"Dollar R2 (Naive expm1):     {r2_score(y_test_actual, y_pred_median):.4f}")
    print(f"Dollar R2 (Duan Smeared):    {r2_score(y_test_actual, y_pred_mean):.4f}")
    print(f"Dollar MAE (Naive expm1):    ${mean_absolute_error(y_test_actual, y_pred_median):,.2f}")
    print(f"Dollar RMSE (Duan Smeared):  ${np.sqrt(mean_squared_error(y_test_actual, y_pred_mean)):,.2f}")

    # 3. Split Conformalized Quantile Regression (CQR) for Guaranteed 80% Coverage
    print("\n3. Training & calibrating Conformalized Quantile Regressors (80% target coverage)...")
    X_tr_trans = point_pipeline[:-1].transform(X_train)
    X_cal_trans = point_pipeline[:-1].transform(X_cal)
    X_te_trans = point_pipeline[:-1].transform(X_test)

    q10_model = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.10, max_iter=400, learning_rate=0.05,
        max_leaf_nodes=63, min_samples_leaf=50, random_state=42
    ).fit(X_tr_trans, y_train)

    q90_model = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.90, max_iter=400, learning_rate=0.05,
        max_leaf_nodes=63, min_samples_leaf=50, random_state=42
    ).fit(X_tr_trans, y_train)

    # Compute CQR non-conformity scores on calibration set
    cal_q10 = q10_model.predict(X_cal_trans)
    cal_q90 = q90_model.predict(X_cal_trans)
    cal_low = np.minimum(cal_q10, cal_q90)
    cal_high = np.maximum(cal_q10, cal_q90)

    alpha = 0.20  # 80% nominal prediction interval
    scores = np.maximum(cal_low - y_cal.values, y_cal.values - cal_high)
    q_level = np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores)
    cqr_offset = float(np.quantile(scores, min(q_level, 1.0), method='higher'))

    # Evaluate calibrated CQR intervals on X_test
    te_q10 = q10_model.predict(X_te_trans)
    te_q90 = q90_model.predict(X_te_trans)
    te_low_dollar = np.expm1(np.minimum(te_q10, te_q90) - cqr_offset)
    te_high_dollar = np.expm1(np.maximum(te_q10, te_q90) + cqr_offset)
    cqr_coverage = ((y_test_actual >= te_low_dollar) & (y_test_actual <= te_high_dollar)).mean() * 100

    print(f"CQR Log-Space Margin Offset:  +/- {cqr_offset:.4f}")
    print(f"CQR Empirical Test Coverage:  {cqr_coverage:.2f}% (Target: 80.00%)")

    # 4. Serialize Production Valuation Bundle
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    artifacts = {
        'point_pipeline': point_pipeline,
        'q10_model': q10_model,
        'q90_model': q90_model,
        'smearing_factor': smearing_factor,
        'cqr_offset': cqr_offset,
        'catalog': catalog
    }
    joblib.dump(artifacts, output_path)
    print(f"\n4. Saved production valuation bundle to '{output_path}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Used Car Valuation Pipeline with CQR")
    parser.add_argument('--data-path', type=str, default='data/raw/used_cars.csv', help='Path to used_cars.csv')
    parser.add_argument(
        '--output-path',
        type=str,
        default='models/used_car_valuation_bundle.joblib',
        help='Output path for serialized joblib bundle'
    )
    args = parser.parse_args()
    train_and_evaluate(args.data_path, args.output_path)