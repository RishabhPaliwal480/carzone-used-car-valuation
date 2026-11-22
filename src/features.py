import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, TargetEncoder

# Feature routing lists used across training and inference
NUM_FEATURES = [
    'milage', 'log_milage', 'model_year', 'vehicle_age', 'horsepower',
    'displacement', 'cylinders', 'miles_per_year', 'hp_per_liter',
    'hp_per_cyl', 'trans_speeds', 'clean_title_flag', 'is_turbo',
    'is_electric_or_hybrid', 'is_automatic', 'is_manual', 'is_luxury'
]

HIGH_CAT_FEATURES = [
    'brand', 'model', 'brand_model', 'engine', 'transmission', 'ext_col', 'int_col'
]

LOW_CAT_FEATURES = [
    'fuel_type_clean', 'accident_clean'
]

LUXURY_BRANDS = [
    'Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Porsche', 'Land', 'Tesla',
    'Jaguar', 'Cadillac', 'Bentley', 'Rolls-Royce', 'Ferrari', 'Lamborghini',
    'McLaren', 'Maserati', 'Aston', 'Rivian', 'Lucid', 'Genesis', 'Alfa',
    'Maybach', 'Bugatti', 'Lotus'
]


def extract_features_v4(X_in: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-driven automotive feature engineering transformation (v4).
    Extracts non-linear mileage ratios, engine horsepower/displacement/cylinders,
    transmission speeds, and preserves missingness indicators.
    """
    X_out = X_in.copy()

    # 1. Vehicle Age & Non-linear Mileage Features
    X_out['vehicle_age'] = 2024 - X_out['model_year']
    X_out['miles_per_year'] = X_out['milage'] / (X_out['vehicle_age'].clip(lower=0) + 1)
    X_out['log_milage'] = np.log1p(X_out['milage'])

    # 2. Richer Engine Parsing (Retain raw 'engine' string for Target Encoding)
    eng = X_out['engine'].fillna('Missing').astype(str)
    X_out['horsepower'] = eng.str.extract(r'(\d+\.?\d*)HP')[0].astype(float)
    X_out['displacement'] = eng.str.extract(r'(\d+\.?\d*)\s*L(?:iter)?', expand=False).astype(float)

    cyl1 = eng.str.extract(r'(\d+)\s*Cylinder', expand=False).astype(float)
    cyl2 = eng.str.extract(r'\b[VIWH](\d+)\b', expand=False).astype(float)
    X_out['cylinders'] = cyl1.fillna(cyl2)

    X_out['hp_per_liter'] = X_out['horsepower'] / (X_out['displacement'].replace(0, np.nan))
    X_out['hp_per_cyl'] = X_out['horsepower'] / (X_out['cylinders'].replace(0, np.nan))
    X_out['is_turbo'] = eng.str.contains('Turbo|Supercharged', case=False, na=False).astype(int)
    X_out['is_electric_or_hybrid'] = (
        eng.str.contains('Electric|Hybrid|Battery', case=False, na=False)
        | X_out['fuel_type'].fillna('').astype(str).str.contains('Hybrid|Electric', case=False)
    ).astype(int)

    # 3. Transmission Parsing (Retain raw 'transmission' string for Target Encoding)
    trans = X_out['transmission'].fillna('Missing').astype(str)
    X_out['trans_speeds'] = trans.str.extract(r'(\d+)[-\s]*Speed', flags=re.IGNORECASE, expand=False).astype(float)
    X_out['is_automatic'] = trans.str.contains(
        'A/T|Automatic|Dual Shift|CVT|Auto-Shift|DCT', case=False, na=False
    ).astype(int)
    X_out['is_manual'] = trans.str.contains('M/T|Manual', case=False, na=False).astype(int)

    # 4. Preserve Missingness Signal in clean_title, accident, and fuel_type
    X_out['clean_title_flag'] = X_out['clean_title'].eq('Yes').astype(int)
    X_out['accident_clean'] = X_out['accident'].fillna('Missing').astype(str)
    X_out['fuel_type_clean'] = X_out['fuel_type'].fillna('Missing').astype(str)

    # 5. Brand-Model Interaction & Complete Luxury/Exotic Segmentation
    X_out['brand_model'] = X_out['brand'].astype(str) + '_' + X_out['model'].astype(str)
    X_out['is_luxury'] = X_out['brand'].isin(LUXURY_BRANDS).astype(int)

    return X_out


def build_preprocessor() -> ColumnTransformer:
    """Builds the Scikit-Learn ColumnTransformer for v4 features."""
    return ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                    ('scaler', StandardScaler())
                ]),
                NUM_FEATURES
            ),
            (
                'high_cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                    ('encoder', TargetEncoder(target_type='continuous', smooth='auto', cv=5, random_state=42))
                ]),
                HIGH_CAT_FEATURES
            ),
            (
                'low_cat',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                LOW_CAT_FEATURES
            )
        ]
    )


def get_feature_transformer() -> FunctionTransformer:
    """Wraps extract_features_v4 in a Scikit-Learn FunctionTransformer."""
    return FunctionTransformer(extract_features_v4)