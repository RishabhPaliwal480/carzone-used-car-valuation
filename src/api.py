import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is on sys.path so `src.features` resolves during unpickling
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features import extract_features_v4

# Structured JSON logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("carzone.api")

# Environment-driven security settings
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "true").lower() == "true"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if origin.strip()
]
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

BUNDLE_PATH = ROOT_DIR / "models" / "used_car_valuation_bundle.joblib"
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_PATH = FRONTEND_DIR / "index.html"
ASSETS_DIR = FRONTEND_DIR / "assets"

bundle: Optional[Dict[str, Any]] = None
_rate_limit_store: Dict[str, deque] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    if BUNDLE_PATH.exists():
        t0 = time.perf_counter()
        bundle = joblib.load(BUNDLE_PATH)
        load_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            json.dumps(
                {
                    "event": "startup",
                    "model_loaded": True,
                    "load_time_ms": load_ms,
                }
            )
        )
    else:
        logger.warning(
            json.dumps(
                {
                    "event": "startup",
                    "model_loaded": False,
                    "detail": "Model bundle not found",
                }
            )
        )
    yield
    bundle = None
    _rate_limit_store.clear()
    logger.info(json.dumps({"event": "shutdown", "status": "clean"}))


app = FastAPI(
    title="CARZONE | Used Car Fair-Market Valuation API",
    description=(
        "Production inference API for used vehicle pricing with Duan's Smearing "
        "bias correction, Conformalized Quantile Regression (80% Prediction Intervals), "
        "Deal Grading, and 5-Year Depreciation forecasting."
    ),
    version="2.3.0",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def log_request_latency(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(latency_ms)
    logger.info(
        json.dumps(
            {
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "client_ip": request.client.host if request.client else "unknown",
            }
        )
    )
    return response


if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


def _enforce_rate_limit(client_ip: str) -> None:
    """Lightweight sliding-window per-IP rate limiter (zero external dependencies)."""
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    timestamps = _rate_limit_store[client_ip]

    while timestamps and timestamps[0] < window_start:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again in a minute.",
        )

    timestamps.append(now)


class VehicleRequest(BaseModel):
    brand: str = Field(..., min_length=1, max_length=60, example="Lexus")
    model: str = Field(..., min_length=1, max_length=100, example="RX 350 Base")
    model_year: int = Field(..., ge=1970, le=2026, example=2016)
    milage: int = Field(..., ge=0, le=500000, example=67500)
    fuel_type: Optional[str] = Field("Gasoline", max_length=40, example="Gasoline")
    engine: str = Field(
        ...,
        min_length=1,
        max_length=120,
        example="295.0HP 3.5L V6 Cylinder Engine Gasoline Fuel",
    )
    transmission: str = Field(..., min_length=1, max_length=80, example="8-Speed A/T")
    ext_col: str = Field("Black", max_length=50, example="Black")
    int_col: str = Field("Black", max_length=50, example="Black")
    accident: Optional[str] = Field("None reported", max_length=80, example="None reported")
    clean_title: Optional[str] = Field("Yes", max_length=30, example="Yes")
    asking_price: Optional[float] = Field(None, gt=0, le=5000000, example=26500.0)


class ValuationResponse(BaseModel):
    expected_market_value_usd: float
    median_listing_value_usd: float
    interval_80_low_usd: float
    interval_80_high_usd: float
    deal_grade: Optional[str] = None
    engineered_features: dict
    depreciation_5yr: Optional[List[dict]] = None


@app.get("/", include_in_schema=False)
def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH)
    if ENABLE_DOCS:
        return RedirectResponse(url="/docs")
    return {"status": "CARZONE API running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": bundle is not None,
    }


@app.get("/catalog")
def get_catalog():
    if bundle is not None and "catalog" in bundle:
        return bundle["catalog"]
    return {"source": "embedded_fallback"}


@app.post("/predict", response_model=ValuationResponse)
def predict_valuation(vehicle: VehicleRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    _enforce_rate_limit(client_ip)

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model bundle not loaded. Run `python -m src.train --data-path data/raw/used_cars.csv` first.",
        )

    point_pipe = bundle["point_pipeline"]
    q10_model = bundle["q10_model"]
    q90_model = bundle["q90_model"]
    smearing_factor = float(bundle.get("smearing_factor", 1.0755))
    cqr_offset = float(bundle.get("cqr_offset", 0.0174))
    reference_year = int(bundle.get("reference_year", 2024))

    row_dict = vehicle.model_dump(exclude={"asking_price"})
    for col in ("fuel_type", "accident", "clean_title"):
        if row_dict[col] in (None, "", "Missing"):
            row_dict[col] = np.nan

    input_df = pd.DataFrame([row_dict])

    # 1. Engineered feature summary
    fe_df = extract_features_v4(input_df)
    fe_row = fe_df.iloc[0]

    def _safe_float(val):
        return None if pd.isna(val) else round(float(val), 2)

    engineered_summary = {
        "vehicle_age": int(fe_row["vehicle_age"]),
        "miles_per_year": round(float(fe_row["miles_per_year"]), 1),
        "horsepower": _safe_float(fe_row["horsepower"]),
        "displacement_l": _safe_float(fe_row["displacement"]),
        "cylinders": _safe_float(fe_row["cylinders"]),
        "is_turbo": bool(fe_row["is_turbo"]),
        "is_luxury": bool(fe_row["is_luxury"]),
    }

    # 2. Point valuation (Log -> Dollar with Duan's Smearing)
    log_pred = float(point_pipe.predict(input_df)[0])
    pred_median = float(np.expm1(log_pred))
    pred_mean = float(np.exp(log_pred) * smearing_factor - 1)

    # 3. Conformalized 80% Prediction Interval
    X_trans = point_pipe[:-1].transform(input_df)
    raw_q10 = float(q10_model.predict(X_trans)[0])
    raw_q90 = float(q90_model.predict(X_trans)[0])
    pi_low = float(np.expm1(min(raw_q10, raw_q90) - cqr_offset))
    pi_high = float(np.expm1(max(raw_q10, raw_q90) + cqr_offset))

    # 4. Deal Grader
    deal_grade = None
    if vehicle.asking_price is not None:
        if vehicle.asking_price <= pi_low * 1.05:
            deal_grade = "Great Deal"
        elif vehicle.asking_price <= pred_mean * 1.10:
            deal_grade = "Fair Market Price"
        else:
            deal_grade = "Overpriced"

    # 5. Vectorized 5-Year Depreciation Forecast (+12,000 miles/yr)
    offsets = np.arange(6)
    sim_df = pd.DataFrame([row_dict] * 6)
    sim_df["model_year"] = int(vehicle.model_year) - offsets
    sim_df["milage"] = int(vehicle.milage) + (offsets * 12000)
    sim_preds = np.exp(point_pipe.predict(sim_df)) * smearing_factor - 1

    depreciation_curve = [
        {
            "year_offset": int(yr),
            "projected_year": reference_year + int(yr),
            "projected_milage": int(vehicle.milage) + (int(yr) * 12000),
            "estimated_market_value": round(float(val), 2),
        }
        for yr, val in enumerate(sim_preds)
    ]

    return ValuationResponse(
        expected_market_value_usd=round(pred_mean, 2),
        median_listing_value_usd=round(pred_median, 2),
        interval_80_low_usd=round(pi_low, 2),
        interval_80_high_usd=round(pi_high, 2),
        deal_grade=deal_grade,
        engineered_features=engineered_summary,
        depreciation_5yr=depreciation_curve,
    )
