# 🚗 CARZONE — Used Car Valuation Platform

An end-to-end, production-ready machine learning web application for predicting fair-market used car prices based on vehicle specs, ownership history, and market attributes. Built with **FastAPI**, **Scikit-Learn**, and **Docker**.

🌐 **Live Demo:** [carzone-used-car-valuation.onrender.com](https://carzone-used-car-valuation.onrender.com) *(Free tier may take ~30s to wake up on first visit)*

---

## 📌 Key Features

* **ML-Powered Valuation Engine:** Tuned `HistGradientBoostingRegressor` (`0.7321` Test Log $R^2$) trained on ~188.5k used vehicle listings, packaged via Joblib with automated feature extraction (`extract_features_v4`) and Duan's smearing bias correction.
* **Uncertainty Quantification & Deal Grading:** 80% Conformalized Quantile Regression (`Q10` – `Q90`) prediction intervals to grade listings as **Great Deal**, **Fair Market Price**, or **Overpriced**.
* **5-Year Depreciation Forecast:** Vectorized multi-year value simulation projecting depreciation at `+12,000 miles/year`.
* **Interactive Showroom & Studio UI:** Single-page CARZONE interface (`frontend/index.html`) with live valuation sliders, brand/fuel/price inventory filtering across 128 catalog vehicles, and dynamic vehicle photo resolution with vector fallbacks.
* **Production-Grade API Hardening:**
  * Environment-configurable CORS origin filtering (`ALLOWED_ORIGINS`).
  * Strict Pydantic input validation (`min_length`, `max_length`, numeric bounds).
  * Sliding-window per-IP rate limiting with `429 Too Many Requests` responses (`RATE_LIMIT_MAX_REQUESTS`).
  * Structured JSON request-latency logging and clean startup/shutdown resource management via FastAPI `lifespan`.
  * Toggleable Swagger/ReDoc documentation (`ENABLE_DOCS`) for production environments.
* **Containerized & Tested:** Multi-stage/slim `Dockerfile` running under a non-root user (`appuser`), backed by automated `pytest` suites.

---

## 🏗️ Project Architecture

```text
carzone-used-car-valuation/
├── data/
│   └── raw/                    # Dataset directory (ignored in Git)
├── models/
│   └── used_car_valuation_bundle.joblib
├── src/
│   ├── __init__.py
│   ├── api.py                  # FastAPI server, rate limiter, lifespan & inference routes
│   ├── features.py             # Domain feature engineering (extract_features_v4)
│   └── train.py                # Model training, CQR calibration & bundle export
├── frontend/
│   └── index.html              # Single-page CARZONE valuation studio & showroom UI
├── tests/
│   └── test_api.py             # Automated pytest API & validation suite
├── Dockerfile                  # Non-root production container configuration
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

* **Backend & API:** Python 3.11, FastAPI, Uvicorn, Pydantic v2
* **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
* **Frontend:** Vanilla HTML5, CSS3, JavaScript (Wikipedia/Wikimedia photo hydration + inline SVGs)
* **DevOps & Testing:** Docker, Pytest, Render

---

## ⚙️ Environment Variables & Configuration

Configure these variables in your `.env` file or hosting provider's environment settings:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ALLOWED_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Comma-separated list of allowed CORS origins (e.g., `https://carzone-used-car-valuation.onrender.com`). |
| `ENABLE_DOCS` | `true` | Set to `false` to disable `/docs`, `/redoc`, and `/openapi.json` in production. |
| `RATE_LIMIT_MAX_REQUESTS` | `60` | Maximum `/predict` requests allowed per client IP per window. |
| `RATE_LIMIT_WINDOW_SEC` | `60` | Sliding-window duration in seconds for rate limiting. |

---

## 🚀 Getting Started Locally

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/RishabhPaliwal480/carzone-used-car-valuation.git
cd carzone-used-car-valuation

python -m venv .venv

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Development Server

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

---

## 🐳 Docker & Cloud Deployment (Render)

### Run Locally via Docker

```bash
docker build -t carzone-valuation .
docker run --rm -p 8000:8000 -e ENABLE_DOCS=false carzone-valuation
```

### Deploy on Render

1. Connect `RishabhPaliwal480/carzone-used-car-valuation` as a **Web Service** on Render using the **Docker** runtime.
2. Set the environment variables `ENABLE_DOCS=false` and `ALLOWED_ORIGINS=https://carzone-used-car-valuation.onrender.com`.
3. Every push to `main` automatically rebuilds and redeploys the live service.

---

## 🧪 Running Tests

```bash
pytest -v
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the single-page CARZONE frontend (`frontend/index.html`). |
| `/health` | `GET` | Health check returning API status and whether the model bundle is loaded. |
| `/catalog` | `GET` | Returns vehicle catalog metadata stored in the model bundle. |
| `/predict` | `POST` | Returns expected market value, 80% CQR interval, deal grade, and 5-year depreciation curve. |
| `/docs` | `GET` | Interactive Swagger API documentation (available when `ENABLE_DOCS=true`). |

### Sample Prediction Payload (`POST /predict`)

```json
{
  "brand": "Lexus",
  "model": "RX 350 Base",
  "model_year": 2016,
  "milage": 67500,
  "fuel_type": "Gasoline",
  "engine": "295.0HP 3.5L V6 Cylinder Engine Gasoline Fuel",
  "transmission": "8-Speed A/T",
  "ext_col": "Green",
  "int_col": "Black",
  "accident": "None reported",
  "clean_title": "Yes",
  "asking_price": 26500
}
```

---

## 📄 License

This project is licensed under the MIT License.