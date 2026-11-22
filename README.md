# 🚗 CARZONE — Used Car Valuation Platform

An end-to-end, production-ready machine learning web application for predicting used car prices based on vehicle specs, ownership history, and market attributes. Built with **FastAPI**, **Scikit-Learn**, **Tailwind CSS**, and **Docker**.

---

## 📌 Features

- **ML-Powered Valuation Engine:** High-accuracy regression model pipeline packaged via Joblib, performing automated feature pre-processing, encoding, and scale transformation.
- **Modern Interactive UI:** Glassmorphic interface built with HTML5, Tailwind CSS, and Alpine.js featuring interactive parameter sliders, quick-fill preset buttons, dynamic visual price gauge, and instant feedback.
- **Vehicle Showroom & Search:** Browsable showcase of sample vehicles with real-time text filter search, sorting by year/mileage/price, and automated unsplash image hydration.
- **Production-Grade API Hardening:**
  - Configurable CORS origin filtering (`ALLOWED_ORIGINS`).
  - Strict input validation via Pydantic schema schemas with numerical sanity constraints.
  - In-memory rate limiting with automated `429 Too Many Requests` responses (`RATE_LIMIT_MAX_REQUESTS`).
  - Structured request-latency logging via FastAPI `lifespan` context handlers.
  - Toggleable API documentation access (`ENABLE_DOCS`) for security in production environments.
- **Containerized & Tested:** Fully dockerized setup running under a non-root security context (`appuser`), backed by comprehensive `pytest` test suites.

---

## 🏗️ Project Architecture

```text
Project1/
├── data/
│   └── raw/
│       └── used_cars.csv
├── models/
│   └── used_car_valuation_bundle.joblib
├── src/
│   ├── __init__.py
│   ├── api.py
│   └── model_loader.py
├── frontend/
│   └── index.html
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

- **Backend Framework:** FastAPI, Uvicorn
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
- **Validation & Security:** Pydantic v2, Slowapi / Custom Rate Limiter
- **Frontend:** HTML5, Tailwind CSS, Alpine.js, Lucide Icons
- **DevOps & Testing:** Docker, Pytest, Git

---

## ⚙️ Environment Variables & Configuration

Create a `.env` file in the root directory (or use environment variables in deployment environments):

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ALLOWED_ORIGINS` | `*` | Comma-separated list of allowed CORS origins (e.g., `https://yourdomain.com`). |
| `ENABLE_DOCS` | `true` | Set to `false` to disable `/docs` and `/redoc` in production. |
| `RATE_LIMIT_MAX_REQUESTS` | `30` | Maximum requests allowed per client IP per window. |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Duration of the rate limit window in seconds. |

---

## 🚀 Getting Started Locally

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Virtual Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate virtual environment
python -m venv .venv

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Development Server

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Access the application in your browser at `http://localhost:8000`.

---

## 🐳 Docker Deployment

The application includes an optimized, multi-stage `Dockerfile` running as a non-privileged user for enhanced security.

### Build Docker Image

```bash
docker build -t carzone-valuation .
```

### Run Container

```bash
docker run -d -p 8000:8000 --name carzone-app --env-file .env carzone-valuation
```

View the live app at `http://localhost:8000`.

---

## 🧪 Running Tests

Execute the automated test suite covering health endpoints, prediction schemas, input validation boundaries, and rate limits:

```bash
pytest -v
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the single-page frontend application (`frontend/index.html`). |
| `/health` | `GET` | API health check and model loading status verification. |
| `/predict` | `POST` | Accepts vehicle attributes in JSON format and returns predicted valuation. |
| `/docs` | `GET` | Interactive Swagger API documentation (when `ENABLE_DOCS=true`). |

### Sample Prediction Payload (`POST /predict`)

```json
{
  "year": 2021,
  "mileage": 28000,
  "engine_size": 2.0,
  "fuel_type": "Petrol",
  "transmission": "Automatic",
  "owner_type": "First",
  "brand": "Toyota"
}
```

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.