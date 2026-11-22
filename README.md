# CARZONE — Used Car Valuation & Fair-Market Engine

CARZONE is an end-to-end web application that estimates the fair market value of used cars. Built on a dataset of over 180,000 vehicle listings, CARZONE goes beyond a simple single-number guess: it provides a **calibrated 80% price range** (low to high) and rates listings as a **Great Deal**, **Fair Price**, or **Overpriced**.

---

## What CARZONE Does

- **Fair Price Estimates:** Predicts realistic dollar values for used cars based on brand, model year, mileage, engine power, and title status.
- **Smart 80% Price Range:** Recognizes that two identical cars can sell for different prices due to unmeasured factors (like interior condition or maintenance records). It gives buyers and sellers a dependable price corridor.
- **Deal Rating System:** Instantly badges listings as a **Great Deal**, **Fair Market Price**, or **Overpriced** by comparing asking prices against calculated market bounds.
- **Interactive Visualizations:** Includes dynamic price distribution charts, a 5-year depreciation forecast, and automatically fetched vehicle photos with fallback illustrations.

---

## Why a Price Range Instead of Just One Number?

In real-life car buying, single price guesses are often slightly off because no dataset tracks every detail—like scratches, clean service records, or how urgently a dealer wants to sell.

Instead of pretending to know the *exact* dollar amount, CARZONE uses a **calibrated confidence range**:
- **80% Confidence Corridor:** The system calculates low-end (10th percentile) and high-end (90th percentile) price limits.
- **Guaranteed Accuracy:** The range is mathematically adjusted so that 80 out of 100 real market sales fall directly within the predicted bracket.

---

## Key Highlights

1. **Smart Data Cleaning:** Automatically detects and filters out extreme pricing typos and unrealistic data errors before training.
2. **Automated Feature Building:** Calculates car age, annual driving pace (miles per year), and engine performance statistics to improve prediction accuracy.
3. **Fast API & Modern Interface:** Powered by a lightweight Python backend (`FastAPI`) and a responsive web dashboard designed with clean CARZONE styling.

---

## Project Structure

```text
Project1/
├── src/
│   ├── features.py          # Preprocessing & car feature calculations
│   ├── train.py             # Model training & price range calibration
│   └── api.py               # Backend server & UI web route
├── frontend/
│   └── index.html           # CARZONE interactive dashboard
├── tests/
│   └── test_api.py          # Automated system & schema tests
├── models/
│   └── used_car_valuation_bundle.joblib  # Trained model bundle
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start (Local Setup)

### 1. Install Dependencies
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Automated Tests
```powershell
pytest -v
```

### 3. Launch the Web App
```powershell
uvicorn src.api:app --reload --port 8000
```

Open your browser and visit:
- **Web App UI:** `http://127.0.0.1:8000`
- **API Documentation:** `http://127.0.0.1:8000/docs`
- **Health Check:** `http://127.0.0.1:8000/health`

---

## Running with Docker

You can package and run the entire application inside a container:

### Build the Image
```powershell
docker build -t carzone-valuation .
```

### Run the Container
```powershell
docker run -p 8000:8000 carzone-valuation
```

Once running, access the dashboard at `http://localhost:8000`.