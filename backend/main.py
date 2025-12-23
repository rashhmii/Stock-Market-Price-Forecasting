import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Configuration ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["stock_forecast_db"]
    history_col = db["prediction_history"]
    print("Connected to MongoDB")
except:
    history_col = None
    print("MongoDB not found. Running without database storage.")

# Load Data Global Variable
DATA_PATH = "nifty50_stock_data.csv"
df_global = None


# --- IMPROVED load_data FUNCTION ---
def load_data():
    global df_global
    if os.path.exists(DATA_PATH):
        try:
            # 1. Read CSV with low_memory=False to suppress the DtypeWarning
            df_global = pd.read_csv(DATA_PATH, low_memory=False)

            # 2. Clean column names
            df_global.columns = [c.strip().replace('"', '') for c in df_global.columns]

            # 3. CRITICAL FIX: Force convert 'Date' column.
            # 'errors="coerce"' turns text like "Adani Ports..." into NaT (Not a Time)
            df_global['Date'] = pd.to_datetime(df_global['Date'], errors='coerce')

            # 4. Remove the rows where Date failed to parse
            initial_count = len(df_global)
            df_global = df_global.dropna(subset=['Date'])
            cleaned_count = len(df_global)

            # 5. Ensure numeric columns are actually numbers (in case garbage got in there too)
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in df_global.columns:
                    df_global[col] = pd.to_numeric(df_global[col], errors='coerce')

            print("--------------------------------------------------")
            print("SUCCESS: CSV Loaded.")
            print(f"Cleaned {initial_count - cleaned_count} bad rows.")
            print(f"Columns found: {list(df_global.columns)}")
            print("--------------------------------------------------")

        except Exception as e:
            print(f"ERROR loading CSV: {e}")
    else:
        print(f"Warning: {DATA_PATH} not found.")


# Load immediately on startup
load_data()


# --- Helper Functions ---
def get_stock_data(symbol):
    if df_global is None:
        return pd.DataFrame()

    # Filter by symbol
    stock_df = df_global[df_global['Symbol'] == symbol].copy()
    stock_df = stock_df.sort_values('Date')

    # Technical Indicators (Pandas version)
    # 1. SMA & Bollinger Bands (20 day window)
    stock_df['SMA_20'] = stock_df['Close'].rolling(window=20).mean()
    stock_df['StdDev'] = stock_df['Close'].rolling(window=20).std()
    stock_df['Upper_Band'] = stock_df['SMA_20'] + (stock_df['StdDev'] * 2)
    stock_df['Lower_Band'] = stock_df['SMA_20'] - (stock_df['StdDev'] * 2)

    # Fill NaN values to avoid errors in JSON
    stock_df = stock_df.fillna(0)

    return stock_df


def train_predict_model(stock_df):
    """
    Trains a Linear Regression model using Scikit-Learn.
    Predicts the 'Close' price based on Open, High, Low, Volume.
    """
    if len(stock_df) < 50:
        return stock_df  # Not enough data

    # Prepare Features
    feature_cols = ['Open', 'High', 'Low', 'Volume']

    # Scikit-learn needs clean data (no NaNs or Infinity)
    model_data = stock_df[feature_cols + ['Close']].replace([np.inf, -np.inf], 0).fillna(0)

    X = model_data[feature_cols]
    y = model_data['Close']

    # Train Model (Using last 80% of data to train)
    split = int(len(model_data) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on entire dataset
    stock_df['prediction'] = model.predict(X)

    return stock_df


# --- API Endpoints ---

@app.get("/")
def home():
    return {"message": "Stock Forecasting API (Pandas Mode) is Running"}


@app.get("/stocks")
def get_stocks():
    if df_global is None:
        return {"stocks": []}
    symbols = sorted(df_global['Symbol'].unique().tolist())
    return {"stocks": symbols}


@app.post("/predict/{symbol}")
def predict_stock(symbol: str):
    # 1. Get Data
    stock_df = get_stock_data(symbol)

    if stock_df.empty:
        raise HTTPException(status_code=404, detail="Stock not found")

    # 2. Run Prediction
    result_df = train_predict_model(stock_df)

    # 3. Format for Frontend (Take last 100 days)
    recent_data = result_df.tail(100)

    chart_data = []
    for _, row in recent_data.iterrows():
        chart_data.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "actual": round(row["Close"], 2),
            "predicted": round(row.get("prediction", 0), 2),  # Handle case if prediction fails
            "upper": round(row["Upper_Band"], 2),
            "lower": round(row["Lower_Band"], 2)
        })

    # 4. Save to MongoDB (Optional)
    if history_col is not None and len(chart_data) > 0:
        try:
            history_col.insert_one({
                "symbol": symbol,
                "timestamp": datetime.now(),
                "latest_price": chart_data[-1]["actual"],
                "predicted_next": chart_data[-1]["predicted"]
            })
        except:
            pass

    last_row = chart_data[-1]
    prev_row = chart_data[-2]

    return {
        "symbol": symbol,
        "current_price": last_row["actual"],
        "predicted_price": last_row["predicted"],
        "trend": "UP" if last_row["predicted"] > prev_row["predicted"] else "DOWN",
        "data": chart_data
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)