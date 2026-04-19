Stock Market Price Forecasting 📈
Project Objective: Building a scalable data-driven model to predict stock movements using PySpark.

📌 Project Overview
Investors require real-time, data-driven insights to navigate the stock market. This project focuses on processing a large-scale financial dataset to identify trends and forecast future prices. By utilizing Apache Spark, we handle the computational load of processing hundreds of thousands of records efficiently.

📊 Dataset Requirement & Selection
This project utilizes the NIFTY-50 Stock Market Data (2000 - 2021), which perfectly aligns with the project's Big Data criteria:

Source: Kaggle - Nifty50 Stock Market Data

Total Records: ~375,000 rows (Meets the 100k - 500k recommendation)

Attributes: 15 columns (Meets the 10-20 column requirement)

File Size: ~60 MB (Meets the 50MB - 200MB requirement)

Key Features: Date, Symbol, Prev Close, VWAP, Volume, Turnover, %Deliverble.

🛠️ Tech Stack
Language: Python

Big Data Engine: PySpark (Spark SQL & MLlib)

Data Handling: Pandas (for final sampling/export)

Visualization: Matplotlib / Plotly

🏗️ Data Processing Pipeline
To demonstrate Big Data capabilities, the following PySpark operations were implemented:

Distributed Ingestion: Loading multiple CSV files into a unified Spark DataFrame.

Feature Engineering: * Implementing Window Functions to calculate 7-day and 30-day Moving Averages.

Computing Daily Returns and Volatility metrics.

Data Cleaning: Handling null values in Trades and Deliverable Volume columns using Spark's .na functions.

Schema Enforcement: Explicitly defining data types for high-precision financial calculations.

🚀 Execution Guide
To replicate this project, follow these steps:

Install Dependencies:

Bash
pip install pyspark kagglehub pandas matplotlib
Download Data: Use the provided script to fetch the Nifty-50 dataset via kagglehub.

Run Spark Job: Execute the main script to process data and generate the finalized CSV.

Visualize: Generate trend prediction graphs to compare actual vs. predicted closing prices.
