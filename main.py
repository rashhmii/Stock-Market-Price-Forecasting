import os
import sys

python_path = sys.executable

os.environ['HADOOP_HOME'] = "C:/hadoop"
os.environ['PATH'] += os.pathsep + "C:/hadoop/bin"

os.environ['PYSPARK_PYTHON'] = python_path
os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, when, year, round, desc, lit
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import lag
import matplotlib.pyplot as plt

#Create SparkSession
spark = SparkSession.builder \
    .appName("StockMarketForecasting") \
    .master("local[*]") \
    .getOrCreate()

#Load dataset
file_path = "C:/Users/Rashmi S/Desktop/BDA LAB PROJECT/nifty50_stock_data.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

#Print Sample rows
print("Sample Rows:")
df.show(5)

#Print schema
print("Schema:")
df.printSchema()

#Print summary
print("Key Indicators Summary:")
df.select("Close", "Volume", "Open", "High", "Low").describe().show()

#Print record count
print(f"Total Rows Loaded: {df.count()}")
print(f"Columns: {len(df.columns)}")

#filter records; filter for rows where 'Date' is actually a date
df_cleaned = df.filter(col("Date").rlike("^[0-9]{4}-[0-9]{2}-[0-9]{2}$"))

#Rename 'Prev Close' to 'PrevClose' (remove space)
#Cast 'Date' string to DateType
#Cast numeric columns to Double/Float (in case they were read as strings)
df_cleaned = df_cleaned \
    .withColumnRenamed("Prev Close", "PrevClose") \
    .withColumnRenamed("%Deliverable", "PercentDeliverable") \
    .withColumn("Date", to_date(col("Date"), "yyyy-MM-dd")) \
    .withColumn("Open", col("Open").cast("double")) \
    .withColumn("High", col("High").cast("double")) \
    .withColumn("Low", col("Low").cast("double")) \
    .withColumn("Close", col("Close").cast("double")) \
    .withColumn("Volume", col("Volume").cast("double"))

#Handle Missing Values
#drop rows where essential data is missing (Open, Close, Volume)
df_cleaned = df_cleaned.dropna(subset=["Date", "Open", "Close", "Volume"])

#remove duplicates
df_cleaned = df_cleaned.dropDuplicates()

print("Data Cleaning Complete.")
df_cleaned.printSchema()
print(f"Cleaned Record Count: {df_cleaned.count()}")

df_cleaned.select("Date", "Symbol", "Close", "Volume").show(5)

#Convert DataFrame to RDD
stock_rdd = df_cleaned.rdd

#filter operation
#Keep only "High Value" stocks where Close price > 1000
high_value_rdd = stock_rdd.filter(lambda row: row['Close'] > 1000)
print(f"Count of high value records: {high_value_rdd.count()}")

#MAP Operation
#Create a Pair RDD of (Symbol, Volume)
symbol_vol_rdd = high_value_rdd.map(lambda row: (row['Symbol'], row['Volume']))

# reduceByKey Operation
# Calculate Total Trading Volume per Stock
total_volume_rdd = symbol_vol_rdd.reduceByKey(lambda x, y: x + y)

#Sort and Take
#Find the Top 5 stocks by total volume
top_5_stocks = total_volume_rdd.sortBy(lambda x: x[1], ascending=False).take(5)

print("\nTop 5 Stocks by Volume (Calculated via RDD):")
for stock, vol in top_5_stocks:
    print(f"{stock}: {vol:,.0f}")

#Column-level Transformations
#Create new useful columns from existing data
df_transformed = df_cleaned \
    .withColumn("Year", year(col("Date"))) \
    .withColumn("Daily_Volatility", round(((col("High") - col("Low")) / col("Open")) * 100, 2))

print("Transformed Data (Added Year and Volatility):")
df_transformed.select("Date", "Symbol", "Year", "Daily_Volatility").show(5)

#Grouping & Aggregation
# Analyze average closing price and total volume per Stock per Year
# We group by 'Symbol' and 'Year', then calculate averages and sums.
yearly_report = df_transformed.groupBy("Symbol", "Year") \
    .agg(
        round(F.avg("Close"), 2).alias("Avg_Yearly_Price"),
        F.max("Volume").alias("Max_Daily_Volume"),
        F.sum("Volume").alias("Total_Yearly_Volume")
    )

#Sorting
#Find the years with the highest trading volume for a specific stock (e.g., RELIANCE)
# We sort by 'Total_Yearly_Volume' in descending order.
sorted_report = yearly_report.orderBy(desc("Total_Yearly_Volume"))

print("Top Trading Years by Volume:")
sorted_report.show(5)

#Joining
#Enrich the data with Sector information.
# Since the CSV didn't have a clean 'Sector' column, we create a small reference DataFrame and join it with our main data.

# Create the reference DataFrame
sector_data = [
    ("RELIANCE", "Energy"),
    ("TCS", "IT"),
    ("INFY", "IT"),
    ("HDFC BANK", "Finance"),
    ("SBIN", "Finance"),
    ("TATA MOTORS", "Automobile")
]
columns = ["Symbol", "Sector"]
sector_df = spark.createDataFrame(sector_data, columns)

#Perform the JOIN (Inner Join)
# This adds the 'Sector' column to our yearly report for the matching symbols.
joined_df = yearly_report.join(sector_df, on="Symbol", how="inner")

print("Joined Data (Yearly Report + Sector):")
joined_df.select("Symbol", "Sector", "Year", "Avg_Yearly_Price").show(5)

# 1. Feature Engineering: Prepare Data for Time Series Forecasting
# We cannot simply randomly shuffle time-series data. We need valid features.
# Feature: 'PrevClose' (The closing price of the previous day)
# We use a Window function to get the previous row's 'Close' value.
windowSpec = Window.partitionBy("Symbol").orderBy("Date")
df_ml = df_cleaned.withColumn("PrevClose", lag("Close", 1).over(windowSpec))

# Drop the first row of each stock (which has null PrevClose)
df_ml = df_ml.dropna()

# 2. Select Features and Label
# Features (X): Open, High, Low, Volume, PrevClose
# Label (y): Close
feature_cols = ["Open", "High", "Low", "Volume", "PrevClose"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Transform the data to have a single 'features' vector column
final_data = assembler.transform(df_ml)

# 3. Split Data into Train and Test Sets
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

print(f"Training Dataset Count: {train_data.count()}")
print(f"Test Dataset Count: {test_data.count()}")

# 4. Train the Model
lr = LinearRegression(featuresCol="features", labelCol="Close")
print("Training the model...")
lr_model = lr.fit(train_data)

# 5. Test (Make Predictions)
predictions = lr_model.transform(test_data)
predictions.select("Date", "Symbol", "Close", "prediction").show(5)

# 6. Evaluate the Model
# We use Root Mean Squared Error (RMSE) to measure accuracy.
evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

print(f"Model Performance Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-Squared (R2): {r2:.4f}")

#Save model
model_path = "file:///C:/Users/Rashmi S/Desktop/BDA LAB PROJECT/model"

try:
    lr_model.write().overwrite().save(model_path)
    print(f"Model saved successfully to: {model_path}")
except Exception as e:
    print(f"Error saving model: {e}")

#Save processed results
#Save as CSV (For Excel / Report)
csv_path = "file:///C:/Users/Rashmi S/Desktop/BDA LAB PROJECT/data/csv_output"
df_cleaned.coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(csv_path)
print(f"Saved CSV to: {csv_path}")

#Save as JSON (For Web Application / API)
json_path = "file:///C:/Users/Rashmi S/Desktop/BDA LAB PROJECT/data/json_output"
df_cleaned.coalesce(1) \
    .write.mode("overwrite") \
    .json(json_path)
print(f"Saved JSON to: {json_path}")

#Save Prediction Results (CSV Format)
#Save the final predictions to CSV so you can open them in Excel, Tableau, or use for your Project Report graphs.
results_path = "file:///C:/Users/Rashmi S/Desktop/BDA LAB PROJECT/results"
predictions.select("Date", "Symbol", "Close", "prediction") \
    .coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(results_path)

print(f"Prediction results saved to folder: {results_path}")

print("Generating Report Graph...")
# We use .orderBy(col("Date").desc()) to get the MOST RECENT 100 days
pdf = predictions.select("Date", "Close", "prediction") \
    .orderBy(col("Date").desc()) \
    .limit(100) \
    .toPandas()

# Sort back to ascending for the plot to look right (Time goes left -> right)
pdf = pdf.sort_values("Date")

plt.figure(figsize=(10, 6))
plt.plot(pdf['Date'], pdf['Close'], label='Actual Price', color='blue')
plt.plot(pdf['Date'], pdf['prediction'], label='Predicted Price', linestyle='--', color='orange')
plt.title(f"Stock Price Prediction (Last 100 Days)")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("prediction_graph.png")
print("Graph saved as 'prediction_graph.png'.")

import pandas as pd
df = pd.read_csv('C:/Users/Rashmi S/Desktop/nifty50_stock_data.csv', low_memory=False)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
data = df[df['Symbol'] == 'RELIANCE'].copy()
data = data.sort_values('Date')

for col in ['Close', 'Open', 'High', 'Low']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

# 1. Moving Average (50-day)
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# 2. Bollinger Bands
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['StdDev'] = data['Close'].rolling(window=20).std()
data['Upper_Band'] = data['SMA_20'] + (data['StdDev'] * 2)
data['Lower_Band'] = data['SMA_20'] - (data['StdDev'] * 2)

# 3. RSI (Relative Strength Index)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Filter last 365 days for clearer charts
plot_data = data.tail(365)

# Plot A: Bollinger Bands
plt.figure(figsize=(12, 6))
plt.plot(plot_data['Date'], plot_data['Close'], label='Price', color='blue', alpha=0.5)
plt.plot(plot_data['Date'], plot_data['SMA_50'], label='50-Day SMA', color='orange')
plt.plot(plot_data['Date'], plot_data['Upper_Band'], label='Upper Band', color='green', linestyle='--')
plt.plot(plot_data['Date'], plot_data['Lower_Band'], label='Lower Band', color='green', linestyle='--')
plt.fill_between(plot_data['Date'], plot_data['Upper_Band'], plot_data['Lower_Band'], color='green', alpha=0.1)
plt.title('Technical Indicators: Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('bollinger_bands.png')
print("Saved bollinger_bands.png")

# Plot B: RSI
plt.figure(figsize=(12, 4))
plt.plot(plot_data['Date'], plot_data['RSI'], label='RSI', color='purple')
plt.axhline(70, linestyle='--', color='red', alpha=0.5)
plt.axhline(30, linestyle='--', color='green', alpha=0.5)
plt.title('Relative Strength Index (RSI)')
plt.grid(True)
plt.savefig('rsi_chart.png')
print("Saved rsi_chart.png")