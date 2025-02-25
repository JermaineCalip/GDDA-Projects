from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima

# ✅ Initialize Spark
spark = SparkSession.builder.appName("RetailTimeSeries").getOrCreate()

# ✅ Load Data
file_path = "online_retail_cleaned.csv"  # Change this to your file path
df = spark.read.csv(file_path, header=True, inferSchema=True)

# ✅ Convert InvoiceDate to timestamp
df = df.withColumn("timestamp", unix_timestamp(col("InvoiceDate")).cast("double"))

# ✅ Select Relevant Columns
df = df.select("timestamp", "TotalRevenue")

# ✅ Convert to Pandas for ARIMA
df_pandas = df.toPandas().sort_values("timestamp")

# ✅ Fit ARIMA Model
arima_model = auto_arima(df_pandas["TotalRevenue"], seasonal=True, m=12, trace=True)
forecast = arima_model.predict(n_periods=100)  # Forecast next 100 points

# ✅ Plot ARIMA Forecast
plt.figure(figsize=(10, 5))
plt.plot(df_pandas["TotalRevenue"].values[:100], label="Actual Sales", color="blue")
plt.plot(forecast, label="Predicted Sales", linestyle="dashed", color="red")
plt.title("Actual vs Predicted Sales (ARIMA)")
plt.xlabel("Time (Sample Points)")
plt.ylabel("Sales")
plt.legend()
plt.show()

# ✅ Convert Sales into Classification Labels
df = df.withColumn("SalesCategory", (col("TotalRevenue") > df.approxQuantile("TotalRevenue", [0.5], 0.01)[0]).cast("int"))

# ✅ Prepare Features for Classification
vector_assembler = VectorAssembler(inputCols=["timestamp"], outputCol="features")
df = vector_assembler.transform(df).select("features", "SalesCategory")

# ✅ Train-Test Split
train, test = df.randomSplit([0.8, 0.2], seed=42)

# ✅ Train Random Forest Classifier
rf = RandomForestClassifier(labelCol="SalesCategory", featuresCol="features", numTrees=20)
model = rf.fit(train)

# ✅ Predictions
predictions = model.transform(test)

# ✅ Evaluate Model
evaluator = MulticlassClassificationEvaluator(labelCol="SalesCategory", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# ✅ Display Accuracy
print(f"Random Forest Classification Accuracy: {accuracy * 100:.2f}%")

# ✅ Confusion Matrix
preds_pd = predictions.select("SalesCategory", "prediction").toPandas()
conf_matrix = pd.crosstab(preds_pd["SalesCategory"], preds_pd["prediction"], rownames=["Actual"], colnames=["Predicted"])

# ✅ Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Random Forest)")
plt.show()

# ✅ Stop Spark Session
spark.stop()
