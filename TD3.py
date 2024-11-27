from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log, when

spark = SparkSession.builder \
    .appName("Credit Card Fraud Detection") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

data = spark.read.csv('creditcard.csv', header=True, inferSchema=True)

# Rename 'Class' to 'isFraud' and drop duplicates
data = data.withColumnRenamed("Class", "isFraud") \
           .dropDuplicates()

# Create new columns 'LogTransactionAmt' and 'FraudLabel'
data = data.withColumn("LogTransactionAmt", log(col("Amount") + 1)) \
           .withColumn("FraudLabel", when(col("isFraud") == 1, "Fraud").otherwise("Non-Fraud"))


# Before writing, ensure the schema is updated by calling .printSchema() 
# to view the current schema.
data.printSchema()
data.write.partitionBy("isFraud").parquet("output_path/partitioned_data")

data = data.repartition(100, "isFraud")

fraud_summary = data.groupBy("isFraud").agg(
    {"Amount": "mean", "Amount": "stddev", "Amount": "max", "Amount": "min"}
)
fraud_summary.show()

total_count = data.count()
fraud_ratio = data.groupBy("isFraud").count().withColumn(
    "FraudPercentage", (col("count") / total_count) * 100
)
fraud_ratio.show()
correlation = data.stat.corr("Amount", "isFraud")
print(f"Correlation between Amount and Fraud: {correlation}")

import matplotlib.pyplot as plt
from pyspark.sql.functions import col

# Assuming 'data' is your PySpark DataFrame containing the transaction data
class_distribution = data.groupBy("isFraud").count() 

# Convert the PySpark DataFrame to a Pandas DataFrame for plotting
fraud_counts = class_distribution.toPandas()

# Create the bar plot
fraud_counts.plot(kind='bar', x='isFraud', y='count', legend=False)
plt.title("Fraud vs Non-Fraud Transaction Counts")
plt.show()