import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
## ETL Pipeline Functions
def extractData(file_path):
    return pd.read_csv(file_path)

def cleanseData(data):
    data = data.drop_duplicates()
    data = data.dropna()
    valid_types = ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    data = data[data['type'].isin(valid_types)]
    data = data[data['amount'] > 0]
    return data

def transformData(data, conversion_rate=1.1):
    data['amount_converted'] = data['amount'] * conversion_rate
    data['timestamp'] = pd.to_datetime(data['step'], unit='h', origin='2021-01-01')
    return data

def aggregateData(data):
    data['day'] = data['timestamp'].dt.date
    data['week'] = data['timestamp'].dt.to_period('W')
    data['month'] = data['timestamp'].dt.to_period('M')
    daily_summary = data.groupby('day').agg({'amount': 'sum', 'amount_converted': 'sum'})
    weekly_summary = data.groupby('week').agg({'amount': 'sum', 'amount_converted': 'sum'})
    monthly_summary = data.groupby('month').agg({'amount': 'sum', 'amount_converted': 'sum'})
    return daily_summary, weekly_summary, monthly_summary

def dataQualityChecks(data):
    issues = {}
    if (data['amount'] < 0).any():
        issues['Negative Amounts'] = (data['amount'] < 0).sum()
    valid_types = ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    invalid_types = data[~data['type'].isin(valid_types)]
    if not invalid_types.empty:
        issues['Invalid Transaction Types'] = len(invalid_types)
    missing_values = data.isnull().sum()
    if missing_values.any():
        issues['Missing Values'] = missing_values
    return issues

def create_summary_reports(data):
    summary = data.describe()
    plt.figure(figsize=(10, 6))
    sns.histplot(data['amount'], bins=50, kde=True, color='blue')
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.show()
    return summary

def main(file_path):
    raw_data = extract_data(file_path)
    cleansed_data = cleanse_data(raw_data)
    transformed_data = transform_data(cleansed_data)
    daily, weekly, monthly = aggregate_data(transformed_data)
    issues = data_quality_checks(transformed_data)
    summary = create_summary_reports(transformed_data)
    print("Data Quality Issues:", issues)
    print("\nSummary Statistics:", summary)
    print("\nDaily Summary:", daily)
    print("\nWeekly Summary:", weekly)
    print("\nMonthly Summary:", monthly)

file_path = "path_to_dataset.csv"
main(file_path)
