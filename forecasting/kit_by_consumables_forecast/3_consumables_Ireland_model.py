# Databricks notebook source
# DBTITLE 1,MODEL SH24-IRELAND: Prophet model to forecast weekly STI test orders
# MAGIC %md
# MAGIC ###steps
# MAGIC 1. Retrieve and preprocess the data
# MAGIC 2. Train and test 
# MAGIC 3. Evaluate model
# MAGIC 4. Train model (all data)
# MAGIC 5. Predict future orders

# COMMAND ----------

#import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

# Use Mae and rmse to evaluate model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt

import itertools
import random
import logging

# Suppress DEBUG and INFO messages for CmdStanPy
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# COMMAND ----------

#Retrieve only preprocessed data gotten from table
df = spark.sql("""select * from prep_sti_order where brand_sk = 5
                    """).toPandas()
df.shape

# COMMAND ----------

#get top orders
toplist = [18, 10, 14, 3]
top_df = df[df['sample_sk'].isin(toplist)]

#get low
low_df = df[~df['sample_sk'].isin(toplist)]
low_df.shape

# COMMAND ----------

# First Function: Data Preprocessing
def preprocess_weekly_date(df):
    df1 = df.copy()
    df1.drop_duplicates(inplace=True)
    df1.dropna(inplace=True)
    df1['date'] = pd.to_datetime(df1['order_created_at'], errors='coerce').dt.date
    df1['date'] = pd.to_datetime(df1['date'])
    df1['count_order'] = 1
    df1['week'] = df1['date'].dt.to_period('W').dt.start_time
    df1 = df1.groupby(['week','sample_sk']).agg({'count_order': 'sum'}).reset_index()
        
    return df1

# Second Function: Filling Missing Dates
def fill_missing_dates(df):
    df = df.groupby(['week']).agg({'count_order': 'sum'}).reset_index()
    df.set_index('week', inplace=True)
    complete_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-MON')
    data_filled = df.reindex(complete_date_range, fill_value=0)
    data_filled.reset_index(inplace=True)
    data_filled.rename(columns={'index': 'ds', 'count_order': 'y'}, inplace=True)
    
    return data_filled


# COMMAND ----------

# Apply the preprocess function
data = top_df.copy()
prep = preprocess_weekly_date(data)
prep = fill_missing_dates(prep)

# remove last week due to incomplete data
# prep = prep.iloc[:-1, :]


# COMMAND ----------

# MAGIC %md
# MAGIC ####Remove Outliers

# COMMAND ----------

from scipy import stats

# Create a copy of the DataFrame 
prep = prep[prep['ds'] > '2021-06-30'].copy()

# Calculate the z-score
prep['z_score'] = np.abs(stats.zscore(prep['y']))

# Identify outliers and non-outliers
outliers = prep[prep['z_score'] > 3]
non_outliers = prep[prep['z_score'] <= 3]

# Find the next highest score among the non-outliers
next_highest_score = non_outliers['y'].max()

# Replace the outliers using loc
prep.loc[prep['z_score'] > 3, 'y'] = next_highest_score

# Drop the 'z_score' column
prep.drop('z_score', axis=1, inplace=True)


# COMMAND ----------

# DBTITLE 1,Split to train and test
#split to train and test
train = prep.iloc[:-18, :]
test =  prep[-18:]

# check size
print(train.shape)
print(test.shape)

# COMMAND ----------

#display train and test data
train.groupby('ds')['y'].sum().plot(legend=True, label='train', figsize=(12,4))
test.groupby('ds')['y'].sum().plot(legend=True, label='test')

# COMMAND ----------

# Instantiate the baseline model and fit the data
model = Prophet()
model.fit(train)

# Create a future data frame
future = model.make_future_dataframe(periods=18, freq='W-MON')
forecast = model.predict(future)

# Setting cross-validation parameters
initial = 5  # Define the number of cross-validation folds
period = '30 days' 
horizon = '30 days' 

# Use baseline model for cross-validation
df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)

# Initialize lists to store metrics for each fold
mae_list = []
rmse_list = []
mape_list = []

# Loop through each fold and calculate metrics
for i in range(initial):
    fold_results = df_cv[df_cv['cutoff'] == df_cv['cutoff'].unique()[i]]

    # Extract the actual and forecasted values for the current fold
    fold_actual = fold_results['y']
    fold_forecast = fold_results['yhat']

    mae_fold = mean_absolute_error(fold_actual, fold_forecast)
    rmse_fold = sqrt(mean_squared_error(fold_actual, fold_forecast))
    mape_fold = mean_absolute_percentage_error(fold_actual, fold_forecast)

    # Append metrics to lists
    mae_list.append(mae_fold)
    rmse_list.append(rmse_fold)
    mape_list.append(mape_fold)

# Print the evaluation metrics for all folds
for i in range(initial):
    print(f"Metrics for Fold {i + 1}:")
    print(f"MAE: {mae_list[i]}")
    print(f"RMSE: {rmse_list[i]}")
    print(f"MAPE: {round(mape_list[i] * 100, 2)}%")
    print()

#Evaluate error of forecast
forecastd = forecast.iloc[-18:][['ds','yhat']]
mae = mean_absolute_error(test['y'], forecastd['yhat'])
rmse = sqrt(mean_squared_error(test['y'], forecastd['yhat']))
mape = mean_absolute_percentage_error(test['y'], forecastd['yhat'])
print( '\n'.join(['MAE: {0}','RMSE: {1}', f"MAPE: {round(mape*100,2)}%" ]).format(mae, rmse, mape) )

# COMMAND ----------

# visualize all the test and forecast test
ax = forecastd.plot(x='ds',y='yhat',label='Forecast test',legend=True,figsize=(10,3))
test.groupby('ds')['y'].sum().plot(label='True test',legend=True)
ax.set_ylim([0, ax.get_ylim()[1]]) 

# COMMAND ----------

# MAGIC %md
# MAGIC ####Note
# MAGIC - Hypertuning and adding regressors didn't improve the model
# MAGIC - A Simple base model gave the least error 

# COMMAND ----------

# DBTITLE 1,Train all data and make four month predictions
# train all data
ireland_model = Prophet()
ireland_model.fit(prep)

# create a future data frame 
future = ireland_model.make_future_dataframe(periods=18, freq='W-MON')
forecast = ireland_model.predict(future)
ireland_pd = forecast.iloc[-18:][['ds','yhat']]

# COMMAND ----------

#plot future predictions
ax = ireland_pd.plot(x='ds', y='yhat', label='Forecast', legend=True, figsize=(10, 3))
ax.set_ylim([0, ax.get_ylim()[1]]) 

# COMMAND ----------

ireland_pd

# COMMAND ----------

# DBTITLE 1,Save predictions
from pyspark.sql import SparkSession

#rename ds and yhat
ireland_pd.rename(columns={'ds': 'week', 'yhat':'preds'}, inplace=True)

#select only the date and preds
ireland_pd = ireland_pd[['week', 'preds']]

# Initialize Spark Session
spark = SparkSession.builder.appName("ireland_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(ireland_pd)

# Save as a Parquet table
spark_df.write.format("parquet").mode("overwrite").saveAsTable("ireland_pred")
