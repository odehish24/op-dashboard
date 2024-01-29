# Databricks notebook source
# DBTITLE 1,SARIMA model to forecast weekly STI test orders
# MAGIC %md
# MAGIC ###steps
# MAGIC 1. Retrieve and preprocess the data for fettle
# MAGIC 2. Train model (split data) 
# MAGIC 3. Forecast test data
# MAGIC 4. Evaluate model
# MAGIC 5. Train model again (all data)
# MAGIC 6. Predict future orders

# COMMAND ----------

#import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt

import itertools
import random
import logging

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress DEBUG and INFO messages for CmdStanPy
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# COMMAND ----------

#Retrieve only freetesting preprocessed data gotten from table
df = spark.sql("""select * from prep_sti_order where brand_sk = 4
                    """).toPandas()
df.shape

# COMMAND ----------

# Function for Data Preprocessing
def preprocess_weekly_date(df):
    df1 = df.copy()
    df1.drop_duplicates(inplace=True)
    df1['date'] = pd.to_datetime(df1['order_created_at'], errors='coerce').dt.date
    df1['date'] = pd.to_datetime(df1['date'])
    df1['count_order'] = 1
    df1['week'] = df1['date'].dt.to_period('W').dt.start_time
    df1 = df1.groupby(['week']).agg({'count_order': 'sum'}).reset_index()
    df1.set_index('week', inplace=True)
    
    # Create a complete date range and reindex
    complete_date_range = pd.date_range(start=df1.index.min(), end=df1.index.max(), freq='W-MON')
    data_filled = df1.reindex(complete_date_range, fill_value=0)
    # data_filled.index.freq = 'W-MON'   
    return data_filled  

# COMMAND ----------

# Apply the preprocess function
prep = preprocess_weekly_date(df)

# remove outliers 
prep['count_order'] = prep['count_order'].apply(lambda x: min(x, 450))

# COMMAND ----------


prep.plot(legend=True, figsize=(10,4))

# COMMAND ----------

#split to train and test
train = prep.iloc[:-18, :]
test =  prep[-18:]

# check size
print(train.shape)
print(test.shape)

# COMMAND ----------

# # Define the p, d, and q parameters to take any value between 0 and 2 (inclusive)
# p = d = q = range(0, 3)

# # Define the seasonal parameters (P, D, Q, S)
# P = D = Q = range(0, 2)
# S = [12]  # Seasonal length is 12 

# # Generate all different combinations of p, d, q, P, D, Q, and S
# pdq = list(itertools.product(p, d, q))
# seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(P, D, Q)) for s in S]

# # Initialize variables to store best params
# best_aic = float('inf')
# best_params = None
# best_seasonal_params = None

# # Grid Search
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             model = SARIMAX(train['count_order'], order=param, seasonal_order=param_seasonal)
#             fit = model.fit(disp=False)
#             if fit.aic < best_aic:
#                 best_aic = fit.aic
#                 best_params = param
#                 best_seasonal_params = param_seasonal
#         except:
#             continue

# print(f"Best SARIMA model parameters: {best_params}x{best_seasonal_params} with AIC: {best_aic}")

# COMMAND ----------

# Fit SARIMA Model
model = SARIMAX(train['count_order'], order=(0,1,2), seasonal_order=(0, 1, 1, 12))  # Adjust p,d,q parameters as needed
model_fit = model.fit()

# Forecast test
forecast = model_fit.forecast(steps=18)

#convert to pd df
forecast_df  = forecast.to_frame(name='forecast')
forecast_df.index.name = 'date'

mae = mean_absolute_error(test['count_order'], forecast)
rmse = np.sqrt(mean_squared_error(test['count_order'], forecast))
mape = mean_absolute_percentage_error(test['count_order'], forecast)

print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {round(mape*100,2)}%")

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 3))

# Plotting 
test['count_order'].groupby(level=0).sum().plot(ax=ax, label='Test')
forecast_df['forecast'].plot(ax=ax, label='Forecast test')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()
plt.show()



# COMMAND ----------

# DBTITLE 1,Train all data and make future predictions
# Fit SARIMA Model
model = SARIMAX(prep['count_order'], order=(0,1,2), seasonal_order=(0, 1, 1, 12))
freetest_model = model.fit()

# Forecast test
forecastd = freetest_model.forecast(steps=18)

freetest_pd = forecastd.to_frame(name='freetest_pd')

# reset index
freetest_pd.reset_index(inplace=True)
freetest_pd.columns = ['week', 'preds']

freetest_pd['week'] = pd.to_datetime(freetest_pd['week'])

# Sort  by 'week' 
freetest_pd.sort_values(by='week', ascending=True, inplace=True)

# Reset the index if you'd like
freetest_pd.reset_index(drop=True, inplace=True)


# COMMAND ----------

freetest_pd

# COMMAND ----------

fig, ax1 = plt.subplots(figsize=(10, 3))
freetest_pd.groupby('week')['preds'].sum().plot(ax=ax1, legend=True, label='Test')
ax1.set_ylim([0, ax1.get_ylim()[1]])
ax1.legend()
plt.show()

# COMMAND ----------

# DBTITLE 1,Save predictions
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("freetest_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(freetest_pd)

# Save as a Parquet table
spark_df.write.format("parquet").mode("overwrite").saveAsTable("freetest_pred")
