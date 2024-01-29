# Databricks notebook source
# DBTITLE 1,Prophet model to forecast weekly STI test orders
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
from prophet import Prophet
from prophet.diagnostics import cross_validation,performance_metrics
from prophet.plot import plot_cross_validation_metric

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt

import itertools
import random
import logging

# Suppress DEBUG and INFO messages for CmdStanPy
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# COMMAND ----------

# DBTITLE 1,Select Fettle brand from Kit ORDERS
#Retrieve only fettle preprocessed data gotten from table
df = spark.sql("""select * from prep_sti_order where brand_sk = 2
                    """).toPandas()
df.shape

# COMMAND ----------

#Get the top 6 test_kit code (16% of product type contributes to 98.6% of the sti test orders)
toplist = [547, 544, 71, 68, 136]
top_df = df[df['test_kit_code'].isin(toplist)]
print(top_df.shape)

#Get the  rest (low orders)
lowlist = [1, 3, 479, 3551, 2595, 2527, 546, 476, 4573, 545, 70, 2119, 69, 955, 2048, 4641, 4572, 4165, 4027, 680, 3143, 952, 683, 3072, 3619]
low_df = df[df['test_kit_code'].isin(lowlist)]
print(low_df.shape)

# COMMAND ----------

# DBTITLE 1,Function for data preprocessing of Fettle kit orders
# First Function: Data Preprocessing
def preprocess_weekly_date(df):
    df1 = df.copy()
    df1.drop_duplicates(inplace=True)
    df1.dropna(inplace=True)
    df1['date'] = pd.to_datetime(df1['order_created_at'], errors='coerce').dt.date
    df1['date'] = pd.to_datetime(df1['date'])
    df1['count_order'] = 1
    df1['week'] = df1['date'].dt.to_period('W').dt.start_time
    df1 = df1.groupby(['week','test_kit_code']).agg({'count_order': 'sum'}).reset_index()
        
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
prep = preprocess_weekly_date(top_df)
prep = fill_missing_dates(prep)

# remove last week if incomplete data
prep = prep.iloc[:-1, :]
prep.tail()

# COMMAND ----------

# DBTITLE 1,Split to train and test
#split to train and test
train = prep.iloc[:-18, :]
test =  prep[-18:]

# check size
print(train.shape)
print(test.shape)

# COMMAND ----------

# DBTITLE 1,Baseline model
initial = '720 days' 
period = '120 days' 
horizon = '122 days' 

#copy train 
data = train.copy()

# Initialize the Prophet model
m = Prophet()

# Fit the model
m.fit(data)

# Make future DataFrame and populate additional regressor columns
future = m.make_future_dataframe(periods=18, freq='W-MON')

# Generate forecast
forecast1 = m.predict(future)

df_cv = cross_validation(m, initial=initial, period=period, horizon = horizon)
pm = performance_metrics(df_cv)

# COMMAND ----------

#Find the percentage error
forecastd = forecast1.iloc[-18:][['ds','yhat']]
mae = round(mean_absolute_error(test['y'].values, forecastd['yhat'].values),2)
mape = round(mean_absolute_percentage_error(test['y'].values, forecastd['yhat'].values)*100,1)
print(f"MAE = {mae}, MAPE = {mape} %")

# COMMAND ----------

# MAGIC %md
# MAGIC ####HyperParameter Tuning
# MAGIC - Find the best hyperparameters which helps in optimizing the prophet model.

# COMMAND ----------

# from sklearn.model_selection import ParameterGrid
# # Set up parameter grid
# param_grid = {  
#     'changepoint_prior_scale': [0.3, 0.4, 0.5],
#     'holidays_prior_scale': [3, 4, 5],
#     'seasonality_prior_scale': [10, 11, 12, 14],
#     'seasonality_mode': ['additive', 'multiplicative']
# }

# # Generate all combinations of parameters
# all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

# # Create a list to store MAPE values for each combination
# mdapes = [] 

# initial = '720 days' 
# period = '120 days' 
# horizon = '122 days'

# # Use cross validation to evaluate all parameters
# for params in all_params:
#     # Fit a model using one parameter combination
#     m3 = Prophet(**params).fit(train)  
#     # Cross-validation
#     df_cv = cross_validation(m3, initial=initial, period=period, horizon = horizon, parallel="processes")
#     # Model performance
#     df_p = performance_metrics(df_cv, rolling_window=1)
#     # Save model performance metrics
#     mdapes.append(df_p['mdape'].values[0])
    
# # Find the best parameters
# best_params = all_params[np.argmin(mdapes)]
# print(best_params)
# print(df_p['mdape'].values[-1])

# COMMAND ----------

# MAGIC %md
# MAGIC ###Train with best params 
# MAGIC - Fit train and test
# MAGIC - Then predict four months orders in the future

# COMMAND ----------

from prophet.make_holidays import make_holidays_df

year_list = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
holiday = make_holidays_df(year_list=year_list, country='GB')

#remove Spring Bank Holiday from holday
holiday = holiday.loc[holiday['holiday'] != 'Spring Bank Holiday']

initial = '720 days' 
period = '120 days' 
horizon = '122 days' 

#copy train 
data = train.copy()
data['month'] = data['ds'].dt.month

# Initialize the Prophet model
m = Prophet(
    changepoint_range =0.95, #including this for  forecast above 11 weeks
    changepoint_prior_scale= 0.5,
    holidays_prior_scale= 3,
    seasonality_prior_scale=10, #14 and 15 improve the model
    seasonality_mode='additive',
    weekly_seasonality=True,
    daily_seasonality=True,
    yearly_seasonality=True,
    holidays=holiday
    )

# Add the extra regressors
m.add_regressor('month')

# Fit the model
m.fit(data)

# Make future DataFrame and populate additional regressor columns
future = m.make_future_dataframe(periods=18, freq='W-MON')
future['month'] = future['ds'].dt.month

# Generate forecast
forecast = m.predict(future)

df_cv = cross_validation(m, initial=initial, period=period, horizon = horizon)
pm = performance_metrics(df_cv)

# COMMAND ----------

#Find the percentage error
forecastd = forecast.iloc[-18:][['ds','yhat']]
mae = round(mean_absolute_error(test['y'].values, forecastd['yhat'].values),2)
mape = round(mean_absolute_percentage_error(test['y'].values, forecastd['yhat'].values)*100,1)
print(f"MAE = {mae}, MAPE = {mape} %")

# COMMAND ----------

# visualize all the test and forecast test
ax = forecastd.plot(x='ds',y='yhat',label='Forecast test',legend=True,figsize=(10,3))
test.groupby('ds')['y'].sum().plot(label='True test',legend=True)
ax.set_ylim([0, ax.get_ylim()[1]]) 

# COMMAND ----------

# MAGIC %md
# MAGIC ####Note
# MAGIC - The changepoint_range default works in shorter forecast horizon of 11 weeks or less, so no need adding it 
# MAGIC - when the forecast is longer a changepoint_range of 0.95 or higher gives lesser error better accuracy
# MAGIC - when the holiday is added with the holiday_prior there slight improvement of the model
# MAGIC - when the forecast horizon is less than or equall to 11 weeks,seasanality mode can be any of the two options gives same result
# MAGIC - when forecast horizon is higher than 11 weeks, seasonality mode is better with multiplicative
# MAGIC - forecast horizon of 9 weeks and below give 7% MAPE, from 10 - 11weeks 8%, 12week 10%, 16 to 18weeks 14%
# MAGIC -from 9 weeks below, the error does not go beyond 7%

# COMMAND ----------

# DBTITLE 1,Train all data and make future predictions
# Copy all data
all_data = prep.copy()

# Add month regressor to all data
all_data['month'] = all_data['ds'].dt.month

fettle_model = Prophet(
    changepoint_range =0.95, 
    changepoint_prior_scale= 0.5,
    holidays_prior_scale= 3,
    seasonality_prior_scale=10, 
    seasonality_mode='additive',
    weekly_seasonality=True,
    daily_seasonality=True,
    yearly_seasonality=True,
    holidays=holiday
    )

# Add regressors and fit the model
fettle_model.add_regressor('month')
fettle_model.fit(all_data)

# create a future data frame 
future = fettle_model.make_future_dataframe(periods=18, freq='W-MON')
future['month'] = future['ds'].dt.month

#make predictions
forecast = fettle_model.predict(future)
fettle_pd = forecast.iloc[-18:][['ds','yhat']]

# COMMAND ----------

#plot future predictions
ax = fettle_pd.plot(x='ds', y='yhat', label='Forecast', legend=True, figsize=(10, 3))
ax.set_ylim([0, ax.get_ylim()[1]]) 

# COMMAND ----------

# DBTITLE 1,Save predictions
from pyspark.sql import SparkSession

#rename ds and yhat
fettle_pd.rename(columns={'ds': 'week', 'yhat':'preds'}, inplace=True)

#select only the date and preds
fettle_pd = fettle_pd[['week', 'preds']]

# Initialize Spark Session
spark = SparkSession.builder.appName("fettle_colour_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(fettle_pd)

# Save as a table
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")
spark_df.write.format("delta").mode("overwrite").saveAsTable("fettle_colour_pred")
