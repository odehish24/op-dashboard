# Databricks notebook source
# MAGIC %md
# MAGIC Using SARIMA Model to Forecast for other product
# MAGIC - Progestogen only pill          
# MAGIC - Chlamydia Treatment            
# MAGIC - Emergency contraception        
# MAGIC - Combined oral contraception    
# MAGIC - Insti kit               
# MAGIC - Condoms - Bolt on              
# MAGIC - Lube - Bolt on                 

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import itertools

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# DBTITLE 1,Retrieve data needed
df = spark.sql('''
            SELECT sh24_uid, product_type, brand_sk, product_sk, order_created_at
            FROM warehouse.sales_events 
            WHERE product_type NOT IN ('STI Test kit', 'STI Test Result Set')
               ''').toPandas()
df.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define all functions

# COMMAND ----------

# DBTITLE 1,Create preprocessing function
# Define Data Preprocessing function
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
    return data_filled  

# COMMAND ----------

# DBTITLE 1,Create a function to generate a forecast df with the trained model
#set no of forecast weeks
steps = 16

#function to generate a forecast df with the trained model
def generate_forecast_df(final_model, steps=steps):
    """
    Parameters:
        final_model: The trained SARIMA model or similar.
        steps: The number of steps to forecast.
    
    Returns:
        DataFrame containing the forecasted values, sorted by 'week'.
    """
    forecast = final_model.forecast(steps=steps)
    new_pd = forecast.reset_index(name='preds')
    new_pd = new_pd.rename(columns={'index': 'week'}).assign(week=lambda x: pd.to_datetime(x['week']))
    new_pd.sort_values(by='week', ascending=True, inplace=True)
    new_pd.reset_index(drop=True, inplace=True)
    
    return new_pd

# new_df = generate_forecast_df(final_model, steps=16)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Progestogen only pill (POP)

# COMMAND ----------

pop = df[df['product_type']=='Progestogen only pill']
pop.shape

# COMMAND ----------

# Apply the data preprocessing function
pop1 = preprocess_weekly_date(pop)

#remove last line due to incomplete data
pop2 = pop1.iloc[:-1]
pop2.plot()

# COMMAND ----------

#split to train and test
train = pop2.iloc[:-16]
test =  pop2.iloc[-16:]

#set no of forecast weeks
steps = 16

# check size
print(train.shape)
print(test.shape)

# COMMAND ----------

# Fit pop SARIMA Model
model = SARIMAX(train['count_order'], order=(2, 2, 1), seasonal_order=(2, 1, 1, 18))
model_fit = model.fit()

#set no of forecast weeks
steps = 16

# Forecast 
forecast = model_fit.forecast(steps=steps)

# Evaluate the error
mae = mean_absolute_error(test['count_order'], forecast)
rmse = np.sqrt(mean_squared_error(test['count_order'], forecast))
mape = mean_absolute_percentage_error(test['count_order'], forecast)

print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {round(mape*100,2)}%")

# COMMAND ----------

# plot
fig, ax = plt.subplots(figsize=(10, 3))

ax.plot(forecast, label='Forecast test')
ax.plot(test.index, test['count_order'], label='True test')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()
plt.show()

# COMMAND ----------

### Train all pop data and Fit SARIMA Model
model = SARIMAX(pop2['count_order'], order=(2, 2, 1), seasonal_order=(2, 1, 1, 18))
pop_model = model.fit()

# Apply generate Forecast df function
pop_pd = generate_forecast_df(pop_model, steps=steps)

#assign  product
pop_pd['product'] = 1
pop_pd.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Chlamydia Treatment (CT)

# COMMAND ----------

#Select only CT data
ct = df[df['product_type']=='Chlamydia Treatment']
ct.shape

# COMMAND ----------

#Apply the preprocessing function
ct1 = preprocess_weekly_date(ct)

#remove last line
ct2 = ct1.iloc[:-1]
ct2.tail()

# COMMAND ----------

#split to train and test
train = ct2.iloc[:-16]
test =  ct2.iloc[-16:]

# check size
print(train.shape)
print(test.shape)

# COMMAND ----------

### Fit ct SARIMA Model
model = SARIMAX(train['count_order'], order=(2, 0, 0), seasonal_order=(0, 1, 2, 18))
model_fit = model.fit()

# Forecast 
forecast = model_fit.forecast(steps=steps)

# Evaluate error
mae = mean_absolute_error(test['count_order'], forecast)
rmse = np.sqrt(mean_squared_error(test['count_order'], forecast))
mape = mean_absolute_percentage_error(test['count_order'], forecast)

print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {round(mape*100,2)}%")

# COMMAND ----------

# plot the test and forecast
fig, ax = plt.subplots(figsize=(10, 3))

ax.plot(forecast, label='Forecast test')
ax.plot(test.index, test['count_order'], label='True test')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()
plt.show()

# COMMAND ----------

### Train all ct data and Forecast with SARIMA Model
model = SARIMAX(ct2['count_order'], order=(2, 0, 0), seasonal_order=(0, 1, 2, 18))
ct_model = model.fit()

# Apply Forecast df function
ct_pd = generate_forecast_df(ct_model, steps=steps)
#assign  product
ct_pd['product'] = 2
ct_pd.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Emergency Contraception (EC)

# COMMAND ----------

ec = df[df['product_type']=='Emergency contraception']
ec.shape

# COMMAND ----------

#Apply the preprocess function
ec1 = preprocess_weekly_date(ec)

#remove last line
ec2 = ec1.iloc[:-1]

#remove outliers at the begining
ec2 = ec2.loc[ec2.index > '2020-04-01']

ec2.plot()

# COMMAND ----------

#split train and test
train = ec2.iloc[:-16]
test =  ec2.iloc[-16:]
# check size
print(train.shape)
print(test.shape)

# COMMAND ----------

### Train and Fit ec SARIMA Model
model = SARIMAX(train['count_order'], order=(1, 1, 0), seasonal_order=(1, 1, 1, 8))
model_fit = model.fit()

# Forecast 
forecast = model_fit.forecast(steps=steps)

# Calculate 
mae = mean_absolute_error(test['count_order'], forecast)
rmse = np.sqrt(mean_squared_error(test['count_order'], forecast))
mape = mean_absolute_percentage_error(test['count_order'], forecast)

print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {round(mape*100,2)}%")

# COMMAND ----------

# plot
fig, ax = plt.subplots(figsize=(10, 3))

ax.plot(forecast, label='Forecast test')
ax.plot(test.index, test['count_order'], label='True test')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()
plt.show()

# COMMAND ----------

### Train all EC forecast with SARIMA Model
model = SARIMAX(ec2['count_order'], order=(1, 1, 0), seasonal_order=(1, 1, 1, 8))
ec_model = model.fit()

# Apply the forecast to df function
ec_pd = generate_forecast_df(ec_model, steps=steps)

#assign  product
ec_pd['product'] = 3
ec_pd.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Combined oral contraception (COC)

# COMMAND ----------

coc = df[df['product_type']=='Combined oral contraception']
coc.shape

# COMMAND ----------

#Apply the preprocess function
coc1 = preprocess_weekly_date(coc)

#remove last line
coc2 = coc1.iloc[:-1]
coc2.plot()

# COMMAND ----------

#split to train and test
train = coc2.iloc[:-16]
test =  coc2.iloc[-16:]

print(train.shape)
print(test.shape)

# COMMAND ----------

### Fit coc SARIMA Model
model = SARIMAX(train['count_order'], order=(1, 1, 0), seasonal_order=(1, 1, 1, 15))
model = model.fit()

# Forecast 
forecast = model.forecast(steps=steps)

# Calculate 
mae = mean_absolute_error(test['count_order'], forecast)
rmse = np.sqrt(mean_squared_error(test['count_order'], forecast))
mape = mean_absolute_percentage_error(test['count_order'], forecast)

print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {round(mape*100,2)}%")

# COMMAND ----------

# plot
fig, ax = plt.subplots(figsize=(10, 3))

ax.plot(forecast, label='Forecast test')
ax.plot(test.index, test['count_order'], label='True test')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()
plt.show()

# COMMAND ----------

### Train all COC and forecast with SARIMA Model
model = SARIMAX(coc2['count_order'], order=(1, 1, 0), seasonal_order=(1, 1, 1, 15))
coc_model = model.fit()

# Apply the forecast df function
coc_pd = generate_forecast_df(coc_model, steps=steps)

#assign  product
coc_pd['product'] = 4
coc_pd.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insti kit 

# COMMAND ----------

inst = df[(df['product_type'] == 'Test at home kit') & (df['product_sk'] == 'insti-hiv-test')]
inst.shape

# COMMAND ----------

#Apply the preprocess function
inst1 = preprocess_weekly_date(inst)

#remove last line
inst1 = inst1.iloc[:-1]
inst1.plot()

# COMMAND ----------

#Remove outliers

# Calculate the mean of 'count_order' where the index > '2023-03-01'
max_value = inst1.loc[inst1.index > '2023-03-01', 'count_order'].max()

# Replace values in 'count_order' that are greater than 300 with the calculated mean
inst1.loc[inst1['count_order'] > 300, 'count_order'] = max_value

#remove the initial zero orders
insti2 = inst1.loc[inst1.index > '2018-12-31']

# COMMAND ----------

#split train and test
train = insti2.iloc[:-16]
test =  insti2.iloc[-16:]

print(train.shape)
print(test.shape)

# COMMAND ----------

### Fit SARIMA Model
model = SARIMAX(train['count_order'], order=(2, 1, 2), seasonal_order=(2, 1, 2, 6))
model_fit = model.fit()

# Forecast 
forecast = model_fit.forecast(steps=steps)

# Evaluate error
mae = mean_absolute_error(test['count_order'], forecast)
rmse = np.sqrt(mean_squared_error(test['count_order'], forecast))
mape = mean_absolute_percentage_error(test['count_order'], forecast)

print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {round(mape*100,2)}%")

# COMMAND ----------

# plot the test and forecast
fig, ax = plt.subplots(figsize=(10, 3))

ax.plot(forecast, label='Forecast test')
ax.plot(test.index, test['count_order'], label='True test')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()
plt.show()

# COMMAND ----------

### Train all insti-kit data Forecast with SARIMA Model
model = SARIMAX(insti2['count_order'], order=(2, 1, 2), seasonal_order=(2, 1, 2, 6))
inst_model = model.fit()

# Apply Forecast df function
inst_pd = generate_forecast_df(inst_model, steps=steps)

#Assign 5 to insti kit product
inst_pd['product'] = 5
inst_pd.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC #### BOLT ON (Condoms and Lubes) #only for sh24

# COMMAND ----------

# DBTITLE 1,Retrieve only Condom bolt on data
cbo = df[df['product_type']=='Condoms - Bolt on']
cbo.shape

# COMMAND ----------

#Apply preprocessing function
cbo1 = preprocess_weekly_date(cbo)

#remove last line
cbo1 = cbo1.iloc[:-1]
cbo1.tail()

# COMMAND ----------

#split to train and test
train = cbo1.iloc[:-16]
test =  cbo1.iloc[-16:]

print(train.shape)
print(test.shape)

# COMMAND ----------

### Fit SARIMA Model
model = SARIMAX(train['count_order'], order=(2, 1, 1), seasonal_order=(2, 1, 2, 8))
model_fit = model.fit()

# Forecast 
forecast = model_fit.forecast(steps=steps)

# Evaluate error
mae = mean_absolute_error(test['count_order'], forecast)
rmse = np.sqrt(mean_squared_error(test['count_order'], forecast))
mape = mean_absolute_percentage_error(test['count_order'], forecast)

print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {round(mape*100,2)}%")

# COMMAND ----------

# plot the test and forecast
fig, ax = plt.subplots(figsize=(10, 3))

ax.plot(forecast, label='Forecast test')
ax.plot(test.index, test['count_order'], label='True test')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()
plt.show()

# COMMAND ----------

### Forecast with cond SARIMA Model
model = SARIMAX(cbo1['count_order'], order=(2, 1, 1), seasonal_order=(2, 1, 2, 8))
cbo_model = model.fit()

# Apply Forecast df function
cbo_pd = generate_forecast_df(cbo_model, steps=steps)

#assign  6 to condom bolt-on product
cbo_pd['product'] = 6
cbo_pd.tail()

# COMMAND ----------

# DBTITLE 1,Lube - Bolt on
lbo = df[df['product_type']=='Lube - Bolt on']
lbo.shape

# COMMAND ----------

#Apply the preprocess function
lbo1 = preprocess_weekly_date(lbo)

#remove last line
lbo1 = lbo1.iloc[:-1]
lbo1.tail()

# COMMAND ----------

#split to train and test
train = lbo1.iloc[:-16]
test =  lbo1.iloc[-16:]

print(train.shape)
print(test.shape)

# COMMAND ----------

### Fit bo SARIMA Model
model = SARIMAX(train['count_order'], order=(2, 1, 1), seasonal_order=(2, 1, 2, 10))
model_fit = model.fit()

# Forecast 
forecast = model_fit.forecast(steps=steps)

# Evaluate error
mae = mean_absolute_error(test['count_order'], forecast)
rmse = np.sqrt(mean_squared_error(test['count_order'], forecast))
mape = mean_absolute_percentage_error(test['count_order'], forecast)

print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {round(mape*100,2)}%")

# COMMAND ----------

# plot the test and forecast
fig, ax = plt.subplots(figsize=(10, 3))

ax.plot(forecast, label='Forecast test')
ax.plot(test.index, test['count_order'], label='True test')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()
plt.show()

# COMMAND ----------

### Forecast with lbo SARIMA Model
model = SARIMAX(lbo1['count_order'], order=(2, 1, 1), seasonal_order=(2, 1, 2, 10))
lbo_model = model.fit()

# Apply Forecast df function
lbo_pd = generate_forecast_df(lbo_model, steps=steps)

#assign  7 to lube bolt-on product
lbo_pd['product'] = 7
lbo_pd.tail()

# COMMAND ----------

####Merge all forecast to one and save table

# COMMAND ----------

merged = pd.concat([pop_pd, ct_pd, ec_pd, coc_pd, inst_pd, cbo_pd, lbo_pd]).reset_index(drop=True)
merged['preds'] = np.ceil(merged['preds'])

# COMMAND ----------

display(merged)

# COMMAND ----------

#Save the merged predictions
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("other_products_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(merged)

# Save as a Parquet table
spark_df.write.format("parquet").mode("overwrite").saveAsTable("other_products_pred")

# COMMAND ----------


