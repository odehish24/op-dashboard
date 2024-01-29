# Databricks notebook source
# MAGIC %md
# MAGIC Using SARIMA Model to Forecast for other product
# MAGIC - Progestogen only pill          
# MAGIC - Chlamydia Treatment            
# MAGIC - Emergency contraception        
# MAGIC - Combined oral contraception    
# MAGIC - Test at home - Insti kit & oraquick            
# MAGIC - Condoms - Bolt on              
# MAGIC - Lube - Bolt on                 

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import timedelta
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
            SELECT sh24_uid, product_type, product_sk, order_created_at
            FROM warehouse.sales_events 
            WHERE product_type NOT IN ('STI Test kit', 'STI Test Result Set')
            AND brand_sk = 2
               ''').toPandas()
df.tail()

# COMMAND ----------

df.product_type.value_counts()

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
# set no of forecast weeks
steps = 16

#function to generate a forecast df with the trained model
def generate_forecast_df(final_model, steps=steps):
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

# Apply the data preprocessing function
pop1 = preprocess_weekly_date(pop)

#remove last line due to incomplete data
pop2 = pop1.iloc[:-1]
pop2.plot(figsize=(9,3))

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

# Forecast 
forecast = model_fit.forecast(steps=steps)

# Evaluate the error
mae = mean_absolute_error(test['count_order'], forecast)
rmse = np.sqrt(mean_squared_error(test['count_order'], forecast))
mape = mean_absolute_percentage_error(test['count_order'], forecast)

print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {round(mape*100,2)}%")

# COMMAND ----------

# plot
fig, ax = plt.subplots(figsize=(12, 3))

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

#assign 1 for pop product
pop_pd['product'] = 1
pop_pd.tail()

# COMMAND ----------

# Add 7 to pred to every alternate week
alternate_weeks_mask = pop_pd['week'].dt.week % 2 != 0

# Add 7 to the 'forecast' column where the mask is True
pop_pd.loc[alternate_weeks_mask, 'preds'] += 7

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(pop_pd['week'], pop_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Chlamydia Treatment (CT)

# COMMAND ----------

#Select only CT data
ct = df[df['product_type']=='Chlamydia Treatment']

#Apply the preprocessing function
ct1 = preprocess_weekly_date(ct)

#remove last line
ct2 = ct1.iloc[:-1]
ct2.tail()

# COMMAND ----------

ct2.plot(figsize=(12, 3))

# COMMAND ----------

# split to train and test
train = ct2.iloc[:-16]
test =  ct2.iloc[-16:]

# check size
print(train.shape)
print(test.shape)

# COMMAND ----------

### Fit ct SARIMA Model
model = SARIMAX(train['count_order'], order=(2, 0, 1), seasonal_order=(0, 1, 1, 8))
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
fig, ax = plt.subplots(figsize=(9, 3))

ax.plot(forecast, label='Forecast test')
ax.plot(test.index, test['count_order'], label='True test')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()
plt.show()

# COMMAND ----------

### Train all ct data and Forecast with SARIMA Model
model = SARIMAX(ct2['count_order'], order=(2, 0, 1), seasonal_order=(0, 1, 1, 8))
ct_model = model.fit()

# Apply Forecast df function
ct_pd = generate_forecast_df(ct_model, steps=steps)
#assign  2 for CT product
ct_pd['product'] = 2
ct_pd.tail()

# COMMAND ----------

# Add to every alternate week and Add to the 'forecast' column where the mask is True
alternate_weeks_mask = ct_pd['week'].dt.week % 2 != 0
ct_pd.loc[alternate_weeks_mask, 'preds'] += 10

alternate_weeks_mask = ct_pd['week'].dt.week % 3 != 0
ct_pd.loc[alternate_weeks_mask, 'preds'] += 5

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(ct_pd['week'], ct_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

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
model = SARIMAX(train['count_order'], order=(2, 1, 0), seasonal_order=(1, 1, 1, 8))
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
model = SARIMAX(ec2['count_order'],  order=(2, 1, 0), seasonal_order=(1, 1, 1, 8))
ec_model = model.fit()

# Apply the forecast to df function
ec_pd = generate_forecast_df(ec_model, steps=steps)

#assign 3 to EC product
ec_pd['product'] = 3
ec_pd.tail()

# COMMAND ----------

# Add to every alternate week and Add to the 'forecast' column where the mask is True
alternate_weeks_mask = ct_pd['week'].dt.week % 3 != 0
ct_pd.loc[alternate_weeks_mask, 'preds'] += 5

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(ec_pd['week'], ec_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Combined oral contraception (COC)

# COMMAND ----------

coc = df[df['product_type']=='Combined oral contraception']

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
model = SARIMAX(train['count_order'], order=(2, 1, 1), seasonal_order=(2, 1, 3, 10))
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
model = SARIMAX(coc2['count_order'], order=(2, 1, 1), seasonal_order=(2, 1, 3, 10))
coc_model = model.fit()

# Apply the forecast df function
coc_pd = generate_forecast_df(coc_model, steps=steps)

#assign  4 to coc product
coc_pd['product'] = 4
coc_pd.tail()

# COMMAND ----------

# Add to every alternate week and Add to the 'forecast' column where the mask is True
alternate_weeks_mask = coc_pd['week'].dt.week % 3 != 0
coc_pd.loc[alternate_weeks_mask, 'preds'] += 5

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(coc_pd['week'], coc_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insti kit 

# COMMAND ----------

inst = df[(df['product_type'] == 'Test at home kit') & (df['product_sk'] == 'insti-hiv-test')]

#Apply the preprocess function
inst1 = preprocess_weekly_date(inst)

#remove last line
inst1 = inst1.iloc[:-1]
inst1.plot(figsize=(9,3))

# COMMAND ----------

#split train and test
train = inst1.iloc[:-16]
test =  inst1.iloc[-16:]

print(train.shape)
print(test.shape)

# COMMAND ----------

### Fit SARIMA Model
model = SARIMAX(train['count_order'], order=(1, 0, 2), seasonal_order=(2, 1, 2, 10))
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
fig, ax = plt.subplots(figsize=(9, 3))

ax.plot(forecast, label='Forecast test')
ax.plot(test.index, test['count_order'], label='True test')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()
plt.show()

# COMMAND ----------

### Train all insti-kit data Forecast with SARIMA Model
model = SARIMAX(inst1['count_order'], order=(2, 1, 2), seasonal_order=(2, 1, 2, 6))
inst_model = model.fit()

# Apply Forecast df function
inst_pd = generate_forecast_df(inst_model, steps=steps)

#Assign 5 to insti kit product
inst_pd['product'] = 5
inst_pd.tail()

# COMMAND ----------

# Add to every alternate week and Add to the 'forecast' column where the mask is True
alternate_weeks_mask = inst_pd['week'].dt.week % 3 != 0
inst_pd.loc[alternate_weeks_mask, 'preds'] += 10

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(inst_pd['week'], inst_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####PATCH

# COMMAND ----------

patch = df[df['product_type']=='Patch']

# COMMAND ----------

# Apply the data preprocessing function
patch1 = preprocess_weekly_date(patch)
patch1 = patch1.iloc[:-1]
patch1.plot()

# COMMAND ----------



# COMMAND ----------

# Select the last 16 weeks
patch2 = patch1.iloc[-16:]

# Convert the index to a datetime index 
patch2.index = pd.to_datetime(patch2.index)

# Get the last date from the index
last_date = patch2.index[-1]
# Create a new date range for the next 16 weeks, starting after the last date
new_dates = pd.date_range(start=last_date + timedelta(days=7), periods=16, freq='W-MON')

# ressign the last 16 orders
new_preds = patch2['count_order'].values

# Create a new DataFrame with the new dates and the count_order values
patch_pd = pd.DataFrame(new_preds, index=new_dates, columns=['preds'])

#Add 8 to last 8 weeks 
patch_pd['preds'].iloc[-8:] += 8

#assign  8 to patch product
patch_pd['product'] = 8
patch_pd.reset_index(inplace=True)
patch_pd.rename(columns={'index': 'week'}, inplace=True)
patch_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ####Injectable

# COMMAND ----------

inject = df[df['product_type']=='Injectable']
inject1 = preprocess_weekly_date(inject)
inject1 = inject1.iloc[:-1]
inject1.plot()

# COMMAND ----------

# # Select the last 16 weeks
# inject2 = inject1.iloc[-16:]

# # Convert the index to a datetime index 
# inject2.index = pd.to_datetime(inject2.index)

# # Get the last date from the index
# last_date = inject2.index[-1]
# # Create a new date range for the next 16 weeks, starting after the last date
# new_dates = pd.date_range(start=last_date + timedelta(days=7), periods=16, freq='W-MON')

# # ressign the last 16 orders
# new_preds = inject2['count_order'].values

# # Create a new DataFrame with the new dates and the count_order values
# inject_pd = pd.DataFrame(new_preds, index=new_dates, columns=['preds'])

# # Add 4 to last 8 weeks 
# inject_pd['preds'].iloc[-8:] += 4

# #assign  12 to injectable product
# inject_pd['product'] = 12
# inject_pd.reset_index(inplace=True)
# inject_pd.rename(columns={'index': 'week'}, inplace=True)
# inject_pd

# COMMAND ----------



# COMMAND ----------

# Add to every alternate week and Add to the 'forecast' column where the mask is True
alternate_weeks_mask = coc_pd['week'].dt.week % 3 != 0
coc_pd.loc[alternate_weeks_mask, 'preds'] += 5

# COMMAND ----------

# MAGIC %md
# MAGIC ####Ring 

# COMMAND ----------

ring = df[df['product_type']=='Ring']
ring1 = preprocess_weekly_date(ring)
ring1 = ring1.iloc[:-1]
ring1.plot()

# COMMAND ----------

# Select the last 16 weeks
ring2 = ring1.iloc[-16:]

# Convert the index to a datetime index 
ring2.index = pd.to_datetime(ring2.index)

# Get the last date from the index
last_date = ring2.index[-1]
# Create a new date range for the next 16 weeks, starting after the last date
new_dates = pd.date_range(start=last_date + timedelta(days=7), periods=16, freq='W-MON')

# ressign the last 16 orders
new_preds = ring2['count_order'].values

# Create a new DataFrame with the new dates and the count_order values
ring_pd = pd.DataFrame(new_preds, index=new_dates, columns=['preds'])

# Add 3 to last 8 weeks 
ring_pd['preds'].iloc[-8:] += 3

#assign  14 to ring product
ring_pd['product'] = 14
ring_pd.reset_index(inplace=True)
ring_pd.rename(columns={'index': 'week'}, inplace=True)
ring_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ####Merge all forecast to one and save table

# COMMAND ----------

# all_pd = spark.sql('''select * from other_products_pred''').toPandas()
# merged = pd.concat([all_pd, phd_pd, wart_pd, pregt_pd, inject_pd, herp_pd, ring_pd]).reset_index(drop=True)
# merged['preds'] = np.ceil(merged['preds'])

# COMMAND ----------

merged = pd.concat([pop_pd, coc_pd, ec_pd, inst_pd, ct_pd]).reset_index(drop=True)
merged['preds'] = np.ceil(merged['preds'])

# COMMAND ----------

# merged = pd.concat([pop_pd, ct_pd, ec_pd, coc_pd, inst_pd, cbo_pd, lbo_pd, patch_pd]).reset_index(drop=True)
# merged['preds'] = np.ceil(merged['preds'])

# COMMAND ----------

display(merged)

# COMMAND ----------

# Save the merged predictions
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("fettle_products_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(merged)

# Save as a table
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")
spark_df.write.format("delta").mode("overwrite").saveAsTable("fettle_products_pred")


# COMMAND ----------


