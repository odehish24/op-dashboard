# Databricks notebook source
# MAGIC %md
# MAGIC Using SARIMA Model to Forecast for other product
# MAGIC - Progestogen only pill          
# MAGIC - Chlamydia Treatment            
# MAGIC - Emergency contraception        
# MAGIC - Combined oral contraception    
# MAGIC - Test at home - Insti kit & oraquick (non for sh24 since 2017)            
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

# MAGIC %md
# MAGIC ####SH24 

# COMMAND ----------

# DBTITLE 1,Retrieve data needed
df = spark.sql('''
            SELECT sh24_uid, product_type, product_sk, order_created_at
            FROM warehouse.sales_events 
            WHERE product_type NOT IN ('STI Test kit', 'STI Test Result Set')
            AND brand_sk = 1
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

#Retrieve pop data
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

# assign 1 for pop product
pop_pd['product'] = 1
# pop_pd.tail()

# COMMAND ----------

# Add Trends to pred to every alternate week
alternate_weeks_mask = pop_pd.index % 2 == 1

# Incrementally add values on alternate weeks
incremental_values = 39 + pop_pd.index[alternate_weeks_mask] * 5

# Incrementally update the "preds" column on alternate weeks
pop_pd.loc[alternate_weeks_mask, 'preds'] += incremental_values.values
pop_pd

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(pop_pd['week'], pop_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Chlamydia Treatment (CT)

# COMMAND ----------

# Select only CT data
ct = df[df['product_type']=='Chlamydia Treatment']

#Apply the preprocessing function
ct1 = preprocess_weekly_date(ct)

#remove last line
ct2 = ct1.iloc[:-1]
ct2.tail()

# COMMAND ----------

ct2.plot(figsize=(9,3))

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

# Train all ct data and Forecast with SARIMA Model
model = SARIMAX(ct2['count_order'], order=(2, 0, 0), seasonal_order=(0, 1, 2, 18))
ct_model = model.fit()

# Apply Forecast df function
ct_pd = generate_forecast_df(ct_model, steps=steps)
#assign  2 for CT product
ct_pd['product'] = 2
ct_pd.tail()

# COMMAND ----------

# Add Trend to 'forecast' column for every alternate week
alternate_weeks_mask = ct_pd['week'].dt.week % 3 != 0
ct_pd.loc[alternate_weeks_mask, 'preds'] += 60

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(ct_pd['week'], ct_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Emergency Contraception (EC)

# COMMAND ----------

# Retrieve EC data
ec = df[df['product_type']=='Emergency contraception']

#Apply the preprocess function
ec1 = preprocess_weekly_date(ec)

#remove last line
ec2 = ec1.iloc[:-1]
ec2.plot(figsize=(9,3))

# COMMAND ----------

#split train and test
train = ec2.iloc[:-16]
test =  ec2.iloc[-16:]
# check size
print(train.shape)
print(test.shape)

# COMMAND ----------

# Train and Fit EC SARIMA Model
model = SARIMAX(train['count_order'], order=(2, 1, 1), seasonal_order=(1, 1, 1, 8))
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
model = SARIMAX(ec2['count_order'], order=(2, 1, 1), seasonal_order=(1, 1, 1, 8))
ec_model = model.fit()

# Apply the forecast to df function
ec_pd = generate_forecast_df(ec_model, steps=steps)

#assign 3 to EC product
ec_pd['product'] = 3
ec_pd.tail()

# COMMAND ----------

# Add the seasonality trend to the preds
ec_pd['week'] = pd.to_datetime(ec_pd['week'])  

ec_pd.loc[ec_pd['week'] == '2023-12-18', 'preds'] -= 250
ec_pd.loc[ec_pd['week'] == '2023-12-25', 'preds'] = 0

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(ec_pd['week'], ec_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Combined oral contraception (COC)

# COMMAND ----------

#Retrieve COC data
coc = df[df['product_type']=='Combined oral contraception']

#Apply the preprocess function
coc1 = preprocess_weekly_date(coc)

#remove last line
coc2 = coc1.iloc[:-1]
coc2.plot(figsize=(9,3))

# COMMAND ----------

#split to train and test
train = coc2.iloc[:-16]
test =  coc2.iloc[-16:]

print(train.shape)
print(test.shape)

# COMMAND ----------

# Fit coc SARIMA Model
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

#assign  4 to coc product
coc_pd['product'] = 4
coc_pd.tail()

# COMMAND ----------

# Add seasonality trend to the pred
alternate_weeks_mask = coc_pd['week'].dt.week % 3 != 0
coc_pd.loc[alternate_weeks_mask, 'preds'] += 60

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(coc_pd['week'], coc_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### BOLT ON (Condoms and Lubes) #only for sh24

# COMMAND ----------

# DBTITLE 1,Retrieve only Condom bolt on data
# Retrieve condom bolt-on data
cbo = df[df['product_type']=='Condoms - Bolt on']

#Apply preprocessing function
cbo1 = preprocess_weekly_date(cbo)

#remove last line
cbo1 = cbo1.iloc[:-1]
cbo1.tail()

# COMMAND ----------

cbo1.plot(figsize=(9,3))

# COMMAND ----------

#split to train and test
train = cbo1.iloc[:-16]
test =  cbo1.iloc[-16:]

print(train.shape)
print(test.shape)

# COMMAND ----------

### Fit SARIMA Model
model = SARIMAX(train['count_order'], order=(1, 1, 2), seasonal_order=(2, 1, 2, 12))
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

# Forecast with condom bolt-on SARIMA Model
model = SARIMAX(cbo1['count_order'], order=(1, 1, 2), seasonal_order=(2, 1, 2, 12))
cbo_model = model.fit()

# Apply Forecast df function
cbo_pd = generate_forecast_df(cbo_model, steps=steps)

#assign  6 to condom bolt-on product
cbo_pd['product'] = 6

# COMMAND ----------

# Add trend to preds
alternate_weeks_mask = cbo_pd['week'].dt.week % 4 == 0
cbo_pd.loc[alternate_weeks_mask, 'preds'] += 115

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(cbo_pd['week'], cbo_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# DBTITLE 1,Lube - Bolt on
# Retrieve lube bolt-on data
lbo = df[df['product_type']=='Lube - Bolt on']

#Apply the preprocess function
lbo1 = preprocess_weekly_date(lbo)

#remove last line
lbo1 = lbo1.iloc[:-1]
lbo1.tail()

# COMMAND ----------

lbo1.plot(figsize=(9,3))

# COMMAND ----------

#split to train and test
train = lbo1.iloc[:-16]
test =  lbo1.iloc[-16:]

print(train.shape)
print(test.shape)

# COMMAND ----------

### Fit lbo SARIMA Model
model = SARIMAX(train['count_order'], order=(1, 1, 2), seasonal_order=(2, 1, 2, 12))
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
model = SARIMAX(lbo1['count_order'], order=(1, 1, 2), seasonal_order=(2, 1, 2, 12))
lbo_model = model.fit()

# Apply Forecast df function
lbo_pd = generate_forecast_df(lbo_model, steps=steps)

#assign  7 to lube bolt-on product
lbo_pd['product'] = 7
lbo_pd.tail()

# COMMAND ----------

# Add trends to preds
alternate_weeks_mask = lbo_pd['week'].dt.week % 4 == 0
lbo_pd.loc[alternate_weeks_mask, 'preds'] += 115

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(lbo_pd['week'], lbo_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####PATCH

# COMMAND ----------

# Retrieve patch data
patch = df[df['product_type']=='Patch']

# Apply the data preprocessing function
patch1 = preprocess_weekly_date(patch)
patch1 = patch1.iloc[:-1]
patch1.plot(figsize=(9,3))

# COMMAND ----------

#split to train and test
train = patch1.iloc[:-16]
test =  patch1.iloc[-16:]

print(train.shape)
print(test.shape)

# COMMAND ----------

# Fit patch SARIMA Model
model = SARIMAX(train['count_order'], order=(1, 1, 2), seasonal_order=(2, 1, 2, 12))
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

### Forecast patch SARIMA Model
model = SARIMAX(patch1['count_order'], order=(1, 1, 2), seasonal_order=(2, 1, 2, 12))
patch_model = model.fit()

# Apply Forecast df function
patch_pd = generate_forecast_df(patch_model, steps=steps)

#assign  8 to patch product
patch_pd['product'] = 8
patch_pd.tail()

# COMMAND ----------

# Add trend to preds
alternate_weeks_mask = patch_pd.index % 3 == 1

# Incrementally add values on alternate weeks
incremental_values =  patch_pd.index[alternate_weeks_mask] + 4

# Incrementally update the "preds" column on alternate weeks
patch_pd.loc[alternate_weeks_mask, 'preds'] += incremental_values.values

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(patch_pd['week'], patch_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Pregnancy test - Bolt on   

# COMMAND ----------

#Retrieve preg data
pregt = df[df['product_type']=='Pregnancy test - Bolt on']

# Apply the data preprocessing function
pregt1 = preprocess_weekly_date(pregt)
pregt1 = pregt1.iloc[:-1]
pregt1.plot(figsize=(9,3))

# COMMAND ----------

#split to train and test
train = pregt1.iloc[:-16]
test =  pregt1.iloc[-16:]

print(train.shape)
print(test.shape)

# COMMAND ----------

### Fit preg test SARIMA Model
model = SARIMAX(train['count_order'], order=(1, 1, 2), seasonal_order=(2, 1, 2, 12))
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

### Forecast preg test SARIMA Model
model = SARIMAX(pregt1['count_order'], order=(1, 1, 2), seasonal_order=(2, 1, 2, 12))
patch_model = model.fit()

# Apply Forecast df function
pregt_pd = generate_forecast_df(patch_model, steps=steps)

#assign 11  to pregt product
pregt_pd['product'] = 11
pregt_pd.tail()

# COMMAND ----------

# Add trend to preds
alternate_weeks_mask = pregt_pd.index % 3 == 1

# Incrementally add values on alternate weeks
incremental_values =  pregt_pd.index[alternate_weeks_mask] + 4

# Incrementally update the "preds" column on alternate weeks
pregt_pd.loc[alternate_weeks_mask, 'preds'] += incremental_values.values

# COMMAND ----------

plt.figure(figsize=(9, 3))
plt.plot(pregt_pd['week'], pregt_pd['preds'])
plt.ylim([0, plt.ylim()[1]])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Photo diagnosis

# COMMAND ----------

# Retrieve photo diagnosis data
phd = df[df['product_type']=='Photo diagnosis']

# Apply the data preprocessing function
phd1 = preprocess_weekly_date(phd)
phd1 = phd1.iloc[:-1]
phd1.plot(figsize=(9,3))

# COMMAND ----------

# Select the last 16 weeks
phd2 = phd1.iloc[-16:]

# Convert the index to a datetime index 
phd2.index = pd.to_datetime(phd2.index)

# Get the last date from the index
last_date = phd2.index[-1]
# Create a new date range for the next 16 weeks, starting after the last date
new_dates = pd.date_range(start=last_date + timedelta(days=7), periods=16, freq='W-MON')

# ressign the last 16 orders
new_preds = phd2['count_order'].values

# Create a new DataFrame with the new dates and the count_order values
phd_pd = pd.DataFrame(new_preds, index=new_dates, columns=['preds'])

# Add 4 to last 8 weeks 
phd_pd['preds'].iloc[-8:] += 4

#assign  9 to phd product
phd_pd['product'] = 9
phd_pd.reset_index(inplace=True)
phd_pd.rename(columns={'index': 'week'}, inplace=True)
phd_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ####Warts treatment

# COMMAND ----------

#Retrieve wart data
wart = df[df['product_type']=='Warts treatment']

# Apply the data preprocessing function
wart1 = preprocess_weekly_date(wart)
wart1 = wart1.iloc[:-1]
wart1.plot(figsize=(9,3))

# COMMAND ----------

# Select the last 16 weeks
wart2 = wart1.iloc[-16:]

# Convert the index to a datetime index 
wart2.index = pd.to_datetime(wart2.index)

# Get the last date from the index
last_date = wart2.index[-1]
# Create a new date range for the next 16 weeks, starting after the last date
new_dates = pd.date_range(start=last_date + timedelta(days=7), periods=16, freq='W-MON')

# ressign the last 16 orders
new_preds = wart2['count_order'].values

# Create a new DataFrame with the new dates and the count_order values
wart_pd = pd.DataFrame(new_preds, index=new_dates, columns=['preds'])

# Add 4 to last 8 weeks 
wart_pd['preds'].iloc[-8:] += 4

#assign  10 to wart product
wart_pd['product'] = 10
wart_pd.reset_index(inplace=True)
wart_pd.rename(columns={'index': 'week'}, inplace=True)
wart_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ####Injectable

# COMMAND ----------

# Retrieve injectable data
inject = df[df['product_type']=='Injectable']
inject1 = preprocess_weekly_date(inject)
inject1 = inject1.iloc[:-1]
inject1.plot(figsize=(9,3))

# COMMAND ----------

# Select the last 16 weeks
inject2 = inject1.iloc[-16:]

# Convert the index to a datetime index 
inject2.index = pd.to_datetime(inject2.index)

# Get the last date from the index
last_date = inject2.index[-1]
# Create a new date range for the next 16 weeks, starting after the last date
new_dates = pd.date_range(start=last_date + timedelta(days=7), periods=16, freq='W-MON')

# ressign the last 16 orders
new_preds = inject2['count_order'].values

# Create a new DataFrame with the new dates and the count_order values
inject_pd = pd.DataFrame(new_preds, index=new_dates, columns=['preds'])

# Add 4 to last 8 weeks 
inject_pd['preds'].iloc[-8:] += 4

#assign  12 to injectable product
inject_pd['product'] = 12
inject_pd.reset_index(inplace=True)
inject_pd.rename(columns={'index': 'week'}, inplace=True)
inject_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ####Herpes treatment

# COMMAND ----------

# Retrieve herp data
herp = df[df['product_type']=='Herpes treatment']
herp1 = preprocess_weekly_date(herp)
herp1 = inject1.iloc[:-1]
herp1.plot(figsize=(9,3))

# COMMAND ----------

# Select the last 16 weeks
herp2 = herp1.iloc[-16:]

# Convert the index to a datetime index 
herp2.index = pd.to_datetime(herp2.index)

# Get the last date from the index
last_date = herp2.index[-1]
# Create a new date range for the next 16 weeks, starting after the last date
new_dates = pd.date_range(start=last_date + timedelta(days=7), periods=16, freq='W-MON')

# ressign the last 16 orders
new_preds = herp2['count_order'].values

# Create a new DataFrame with the new dates and the count_order values
herp_pd = pd.DataFrame(new_preds, index=new_dates, columns=['preds'])

# Add 4 to last 8 weeks 
herp_pd['preds'].iloc[-8:] += 4

#assign  13 to herp product
herp_pd['product'] = 13
herp_pd.reset_index(inplace=True)
herp_pd.rename(columns={'index': 'week'}, inplace=True)
herp_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ####Ring 

# COMMAND ----------

#Retrieve ring data
ring = df[df['product_type']=='Ring']
ring1 = preprocess_weekly_date(ring)
ring1 = ring1.iloc[:-1]
ring1.plot(figsize=(9,3))

# COMMAND ----------

# MAGIC %md
# MAGIC ####### NOTE:
# MAGIC - insufficent data to make forecast for the ring products

# COMMAND ----------

# MAGIC %md
# MAGIC ####Merge all SH24 forecast to one and save table

# COMMAND ----------

merged = pd.concat([pop_pd, ct_pd, ec_pd, coc_pd, lbo_pd, patch_pd, phd_pd, wart_pd, pregt_pd, inject_pd, herp_pd]).reset_index(drop=True)
merged['preds'] = np.ceil(merged['preds'])

# COMMAND ----------

# Assign 1 to sh24 brand
merged['brand_sk'] = 1
merged

# COMMAND ----------

# Save the merged predictions
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("other_products_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(merged)

# Save as a table
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")
spark_df.write.format("delta").mode("overwrite").saveAsTable("other_products_pred")

# COMMAND ----------

# MAGIC %md
# MAGIC #### FREETESTING-HIV

# COMMAND ----------

df = spark.sql('''
            SELECT sh24_uid, product_type, brand_sk, product_sk, order_created_at
            FROM warehouse.sales_events 
            WHERE product_type NOT IN ('STI Test kit', 'STI Test Result Set')
            AND brand_sk NOT IN (1,2)
               ''').toPandas()
df.tail()

# COMMAND ----------

# Retrieve insti data
inst = df[(df['product_type'] == 'Test at home kit') & (df['product_sk'] == 'insti-hiv-test')]

#Apply the preprocess function
inst1 = preprocess_weekly_date(inst)

#remove last line
inst1 = inst1.iloc[:-1]
inst1.plot(figsize=(9,3))

# COMMAND ----------

# Select the last 16 weeks
inst2 = inst1.iloc[-16:]

# Convert the index to a datetime index 
inst2.index = pd.to_datetime(inst2.index)

# Get the last date from the index
last_date = inst2.index[-1]

# Create a new date range for the next 16 weeks, starting after the last date
new_dates = pd.date_range(start=last_date + timedelta(days=7), periods=16, freq='W-MON')

# ressign the last 16 orders
new_preds = inst2['count_order'].values

# Create a new DataFrame with the new dates and the count_order values
inst_pd = pd.DataFrame(new_preds, index=new_dates, columns=['preds'])
inst_pd.reset_index(inplace=True)
inst_pd.rename(columns={'index': 'week'}, inplace=True)

# COMMAND ----------

#Add the NHFTW projection
inst_pd.loc[inst_pd['week'] == '2024-02-05', 'preds'] += 11000
inst_pd.loc[inst_pd['week'] == '2024-02-12', 'preds'] += 4000

#Assign 5 to insti kit product
inst_pd['product'] = 5

# Assign freetest brand
inst_pd['brand_sk'] = 4

# COMMAND ----------

# MAGIC %md
# MAGIC #### IRELAND - CT

# COMMAND ----------

# Select only CT data
ct = df[df['product_type']=='Chlamydia Treatment']

# Apply the preprocessing function
ct1 = preprocess_weekly_date(ct)

# remove last line
# ct1 = ct1.iloc[:-1]

# COMMAND ----------

# Select the last 16 weeks
ct2 = ct1.iloc[-15:]

# Convert the index to a datetime index 
ct2.index = pd.to_datetime(ct2.index)

# Get the last date from the index
last_date = ct2.index[-1]
# Create a new date range for the next 16 weeks, starting after the last date
new_dates = pd.date_range(start=last_date + timedelta(days=7), periods=15, freq='W-MON')

# ressign the last 16 orders
new_preds = ct2['count_order'].values

# Create a new DataFrame with the new dates and the count_order values
ct_pd = pd.DataFrame(new_preds, index=new_dates, columns=['preds'])
ct_pd.reset_index(inplace=True)
ct_pd.rename(columns={'index': 'week'}, inplace=True)

# COMMAND ----------

ct_pd['product'] = 2
ct_pd['brand_sk'] = 5
ct_pd

# COMMAND ----------

all_pd = spark.sql('''select * from other_products_pred''').toPandas()
merged = pd.concat([all_pd, ct_pd]).reset_index(drop=True)
merged['preds'] = np.ceil(merged['preds'])

# COMMAND ----------

# Save the merged predictions
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("other_products_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(merged)

# Save as a table
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")
spark_df.write.format("delta").mode("overwrite").saveAsTable("other_products_pred")
