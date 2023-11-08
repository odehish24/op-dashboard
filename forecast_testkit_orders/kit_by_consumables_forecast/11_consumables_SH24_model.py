# Databricks notebook source
# MAGIC %md
# MAGIC ####XGBOOST model for SH24 brand
# MAGIC - The data = STI Test kit orders from sales_events, sti_test_orders, testkit_colour_sample table
# MAGIC - seperate the top test sample from the rest
# MAGIC - Add feature engineering
# MAGIC - Hypertuning
# MAGIC - Train all data
# MAGIC - Save model

# COMMAND ----------

#import libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from pandas.tseries.offsets import DateOffset

import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# DBTITLE 1,Select SH24 brand from STI TEST ORDERS
#Select only SH24 brand from the prep data and convert to pandas df
df = spark.sql("""select * from prep_sti_order where brand_sk = 1
                    """).toPandas()

# COMMAND ----------

#display 
df.tail()

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Separate the test_sample size into top and low 
#Get the top 6 test_sample(16% of product type contributes to 98.6% of the sti test orders)
toplist = [31, 18, 10, 23, 14, 3]

top_df = df[df['sample_sk'].isin(toplist)]
print(top_df.shape)

#Get the  rest (low orders)
low_df = df[~df['sample_sk'].isin(toplist)]
print(low_df.shape)

# COMMAND ----------

# DBTITLE 1,Function for data preprocessing of sh24 kit orders
#define the function 

def preprocessing(df):
    # Copy the DataFrame to prevent modifications to the original
    df_copy = df.copy()

    # Drop rows with any NaN values and duplicates
    df_copy.dropna(inplace=True)
    df_copy.drop_duplicates(inplace=True)

    # Convert order_created_at to datetime and extract the date, then convert to datetime64[ns]
    df_copy['date'] = pd.to_datetime(df_copy['order_created_at']).dt.date
    df_copy['date'] = pd.to_datetime(df_copy['date'])

    # Group by date, counting the number of occurrences (note: each row represent an order)
    df_grouped = df_copy.groupby('date').size().reset_index(name='count_order')
    
    # Create a DataFrame with a complete date range
    complete_date_range = pd.date_range(start=df_grouped['date'].min(), end=df_grouped['date'].max(), freq='D')
    df_complete = pd.DataFrame({'date': complete_date_range})

    # Merge with the grouped data, filling missing counts with 0
    final_df = pd.merge(df_complete, df_grouped, on='date', how='left').fillna(0)

    # Rename the columns as required
    final_df.set_index('date', inplace=True)

    return final_df

# COMMAND ----------

#Apply the preprocessing function to the top df
prep = preprocessing(top_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####1. Split train and test

# COMMAND ----------

# DBTITLE 1,Split data into train and test
from pandas.tseries.offsets import DateOffset

# Sort the DataFrame by the date index 
prep.sort_index(inplace=True)

# Get the latest date in the DataFrame
latest_date = prep.index.max()

# Calculate the split_date as 4-months(122 days) from the latest date
split_date = latest_date - DateOffset(days=122)

# Split prep into training and testing sets based on the calculated split_date
train_df = prep[prep.index <= split_date]
test_df = prep[prep.index > split_date]

print(train_df.shape)
print(test_df.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ####2. Feature Engineering
# MAGIC - NOTE: Multiple features were experimented before arriving at these three features that improved the model 
# MAGIC - Create date features
# MAGIC - Create x_mas season feature
# MAGIC - Create lag feature of 364days ago

# COMMAND ----------

# DBTITLE 1,Date features
def date_features(df):
    
    df = df.copy()
    df['week'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype('int64')
    df['is_weekend'] = df.index.dayofweek.isin([5]).astype(int)

    return df

# COMMAND ----------

# DBTITLE 1,Create xmas season effect
def is_xmas_season(df):
    df.index = pd.to_datetime(df.index)
    df['is_xmas_season'] = ((df.index.month == 12) & (df.index.day > 24)) | ((df.index.month == 1) & (df.index.day == 1))
    df['is_xmas_season'] = df['is_xmas_season'].astype(int)
    return df

# COMMAND ----------

# DBTITLE 1,Create one year lag
def add_lags(df):
    target_map = df['count_order'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Cross Validation
# MAGIC - To improve the training of the data implement cross validation
# MAGIC - Split the data into 6 training parts
# MAGIC - Specify the gap and test size 

# COMMAND ----------

# Time series cross-validation for train and test split
from sklearn.model_selection import TimeSeriesSplit

# specify the split, gap and test size
tss = TimeSeriesSplit(n_splits=6, test_size=122, gap=1)

# COMMAND ----------

#Apply best param
max_depth = 4 
reg_alpha = 0.4
reg_lambda = 0.09
n_estimators = 295
learning_rate = 0.02
gamma = 0.3      

fold = 0
scores = []

data = train_df.copy()
data = data.sort_index()

#Apply the lag feature
add_lags(data)

for train_idx, val_idx in tss.split(data):
    train1 = data.iloc[train_idx]
    test1 = data.iloc[val_idx]
    
    #Apply the x_mas season feature
    train2 = is_xmas_season(train1)
    test2 = is_xmas_season(test1)
    
    #Apply the date features
    train3 = date_features(train2)
    test3 = date_features(test2)

    FEATURES = ['lag1', 'week', 'quarter', 'month', 'dayofyear', 'weekofyear', 'is_xmas_season']
    TARGET = 'count_order'

    X_train = train3[FEATURES].values
    y_train = train3[TARGET].values

    X_test = test3[FEATURES].values
    y_test = test3[TARGET].values

    model = xgb.XGBRegressor( 
                            max_depth = max_depth,  
                            reg_alpha = reg_alpha,
                            reg_lambda = reg_lambda,
                            n_estimators = n_estimators,
                            learning_rate = learning_rate,
                            gamma = gamma,
                            early_stopping_rounds = 50           
        )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=100)

    y_pred = model.predict(X_test)
    
    test1['y_pred'] = y_pred
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)

mae = round(mean_absolute_error(test1['y_pred'].values , test1['count_order'].values),2)
mape = round(mean_absolute_percentage_error(test1['y_pred'].values, test1['count_order'].values)*100,1)
print(f"MAE = {mae}, MAPE = {mape} %")

# COMMAND ----------

# MAGIC %md
# MAGIC ####4. Search for best parameters 
# MAGIC -  Using Bayesian Optimization For XGBoost
# MAGIC

# COMMAND ----------

# from hyperopt import hp, fmin, tpe, Trials

# # Define the search space for hyperparameters
# space = {
#     'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
#     'max_depth': hp.quniform('max_depth', 3, 10, 1),
#     'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
#     'gamma': hp.uniform('gamma', 0, 1),
#     'reg_lambda': hp.uniform('reg_lambda', 0, 1),
#     'reg_alpha': hp.uniform('reg_alpha', 0, 1)
# }

# tss = TimeSeriesSplit(n_splits=6)

# # Define the objective function
# def objective(params):
#     model = xgb.XGBRegressor(
#         n_estimators=int(params['n_estimators']),
#         max_depth=int(params['max_depth']),
#         learning_rate=params['learning_rate'],
#         gamma=params['gamma'],
#         reg_alpha=params['reg_alpha'],
#         reg_lambda=params['reg_lambda'],
#         objective='reg:squarederror'
#     )

#     mape_scores = []
#     for train_idx, val_idx in tss.split(X_train):
#         X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
#         y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

#         model.fit(X_train_fold, y_train_fold)
#         y_preds = model.predict(X_val_fold)
#         mape = mean_absolute_percentage_error(y_val_fold, y_preds)
#         mape_scores.append(mape)

#     avg_mape = np.mean(mape_scores)
#     return avg_mape

# # Perform hyperparameter optimization
# trials = Trials()
# best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

# print("Best hyperparameters:", best)

# COMMAND ----------

# DBTITLE 1,Apply all necessary features and best parameters
#Apply best param   
max_depth = 4 
reg_alpha = 0.4
reg_lambda = 0.09
n_estimators = 269
learning_rate = 0.02
gamma = 0.3    

fold = 0
scores = []

data = prep.copy()
data = data.sort_index()

#Apply the lag feature
add_lags(data)

for train_idx, val_idx in tss.split(data):
    train1 = data.iloc[train_idx]
    test1 = data.iloc[val_idx]
    
    #Apply the x_mas season feature
    train2 = is_xmas_season(train1)
    test2 = is_xmas_season(test1)
    
    #Apply the date features
    train = date_features(train2)
    test = date_features(test2)

    FEATURES = ['lag1', 'week', 'quarter', 'month', 'dayofyear', 'weekofyear', 'is_xmas_season']
    TARGET = 'count_order'

    X_train = train[FEATURES].values
    y_train = train[TARGET].values

    X_test = test[FEATURES].values
    y_test = test[TARGET].values

    model2 = xgb.XGBRegressor( 
                            max_depth = max_depth,  
                            reg_alpha = reg_alpha,
                            reg_lambda = reg_lambda,
                            n_estimators = n_estimators,
                            learning_rate = learning_rate,
                            gamma = gamma,
                            early_stopping_rounds = 50           
        )
    model2.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=100)

    y_pred2 = model2.predict(X_test)
    
    test1['y_pred'] = y_pred2
    score = np.sqrt(mean_squared_error(y_test, y_pred2))
    scores.append(score)

# Evaluate the error
mae = round(mean_absolute_error(test1['y_pred'].values , test1['count_order'].values),2)
mape = round(mean_absolute_percentage_error(test1['y_pred'].values, test1['count_order'].values)*100,1)
print(f"MAE = {mae}, MAPE = {mape} %")

# COMMAND ----------

# DBTITLE 1,Plot
ax = test1[['count_order','y_pred']].plot(figsize=(12,4))
ax.set_ylim([0, ax.get_ylim()[1]]) 

# COMMAND ----------

#check for features that are important to the model
from sklearn.inspection import permutation_importance

# Conduct permutation importance on the test set
result = permutation_importance(model2, X_test, y_test, n_repeats=30, random_state=0)

# Sort feature indices by importance
sorted_idx = result.importances_mean.argsort()

# Plot feature importance
plt.barh(range(X_test.shape[1]), result.importances_mean[sorted_idx])
plt.yticks(range(X_test.shape[1]), [FEATURES[i] for i in sorted_idx])
plt.title('SH24 Feature Importance')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Make Four months prediction
# MAGIC - Create future df for 122 days
# MAGIC - Train all data 
# MAGIC - fit the model

# COMMAND ----------

# DBTITLE 1,Create future df
# Copy prep to all_data and add 'isFuture' column
all_data = prep.copy()
all_data['isFuture'] = False

# Sort all_data by index to ensure the last date is the maximum
all_data.sort_index(inplace=True)

# Get the last date from all_data
last_date = all_data.index.max()

# Calculate the future date 122 days from the last date of all_data
future_end_date = last_date + DateOffset(days=122)

# Create a dynamic future date range
future = pd.date_range(start=last_date + DateOffset(days=1), end=future_end_date, freq='D')
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True

# Merge future df with all_data
merged_df = pd.concat([all_data, future_df])

# COMMAND ----------

# DBTITLE 1,Train final model will at data

# Add feature engineering functions to merged_df
merged_df2 = add_lags(merged_df)
merged_df2 = date_features(merged_df2)
merged_df2 = is_xmas_season(merged_df2)

#Extract the past data
past_df = merged_df2.query('not isFuture')
past_df.dropna(axis=0, inplace =True)

#Extract the future_df part for prediction
future_pd = merged_df2.query('isFuture')

FEATURES = ['lag1', 'week', 'quarter', 'month', 'dayofyear', 'weekofyear', 'is_xmas_season']
TARGET = 'count_order'

X_all = past_df[FEATURES].values
y_all = past_df[TARGET].values


sh24_model = xgb.XGBRegressor( 
                            max_depth = max_depth,  
                            reg_alpha = reg_alpha,
                            reg_lambda = reg_lambda,
                            n_estimators = n_estimators,
                            learning_rate = learning_rate,
                            gamma = gamma
                        )

sh24_model.fit(X_all, y_all, verbose=100)

#Use the SH24 model to Predict 4 months in future
future_pd['preds'] = sh24_model.predict(future_pd[FEATURES].values)

# COMMAND ----------

#plot forecast
future_pd['preds'].plot(figsize=(12,4))

# COMMAND ----------

# DBTITLE 1,Save predictions
from pyspark.sql import SparkSession

#select only the date and preds
sh24_pd = future_pd[['preds']]

# Convert daily index  to week starting from Monday
sh24_pd['week'] = sh24_pd.index.to_period('W').start_time
sh24_pd = sh24_pd.groupby(['week']).agg({'preds': 'sum'}).reset_index()

# Initialize Spark Session
spark = SparkSession.builder.appName("sh24_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(sh24_pd)

# Save as a Parquet table
spark_df.write.format("parquet").mode("overwrite").saveAsTable("sh24_pred")

# COMMAND ----------

# DBTITLE 1,Save SH24 model
# Save the model to DBFS using pickle module
import pickle

with open('/dbfs/tmp/sh24_xgb_model.pkl', 'wb') as file:
    pickle.dump(sh24_model, file)
