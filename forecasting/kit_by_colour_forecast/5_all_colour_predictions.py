# Databricks notebook source
# MAGIC %md
# MAGIC ###Kit by Colour Forecast
# MAGIC - Retrieve pred table from all brands
# MAGIC - Assign test_samples, lab and brand
# MAGIC - Calculate consumables 

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import timedelta

# COMMAND ----------

# DBTITLE 1,Function that extract previous data 
#create function that extract previous 18 weeks data
def previous_weeks_data(df):
    df1 = df.copy()
    df1['order_created_at'] = pd.to_datetime(df1.loc[:, 'order_created_at'])
    last_date = df1['order_created_at'].max()
    weeks_ago = last_date - pd.Timedelta(weeks=18)
    return df1[df1['order_created_at'] >= weeks_ago]

# COMMAND ----------

# DBTITLE 1,function to get the distribution of the top_test_kit_code
# get percentage of testkitcode value counts from previous weeks
def get_testkitcode_percentage(df, testkitcode_column):
    value_counts = df[testkitcode_column].value_counts(normalize=True)
    testkitcode_percentages = value_counts.to_dict()
    return testkitcode_percentages


# function to extract testkitcode percentage to the pred df
def add_testkitcode_percentage(df, pred_column, testkitcode_percentages):
    for new_col, percentage in testkitcode_percentages.items():
        df[new_col] = df[pred_column] * (percentage)        
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC #####Functions to process Low testkit colour forecast values

# COMMAND ----------

def preprocess_weekly_date(df):
    df1 = df.copy()
    
    df1 = df1.drop_duplicates()
    df1.dropna(inplace=True)

    # Convert 'order_created_at' column to datetime type
    df1['date'] = pd.to_datetime(df1.loc[:, 'order_created_at'], errors='coerce').dt.date
    df1['date'] = pd.to_datetime(df1.loc[:, 'date'])

    # Create the 'week' column by using the week starting from Monday ('W-MON' - didn't start monday but tuesday)
    df1['week'] = df1['date'].dt.to_period('W').dt.start_time

    df1 = df1.groupby([ 'week','test_kit_code']).size().reset_index(name='count_order')
    return df1

# COMMAND ----------

#function to pivot the weekly df
def pivot_data(df):
    df = df.copy()
    # Pivot the Df, using the test_kit_code
    pivot_df = df.pivot(index='week', columns='test_kit_code', values='count_order')

    # Reset the index to make 'week' a regular column
    pivot_df.reset_index(inplace=True)

    # Fill missing values with 0
    pivot_df.fillna(0, inplace=True)
    
    return pivot_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##SH24

# COMMAND ----------

# Retrieve Sh24 data
sh24_pred = spark.sql("""select * from sh24_colour_pred""").toPandas()
sh = spark.sql("""select * from prep_sti_order where brand_sk = 1 and order_created_at > '2023-06-01'
                    """).toPandas()

# Get the top 6 test_kit code 
toplist = [547, 544, 71, 68, 479]
sh_top = sh[sh['test_kit_code'].isin(toplist)]

# Get the  rest (low orders)
lowlist = [1,2,3,136,272,408,4640,3617,545,546,2595,4641,3619,680,683,952,5049,1979,4027,955,3141,69,70,2119,4165,3143,204,4301,207,2255,6620,5597,477,7645,6621,478,1503,4572,476,4573,3551,2527]
sh_low = sh[sh['test_kit_code'].isin(lowlist)]

# COMMAND ----------

# remove last line
sh24_pred.drop(sh24_pred.index[-1], inplace=True)
sh24_pred

# COMMAND ----------

# DBTITLE 1,Apply Functions to top orders
# SH24 Apply the previous_weeks_data function for sh_top
sh_top_previous_weeks = previous_weeks_data(sh_top)

# Apply get_sample_percentage sample functn
get_percentage = get_testkitcode_percentage(sh_top_previous_weeks, 'test_kit_code')

# Apply add_sample_percentage function
sh_top_pd = add_testkitcode_percentage(sh24_pred, 'preds', get_percentage)

# drop column not needed anymore
sh_top_pd.drop("preds", axis=1, inplace=True)

# COMMAND ----------

# DBTITLE 1,#Apply for sh_low
#Apply the previous  week for sh_low
sh_low_p = previous_weeks_data(sh_low)

# Apply the weekly preprocess function
sh_low_p= preprocess_weekly_date(sh_low_p)

# pivot the df
sh_low_pd = pivot_data(sh_low_p)

#remove the first row of the week due to incomplete data
sh_low_pd = sh_low_pd.drop(sh_low_pd.index[0]).reset_index(drop=True)

# drop columns not needed anymore
sh_low_pd.drop('week', axis=1, inplace=True)

# drop the last line
sh_low_pd = sh_low_pd[:-1]

# COMMAND ----------

# DBTITLE 1,Merge top and low testkit colour pred

# Merge both sh_top and sh_low togther
sh_join_pd = sh_top_pd.join(sh_low_pd)
sh_join_pd.tail()


# COMMAND ----------

# DBTITLE 1,SH24 Assign the brand and lab to the df
# Create a function to assign lab and brand
def sh24_lab_brand(df):
    # unpivot df
    df = pd.melt(df, id_vars=['week'], var_name='test_kit_code', value_name='count_preds')
    df['count_preds'] = np.ceil(df['count_preds'])
    
    #Assign lab %
    df['sps'] = (df['count_preds'] * 0.58).astype(int)  
    df['tdl'] = df['count_preds'] - df['sps']
    df.drop('count_preds', axis=1, inplace=True)

    # First, add a unique ID column to preserve the relationship between the new rows
    df['id'] = df.index

    # Now melt the DataFrame to unpivot the sps and tdl columns
    df = df.melt(id_vars=['id', 'week', 'test_kit_code'], var_name='lab', value_name='count_preds')

    # drop id
    df.drop('id', axis=1, inplace=True)

    # Replace 'sps' with 2 and 'tdl' with 0 
    df['lab'] = df['lab'].replace({'sps': 2, 'tdl': 0})
    df['lab'] = df['lab'].astype(int)

    # assign 1 for sh24 brand
    df['brand'] = 1
    return df

# COMMAND ----------

# Apply the sh24_lab_brand function
sh24_df = sh24_lab_brand(sh_join_pd)
sh24_df.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC ###FETTLE

# COMMAND ----------

fettle_pd = spark.sql("""select * from fettle_colour_pred""").toPandas()

fe = spark.sql("""select * from prep_sti_order where brand_sk = 2 and order_created_at > '2023-06-01'""").toPandas()  

# get top list
toplist = [547, 544, 71, 68, 136]
fe_top = fe[fe['test_kit_code'].isin(toplist)]
print(fe_top.shape)

# Get the  rest (low orders)
lowlist = [1,2,3,272,479,408,4640,3617,545,546,2595,4641,3619,680,683,952,5049,1979,4027,955,3141,69,70,2119,4165,3143,204,4301,207,2255,6620,5597,477,7645,6621,478,1503,4572,476,4573,3551,2527]
fe_low = fe[fe['test_kit_code'].isin(lowlist)]
print(fe_low.shape)

# COMMAND ----------

fettle_pd

# COMMAND ----------

# DBTITLE 1,Apply to top fettle orders
# Apply the previous_weeks_data function
fe_top_previous_weeks = previous_weeks_data(fe_top)

# Apply get_sample_percentage sample functn
get_percentage = get_testkitcode_percentage(fe_top_previous_weeks, 'test_kit_code')

# Apply add_sample_percentage function
fe_top_pd = add_testkitcode_percentage(fettle_pd, 'preds', get_percentage)

# #drop column not needed anymore
fe_top_pd.drop('preds', axis=1, inplace=True)


# COMMAND ----------

fe_top_pd

# COMMAND ----------

# DBTITLE 1,Apply to low orders
#Apply the previous  week for fe_low
fe_low_p = previous_weeks_data(fe_low)

# Apply the weekly preprocess function
fe_low_p= preprocess_weekly_date(fe_low_p)

# pivot the df
fe_low_pd = pivot_data(fe_low_p)
fe_low_pd.drop('week', axis=1, inplace=True)

# drop the first and last rows
fe_low_pd = fe_low_pd.iloc[1:-1].reset_index(drop=True)

# COMMAND ----------

# DBTITLE 1,Merge

# Merge both sh_top and sh_low togther
fe_join_pd = fe_top_pd.join(fe_low_pd)
fe_join_pd.tail()

# COMMAND ----------

# DBTITLE 1,Assign brand and lab
# function unpivot and assign lab and brand to fettle
def fettle_unpivot_assign_lab_brand(df):
    # Unpivot 
    df = pd.melt(df, id_vars=['week'], var_name='test_kit_code', value_name='count_preds')
    df['count_preds'] = np.ceil(df['count_preds'])
    
    # Assign 0 for TDL lab
    df['lab'] = 0
    
    # Assign 2 for fettle brand
    df['brand'] = 2
    return df

# Group by 'week' and apply the lab and brand assignment function
fettle_df = fe_join_pd.groupby(['week']).apply(fettle_unpivot_assign_lab_brand)
fettle_df.reset_index(drop=True, inplace=True)

# COMMAND ----------

fettle_df

# COMMAND ----------

# MAGIC %md
# MAGIC ###IRELAND

# COMMAND ----------

# # SH24-Ireland
ireland_pred = spark.sql("""select * from ireland_colour_pred""").toPandas()
ire = spark.sql("""select * from prep_sti_order where brand_sk = 5 and order_created_at > '2023-06-01' """).toPandas() 

# get top orders
toplist = [547, 71, 3551, 2527, 683, 207]
ire_top = ire[ire['test_kit_code'].isin(toplist)]

#get low
lowlist = [544, 68, 479, 3, 1, 136, 2595, 546, 476, 4573, 545, 70, 2119, 69, 955, 2048, 4641, 4572, 4165, 4027, 680, 3143, 952, 3072, 3619, 4694, 3087, 905, 747, 359, 319, 294, 217, 183, 167, 149, 147, 125, 108, 46, 37, 35, 34, 27, 25, 23, 22, 20, 16, 12, 10]
ire_low = ire[ire['test_kit_code'].isin(lowlist)]
ire_low.shape

# COMMAND ----------

# DBTITLE 1,Apply to top
# Apply the previous_weeks_data function
ire_top_previous_weeks = previous_weeks_data(ire_top)

# Apply get_sample_percentage sample functn
get_percentage = get_testkitcode_percentage(ire_top_previous_weeks, 'test_kit_code')

# Apply add_sample_percentage function
ire_top_pd = add_testkitcode_percentage(ireland_pred, 'preds', get_percentage)

# drop column not needed anymore
ire_top_pd.drop('preds', axis=1, inplace=True)

# COMMAND ----------

# remove red 547 from 580 from 2023-12-04 to 2024-02-26j
# remove pink 71, from 200
# add lime 683, add 500
# add biscuit 207, add 200

ire_top_pd['week'] = pd.to_datetime(ire_top_pd['week'])

# Define the date range
start_date = '2023-12-04'
end_date = '2024-02-26'

# Create a mask to identify rows within the specified date range
mask = (ire_top_pd['week'] >= start_date) & (ire_top_pd['week'] <= end_date)

# Subtract from the '547' column for rows within the date range
ire_top_pd.loc[mask, 547] -= 600
ire_top_pd.loc[mask, 71] -= 150
ire_top_pd.loc[mask, 683] += 500
ire_top_pd.loc[mask, 207] += 210

# COMMAND ----------

ire_top_pd

# COMMAND ----------

# DBTITLE 1,Apply to low and merge
#Apply the previous  week for fe_low
ire_low_p = previous_weeks_data(ire_low)

# Apply the weekly preprocess function
ire_low_p= preprocess_weekly_date(ire_low_p)

# pivot the df
ire_low_pd = pivot_data(ire_low_p)
ire_low_pd.drop('week', axis=1, inplace=True)

#match the length
length = len(ire_top_pd)
ire_low_pd = ire_low_pd[:length]

# Merge both sh_top and sh_low togther
ire_join_pd = ire_top_pd.join(ire_low_pd)

# COMMAND ----------

ire_join_pd.tail()

# COMMAND ----------

# function unpivot and assign lab and brand to fettle
def ireland_unpivot_assign_lab_brand(df):
    # Unpivot 
    df = pd.melt(df, id_vars=['week'], var_name='test_kit_code', value_name='count_preds')
    df['count_preds'] = np.ceil(df['count_preds'])
    
    # Assign 4 for Enfer lab
    df['lab'] = 4
    
    # Assign 2 for fettle brand
    df['brand'] = 5
    return df

# Group by 'week' and apply the lab and brand assignment function
ireland_df = ire_join_pd.groupby(['week']).apply(ireland_unpivot_assign_lab_brand)
ireland_df.reset_index(drop=True, inplace=True)

# COMMAND ----------

ireland_df

# COMMAND ----------

# MAGIC %md
# MAGIC ###FREETESTING

# COMMAND ----------

# DBTITLE 1,Retrieve freetesting data
freetest_pred = spark.sql("""select * from freetest_colour_pred""").toPandas()
ft = spark.sql("""select * from prep_sti_order where brand_sk = 4 and order_created_at > '2023-06-01'""").toPandas()

# get top orders
toplist = [3, 1]
ft_top = ft[ft['test_kit_code'].isin(toplist)]

# COMMAND ----------

# DBTITLE 1,Apply functions
# Apply the previous_weeks_data function
ft_top_previous_weeks = previous_weeks_data(ft_top)

# Apply get_sample_percentage sample function
get_percentage = get_testkitcode_percentage(ft_top_previous_weeks, 'test_kit_code')

# Apply add_sample_percentage function
ft_top_pd = add_testkitcode_percentage(freetest_pred, 'preds', get_percentage)

#drop column not needed anymore
ft_top_pd.drop('preds', axis=1, inplace=True)

# COMMAND ----------

# add NHIV week forecast 
ft_top_pd.loc[ft_top_pd['week'] == '2024-02-05', 1] = 623   
ft_top_pd.loc[ft_top_pd['week'] == '2024-02-05', 3] = 8440 

ft_top_pd.loc[ft_top_pd['week'] == '2024-02-12', 1] = 459 
ft_top_pd.loc[ft_top_pd['week'] == '2024-02-12', 3] = 2630 

# COMMAND ----------

def freetest_lab_brand(df):
    # Unpivot 
    df = pd.melt(df, id_vars=['week'], var_name='test_kit_code', value_name='count_preds')
    df['count_preds'] = np.ceil(df['count_preds'])
    
    # Assign 2 for sps lab
    df['lab'] = 2
    
    # Assign 4 for freetest brand
    df['brand'] = 4
     
    # df.drop('preds', axis=1, inplace=True)
    
    return df

#Apply 
freetest_df = freetest_lab_brand(ft_top_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC #### IRELAND HEP C

# COMMAND ----------

# Create the DataFrame with 17 rows and 4 columns
hepc_df = pd.DataFrame({
    'test_kit_code': [2048] * 17,
    'count_preds': np.random.randint(19, 27, size=17),  
    'lab': [4] * 17,
    'brand': [6] * 17
})

# Create the 'week' column for 17 weeks
start_date = pd.Timestamp('2023-12-04')
week_column = [start_date + timedelta(weeks=i) for i in range(17)]

# Insert 'week' column at the beginning
hepc_df.insert(0, 'week', week_column)

# Set the data types for each column
hepc_df = hepc_df.astype({
    'week': 'datetime64[ns]',
    'test_kit_code': 'int',
    'count_preds': 'int',
    'lab': 'int',
    'brand': 'int'
})
hepc_df.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Merge all brands SH24, Fettle, Ireland, Freetesting, HepC

# COMMAND ----------

all_df = pd.concat([sh24_df, fettle_df, ireland_df, freetest_df, hepc_df]).reset_index(drop=True)
all_df['test_kit_code'] = all_df['test_kit_code'].astype(int)
all_df 

# COMMAND ----------

# MAGIC %md
# MAGIC ####Test kit colour Predictions Saved

# COMMAND ----------

# Save the predictions
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("testkitcolour_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(all_df)

# Save as a table
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")
spark_df.write.format("delta").mode("overwrite").saveAsTable("colour_pred")

