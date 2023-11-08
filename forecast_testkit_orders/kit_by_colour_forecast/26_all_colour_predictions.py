# Databricks notebook source
# MAGIC %md
# MAGIC ###Kit Consumables Predictions
# MAGIC - Retrieve pred table from all brands
# MAGIC - Assign test_samples, lab and brand
# MAGIC - Calculate consumables 

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import timedelta

# COMMAND ----------

# DBTITLE 1,Retrieve brands forecast
# Sh24
sh24_pred = spark.sql("""select * from sh24_colour_pred
                    """).toPandas()
sh = spark.sql("""select * from prep_sti_order where brand_sk = 1
                    """).toPandas()

# Fettle
fettle_pred = spark.sql("""select * from fettle_colour_pred
                    """).toPandas()
fe = spark.sql("""select * from prep_sti_order where brand_sk = 2
                    """).toPandas()              

# # SH24-Ireland
# ireland_pred = spark.sql("""select * from ireland_colour_pred
#                     """).toPandas()
# ire = spark.sql("""select * from prep_sti_order where brand_sk = 5
#                     """).toPandas() 

# # freetesting
# freetest_pred = spark.sql("""select * from freetest_colour_pred
#                     """).toPandas()

# COMMAND ----------

#Get the top 6 test_kit code (16% of product type contributes to 98.6% of the sti test orders)
toplist = [547, 544, 71, 68, 479]
sh_top = sh[sh['test_kit_code'].isin(toplist)]

#Get the  rest (low orders)
lowlist = [3, 1, 3551, 2595, 2527, 546, 476, 4573, 545, 70, 2119, 69, 955, 2048, 4641, 4572, 4165, 4027, 680, 3143, 952, 683, 3072, 3619]
sh_low = sh[sh['test_kit_code'].isin(lowlist)]

##Fettle
fe_top = fe[fe['test_kit_code'].isin(toplist)]

#Get the  rest (low orders)
fe_low = fe[fe['test_kit_code'].isin(lowlist)]

# # Ireland
# irelist = [18, 10, 14, 3]
# ire_top = ire[ire['test_kit_code'].isin(irelist)]

# #Get the  rest (low orders)
# ire_low = ire[~ire['test_kit_code].isin(irelist)]
# print(ire_low.shape)

# COMMAND ----------

# DBTITLE 1,Function that extract previous data 
#Retrieve length of the predictions 
pred_weeks = sh24_pred.shape[0]

#create function that extract previous month same s pred length data
def previous_weeks_data(df):
    df1 = df.copy()
    df1['order_created_at'] = pd.to_datetime(df1.loc[:, 'order_created_at'])
    last_date = df1['order_created_at'].max()
    weeks_ago = last_date - pd.Timedelta(weeks=pred_weeks)
    return df1[df1['order_created_at'] >= weeks_ago]

# COMMAND ----------

# DBTITLE 1,function to get the distribution of the top_test_kit_code
#get percentage of testkit_code value counts from previous months
def get_testkitcode_percentage(df, column_name):
    value_counts = df[column_name].value_counts()
    total_rows = len(df)
    percentage_counts = (value_counts / total_rows) 
    return percentage_counts.to_dict()


#function to add test_sample percentage to the pred df
def add_testkitcode_percentage(df, column_name, percentages):
    for new_col, percentage in percentages.items():
        df[new_col] = df[column_name] * (percentage)        
    return df


# COMMAND ----------

# MAGIC %md
# MAGIC #####Get the other sample_test distr

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
    # Pivot the Df, using the sample_sk
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

# DBTITLE 1,SH24 Apply Functions
# SH24 Apply the previous_weeks_data function for sh_top
sh_top_previous_weeks = previous_weeks_data(sh_top)

# Apply get_sample_percentage sample functn
get_percentage = get_testkitcode_percentage(sh_top_previous_weeks, 'test_kit_code')

# Apply add_sample_percentage function
sh_top_pd = add_testkitcode_percentage(sh24_pred, 'preds', get_percentage)

# #drop column not needed anymore
sh_top_pd.drop("preds", axis=1, inplace=True)

# remove 10% from 547 -red kit
sh_top_pd.iloc[:, 1] = sh_top_pd.iloc[:, 1] * 0.85
sh_top_pd.iloc[:, 1] = sh_top_pd.iloc[:, 3] * 0.9
sh_top_pd.iloc[:, 1] = sh_top_pd.iloc[:, 5] * 0.9
sh_top_pd.iloc[:, 1] = sh_top_pd.iloc[:, 4] * 0.93

# COMMAND ----------

sh_top_pd.tail()

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Merge top and low testkit colour pred
#Apply the previous  week for sh_low
sh_low_p = previous_weeks_data(sh_low)

# Apply the weekly preprocess function
sh_low_p= preprocess_weekly_date(sh_low_p)

# pivot the df
sh_low_pd = pivot_data(sh_low_p)

# drop columns not needed anymore
sh_low_pd.drop('week', axis=1, inplace=True)

# Merge both sh_top and sh_low togther
sh_join_pd = sh_top_pd.join(sh_low_pd)
sh_join_pd.tail()


# COMMAND ----------

# DBTITLE 1,SH24 Assign the brand and lab to the df
# Create a function to assign lab (59% assigned to SPS(2) and the rest to TDL(0))
def sh24_assign_lab_brand(df):
    #unpivot df
    df = pd.melt(df, id_vars=['week'], var_name='test_kit_code', value_name='count_preds')
    df['count_preds'] = np.ceil(df['count_preds'])
    
    num_rows = len(df)
    num_sps = int(np.ceil(0.59 * num_rows))
    lab_values = [2] * num_sps + [0] * (num_rows - num_sps)
    np.random.shuffle(lab_values)
    df['lab'] = lab_values
    
    #assign 1 for sh24 brand
    df['brand'] = 1
    return df

# Group by 'week' and apply the lab and brand assignment function
sh24_df = sh_join_pd.groupby(['week']).apply(sh24_assign_lab_brand)
sh24_df.reset_index(drop=True, inplace=True)
sh24_df.shape

# COMMAND ----------

sh24_df = sh24_df[sh24_df['week'] > '2023-10-01']


# COMMAND ----------

sh24_df

# COMMAND ----------

# MAGIC %md
# MAGIC ###FETTLE

# COMMAND ----------

# Apply the previous_weeks_data function
fe_top_previous_weeks = previous_weeks_data(fe_top)

# Apply get_sample_percentage sample functn
get_percentage = get_testkitcode_percentage(fe_top_previous_weeks, 'test_kit_code')

# Apply add_sample_percentage function
fe_top_pd = add_testkitcode_percentage(fettle_pred, 'preds', get_percentage)

#drop column not needed anymore
fe_top_pd.drop('preds', axis=1, inplace=True)


# COMMAND ----------

fe_top_pd

# COMMAND ----------

# DBTITLE 1,Merge
#Apply the previous  week for fe_low
fe_low_p = previous_weeks_data(fe_low)

# Apply the weekly preprocess function
fe_low_p= preprocess_weekly_date(fe_low_p)

# pivot the df
fe_low_pd = pivot_data(fe_low_p)
fe_low_pd.drop('week', axis=1, inplace=True)

#Merge both sh_top and sh_low togther
fe_join_pd = fe_top_pd.join(fe_low_pd)
fe_join_pd.tail()

# COMMAND ----------

fe_join_pd

# COMMAND ----------

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

# Apply the previous_weeks_data function
ire_top_previous_weeks = previous_weeks_data(ire_top)

# Apply get_sample_percentage sample functn
get_percentage = get_testkitcode_percentage(ire_top_previous_weeks, 'test_kit_code')

# Apply add_sample_percentage function
ire_top_pd = add_testkitcode_percentage(ireland_pred, 'preds', get_percentage)

#drop column not needed anymore
ire_top_pd.drop('preds', axis=1, inplace=True)

# COMMAND ----------

#Apply the previous  week for fe_low
ire_low_p = previous_weeks_data(ire_low)

# Apply the weekly preprocess function
ire_low_p= preprocess_weekly_date(ire_low_p)

# pivot the df
ire_low_pd = pivot_data(ire_low_p)
ire_low_pd.drop('week', axis=1, inplace=True)

#Merge both sh_top and sh_low togther
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

def freetest_sample_lab_brand(df):
    # assign sample_sk
    df['test_kit_code'] = green,yellow

    df['count_preds'] = np.ceil(df['preds'])
    
    # Assign 2 for sps lab
    df['lab'] = 2
    
    # Assign 4 for freetest brand
    df['brand'] = 4
     
    df.drop('preds', axis=1, inplace=True)
    
    return df

#Apply 
freetest_df = freetest_sample_lab_brand(freetest_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC #### IRELAND HEP C

# COMMAND ----------

# Create the DataFrame with 18 rows and 4 columns
hepc_df = pd.DataFrame({
    'test_kit_code': [2048]*18,
    'count_preds': [21]*18,
    'lab': [4]*18,
    'brand': [6]*18
})

# Create the 'week' column starting from '2023-10-02' for 18 weeks
start_date = pd.Timestamp('2023-10-02')
week_column = [start_date + timedelta(weeks=i) for i in range(18)]

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
# MAGIC ###Merge all brands SH24, Fettle, Ireland. Freetesting 

# COMMAND ----------

all_df = pd.concat([sh24_df, fettle_df]).reset_index(drop=True)
all_df['test_kit_code'] = all_df['test_kit_code'].astype(int)
all_df 

# COMMAND ----------

# all_df = pd.concat([sh24_df, fettle_df, ireland_df, freetest_df, hepc_df]).reset_index(drop=True)
# all_df['test_kit_code'] = all_df['test_kit_code'].astype(int)
# all_df 

# COMMAND ----------

# MAGIC %md
# MAGIC ####Test kit colour Predictions Assigned

# COMMAND ----------

# df_bom = spark.sql("""
#                    SELECT sample_sk, brand_sk, bm.lab_enum, consumable_sk, c.consumable, count1 
#                    FROM bill_of_materials bm 
#                    LEFT JOIN consumables c on c.con_sk = bm.consumable_sk 
#                     """).toPandas()

# COMMAND ----------

# #Merge all_df with bill_of_materials table
# merged_df = pd.merge(all_df, df_bom, 
#                      left_on=['sample_sk', 'brand', 'lab'], 
#                      right_on=['sample_sk', 'brand_sk', 'lab_enum'], 
#                      how='left')

# # Performing the aggregation
# grouped_df = merged_df.groupby(['week', 'sample_sk', 'brand', 'lab', 'consumable_sk', 'consumable'])
# result = grouped_df.apply(lambda x: (x['count_preds'] * x['count1']).sum()).reset_index(name='total_count')


# COMMAND ----------

display(all_df)

# COMMAND ----------

#Save the predictions
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("testkitcolour_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(all_df)

# Save as a Parquet table
spark_df.write.format("parquet").mode("overwrite").saveAsTable("colour_pred")
