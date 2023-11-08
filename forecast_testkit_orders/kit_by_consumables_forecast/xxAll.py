# Databricks notebook source
# MAGIC %md
# MAGIC ###Kit Consumables Predictions
# MAGIC - Retrieve pred table from all brands
# MAGIC - Assign test_samples, lab and brand
# MAGIC - Calculate consumables 

# COMMAND ----------

# import pandas as pd
# import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Retrieve brands forecast
# Sh24
sh24_pred = spark.sql("select * from sh24_pred")
sh = spark.sql("select * from prep_sti_order where brand_sk = 1")

# Fettle
fettle_pred = spark.sql("select * from fettle_pred")
fe = spark.sql("select * from prep_sti_order where brand_sk = 2")           

# SH24-Ireland
ireland_pred = spark.sql("select * from ireland_pred")
ire = spark.sql("select * from prep_sti_order where brand_sk = 5")

# Freetesting-HIV
freetest_pred = spark.sql("select * from freetest_pred")

# COMMAND ----------

# # Sh24
# sh24_pred = spark.sql("""select * from sh24_pred
#                     """).toPandas()
# sh = spark.sql("""select * from prep_sti_order where brand_sk = 1
#                     """).toPandas()

# # Fettle
# fettle_pred = spark.sql("""select * from fettle_pred
#                     """).toPandas()
# fe = spark.sql("""select * from prep_sti_order where brand_sk = 2
#                     """).toPandas()              

# # SH24-Ireland
# ireland_pred = spark.sql("""select * from ireland_pred
#                     """).toPandas()
# ire = spark.sql("""select * from prep_sti_order where brand_sk = 5
#                     """).toPandas() 

# # freetesting
# freetest_pred = spark.sql("""select * from freetest_pred
#                     """).toPandas()

# COMMAND ----------

# #Get the top 6 test_sample(16% of product type contributes to 98.6% of the orders)
# toplist = [31, 18, 10, 23, 14, 3]

# sh_top = sh[sh['sample_sk'].isin(toplist)]

# #Get the  rest (low orders)
# sh_low = sh[~sh['sample_sk'].isin(toplist)]


# ##Fettle
# fe_top = fe[fe['sample_sk'].isin(toplist)]

# #Get the  rest (low orders)
# fe_low = fe[~fe['sample_sk'].isin(toplist)]

# # Ireland
# irelist = [18, 10, 14, 3]
# ire_top = ire[ire['sample_sk'].isin(irelist)]

# #Get the  rest (low orders)
# ire_low = ire[~ire['sample_sk'].isin(irelist)]
# print(ire_low.shape)

# COMMAND ----------

from pyspark.sql import functions as F

# Define the lists
toplist = [31, 18, 10, 23, 14, 3]
irelist = [18, 10, 14, 3]

# Convert the Python lists to Spark literals
toplist_lit = F.array([F.lit(i) for i in toplist])
irelist_lit = F.array([F.lit(i) for i in irelist])

# Sh24
sh_top = sh.filter(F.col("sample_sk").isin(toplist))
sh_low = sh.filter(~F.col("sample_sk").isin(toplist))

# Fettle
fe_top = fe.filter(F.col("sample_sk").isin(toplist))
fe_low = fe.filter(~F.col("sample_sk").isin(toplist))

# Ireland
ire_top = ire.filter(F.col("sample_sk").isin(irelist))
ire_low = ire.filter(~F.col("sample_sk").isin(irelist))

print(ire_low.count())

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Function that extract previous data 
# #Retrieve length of the predictions 
# pred_weeks = sh24_pred.shape[0]

# #create function that extract previous month same s pred length data
# def previous_weeks_data(df):
#     df['order_created_at'] = pd.to_datetime(df['order_created_at'])
#     last_date = df['order_created_at'].max()
#     weeks_ago = last_date - pd.Timedelta(weeks=pred_weeks)
#     return df[df['order_created_at'] >= weeks_ago]

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import timedelta

# Retrieve length of the predictions
pred_weeks = sh24_pred.count()

# Function to get data for the previous weeks
def previous_weeks_data(df):
    # Convert to timestamp if not already
    df = df.withColumn("order_created_at", F.to_timestamp("order_created_at"))
    
    # Get the max date
    last_date = df.select(F.max("order_created_at")).collect()[0][0]
    
    # Calculate the date for pred_weeks ago
    weeks_ago = last_date - timedelta(weeks=pred_weeks)
    df = df.filter(F.col("order_created_at") >= weeks_ago)
    return df

# Using the function
# previous_weeks_df = previous_weeks_data_spark(sh)


# COMMAND ----------

# DBTITLE 1,function to get the distribution of the top_test_sample
# #get percentage of sample_test value counts from previous months
# def get_sample_percentage(df, column_name):
#     value_counts = df[column_name].value_counts()
#     total_rows = len(df)
#     percentage_counts = (value_counts / total_rows) 
#     return percentage_counts.to_dict()


# #function to add test_sample percentage to the pred df
# def add_sample_percentage(df, column_name, percentages):
#     for new_col, percentage in percentages.items():
#         df[new_col] = df[column_name] * (percentage)        
#     return df


# COMMAND ----------

from pyspark.sql import functions as F

# Function to get percentage of sample_test value counts from previous months
def get_sample_percentage(df, column_name):
    total_rows = df.count()
    value_counts = df.groupBy(column_name).count()
    value_counts = value_counts.withColumn('percentage', (F.col('count') / total_rows))
    percentage_dict = {row[column_name]: row['percentage'] for row in value_counts.collect()}
    return percentage_dict

# Function to add test_sample percentage to the pred df
def add_sample_percentage(df, column_name, percentages):
    for new_col, percentage in percentages.items():
        new_col_str = str(new_col)  # Ensure the column name is a string
        df = df.withColumn(new_col_str, F.col(column_name) * F.lit(percentage))
    return df

# Example usage
# sample_percentages = get_sample_percentage_spark(sh, 'sample_sk')
# sh_with_percentages = add_sample_percentage_spark(sh24_pred, 'your_column_name', sample_percentages)


# COMMAND ----------

# MAGIC %md
# MAGIC #####Get the other sample_test distr

# COMMAND ----------

# def preprocess_weekly_date(df):
#     df1 = df.copy()
    
#     df1 = df1.drop_duplicates()
#     df1.dropna(inplace=True)

#     # Convert 'order_created_at' column to datetime type
#     df1['date'] = pd.to_datetime(df1['order_created_at'], errors='coerce').dt.date
#     df1['date'] = pd.to_datetime(df1['date'])

#     # Create the 'week' column by using the week starting from Monday ('W-MON' - didn't start monday but tuesday)
#     df1['week'] = df1['date'].dt.to_period('W').dt.start_time

#     df1 = df1.groupby([ 'week','sample_sk']).size().reset_index(name='count_order')
#     return df1

# COMMAND ----------



def preprocess_weekly_date(df):
    df1 = df.dropDuplicates()

    df1 = df1.na.drop()

    df1 = df1.withColumn("date", F.to_date("order_created_at"))
    
    df1 = df1.withColumn("week", F.date_trunc("week", "date"))

    df1 = df1.groupBy("week", "sample_sk").count().withColumnRenamed("count", "count_order")
    return df1

# Example usage
# preprocessed_df = preprocess_weekly_date_spark(sh)


# COMMAND ----------

# #function to pivot the weekly df
# def pivot_data(df):
#     df = df.copy()
#     # Pivot the Df, using the sample_code 
#     pivot_df = df.pivot(index='week', columns='sample_sk', values='count_order')

#     # Reset the index to make 'week' a regular column
#     pivot_df.reset_index(inplace=True)

#     # Fill missing values with 0
#     pivot_df.fillna(0, inplace=True)

#     return pivot_df

# COMMAND ----------

def pivot_data(df):
    pivot_df = df.groupBy("week").pivot("sample_sk").agg(F.sum("count_order").alias("count_order"))

    # Fill missing values with 0
    pivot_df = pivot_df.na.fill(0)
    return pivot_df

# Example usage
# pivoted_df = pivot_data_spark(preprocessed_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ##SH24

# COMMAND ----------

# DBTITLE 1,SH24 Apply Functions
# SH24 Apply the previous_weeks_data function for sh_top
sh_top_previous_weeks = previous_weeks_data(sh_top)

# Apply get_sample_percentage sample functn
sample_percentage = get_sample_percentage(sh_top_previous_weeks, 'sample_sk')

# Apply add_sample_percentage function
sh_top_pd = add_sample_percentage(sh24_pred, 'preds', sample_percentage)

#drop column not needed anymore
sh_top_pd = sh_top_pd.drop("preds")


# COMMAND ----------

# Example usage
# sample_percentages = get_sample_percentage_spark(sh, 'sample_sk')
# sh_with_percentages = add_sample_percentage_spark(sh24_pred, 'your_column_name', sample_percentages)
# def preprocess_weekly_date(df):
#     df1 = df.dropDuplicates()

#     df1 = df1.na.drop()

#     df1 = df1.withColumn("date", F.to_date("order_created_at"))
    
#     df1 = df1.withColumn("week", F.date_trunc("week", "date"))

#     df1 = df1.groupBy("week", "sample_sk").count().withColumnRenamed("count", "count_order")
#     return df1




# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Merge top and low test samples pred
#Apply the previous  week for sh_low
sh_low_p = previous_weeks_data(sh_low)

# Apply the weekly preprocess function
sh_low_p= preprocess_weekly_date(sh_low_p)

# pivot the df
sh_low_pd = pivot_data(sh_low_p)
# sh_low_pd.drop('week', axis=1, inplace=True)

# #Merge both sh_top and sh_low togther
# sh_join_pd = sh_top_pd.join(sh_low_pd)
# sh_join_pd.tail()


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Create a temporary index column for both DataFrames
windowSpec = Window().orderBy(F.lit('A'))

sh_top_pd = sh_top_pd.withColumn("row_idx", F.row_number().over(windowSpec))

# Drop the 'week' column from sh_low_pd
sh_low_pd = sh_low_pd.drop("week")

sh_low_pd = sh_low_pd.withColumn("row_idx", F.row_number().over(windowSpec))

# Perform a full outer join on the temporary index column
sh_join_pd = sh_top_pd.join(sh_low_pd, "row_idx", "left")

# Drop the temporary index column
sh_join_pd = sh_join_pd.drop("row_idx")

# COMMAND ----------

display(sh_join_pd)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,SH24 Assign the brand and lab to the df
# # Create a function to assign lab (59% assigned to SPS(2) and the rest to TDL(0))
# def sh24_assign_lab_brand(df):
#     #unpivot df
#     df = pd.melt(df, id_vars=['week'], var_name='sample_sk', value_name='count_preds')
#     df['count_preds'] = np.ceil(df['count_preds'])
    
#     num_rows = len(df)
#     num_sps = int(np.ceil(0.59 * num_rows))
#     lab_values = [2] * num_sps + [0] * (num_rows - num_sps)
#     np.random.shuffle(lab_values)
#     df['lab'] = lab_values
    
#     #assign 1 for sh24 brand
#     df['brand'] = 1
#     return df

# # Group by 'week' and apply the lab and brand assignment function
# sh24_df = sh_join_pd.groupby(['week']).apply(sh24_assign_lab_brand)
# sh24_df.reset_index(drop=True, inplace=True)
# sh24_df.shape

# COMMAND ----------

from pyspark.sql import functions as F
import random

# Function to assign lab (59% assigned to SPS(2) and the rest to TDL(0))
def sh24_assign_lab_brand(df):
    # List of columns to unpivot, excluding 'week'
    cols_to_unpivot = [col for col in df.columns if col != 'week']
    
    # Create an array of structs
    array_of_structs = F.array(*[F.struct(F.lit(c).alias("sample_sk"), F.col(c).alias("count_preds")) for c in cols_to_unpivot])
    
    # Explode the array to unpivot
    df = df.withColumn("exploded", F.explode(array_of_structs)).select("week", "exploded.*")

    # Round up the count_preds
    df = df.withColumn("count_preds", F.ceil("count_preds"))
    
    # Calculate the number of rows and generate lab values
    num_rows = df.count()
    num_sps = int(0.59 * num_rows)
    lab_values = [2] * num_sps + [0] * (num_rows - num_sps)
    random.shuffle(lab_values)
    
    # Create a DataFrame from the lab values and join it
    lab_df = spark.createDataFrame([(l,) for l in lab_values], ["lab"])
    df = df.withColumn("row_id", F.monotonically_increasing_id())
    lab_df = lab_df.withColumn("row_id", F.monotonically_increasing_id())
    df = df.join(lab_df, "row_id").drop("row_id")

    # Assign 1 for sh24 brand
    df = df.withColumn("brand", F.lit(1))
    
    return df

# Assuming sh_join_pd is your DataFrame
sh24_df = sh24_assign_lab_brand(sh_join_pd)

# Show the resulting DataFrame
sh24_df.show()

# Print the shape
print((sh24_df.count(), len(sh24_df.columns)))


# COMMAND ----------



# COMMAND ----------

display(sh24_df)

# COMMAND ----------

sh24_df

# COMMAND ----------

# MAGIC %md
# MAGIC ###FETTLE

# COMMAND ----------

# Apply the previous_weeks_data function
fe_top_previous_weeks = previous_weeks_data(fe_top)

# Apply get_sample_percentage sample functn
sample_percentage = get_sample_percentage(fe_top_previous_weeks, 'sample_sk')

# Apply add_sample_percentage function
fe_top_pd = add_sample_percentage(fettle_pred, 'preds', sample_percentage)

#drop column not needed anymore
fe_top_pd.drop('preds', axis=1, inplace=True)


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

# function unpivot and assign lab and brand to fettle
def fettle_unpivot_assign_lab_brand(df):
    # Unpivot 
    df = pd.melt(df, id_vars=['week'], var_name='sample_sk', value_name='count_preds')
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

# MAGIC %md
# MAGIC ###IRELAND

# COMMAND ----------

# Apply the previous_weeks_data function
ire_top_previous_weeks = previous_weeks_data(ire_top)

# Apply get_sample_percentage sample functn
sample_percentage = get_sample_percentage(ire_top_previous_weeks, 'sample_sk')

# Apply add_sample_percentage function
ire_top_pd = add_sample_percentage(ireland_pred, 'preds', sample_percentage)

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
    df = pd.melt(df, id_vars=['week'], var_name='sample_sk', value_name='count_preds')
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
    df['sample_sk'] = 3

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



# COMMAND ----------

freetest_df.tail() 

# COMMAND ----------

# MAGIC %md
# MAGIC ###Merge all brands SH24, Fettle, Ireland. Freetesting 

# COMMAND ----------

all_df = pd.concat([sh24_df, fettle_df, ireland_df, freetest_df]).reset_index(drop=True)
all_df 

# COMMAND ----------

# MAGIC %md
# MAGIC ####Consumables Predictions Assigned

# COMMAND ----------

df_bom = spark.sql("""
                   SELECT sample_sk1, brand1, lab1, consumable1, c.consumable, count1 
                   FROM bill_of_materials bm 
                   LEFT JOIN consumables c on c.con_sk = bm.consumable1 
                    """).toPandas()

# COMMAND ----------

#Merge all_df with bill_of_mat table
merged_df = pd.merge(all_df, df_bom, 
                     left_on=['sample_sk', 'brand', 'lab'], 
                     right_on=['sample_sk1', 'brand1', 'lab1'], 
                     how='left')

# Performing the aggregation
grouped_df = merged_df.groupby(['week', 'sample_sk', 'brand', 'lab', 'consumable1', 'consumable'])
result = grouped_df.apply(lambda x: (x['count_preds'] * x['count1']).sum()).reset_index(name='total_count')


# COMMAND ----------

display(result)

# COMMAND ----------

#Save the predictions
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("sti_test_kit_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(result)

# Save as a Parquet table
spark_df.write.format("parquet").mode("overwrite").saveAsTable("sti_test_kit_pred")
