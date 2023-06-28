# Databricks notebook source
# DBTITLE 1,FETTLE-TDL-fetch from google sheet
from pyspark import SparkFiles

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=1309072669&single=true&output=csv"
spark.sparkContext.addFile(path)

df1 = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#drop the columns not needed
df1 = df1.drop('test_sample','No of Instruction')

# COMMAND ----------

# Fettle -transpose consumable columns into rows and count
import pyspark.sql.functions as F

# Create a list of new column names, excluding "test_kit_code", "brand" and "lab"
fettle_tdl_list = [col for col in df1.columns if col not in ["test_kit_code", "brand", "lab"]]

# Combine all columns into a single array column
df2 = df1.withColumn("array_col", F.array([F.struct(F.lit(c).alias("consumable"), F.col(c).alias("count")) for c in fettle_tdl_list]))

# Explode the array into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))

fettle_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

# COMMAND ----------

# # Save as a Parquet file
# fettle_df.write.parquet("dbfs:/path/to/fettle_df.parquet")

# Retrieve fettle_df from the Parquet file
fettle_df = spark.read.parquet("dbfs:/path/to/fettle_df.parquet")
display(fettle_df)

# COMMAND ----------

# DBTITLE 1,SH24_SPS - fetch from google sheet
#sh24-sps
from pyspark import SparkFiles

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=1239395583&single=true&output=csv"
spark.sparkContext.addFile(path)

sh_sps_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#drop the columnns not needed
sh_sps_df = sh_sps_df.drop('test_sample','No of Instruction')

# COMMAND ----------

#sh24-sps transpose columns
import pyspark.sql.functions as F

# Create a list of new column names, excluding "test_kit_code", "brand" and "lab"
sh_sps_df_list = [col for col in sh_sps_df.columns if col not in ["test_kit_code", "brand", "lab"]]

# Combine all columns into a single array column
df2 = sh_sps_df.withColumn("array_col", F.array([F.struct(F.lit(c).alias("consumable"), F.col(c).alias("count")) for c in sh_sps_df_list]))

# Explode the array into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))

sh_sps_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

# COMMAND ----------

# # Save as a Parquet file
# sh_sps_df.write.parquet("dbfs:/path/to/sh_sps_df.parquet")

# Retrieve sh_sps_df from the Parquet file
sh_sps_df = spark.read.parquet("dbfs:/path/to/sh_sps_df.parquet")
display(sh_sps_df)

# COMMAND ----------

# DBTITLE 1,SH24_TDL - fetch from google sheet
#sh24-tdl
from pyspark import SparkFiles

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=145389625&single=true&output=csv"
spark.sparkContext.addFile(path)

sh_tdl_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#drop the columnns not needed
sh_tdl_df = sh_tdl_df.drop('test_sample','No of Instruction')

# COMMAND ----------

#sh24_tdl transpose columns
import pyspark.sql.functions as F

# Create a list of new column names, excluding "test_kit_code", "brand" and "lab"
sh_tdl_df_list = [col for col in sh_tdl_df.columns if col not in ["test_kit_code", "brand", "lab"]]

# Combine all columns into a single array column
df2 = sh_tdl_df.withColumn("array_col", F.array([F.struct(F.lit(c).alias("consumable"), F.col(c).alias("count")) for c in sh_tdl_df_list]))

# Explode the array into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))

sh_tdl_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

# COMMAND ----------

# # Save as a Parquet file
# sh_tdl_df.write.parquet("dbfs:/path/to/sh_tdl_df.parquet")

# Retrieve sh_tdl_df from the Parquet file
sh_tdl_df = spark.read.parquet("dbfs:/path/to/sh_tdl_df.parquet")

display(sh_tdl_df)

# COMMAND ----------

# DBTITLE 1,IRELAND_ENFER - fetch from google sheet
#ireland_enfer
from pyspark import SparkFiles

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=1671454770&single=true&output=csv"
spark.sparkContext.addFile(path)

ireland_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#drop the columnns not needed
ireland_df = ireland_df.drop('test_sample','No of Instruction')

# COMMAND ----------

#ireland transpose columns
import pyspark.sql.functions as F

# Create a list of new column names, excluding "test_kit_code", "brand" and "lab"
ireland_df_list = [col for col in ireland_df.columns if col not in ["test_kit_code", "brand", "lab"]]

# Combine all columns into a single array column
df2 = ireland_df.withColumn("array_col", F.array([F.struct(F.lit(c).alias("consumable"), F.col(c).alias("count")) for c in ireland_df_list]))

# Explode the array into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))

ireland_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

# COMMAND ----------

# # Save as a Parquet file
# ireland_df.write.parquet("dbfs:/path/to/ireland_df.parquet")

# Read the Parquet file
ireland_df = spark.read.parquet("dbfs:/path/to/ireland_df.parquet")
display(ireland_df)

# COMMAND ----------

# DBTITLE 1,FREETESTING(HIV) - fetch from google
#freetesting
from pyspark import SparkFiles

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=305956146&single=true&output=csv"
spark.sparkContext.addFile(path)

freetest_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#drop the columns not needed
freetest_df = freetest_df.drop('test_sample','No of Instruction')

# COMMAND ----------

#freetest transpose columns
import pyspark.sql.functions as F

# Create a list of new column names, excluding "test_kit_code", "brand" and "lab"
freetest_df_list = [col for col in freetest_df.columns if col not in ["test_kit_code", "brand", "lab"]]

# Combine all columns into a single array column
df2 = freetest_df.withColumn("array_col", F.array([F.struct(F.lit(c).alias("consumable"), F.col(c).alias("count")) for c in freetest_df_list]))

# Explode the array into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))

freetest_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

# COMMAND ----------

# # Save as a Parquet file
# freetest_df.write.parquet("dbfs:/path/to/freetest_df.parquet")

# Read the Parquet file
freetest_df = spark.read.parquet("dbfs:/path/to/freetest_df.parquet")
display(freetest_df)

# COMMAND ----------

# DBTITLE 1,HEPC_IRELAND-ENFER - fetch from google
#Hep_c_ireland
from pyspark import SparkFiles

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=1753729190&single=true&output=csv"
spark.sparkContext.addFile(path)

hepc_ireland_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#drop the columnns not needed
hepc_ireland_df = hepc_ireland_df.drop('test_sample','No of Instruction')

# COMMAND ----------

#hepc_ireland transpose columns
import pyspark.sql.functions as F

# Create a list of new column names, excluding "test_kit_code", "brand" and "lab"
hepc_ireland_list = [col for col in hepc_ireland_df.columns if col not in ["test_kit_code", "brand", "lab"]]

# Combine all columns into a single array column
df2 = hepc_ireland_df.withColumn("array_col", F.array([F.struct(F.lit(c).alias("consumable"), F.col(c).alias("count")) for c in hepc_ireland_list]))

# Explode the array into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))

hepc_ireland_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

# COMMAND ----------

# Save as a Parquet file
hepc_ireland_df.write.mode('overwrite').parquet("dbfs:/path/to/hepc_ireland_df.parquet")

# Read the Parquet file
hepc_ireland_df = spark.read.parquet("dbfs:/path/to/hepc_ireland_df.parquet")
display(hepc_ireland_df)

# COMMAND ----------

# DBTITLE 1,ARAS_ROMANIA
#aras_romania
from pyspark import SparkFiles

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=136572926&single=true&output=csv"
spark.sparkContext.addFile(path)

aras_romania_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#drop the columnns not needed
aras_romania_df = aras_romania_df.drop('test_sample','No of Instruction')

# COMMAND ----------

#aras_romania transpose columns
import pyspark.sql.functions as F

# Create a list of new column names, excluding "test_kit_code", "brand" and "lab"
aras_romania_list = [col for col in aras_romania_df.columns if col not in ["test_kit_code", "brand", "lab"]]

# Combine all columns into a single array column
df2 = aras_romania_df.withColumn("array_col", F.array([F.struct(F.lit(c).alias("consumable"), F.col(c).alias("count")) for c in aras_romania_list]))

# Explode the array into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))

# Select the values from the new construct column
aras_romania_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

# COMMAND ----------

# # Save as a Parquet file
# aras_romania_df.write.mode('overwrite').parquet("dbfs:/path/to/aras_romania_df.parquet")

# Read the Parquet
aras_romania_df = spark.read.parquet("dbfs:/path/to/aras_romania_df.parquet")


# COMMAND ----------

# DBTITLE 1,Combine all seven dfs to one
# combine all 7 tables
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

combined_df = fettle_df.unionByName(sh_sps_df).unionByName(sh_tdl_df).unionByName(ireland_df).unionByName(freetest_df).unionByName(hepc_ireland_df).unionByName(aras_romania_df)
display(combined_df)

# COMMAND ----------

# # Save the DataFrame as a Parquet file
# combined_df.write.mode('overwrite').parquet("dbfs:/path/to/new_combined_df.parquet")

# Read the Parquet file into a DataFrame
combined_df = spark.read.parquet("dbfs:/path/to/new_combined_df.parquet")

# COMMAND ----------

print(combined_df.count())

# COMMAND ----------

#DATA PREPROCESSING BEGINS

# COMMAND ----------

#check no of rows  with count column 0
print(combined_df.filter(combined_df['count'] == 0).count())

# COMMAND ----------

#create a new df that has no count 0 rows from the combined_df 
no_zero_count_df = combined_df.filter(combined_df['count'] != 0)
print(no_zero_count_df.count())

# COMMAND ----------

# DBTITLE 1,Rename columns by adding 1 to the ending
import pyspark.sql.functions as F

#rename the column names add 1 this will differentiate it when joined with kit_created columns later
bill_of_materials_df = no_zero_count_df.select([F.col(c).alias(c+'1') for c in no_zero_count_df.columns])

# COMMAND ----------

# # Save bill_of_materials as a Parquet file
# bill_of_materials_df.write.mode("overwrite").parquet("dbfs:/path/to/bill_of_materials_df.parquet")

# Read the Parquet file
bill_of_materials_df = spark.read.parquet("dbfs:/path/to/bill_of_materials_df.parquet")

# COMMAND ----------

# DBTITLE 1,create a lab table 
#create a lab table
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Define the data
data = [("TDL", 0), ("SPS", 2), ("MEDLAB", 3), ("ENFER", 4)]

# Define the schema
schema = ["lab", "lab_id"]

# Create DataFrame
lab_df = spark.createDataFrame(data, schema)

#create a file and Temp table to save lab table
lab_df.write.mode('overwrite').parquet('dbfs:/path/to/lab.parquet')

# Read the Parquet file into a DataFrame
lab_df = spark.read.parquet('dbfs:/path/to/lab.parquet')


# COMMAND ----------

# DBTITLE 1,create a brand table 
#create a brand table
data = [("SH24", 0), ("FETTLE", 1), ("ARAS_ROMANIA", 3), ("FREETESTING_HIV", 4), 
        ("SH24_IRELAND", 5), ("HEP_C_IRELAND", 6)]

# Define the schema
schema = ["brand", "brand_id"]

# Create DataFrame
brand_df = spark.createDataFrame(data, schema)

#create a file and Temp table to save combined_df table
brand_df.write.mode('overwrite').parquet('dbfs:/path/to/brand.parquet')

# Read the Parquet file into a DataFrame
brand_df = spark.read.parquet('dbfs:/path/to/brand.parquet')


# COMMAND ----------

# DBTITLE 1,create a consumable table - from google sheet
# create a consumables table from google sheet
from pyspark import SparkFiles

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=1111658249&single=true&output=csv"
spark.sparkContext.addFile(path)

consumable_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#create a file and Temp table to save combined_df table
consumable_df.write.mode('overwrite').parquet('dbfs:/path/to/consumable.parquet')

# Read the Parquet file into a DataFrame
consumable_df = spark.read.parquet('dbfs:/path/to/consumable.parquet')

# COMMAND ----------

#END HERE
#CONTINUE FROM 2.stock notebook
