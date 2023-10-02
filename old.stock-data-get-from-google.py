# Databricks notebook source
# MAGIC %md
# MAGIC ###1. CREATE BILL OF MATERIALS TABLE 

# COMMAND ----------

from pyspark import SparkFiles
import pyspark.sql.functions as F

# COMMAND ----------

# DBTITLE 1,FETTLE-TDL-fetch from google sheet
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

# Transpose the columns into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))

fettle_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

#remove zero count rows
fettle_df = fettle_df.filter(fettle_df['count']!=0)

# COMMAND ----------

# Save as a Parquet file
fettle_df.write.mode('overwrite').parquet("dbfs:/path/to/fettle_df.parquet")

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

# Transpose the array into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))
sh_sps_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

sh_sps_df = sh_sps_df.filter(sh_sps_df['count']!=0)

# COMMAND ----------

# Save as a Parquet file
sh_sps_df.write.mode('overwrite').parquet("dbfs:/path/to/sh_sps_df.parquet")

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

# Transpose the array into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))
sh_tdl_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

sh_tdl_df = sh_tdl_df.filter(sh_tdl_df['count']!=0)

# COMMAND ----------

# Save as a Parquet file
sh_tdl_df.write.mode('overwrite').parquet("dbfs:/path/to/sh_tdl_df.parquet")

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

# Transpose the columns into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))
ireland_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

ireland_df = ireland_df.filter(ireland_df['count']!=0)

# COMMAND ----------

# Save as a Parquet file
ireland_df.write.mode('overwrite').parquet("dbfs:/path/to/ireland_df.parquet")


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

freetest_df = freetest_df.filter(freetest_df['count']!=0)

# COMMAND ----------

# Save as a Parquet file
freetest_df.write.mode('overwrite').parquet("dbfs:/path/to/freetest_df.parquet")

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

# Transpose the columns into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))
hepc_ireland_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

hepc_ireland_df = hepc_ireland_df.filter(hepc_ireland_df['count']!=0)

# COMMAND ----------

# Save as a Parquet file
hepc_ireland_df.write.mode('overwrite').parquet("dbfs:/path/to/hepc_ireland_df.parquet")

# COMMAND ----------

# DBTITLE 1,ARAS_ROMANIA
#aras_romania

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=136572926&single=true&output=csv"
spark.sparkContext.addFile(path)

aras_romania_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#drop the columnns not needed
aras_romania_df = aras_romania_df.drop('test_sample','No of Instruction')

# COMMAND ----------

#aras_romania transpose columns

# Create a list of new column names, excluding "test_kit_code", "brand" and "lab"
aras_romania_list = [col for col in aras_romania_df.columns if col not in ["test_kit_code", "brand", "lab"]]

# Combine all columns into a single array column
df2 = aras_romania_df.withColumn("array_col", F.array([F.struct(F.lit(c).alias("consumable"), F.col(c).alias("count")) for c in aras_romania_list]))

# Transpose the columns into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))
aras_romania_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

aras_romania_df = aras_romania_df.filter(aras_romania_df['count']!=0)

# Save as a Parquet file
aras_romania_df.write.mode('overwrite').parquet("dbfs:/path/to/aras_romania_df.parquet")

# COMMAND ----------

# DBTITLE 1,IRELAND - MEDLAB
#ireland_medlab
from pyspark import SparkFiles

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=187790868&single=true&output=csv"
spark.sparkContext.addFile(path)

medlab_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#drop the columnns not needed
medlab_df = medlab_df.drop('test_sample','No of Instruction')

# COMMAND ----------

#medlab transpose columns
import pyspark.sql.functions as F

# Create a list of new column names, excluding "test_kit_code", "brand" and "lab"
medlab_df_list = [col for col in medlab_df.columns if col not in ["test_kit_code", "brand", "lab"]]

# Combine all columns into a single array column
df2 = medlab_df.withColumn("array_col", F.array([F.struct(F.lit(c).alias("consumable"), F.col(c).alias("count")) for c in medlab_df_list]))

# Transpose the columns into multiple rows
df3 = df2.select("test_kit_code", "brand", "lab", F.explode("array_col").alias("new_col"))
medlab_df = df3.select("test_kit_code", "brand", "lab", "new_col.consumable", "new_col.count")

medlab_df = medlab_df.filter(medlab_df['count']!=0)

# COMMAND ----------

# Save as a Parquet file
medlab_df.write.mode('overwrite').parquet("dbfs:/path/to/medlab_df.parquet")
display(medlab_df)

# COMMAND ----------

# DBTITLE 1,Combine all eight dfs to one
# combine all 8 tables
combined_df = fettle_df.unionByName(sh_sps_df).unionByName(sh_tdl_df).unionByName(ireland_df).unionByName(freetest_df).unionByName(hepc_ireland_df).unionByName(aras_romania_df).unionByName(medlab_df)
display(combined_df)

# COMMAND ----------

# Save the DataFrame as a Parquet file
combined_df.write.mode('overwrite').parquet("dbfs:/path/to/new_combined_df.parquet")

# COMMAND ----------

#DATA PREPROCESSING BELOW

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

# Save as a Parquet file
bill_of_materials_df.write.mode("overwrite").parquet("dbfs:/path/to/bill_of_materials_df.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ###2. CREATE TESTKIT_CODE_COLOUR, LAB, BRAND, CONSUMABLE TABLE

# COMMAND ----------

# DBTITLE 1,Create testkit_code_colour table
# create a testKit_code_colour table from google sheet
from pyspark import SparkFiles

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=1997112257&single=true&output=csv"
spark.sparkContext.addFile(path)

testkit_code_colour_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#select only the columnns needed
testkit_code_colour_df = testkit_code_colour_df.select('test_kit_code', 'colour')

#save table
testkit_code_colour_df.write.mode('overwrite').parquet('dbfs:/path/to/code_colour.parquet')

# COMMAND ----------

# DBTITLE 1,Testkit_code_colour_test_sample
path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=1997112257&single=true&output=csv"
spark.sparkContext.addFile(path)

testkit_code_colour_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#save as table
testkit_code_colour_df.write.saveAsTable("testkit_colour_sample")

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

#save lab table
lab_df.write.mode('overwrite').parquet('dbfs:/path/to/lab.parquet')

# COMMAND ----------

# DBTITLE 1,create a brand table 
#create a brand table
data = [("SH24", 0), ("FETTLE", 1), ("ARAS_ROMANIA", 3), ("FREETESTING_HIV", 4), 
        ("SH24_IRELAND", 5), ("HEP_C_IRELAND", 6)]

# Define the schema
schema = ["brand", "brand_id"]

# Create DataFrame
brand_df = spark.createDataFrame(data, schema)

#save brand table
brand_df.write.mode('overwrite').parquet('dbfs:/path/to/brand.parquet')

# COMMAND ----------

# DBTITLE 1,create a consumable table - from google sheet
# create a consumables table from google sheet
from pyspark import SparkFiles

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRORkKyB9jbaQ7TfpldxQI5zR_tFr4IXmHVPOfy56dORTyzxkumgZqV9k8kd_JwAD0kjAHm3XhfSxLM/pub?gid=1111658249&single=true&output=csv"
spark.sparkContext.addFile(path)

consumable_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#save table
consumable_df.write.mode('overwrite').parquet('dbfs:/path/to/consumable.parquet')

# COMMAND ----------

# DBTITLE 1,Create Tables from parquet file
#create table for bill_of_materials
spark.sql("CREATE TABLE IF NOT EXISTS bill_of_materials USING parquet OPTIONS (path 'dbfs:/path/to/bill_of_materials_df.parquet')")

#create table for consumables
spark.sql("CREATE TABLE IF NOT EXISTS consumables USING parquet OPTIONS (path 'dbfs:/path/to/consumable.parquet')")

#create table for testkit_code_colour
spark.sql("CREATE TABLE IF NOT EXISTS testkit_code_colour USING parquet OPTIONS (path 'dbfs:/path/to/code_colour.parquet')")

#create table for lab
spark.sql("CREATE TABLE IF NOT EXISTS lab USING parquet OPTIONS (path 'dbfs:/path/to/lab.parquet')")

#create table for brand
spark.sql("CREATE TABLE IF NOT EXISTS brand USING parquet OPTIONS (path 'dbfs:/path/to/brand.parquet')")

# COMMAND ----------

# MAGIC %md
# MAGIC CONTINUE FROM 2.stock notebook
