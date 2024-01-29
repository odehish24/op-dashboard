# Databricks notebook source
# MAGIC %md
# MAGIC ###1. CREATE BILL OF MATERIALS TABLE 

# COMMAND ----------

from pyspark import SparkFiles
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import regexp_replace

# COMMAND ----------

# DBTITLE 1,Extract test-sample-type from required_klasses to determine the kit consumables
#Query to get all testkit_codes and required_klasses
df = spark.sql("""
    SELECT test_kit_code, required_klasses
    FROM raw_admin.testkit_metadata 
    """)

#remove the {} curly braces from the required_klasses
df = df.withColumn('required_klasses', regexp_replace('required_klasses', '[{}]', ''))

# Replace "::" with "_" to create test_names column
df = df.withColumn('test_names', F.regexp_replace('required_klasses', '::', '_'))

#count number of test in test_names columns
df = df.withColumn('test_count', F.size(F.split('test_names', ',')))

# drop column not needed
df = df.drop('required_klasses')

# COMMAND ----------

# save as table
# df.write.format("delta").mode("overwrite").saveAsTable("prod.default.test_names_count")

# COMMAND ----------

display(df)

# COMMAND ----------

# create a function that extract test-sample-type from required_klasses
def extract_test_sample(row):
    keywords = ['Treponemal', 'RPR', 'Blood', 'Urine', 'Oral', 'Anal', 'Vaginal']
    found_keywords = []
    blood_keywords = ['Treponemal', 'RPR', 'Blood']
    blood_found = False

    for keyword in keywords:
        if keyword in row:
            if keyword in blood_keywords and not blood_found:
                found_keywords.append('Blood')
                blood_found = True
            elif keyword not in blood_keywords and keyword not in found_keywords:
                found_keywords.append(keyword)
                
    # Return unique keywords as a comma-separated string
    return ','.join(found_keywords)

# Define the UDF
extract_test_sample_udf = udf(extract_test_sample, StringType())

# Create the new column (test_sample) by applying the UDF
df = df.withColumn('test_sample', extract_test_sample_udf(df['required_klasses']))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Retrieve data from google sheet

# COMMAND ----------

# DBTITLE 1,Create function that retrieve data from Google sheet and transform
def load_and_transform_data(spark, path):
    
    # Add the file to the Spark context
    spark.sparkContext.addFile(path)

    # Read the file from the provided path
    df = spark.read.csv("file://" + SparkFiles.get("pub"), header=True, inferSchema=True)

    # Exclude specified columns and combine others into an array column
    brand_list = [col for col in df.columns if col not in ["sample_sk", "brand_sk", "lab_enum"]]

    df2 = df.withColumn("array_col", F.array([F.struct(F.lit(c).alias("consumable_sk"), F.col(c).alias("count1")) for c in brand_list]))

    # Transpose the array into multiple rows
    df3 = df2.select("sample_sk", "brand_sk", "lab_enum", F.explode("array_col").alias("new_col"))
    df4 = df3.select("sample_sk", "brand_sk", "lab_enum", "new_col.consumable_sk", "new_col.count1")

    # Filter out rows with 'count' equal to 0
    df5 = df4.filter(df4['count1'] != 0)

    #change dtype to int
    brand_df = df5.withColumn("consumable_sk", F.col("consumable_sk").cast("int"))

    #rename the column names add 1 this will differentiate it when joined with kit_created columns later
    # brand_df = df5.select([F.col(c).alias(c+'1') for c in df5.columns])

    return brand_df


# COMMAND ----------

# DBTITLE 1,SH:24 SPS 
#SH:24 SPS 
path1 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=1239395583&single=true&output=csv"

#call function
sh24_sps = load_and_transform_data(spark, path1)
sh24_sps.createOrReplaceTempView('sh24_sps')


# COMMAND ----------

#SH:24 TDL 
path2 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=538017527&single=true&output=csv"

#call function
sh24_tdl = load_and_transform_data(spark, path2)
sh24_tdl.createOrReplaceTempView('sh24_tdl')
# display(sh24_tdl)

# COMMAND ----------

#Fettle TDL 
path3 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=1852084597&single=true&output=csv"

#call function
fettle = load_and_transform_data(spark, path3)
fettle.createOrReplaceTempView('fettle')
# display(fettle)

# COMMAND ----------

# DBTITLE 1,IRELAND_ENFER
#IRELAND ENFER
path4 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=1671454770&single=true&output=csv"

#call function
ireland_enfer = load_and_transform_data(spark, path4)
ireland_enfer.createOrReplaceTempView('ireland_enfer')

# display(ireland_enfer)

# COMMAND ----------

# DBTITLE 1,IRELAND MEDLAB 
#IRELAND MEDLAB 
path5 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=187790868&single=true&output=csv"

#call function
ireland_medlab = load_and_transform_data(spark, path5)
ireland_medlab.createOrReplaceTempView('ireland_medlab')
# display(ireland_medlab)

# COMMAND ----------

# DBTITLE 1,HEP_C_IRELAND ENFER
#HEP_C_IRELAND ENFER
path6 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=1753729190&single=true&output=csv"

#call function
hepc = load_and_transform_data(spark, path6)
hepc.createOrReplaceTempView('hepc')
# display(hepc)

# COMMAND ----------

# DBTITLE 1,FREETESTING SPS
#FREETESTING SPS
path7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=305956146&single=true&output=csv"

#call function
freetesting = load_and_transform_data(spark, path7)
freetesting.createOrReplaceTempView('freetesting')
# display(freetesting)

# COMMAND ----------

# DBTITLE 1,ARAS ROMANIA TDL
#ARAS ROMANIA TDL

path8 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=136572926&single=true&output=csv"

#call function
aras = load_and_transform_data(spark, path8)
aras.createOrReplaceTempView('aras')
# display(aras)

# COMMAND ----------

# Combine all 8 dataframes
combined_df = (sh24_sps
               .unionByName(sh24_tdl)
               .unionByName(fettle)
               .unionByName(ireland_enfer)
               .unionByName(ireland_medlab)
               .unionByName(hepc)
               .unionByName(freetesting)
               .unionByName(aras))

# Save as table new bill_of materials
# combined_df.write.saveAsTable("bill_of_materials")


# COMMAND ----------

# sh24_sps.write.mode("overwrite").saveAsTable("bill_of_materials")
# sh24_tdl.write.mode("append").saveAsTable("bill_of_materials")
# fettle.write.mode("append").saveAsTable("bill_of_materials")
# ireland_enfer.write.mode("append").saveAsTable("bill_of_materials")
# ireland_medlab.write.mode("append").saveAsTable("bill_of_materials")
# hepc.write.mode("append").saveAsTable("bill_of_materials")
# freetesting.write.mode("append").saveAsTable("bill_of_materials")
# aras.write.mode("append").saveAsTable("bill_of_materials")

# COMMAND ----------

# MAGIC %md
# MAGIC ###2. CREATE TESTKIT_COLOUR_SAMPLE AND CONSUMABLE TABLE

# COMMAND ----------

# DBTITLE 1,Get sample_sites_medium from google sheet
#sample_sites_medium

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTsuq_Y4edq2OUMVeAizZ0IptsJXM6F3cMvcKHI-js3MAwDUplbRRoYZd7SIVXpcbmkvc16ibcoLaFV/pub?gid=527444052&single=true&output=csv"
spark.sparkContext.addFile(path)

sample_sites = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

# save as table
sample_sites.write.saveAsTable("sample_sites_medium")

# COMMAND ----------

# DBTITLE 1,Testkitcode_colour_sample
path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTsuq_Y4edq2OUMVeAizZ0IptsJXM6F3cMvcKHI-js3MAwDUplbRRoYZd7SIVXpcbmkvc16ibcoLaFV/pub?gid=1997112257&single=true&output=csv"
spark.sparkContext.addFile(path)

testkitcode_colour_sample = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

# save as table
testkitcode_colour_sample.write.saveAsTable("testkitcode_colour_sample")

df = spark.sql('''
               select test_kit_code, case when colour = 'NULL' then 'Custom' else colour end as colour, sample_sk from ttcs
               ''')

# COMMAND ----------

display(testkitcode_colour_sample)

# COMMAND ----------

# DBTITLE 1,create a consumable table - from google sheet
# create a consumables table from google sheet

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=1111658249&single=true&output=csv"
spark.sparkContext.addFile(path)

consumable_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

# save table
consumable_df.write.saveAsTable("consumables")

# COMMAND ----------

# DBTITLE 1,SAVE ALL Tables 
#save as table testkit_code_colour_sample
testkit_colour_sample.write.saveAsTable("testkitcode_colour_sample")

#save as table
test_sample.write.saveAsTable("sample_sites_medium")

#new bill of materials
df.write.saveAsTable("bill_of_materials")

#create table for consumables
consumable_df.write.saveAsTable("consumables")


# COMMAND ----------

# DBTITLE 1,Remove all parquet files
# MAGIC %fs rm -r dbfs:/path/to

# COMMAND ----------

# MAGIC %md
# MAGIC CONTINUE FROM preprocessing_stock_dashboard
