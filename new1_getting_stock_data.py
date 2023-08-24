# Databricks notebook source
# MAGIC %md
# MAGIC ###1. CREATE BILL OF MATERIALS TABLE 

# COMMAND ----------

from pyspark import SparkFiles
import pyspark.sql.functions as F

# COMMAND ----------

# DBTITLE 1,Create function that retrieve data from Google sheet and transform
def load_and_transform_data(spark, path):
    
    # Add the file to the Spark context
    spark.sparkContext.addFile(path)

    # Read the file from the provided path
    df = spark.read.csv("file://" + SparkFiles.get("pub"), header=True, inferSchema=True)

    # Exclude specified columns and combine others into an array column
    brand_list = [col for col in df.columns if col not in ["sample_sk", "brand", "lab"]]

    df2 = df.withColumn("array_col", F.array([F.struct(F.lit(c).alias("consumable"), F.col(c).alias("count")) for c in brand_list]))

    # Transpose the array into multiple rows
    df3 = df2.select("sample_sk", "brand", "lab", F.explode("array_col").alias("new_col"))
    df4 = df3.select("sample_sk", "brand", "lab", "new_col.consumable", "new_col.count")

    # Filter out rows with 'count' equal to 0
    df5 = df4.filter(df4['count'] != 0)

    #rename the column names add 1 this will differentiate it when joined with kit_created columns later
    brand_df = df5.select([F.col(c).alias(c+'1') for c in df5.columns])

    return brand_df


# COMMAND ----------

# DBTITLE 1,SH:24 SPS 
#SH:24 SPS 
path1 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=1239395583&single=true&output=csv"

#call function
sh24_sps = load_and_transform_data(spark, path1)
sh24_sps.createOrReplaceTempView('sh24_sps')
display(sh24_sps)

# COMMAND ----------

#SH:24 TDL 
path2 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=538017527&single=true&output=csv"

#call function
sh24_tdl = load_and_transform_data(spark, path2)
sh24_tdl.createOrReplaceTempView('sh24_tdl')
display(sh24_tdl)

# COMMAND ----------

#Fettle TDL 
path3 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=1852084597&single=true&output=csv"

#call function
fettle = load_and_transform_data(spark, path3)
fettle.createOrReplaceTempView('fettle')
display(fettle)

# COMMAND ----------

# DBTITLE 1,IRELAND_ENFER
#IRELAND ENFER
path4 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=1671454770&single=true&output=csv"

#call function
ireland_enfer = load_and_transform_data(spark, path4)
ireland_enfer.createOrReplaceTempView('ireland_enfer')

display(ireland_enfer)

# COMMAND ----------

# DBTITLE 1,IRELAND MEDLAB 
#IRELAND MEDLAB 
path5 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=187790868&single=true&output=csv"

#call function
ireland_medlab = load_and_transform_data(spark, path5)
ireland_medlab.createOrReplaceTempView('ireland_medlab')
display(ireland_medlab)

# COMMAND ----------

# DBTITLE 1,HEP_C_IRELAND ENFER
#HEP_C_IRELAND ENFER
path6 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=1753729190&single=true&output=csv"

#call function
hepc = load_and_transform_data(spark, path6)
hepc.createOrReplaceTempView('hepc')
display(hepc)

# COMMAND ----------

# DBTITLE 1,FREETESTING SPS
#FREETESTING SPS
path7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=305956146&single=true&output=csv"

#call function
freetesting = load_and_transform_data(spark, path7)
freetesting.createOrReplaceTempView('freetesting')
display(freetesting)

# COMMAND ----------

# DBTITLE 1,ARAS ROMANIA TDL
#ARAS ROMANIA TDL

path8 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=136572926&single=true&output=csv"

#call function
aras = load_and_transform_data(spark, path8)
aras.createOrReplaceTempView('aras')
display(aras)

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
combined_df.write.saveAsTable("bill_of_mat")


# COMMAND ----------

# DBTITLE 1,Another way to get the new bill of mat
sql_statement = '''
    WITH CTE AS (
    select bm.test_kit_code1,
        brand1,
        lab1,
        consumable1,
        tc.sample_sk AS sample_sk1,
        count1
        from bill_of_materials bm
        left join testkit_colour_sample tc ON tc.test_kit_code = bm.test_kit_code1
    ) 
    SELECT 
        sample_sk1,
        brand1,
        lab1,
        consumable1,
        count1
    FROM CTE
    GROUP BY
        sample_sk1,
        brand1,
        lab1,
        consumable1,
        count1    
'''
df = spark.sql(sql_statement)
df.write.saveAsTable("bill_of_mat")

# COMMAND ----------

display(df)

# COMMAND ----------

#errorrrrrrrrrrrrrrrrrrr of caching
display(sh24_sps)

# COMMAND ----------

# MAGIC %md
# MAGIC ###2. CREATE TESTKIT_COLOUR_SAMPLE AND CONSUMABLE TABLE

# COMMAND ----------

# DBTITLE 1,Testkit_code_colour_test_sample
path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=1997112257&single=true&output=csv"
spark.sparkContext.addFile(path)

testkit_colour_sample = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#save as table
testkit_colour_sample.write.saveAsTable("testkit_colour_sample")

# COMMAND ----------

# DBTITLE 1,create a consumable table - from google sheet
# create a consumables table from google sheet

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_hw28G0N5LMZiLDUYPCOm7gMIMezhmSRRmyl-tCuY1i_ughdGZELy-KeSgojmlYvt_htb9bdBkzVp/pub?gid=1111658249&single=true&output=csv"
spark.sparkContext.addFile(path)

consumable_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#save table
consumable_df.write.saveAsTable("consumables")

# COMMAND ----------

# DBTITLE 1,Create Tables 
#save as table testkit_code_colour_sample
testkit_colour_sample.write.saveAsTable("testkit_colour_sample")

#new bill of mat
df.write.saveAsTable("bill_of_mat")

#create table for consumables
consumable_df.write.saveAsTable("consumables")


# COMMAND ----------

# MAGIC %md
# MAGIC CONTINUE FROM new2_preprocessing_stock_dashboard
