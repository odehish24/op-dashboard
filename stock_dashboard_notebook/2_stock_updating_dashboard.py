# Databricks notebook source
# MAGIC %md
# MAGIC ### UPDATING DAILY KIT DISPATCHED TABLE
# MAGIC - 1. Get only new kit created from the last date on the existing kit_created table
# MAGIC - 2. Ensure data quality checks

# COMMAND ----------

# import necessary libriaries
from pyspark import SparkFiles
from pyspark.sql.functions import sum, col, when, lit, regexp_replace
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC This query below get the daily kit created with their respective brand, lab, kit_type and test_kit_code
# MAGIC - brand1 mapping: brand1 is transformed from the brand column in episodes_of_care table to map string names to integer identifiers.
# MAGIC - brand_sk transformed from the testing_service column in distribution_centres table map it brand codes to correspond with sk in brand table.
# MAGIC - total_count: creates record of one row as count o an order

# COMMAND ----------

# DBTITLE 1,Get daily kit Dispatched
kit_created1 = spark.sql("""
    SELECT 
        tt.id,
        CAST(tt.dispatched_at AS timestamp) AS dispatched_at, 
        tt.test_kit_code,
        tcs.sample_sk,
        tt.lab,
        tt.kit_type,
        CAST(
            CASE eoc.brand 
                WHEN 'sh24' THEN 1
                WHEN 'fettle' THEN 2
                WHEN 'aras_romania' THEN 3
                WHEN 'freetesting_hiv' THEN 4
                WHEN 'sh24_ireland' THEN 5
                WHEN 'hep_c_ireland' THEN 6
            END AS int
        ) AS brand1,
        CASE dc.testing_service
            WHEN 0 THEN 1
            WHEN 1 THEN 2
            ELSE dc.testing_service
        END AS brand_sk,
        COUNT(*) AS total_count
    FROM raw_admin.test_kits tt
        LEFT JOIN default.testkitcode_colour_sample tcs ON tcs.test_kit_code = tt.test_kit_code
        LEFT JOIN raw_admin.sti_test_orders sto ON tt.sti_test_order_id = sto.id
        LEFT JOIN raw_admin.episodes_of_care eoc ON sto.episode_of_care_id = eoc.id
        LEFT JOIN raw_admin.batches b ON b.id = tt.batch_id
        LEFT JOIN raw_admin.distribution_centres dc ON dc.id = b.distribution_centre_id
    WHERE tt.dispatched_at >
                (SELECT MAX(dispatched_at) FROM default.kit_created)
    GROUP BY
        tt.id,
        CAST(tt.dispatched_at AS timestamp), 
        tt.test_kit_code,
        tcs.sample_sk,
        tt.lab,
        tt.kit_type,
        brand1,
        brand_sk
""")
kit_created1.createOrReplaceTempView("kit_created1")

# COMMAND ----------

# MAGIC %md
# MAGIC The purpose of this code below is to handle null values in kit_created1 DataFrame. Specifically, it performs the following operations:
# MAGIC
# MAGIC - Filtering NULL Values: Removes rows where the built_at column contains null values.
# MAGIC - Handling Missing Brands: Fills missing values in the brand2 column with corresponding values from the brand1 column.
# MAGIC - Column Removal: Deletes the now redundant brand1 column.
# MAGIC - Further Filtering: Filters out rows where the brand2 column is still null after the above transformations.

# COMMAND ----------

# DBTITLE 1,Handling Null Values
kit_created2 = spark.sql('''
    SELECT
        id, 
        dispatched_at, 
        test_kit_code, 
        sample_sk, 
        lab AS lab_enum, 
        kit_type,
        COALESCE(brand_sk, brand1) AS brand_sk,
        total_count
    FROM kit_created1
        WHERE dispatched_at IS NOT NULL OR dispatched_at !='NULL' AND COALESCE(brand_sk, brand1) IS NOT NULL   
    ''')
# Update main table
kit_created2.write.mode("append").saveAsTable("kit_created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate consumables used per kit created

# COMMAND ----------

# DBTITLE 1,Retrieve the latest kits created to determine the consumables used
sql_statement = """
        SELECT 
            kc.dispatched_at,
            kc.test_kit_code,
            kc.sample_sk,
            kc.brand_sk,
            kc.lab_enum,
            kc.kit_type,
            bom.consumable_sk,
            SUM(kc.total_count * bom.count1) AS used
        FROM default.kit_created AS kc
        LEFT JOIN default.bill_of_materials AS bom
            ON (kc.sample_sk = bom.sample_sk) 
            AND (kc.brand_sk = bom.brand_sk) 
            AND (kc.lab_enum = bom.lab_enum)
        WHERE dispatched_at >  
                    (SELECT MAX(dispatched_at) FROM default.consumable_used)       
        GROUP BY 
            kc.dispatched_at, 
            kc.test_kit_code, 
            kc.sample_sk, 
            kc.brand_sk, 
            kc.lab_enum, 
            kc.kit_type, 
            bom.consumable_sk
"""
consumable_used1 = spark.sql(sql_statement)

# Update consumable_used table
consumable_used1.write.mode("append").saveAsTable("consumable_used")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate Stock level based on Stock Usage

# COMMAND ----------

# MAGIC %md
# MAGIC - Get the latest supplies from google sheet 
# MAGIC - Clean and prepare the data then save

# COMMAND ----------

# DBTITLE 1,Get Quantity in Stock - from supplies tracker google sheet
# Get quantity in stock from supplies tracker as at 16 october 2023
path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQvualKidxOjIjDPQEiExsiFA18wIIno8y0qt_xTStd-FptmwHgtfN_PIbpM8nq5UtfnDaVc_ngSndF/pub?gid=1323386964&single=true&output=csv"

# Load CSV file using a Python pandas DataFrame
import pandas as pd
pdf = pd.read_csv(path)

# Convert pandas DataFrame to Spark DataFrame
from pyspark.sql.types import *
schema = StructType([StructField(col, StringType(), True) for col in pdf.columns])
sdf = spark.createDataFrame(pdf, schema)

# save as temporary view
sdf.createOrReplaceTempView("qty_temp_view")

sql_statement = """
    SELECT 
        CAST(con_sk AS INT) AS con_sk, 
        CAST(REGEXP_REPLACE(total_qty, ',', '') AS INT) AS total_qty
    FROM 
        qty_temp_view
    WHERE con_sk NOT IN (26, 27)
"""
qty_in_df = spark.sql(sql_statement)

# Save as a table
qty_in_df.write.mode("overwrite").saveAsTable("default.qty_in_stock")

# COMMAND ----------

display(qty_in_df)
