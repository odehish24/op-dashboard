# Databricks notebook source
# MAGIC %md
# MAGIC ### CREATE DAILY KIT CREATED TABLE
# MAGIC - 1. create brandlookup table that would map distribution_centres table testing_service attribute to match brand table sk attribute
# MAGIC - 2. get daily kit created in 2023
# MAGIC - 3. calculate consumable used 

# COMMAND ----------

#import needed libraries
from pyspark import SparkFiles
from pyspark.sql.functions import sum, col, when, lit, regexp_replace


# COMMAND ----------

# DBTITLE 1,Brand hookup table with testing_service
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW brand_lookup AS
# MAGIC SELECT
# MAGIC     sk,
# MAGIC     testing_service
# MAGIC FROM VALUES
# MAGIC     (1, 0),
# MAGIC     (2, 1),
# MAGIC     (3, 3),
# MAGIC     (4, 4),
# MAGIC     (5, 5),
# MAGIC     (6, 6)
# MAGIC AS brand_lookup(sk, testing_service);

# COMMAND ----------

# DBTITLE 1,Get daily kit created in 2023
#retrieve all daily kit created from Jan 2023 - to date 
# kit_type 1 means custom kit, don't have batchid
kit_created = spark.sql("""
    SELECT 
        tt.id,
        CASE kit_type 
        WHEN 0 THEN tt.created_at 
        WHEN 1 THEN tt.packaged_at
    END AS built_at,
        tt.test_kit_code,
        tcs.sample_sk,
        tt.lab,
        tt.kit_type,
        eoc.brand AS brand1,
        bl.sk AS brand2,
        COUNT(*) AS count
    FROM raw_admin.test_kits tt
        LEFT JOIN testkit_colour_sample tcs ON tcs.test_kit_code = tt.test_kit_code
        LEFT JOIN raw_admin.sti_test_orders sto ON tt.sti_test_order_id = sto.id
        LEFT JOIN raw_admin.episodes_of_care eoc ON sto.episode_of_care_id = eoc.id
        LEFT JOIN raw_admin.batches b ON b.id = tt.batch_id
        LEFT JOIN raw_admin.distribution_centres dc ON dc.id = b.distribution_centre_id
        LEFT JOIN brand_lookup bl ON bl.sk = dc.testing_service
    GROUP BY
        tt.id,
        built_at,
        tt.test_kit_code,
        tcs.sample_sk,
        tt.lab,
        tt.kit_type,
        brand1,
        brand2
""")


# COMMAND ----------

# DBTITLE 1,Assign brand1 to the corresponding brand sk values 
#the episodes_of_care table brand attribute renamed as brand1 above is mapped to the corresponding brand sk values

brand_mapping = {
    'sh24': 1,
    'fettle': 2,
    'aras_romania': 3,
    'freetesting_hiv': 4,
    'sh24_ireland': 5,
    'hep_c_ireland': 6
}

expr = col("brand1")

for k, v in brand_mapping.items():
    expr = when(expr == lit(k), lit(v)).otherwise(expr)

kit_created = kit_created.withColumn("brand1", expr)

# COMMAND ----------

# DBTITLE 1,Handling Null Values
#drop all null values in built_at (likely custom kits that has not been built)
kit_created = kit_created.filter(kit_created['built_at'] != 'NULL')

#Handling Null values in brand: assign values from brand1 into null values in brand2 
kit_created = kit_created.withColumn("brand2", when(col("brand2").isNull(), col("brand1")).otherwise(col("brand2")))

#delete brand1 column, not needed anymore
kit_created = kit_created.drop("brand1")

#drop row with null value in brand2
kit_created = kit_created.filter(col("brand2").isNotNull())

#save as table
kit_created.write.mode("overwrite").saveAsTable("kit_created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate consumables used per kit created

# COMMAND ----------

sql_statement = """
    SELECT 
        kc.built_at,
        kc.test_kit_code,
        kc.sample_sk,
        kc.brand2,
        kc.lab,
        kc.kit_type,
        bom.consumable1,
        SUM(kc.count * bom.count1) AS used
    FROM kit_created AS kc
    LEFT JOIN bill_of_materials AS bom
    ON (kc.sample_sk = bom.sample_sk1) 
    AND (kc.brand2 = bom.brand1) 
    AND (kc.lab = bom.lab1)
    GROUP BY 
        kc.built_at, 
        kc.test_kit_code, 
        kc.sample_sk, 
        kc.brand2, 
        kc.lab, 
        kc.kit_type, 
        bom.consumable1
"""

consumable_used1 = spark.sql(sql_statement)

#Create consumable_used table
consumable_used1.write.mode("overwrite").saveAsTable("consumable_used")

# COMMAND ----------

# DBTITLE 1,Get the names of brand, lab, colour, consumable to replace the ids
#replace the ids of brand, colour, lab, consumable with their full names 
sql_statement = '''
    SELECT 
        cu.built_at, 
        c.consumable, 
        b.name AS brand, 
        l.name AS lab, 
        CASE 
            WHEN cu.kit_type = 1 THEN 'Custom_kit' ELSE tcs.colour END AS colour, 
        cu.used
    FROM consumable_used cu
        LEFT JOIN consumables c ON c.con_sk = cu.consumable1
        LEFT JOIN warehouse.brands b ON b.sk = cu.brand2
        LEFT JOIN warehouse.labs l ON l.enum = cu.lab
        LEFT JOIN testkit_colour_sample tcs on tcs.test_kit_code = cu.test_kit_code;
'''
consumable_used_tb = spark.sql(sql_statement)

#save df to consumable_used_tb table
consumable_used_tb.write.mode("overwrite").saveAsTable("consumable_used_tb")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate Stock level based on Stock Usage

# COMMAND ----------

# DBTITLE 1,Get Quantity in Stock - from supplies tracker google sheet
#Get quantity in stock
path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQvualKidxOjIjDPQEiExsiFA18wIIno8y0qt_xTStd-FptmwHgtfN_PIbpM8nq5UtfnDaVc_ngSndF/pub?gid=1323386964&single=true&output=csv"
spark.sparkContext.addFile(path)

qty_df = spark.read.csv("file://"+SparkFiles.get("pub"), header=True, inferSchema= True)

#select only the columnns needed
qty_df = qty_df.select('Con_Id', 'Consumable', 'Con_Type', 'Total_Qty')

# Replace commas in the "Total_Qty" column
qty_df = qty_df.withColumn("Total_Qty", regexp_replace(col("Total_Qty"), ",", ""))

# Convert "Total_Qty" column string to integer
qty_df = qty_df.withColumn("Total_Qty", col("Total_Qty").cast("int"))

# Save as a temp table
qty_df.createOrReplaceTempView("qty_df")

# COMMAND ----------

# DBTITLE 1,Get current quantity in stock
#Retrieve consumable_used_tb from 18Sept2023 to date
consumable_used_sept18 = consumable_used.filter(consumable_used.built_at > '2023-09-18 20:00:00')

# display only consumables and the total used from 18sept23 to date
consumable_used_sept18 = consumable_used_sept18.groupBy("consumable").agg(sum("used").alias("used"))

#rename consumable to avoid conflict
consumable_used_sept18 = consumable_used_sept18.withColumnRenamed("consumable", "con")

#Join qty_df with con_used_sept18)
joined_df = qty_df.join(consumable_used_sept18, qty_df.Consumable == consumable_used_sept18.con, how='left')

# Substract qty used from remaining total qty
df = (joined_df
             .groupBy('Consumable','Total_Qty')
             .agg((sum(joined_df['Total_Qty'] - joined_df['used'])).alias('In_Stock'))
             )

#replace null in_stock with value from total_qty
df = df.withColumn("In_Stock", when(col("In_Stock").isNull(), col("Total_Qty")).otherwise(col("In_Stock")))

#drop Total_Qty not needed anymore
Qty_In_Stock_df = df.drop("Total_Qty")

# save to table
Qty_In_Stock_df.write.format("parquet").mode("overwrite").saveAsTable("qty_in_stock")
