# Databricks notebook source
# MAGIC %md
# MAGIC START FROM new1.getting_stock_data
# MAGIC
# MAGIC ###3. CREATE DAILY KIT CREATED TABLE

# COMMAND ----------

# DBTITLE 1,Get daily kit created in 2023
#retrieve all daily kit created from Jan 2023 - to date 
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
        dc.testing_service AS brand2,
        COUNT(*) AS count
    FROM raw_admin.test_kits tt
        LEFT JOIN testkit_colour_sample tcs ON tcs.test_kit_code = tt.test_kit_code
        LEFT JOIN raw_admin.sti_test_orders sto ON tt.sti_test_order_id = sto.id
        LEFT JOIN raw_admin.episodes_of_care eoc ON sto.episode_of_care_id = eoc.id
        LEFT JOIN raw_admin.batches b ON b.id = tt.batch_id
        LEFT JOIN raw_admin.distribution_centres dc ON dc.id = b.distribution_centre_id
    WHERE tt.created_at > '2023-01-01 00:00:00.000'
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
kit_created.count()

# COMMAND ----------

# DBTITLE 1,Change brand1 values to brand ids to be identical with brand2
from pyspark.sql.functions import col, when, lit

brand_mapping = {
    'fettle': 1,
    'sh24': 0,
    'aras_romania': 3,
    'freetesting_hiv': 4,
    'sh24_ireland': 5,
    'hep_c_ireland': 6
}

expr = col("brand1")

for k, v in brand_mapping.items():
    expr = when(expr == lit(k), lit(v)).otherwise(expr)

kit_created1 = kit_created.withColumn("brand1", expr)

# COMMAND ----------

#drop all null values in built_at (likely custom kits that has not been built)
kit_created1 = kit_created1.filter(kit_created1['built_at'] != 'NULL')

# COMMAND ----------

#assign values from brand1 into null values in brand2 
from pyspark.sql.functions import col, when

kit_created2 = kit_created1.withColumn("brand2", when(col("brand2").isNull(), col("brand1")).otherwise(col("brand2")))

#delete brand1 column, not needed anymore
kit_created2 = kit_created2.drop("brand1")

#drop row with null value in brand2
kit_created3 = kit_created2.filter(col("brand2").isNotNull())

# COMMAND ----------

display(kit_created3)

# COMMAND ----------

#save df as table
kit_created3.write.format("parquet").mode("overwrite").saveAsTable("kit_created")

kit_created3.createOrReplaceTempView("kit_created")

# COMMAND ----------

# MAGIC %md
# MAGIC ###4. Calculate consumables used per kit created

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW consumable_used AS
# MAGIC SELECT 
# MAGIC     kc.built_at,
# MAGIC     kc.test_kit_code,
# MAGIC     kc.sample_sk,
# MAGIC     kc.brand2,
# MAGIC     kc.lab,
# MAGIC     kc.kit_type,
# MAGIC     bom.consumable1,
# MAGIC     SUM(kc.count * bom.count1) AS used
# MAGIC FROM kit_created AS kc
# MAGIC LEFT JOIN bill_of_mat AS bom
# MAGIC   ON (kc.sample_sk = bom.sample_sk1) 
# MAGIC   AND (kc.brand2 = bom.brand1) 
# MAGIC   AND (kc.lab = bom.lab1)
# MAGIC GROUP BY kc.built_at, kc.test_kit_code, kc.sample_sk, kc.brand2, kc.lab, kc.kit_type, bom.consumable1;
# MAGIC

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
      LEFT JOIN consumables c ON c.con_id = cu.consumable1
      LEFT JOIN brand_lookup bl ON bl.testing_service = cu.brand2
      LEFT JOIN warehouse.brands b ON b.sk = bl.sk
      LEFT JOIN warehouse.labs l ON l.enum = cu.lab
      LEFT JOIN testkit_colour_sample tcs on tcs.test_kit_code = cu.test_kit_code;
'''
df = spark.sql(sql_statement)
display(df)

# COMMAND ----------

#save df to consumable_used_tb table
df.write.format("parquet").mode("overwrite").saveAsTable("consumable_used_tb")

# COMMAND ----------

# MAGIC %md
# MAGIC ###5. Calculate Stock level based on Stock Usage

# COMMAND ----------

# DBTITLE 1,Get Quantity in Stock - from supplies tracker google sheet
#Get quantity in stock

from pyspark import SparkFiles
from pyspark.sql.functions import col, regexp_replace

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQvualKidxOjIjDPQEiExsiFA18wIIno8y0qt_xTStd-FptmwHgtfN_PIbpM8nq5UtfnDaVc_ngSndF/pub?gid=1572502399&single=true&output=csv"
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
from pyspark.sql.functions import sum, col, when

#Get stock used from 17june2023 to date
june17 = df.filter(df.built_at > '2023-06-17 00:00:00')

# display only consumables and the total used from 17june23 to date
june17 = june17.groupBy("consumable").agg(sum("used").alias("used"))

#rename consumable to avoid conflict
june17 = june17.withColumnRenamed("consumable", "con")

########### Join qty_df with june17 ################
joined_df = qty_df.join(june17, qty_df.Consumable == june17.con, how='left')

# Substract qty used from remaining total qty
df = (joined_df
             .groupBy('Consumable','Total_Qty')
             .agg((sum(joined_df['Total_Qty'] - joined_df['used'])).alias('In_Stock'))
             )

#replace null in_stock with value from total_qty
df = df.withColumn("In_Stock", when(col("In_Stock").isNull(), col("Total_Qty")).otherwise(col("In_Stock")))

Qty_In_Stock_df = df.drop("Total_Qty")

# COMMAND ----------

# save df to table
Qty_In_Stock_df.write.format("parquet").mode("overwrite").saveAsTable("qty_in_stock")

# COMMAND ----------

# MAGIC %md
# MAGIC ###notebook visuals

# COMMAND ----------

# DBTITLE 1,display quantity remaining in stock
display(Qty_In_Stock_df.orderBy("consumable", ascending=True))

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from kit_created

# COMMAND ----------

# MAGIC %sql
# MAGIC select sum(used) from consumable_used

# COMMAND ----------

from pyspark.sql.functions import date_trunc, sum

# group by month
monthly_used = df.groupBy(
    date_trunc('Month', df.built_at).alias('Date'),
    df.colour,
    df.consumable,
    df.brand,
    df.lab
).agg(
    sum(df.used).alias('Used')
)

# COMMAND ----------

display(monthly_used)

# COMMAND ----------

# DBTITLE 1,Get latest day Used
from pyspark.sql.functions import col, sum, to_date, max

# Get the latest date from the df
max_date = df.select(max(to_date(col("built_at")))).first()[0]

# Filter the rows for the most recent day and perform aggregation
last_used_df = df \
    .filter(to_date(col("built_at")) == max_date) \
    .groupBy("consumable") \
    .agg(sum("used").alias("total_used")) \
    .orderBy("consumable")

display(last_used_df)

# COMMAND ----------

from pyspark.sql.functions import col, sum, to_date, max

# Get the colour latest date from the df

colour_used_df = df \
    .filter(to_date(col("built_at")) == max_date) \
    .groupBy("colour", "consumable") \
    .agg(sum("used").alias("total_used"))

display(colour_used_df)
