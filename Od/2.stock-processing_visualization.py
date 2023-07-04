# Databricks notebook source
# MAGIC %md
# MAGIC CONTINUE FROM 1.stock notebook

# COMMAND ----------

# DBTITLE 1,Get daily kit created in 2023
#retrieve all daily kit created from Jan 2023 - to date 
kit_created = spark.sql("""
    SELECT 
        tt.id,
        tt.packaged_at,
        tt.test_kit_code,
        tt.lab,
        tt.kit_type,
        eoc.brand AS brand1,
        dc.testing_service AS brand2,
        COUNT(*) AS count
    FROM raw_admin.test_kits tt
        LEFT JOIN raw_admin.sti_test_orders sto ON tt.sti_test_order_id = sto.id
        LEFT JOIN raw_admin.episodes_of_care eoc ON sto.episode_of_care_id = eoc.id
        LEFT JOIN raw_admin.batches b ON b.id = tt.batch_id
        LEFT JOIN raw_admin.distribution_centres dc ON dc.id = b.distribution_centre_id
    WHERE tt.created_at > '2023-01-01 00:00:00.000'
    GROUP BY
        tt.id,
        tt.packaged_at,
        tt.test_kit_code,
        tt.lab,
        tt.kit_type,
        brand1,
        brand2
""")
print(kit_created.count())

# COMMAND ----------

# #get list of unique values in brand1 column
# unique_values = kit_created.select("brand1").distinct().rdd.map(lambda row: row[0]).collect()
# print(unique_values)

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
# kit_created1.show(n=5)

# COMMAND ----------

# #check for NULL values (empty cells)
# from pyspark.sql.functions import col, sum as spark_sum

# null_counts = kit_created1.select([spark_sum(col(c).isNull().cast('integer')).alias(c) for c in kit_created1.columns])
# null_counts.show()

# COMMAND ----------

# #check for 'NULL' string values in each columns
# from pyspark.sql.functions import col, sum, lit, when

# column_counts = kit_created1.select(*[
#     sum(when(col(col_name) == 'NULL', lit(True)).otherwise(lit(False)).cast("integer")).alias(col_name)
#     for col_name in kit_created1.columns
# ])
# column_counts.show()

# COMMAND ----------

# #check for null values in both brand columns
# from pyspark.sql.functions import col

# df_with_null = kit_created1.filter(col("brand1").isNotNull() & col("brand2").isNull())
# df_with_null.count()

# COMMAND ----------

#assign values from brand1 into null values in brand2 
from pyspark.sql.functions import col, when

kit_created2 = kit_created1.withColumn("brand2", when(col("brand2").isNull(), col("brand1")).otherwise(col("brand2")))

#delete brand1 column, not needed anymore
kit_created2 = kit_created2.drop("brand1")

# COMMAND ----------

# #check row where brand2 has null value
# kit_created2.filter(col("brand2").isNull()).show()

#drop row with null value in brand2
kit_created3 = kit_created2.filter(col("brand2").isNotNull())

# COMMAND ----------

#Reassign medlab 3 to Enfer 4 (medlab no longer in use)
from pyspark.sql.functions import when

kit_created_df = kit_created3.withColumn("lab", when(kit_created3.lab == 3, 4).otherwise(kit_created3.lab))

# COMMAND ----------

#save df as table
kit_created_df.write.format("parquet").mode("overwrite").saveAsTable("kit_created")

kit_created_df.createOrReplaceTempView("kit_created")

# COMMAND ----------

# kit_created_df = spark.read.table("kit_created")

# Retrieve bill_of_materials 
bill_of_materials_df = spark.read.parquet("dbfs:/path/to/bill_of_materials_df.parquet")

# Retrieve lab table from the Parquet file into a DataFrame
lab_df = spark.read.parquet('dbfs:/path/to/lab.parquet')

# Retrieve brand table from the Parquet file into a DataFrame
brand_df = spark.read.parquet('dbfs:/path/to/brand.parquet')

# Retrieve consumable the Parquet file into a DataFrame
consumable_df = spark.read.parquet('dbfs:/path/to/consumable.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Calculate consumables used per kit created

# COMMAND ----------

# DBTITLE 1,Get consumables used by joining kit_created with the bill_of_materials 
from pyspark.sql.functions import sum

# Join on 'test_kit_code', 'brand' and 'lab' columns on kit_created and bill_of_materials
joined_df = kit_created_df.join(bill_of_materials_df, 
                    on=(kit_created_df.test_kit_code == bill_of_materials_df.test_kit_code1) 
                       & (kit_created_df.brand2 == bill_of_materials_df.brand1) 
                       & (kit_created_df.lab == bill_of_materials_df.lab1), 
                    how='left')

# Remove the duplicate columns from the joined_df
joined_df2 = joined_df.drop('test_kit_code1', 'brand1', 'lab1')

# Calculate the number of consumables used by multiplying 'count' and 'count1' columns 
consumable_used_df = (joined_df2
             .groupBy('packaged_at', 'test_kit_code', 'brand2', 'lab', 'kit_type','consumable1')
             .agg((sum(joined_df2['count'] * joined_df2['count1'])).alias('used'))
             )
consumable_used_df.createOrReplaceTempView("consumable_used")

# COMMAND ----------

#replace the ids of brand, lab, consumable with the full names 
consumable_used_df2 = spark.sql("""
   SELECT cu.packaged_at, tr.internal_name AS colour, c.consumable, b.brand, l.lab, cu.kit_type, cu.used
   FROM consumable_used cu
   LEFT JOIN consumables c ON c.con_id = cu.consumable1
   LEFT JOIN brand b ON b.brand_id = cu.brand2
   LEFT JOIN lab l ON l.lab_id = cu.lab
   LEFT JOIN raw_admin.test_regimes tr on tr.test_kit_code = cu.test_kit_code
   
""")
consumable_used_df2.count()

# COMMAND ----------

#assign custom_kit to colour column with kit_type = 1
from pyspark.sql.functions import when, col

consumable_used_df3 = consumable_used_df2.withColumn(
    "colour", when(col("kit_type") == 1, 'Custom_kit').otherwise(col("colour"))
)

#drop kit_type column no longer needed
consumable_used_df3 = consumable_used_df3.drop("kit_type")

# COMMAND ----------

#save df to table
consumable_used_df3.write.format("parquet").mode("overwrite").saveAsTable("consumable_used_tb")

# COMMAND ----------

#Number of null colour kit_created that is not custom kit
# print(consumable_used_df3.filter(col("colour") == "NULL").count())

# COMMAND ----------

# MAGIC %md
# MAGIC BREAK \
# MAGIC CONTINUE 

# COMMAND ----------

# DBTITLE 1,Get Quantity in Stock from supplies - from google sheet
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

# Save as a Parquet file
qty_df.write.mode('overwrite').parquet("dbfs:/path/to/qty_df.parquet")

# COMMAND ----------

# Retrieve from the Parquet file 
# qty_df = spark.read.parquet('dbfs:/path/to/qty_df.parquet')

# COMMAND ----------

# DBTITLE 1,Get current quantity in stock
from pyspark.sql.functions import sum, col, when

#Get stock used from 17june2023 to date
june17 = consumable_used_df3.filter(consumable_used_df3.packaged_at > '2023-06-17 00:00:00')

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
# MAGIC #df visuals

# COMMAND ----------

# DBTITLE 1,display quantity remaining in stock
display(Qty_In_Stock_df.orderBy("consumable", ascending=True))

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from kit_created

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from consumable_used

# COMMAND ----------

from pyspark.sql.functions import date_trunc, sum

# group by month
monthly_used = consumable_used_df3.groupBy(
    date_trunc('Month', consumable_used_df3.packaged_at).alias('Date'),
    consumable_used_df3.colour,
    consumable_used_df3.consumable,
    consumable_used_df3.brand,
    consumable_used_df3.lab
).agg(
    sum(consumable_used_df3.used).alias('Used')
)
# print(monthly_used.count())

# COMMAND ----------

display(monthly_used)

# COMMAND ----------

# DBTITLE 1,Get latest day Used
from pyspark.sql.functions import col, sum, to_date, max

# Get the latest date from the df
max_date = consumable_used_df3.select(max(to_date(col("packaged_at")))).first()[0]

# Filter the rows for the most recent day and perform aggregation
last_used_df = consumable_used_df3 \
    .filter(to_date(col("packaged_at")) == max_date) \
    .groupBy("consumable") \
    .agg(sum("used").alias("total_used")) \
    .orderBy("consumable")

display(last_used_df)


# COMMAND ----------

from pyspark.sql.functions import col, sum, to_date, max

# Get the colour latest date from the df

colour_used_df = consumable_used_df3 \
    .filter(to_date(col("packaged_at")) == max_date) \
    .groupBy("colour", "consumable") \
    .agg(sum("used").alias("total_used"))
    .orderBy("total_used") 

display(colour_used_df)

# COMMAND ----------

from pyspark.sql.functions import col, sum, to_date, max

# Get the colour latest date from the df

colour_used_df = consumable_used_df3 \
    .filter(to_date(col("packaged_at")) == max_date) \
    .groupBy("colour", "consumable") \
    .agg(sum("used").alias("total_used")) \
    .orderBy("total_used") 

display(colour_used_df)
