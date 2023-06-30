# Databricks notebook source
#CONTINUE FROM 2.STOCK

# COMMAND ----------

# DBTITLE 1,Get Quantity in Stock - from google sheet
#quantity in stock

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

# Retrieve qty_df from the Parquet file
qty_df = spark.read.parquet("dbfs:/path/to/qty_df.parquet")

# Retrieve the consumable_used from the Parquet file 
consumable_used_df = spark.read.parquet('dbfs:/path/to/consumable_used.parquet')

#drop kit_type column no longer needed
consumable_used_df = consumable_used_df.drop("kit_type")


# COMMAND ----------

spark.sql("CREATE TABLE IF NOT EXISTS consumable_used_table USING parquet OPTIONS (path 'dbfs:/path/to/consumable_used.parquet')")


# COMMAND ----------

from pyspark.sql.functions import date_trunc, sum

# group by month
monthly_used = consumable_used_df.groupBy(
    date_trunc('Month', consumable_used_df.created_at).alias('Date'),
    consumable_used_df.colour,
    consumable_used_df.consumable,
    consumable_used_df.brand,
    consumable_used_df.lab
).agg(
    sum(consumable_used_df.used).alias('Used')
)
print(monthly_used.count())

# COMMAND ----------

display(monthly_used)

# COMMAND ----------

#Get stock used from june2023 to date
june = consumable_used_df.filter(consumable_used_df.created_at > '2023-06-01 00:00:00')
display(june)

# COMMAND ----------

# DBTITLE 1,Calculate Quantity in stock from consumables used from 17 June to date
from pyspark.sql.functions import sum, col, when

#Get stock used from 17june2023 to date
june17 = consumable_used_df.filter(consumable_used_df.created_at > '2023-06-17 00:00:00')

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

display(Qty_In_Stock_df)
