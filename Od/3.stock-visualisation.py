# Databricks notebook source
#CONTINUE FROM 2.STOCK

# COMMAND ----------

# DBTITLE 1,Quantity in Stock - from google sheet
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
# qty_df.write.mode('overwrite').parquet("dbfs:/path/to/qty_df.parquet")

# COMMAND ----------

# Retrieve qty_df from the Parquet file
qty_df = spark.read.parquet("dbfs:/path/to/qty_df.parquet")


# COMMAND ----------

# Retrieve the consumable_used from the Parquet file 
consumable_used_df = spark.read.parquet('dbfs:/path/to/consumable_used.parquet')

#drop kit_type column no longer needed
consumable_used_df = consumable_used_df.drop("kit_type")

# COMMAND ----------

display(consumable_used_df)

# COMMAND ----------

#Get stock used from june2023 to date
june = consumable_used_df.filter(consumable_used_df.created_at > '2023-06-01 00:00:00')
display(june)

# COMMAND ----------

# DBTITLE 1,Get consumables used from 17 June 2023
from pyspark.sql.functions import sum

#Get stock used from 17june2023 to date
june17 = consumable_used_df.filter(consumable_used_df.created_at > '2023-06-17 00:00:00')

# display only consumables and the total used from 17june23 to date
jun17_df = june17.groupBy("consumable").agg(sum("used").alias("used"))
display(jun17_df)

# COMMAND ----------

#calculate remaining quantity in stock after usage deduction
from pyspark.sql.functions import sum, col, when

#rename consumable to avoid conflict
jun17_df = jun17_df.withColumnRenamed("consumable", "con")

joined_df = qty_df.join(jun17_df, qty_df.Consumable == jun17_df.con, how='left')

# Substract used from total qty
df = (joined_df
             .groupBy('Consumable','Total_Qty')
             .agg((sum(joined_df['Total_Qty'] - joined_df['used'])).alias('In_Stock'))
             )

#replace null in_stock with value from total_qty
df = df.withColumn("In_Stock", when(col("In_Stock").isNull(), col("Total_Qty")).otherwise(col("In_Stock")))

Qty_In_Stock_df = df.drop("Total_Qty")
display(Qty_In_Stock_df)
