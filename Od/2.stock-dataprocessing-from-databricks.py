# Databricks notebook source
#CONTINUE FROM 1.stock notebook

# COMMAND ----------

# DBTITLE 1,Get daily kit created in 2023
#retrieve all daily kit created from Jan 2023 - date (2023-06-27) 
kit_created = spark.sql("""
    SELECT 
        tt.id,
        tt.created_at,
        tt.test_kit_code,
        tt.lab,
        eoc.brand AS brand1,
        dc.testing_service AS brand2,
        COUNT(*) AS count
    FROM raw_admin.test_kits tt
        LEFT JOIN raw_admin.batches b ON b.id = tt.batch_id
        LEFT JOIN raw_admin.distribution_centres dc ON dc.id = b.distribution_centre_id
        LEFT JOIN raw_admin.sti_test_orders sto ON tt.sti_test_order_id = sto.id
        LEFT JOIN raw_admin.episodes_of_care eoc ON sto.episode_of_care_id = eoc.id
    WHERE tt.created_at > '2023-01-01 00:00:00.000'
    GROUP BY
        tt.id,
        tt.created_at,
        tt.test_kit_code,
        tt.lab,
        tt.kit_type,
        brand1,
        brand2
""")


# COMMAND ----------

print(kit_created.count())

# COMMAND ----------

#get list of unique values in brand1 column
unique_values = kit_created.select("brand1").distinct().rdd.map(lambda row: row[0]).collect()
print(unique_values)


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

kit_created1.show(n=5)

# COMMAND ----------

# save as parquet file
# kit_created1.write.mode('overwrite').parquet('dbfs:/path/to/kit_created1.parquet')

# Read the Parquet file into a DataFrame
kit_created1 = spark.read.parquet('dbfs:/path/to/kit_created1.parquet')

# COMMAND ----------

#check for NULL values (empty cells)
from pyspark.sql.functions import col, sum as spark_sum

null_counts = kit_created1.select([spark_sum(col(c).isNull().cast('integer')).alias(c) for c in kit_created1.columns])
null_counts.show()

# COMMAND ----------

#check for 'NULL' string values in each columns
from pyspark.sql.functions import col, sum, lit, when

column_counts = kit_created1.select(*[
    sum(when(col(col_name) == 'NULL', lit(True)).otherwise(lit(False)).cast("integer")).alias(col_name)
    for col_name in kit_created1.columns
])
column_counts.show()


# COMMAND ----------

#check no of similar rows in brand1 and brand2 that is not null
from pyspark.sql.functions import col

similar_df = kit_created1.filter(col("brand1").isNotNull() & col("brand2").isNotNull()) \
                           .filter(col("brand1") == col("brand2"))
similar_df.count()

# COMMAND ----------

#check for null values in both column
from pyspark.sql.functions import col

df_with_null = kit_created1.filter(col("brand1").isNotNull() & col("brand2").isNull())
df_with_null.count()

# COMMAND ----------

#assign values from brand1 into brand2 with null values
from pyspark.sql.functions import col, when

kit_created2 = kit_created1.withColumn("brand2", when(col("brand2").isNull(), col("brand1")).otherwise(col("brand2")))

# COMMAND ----------

#check again for NULL values(empty cells)
from pyspark.sql.functions import col, sum as spark_sum

null_counts = kit_created2.select([spark_sum(col(c).isNull().cast('integer')).alias(c) for c in kit_created2.columns])
null_counts.show()

# COMMAND ----------

#delete brand1 column, not needed anymore
kit_created2 = kit_created2.drop("brand1")
kit_created2.show(n=5)

# COMMAND ----------

#check row where brand2 has null value
kit_created2.filter(col("brand2").isNull()).show()

# COMMAND ----------

#drop row with null value in brand2
kit_created3 = kit_created2.filter(col("brand2").isNotNull())

# COMMAND ----------

#Reassign medlab 3 to Enfer 4 (medlab no longer in use)
from pyspark.sql.functions import when

kit_created4 = kit_created3.withColumn("lab", when(kit_created3.lab == 3, 4).otherwise(kit_created3.lab))

# COMMAND ----------

# #save or update file 
kit_created4.write.mode('overwrite').parquet('dbfs:/path/to/kit_created1.parquet')


# COMMAND ----------

#BREAK TIME -- CONTINUE BELOW --

# COMMAND ----------

# Retrieve kit_created from the Parquet file 
kit_created = spark.read.parquet('dbfs:/path/to/kit_created1.parquet')

# Create a temporary view f
kit_created.createOrReplaceTempView('kit_created')
kit_created.count()

# COMMAND ----------

# Retrieve bill_of_materials 
bill_of_materials_df = spark.read.parquet("dbfs:/path/to/bill_of_materials_df.parquet")

# Create a temporary view 
bill_of_materials_df.createOrReplaceTempView("bill_of_materials")
bill_of_materials_df.count()

# COMMAND ----------

# DBTITLE 1,Get consumables used by joining kit_created with the bill_of_materials 
from pyspark.sql.functions import sum

# Join on 'test_kit_code', 'brand' and 'lab' columns on kit_created and bill_of_materials
joined_df = kit_created.join(bill_of_materials_df, 
                    on=(kit_created.test_kit_code == bill_of_materials_df.test_kit_code1) 
                       & (kit_created.brand2 == bill_of_materials_df.brand1) 
                       & (kit_created.lab == bill_of_materials_df.lab1), 
                    how='left')

# Remove the duplicate columns from the DataFrame
joined_df2 = joined_df.drop('test_kit_code1', 'brand1', 'lab1')

# Calculate the number of consumables used by multiplying count by count1 
consumable_used_df = (joined_df2
             .groupBy('created_at', 'test_kit_code', 'brand2', 'lab', 'kit_type','consumable1')
             .agg((sum(joined_df2['count'] * joined_df2['count1'])).alias('used'))
             )



# COMMAND ----------

consumable_used_df.count()

# COMMAND ----------

# Create a temporary view
consumable_used_df.createOrReplaceTempView("consumable_used")

# Retrieve lab table from the Parquet file into a DataFrame
lab_df = spark.read.parquet('dbfs:/path/to/lab.parquet')

# Create a temporary view
lab_df.createOrReplaceTempView("lab")

# Retrieve brand table from the Parquet file into a DataFrame
brand_df = spark.read.parquet('dbfs:/path/to/brand.parquet')

# Create a temporary view
brand_df.createOrReplaceTempView("brand")

# Retrieve consumable the Parquet file into a DataFrame
consumable_df = spark.read.parquet('dbfs:/path/to/consumable.parquet')

# Create a temporary view
consumable_df.createOrReplaceTempView("consumable")


# COMMAND ----------

#replace the ids of brand, lab, consumable with the full name 
consumable_used_df2 = spark.sql("""
   SELECT cu.created_at, tr.internal_name AS colour, c.consumable, b.brand, l.lab, cu.kit_type, cu.used
   FROM consumable_used cu
   LEFT JOIN consumable c ON c.con_id = cu.consumable1
   LEFT JOIN brand b ON b.brand_id = cu.brand2
   LEFT JOIN lab l ON l.lab_id = cu.lab
   LEFT JOIN raw_admin.test_regimes tr on tr.test_kit_code = cu.test_kit_code
   
""")
consumable_used_df2.count()

# COMMAND ----------

#assign custom_kit to colour column with kit_type = 1
from pyspark.sql.functions import when, col

consumable_used_df3 = consumable_used_df2.withColumn(
    "colour",
    when(col("kit_type") == 1, 'Custom_kit').otherwise(col("colour"))
)

# COMMAND ----------

#No of null colour kit_created that is not custom kit
print(consumable_used_df3.filter(col("colour") == "NULL").count())

# COMMAND ----------

display(consumable_used_df3)

# COMMAND ----------

# # save file 
# consumable_used_df3.write.mode('overwrite').parquet('dbfs:/path/to/consumable_used.parquet')


# COMMAND ----------

#END HERE
#CONTINUE ON 3.stock notebook
