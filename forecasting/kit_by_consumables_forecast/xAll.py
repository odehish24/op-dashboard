# Databricks notebook source


# COMMAND ----------

all_df = spark.sql("""
               SELECT
                    week,  
                    tc.sample_sk, 
                    lab, 
                    brand, 
                    cast(count_preds AS INT) 
                FROM colour_pred cp 
                JOIN testkitcode_colour_sample tc on tc.test_kit_code = cp.test_kit_code 
               """).toPandas()

# COMMAND ----------

        SELECT 
            cp.week,
            cp.test_kit_code,
            tc.sample_sk,
            cp.brand,
            cp.lab,
            bom.consumable_sk,
            SUM(cp.count_pred * bom.count1) AS used
        FROM default.colour_pred cp 
        LEFT JOIN default.bill_of_materials AS bom
            ON (tc.sample_sk = bom.sample_sk) 
            AND (cp.brand = bom.brand_sk) 
            AND (cp.lab = bom.lab_enum)     
        GROUP BY 1,2,3,4,5,6

# COMMAND ----------

df_bom = spark.sql("""
                   SELECT sample_sk1, brand1, lab1, consumable1, c.consumable, count1 
                   FROM bill_of_materials bm 
                   LEFT JOIN consumables c on c.con_sk = bm.consumable1 
                    """).toPandas()

# COMMAND ----------

#Merge all_df table with bill_of_mat table
merged_df = pd.merge(all_df, df_bom, 
                     left_on=['sample_sk', 'brand', 'lab'], 
                     right_on=['sample_sk1', 'brand1', 'lab1'], 
                     how='left')

# Performing the aggregation
grouped_df = merged_df.groupby(['week', 'sample_sk', 'brand', 'lab', 'consumable1', 'consumable'])
result = grouped_df.apply(lambda x: (x['count_preds'] * x['count1']).sum()).reset_index(name='total_count')


# COMMAND ----------

display(result)

# COMMAND ----------

#Save the predictions
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("sti_test_kit_prediction").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(result)

# Save as a Parquet table
spark_df.write.format("parquet").mode("overwrite").saveAsTable("sti_test_kit_pred")
