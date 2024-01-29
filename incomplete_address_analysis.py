# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession

# COMMAND ----------

# Spark session creation
spark = SparkSession.builder.appName("YourAppName").getOrCreate()

# Google Sheets link
path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSAv1XUaU1E2tlZb_5O_6Y4kF-NrhFb1Wo2I2OlBs1BfKX8__FXuO1HpNPS-GIpWDSbOKJn1grnZ83g/pub?gid=2143276313&single=true&output=csv"

# Read CSV data using pandas
df= pd.read_csv(path)

# COMMAND ----------

df

# COMMAND ----------

# Split the "Incomplete address" column
split_data = df['Incomplete address'].str.split('/', expand=True)

# Assign the split columns to the DataFrame
df['order_table'] = split_data[3]
df['id'] = split_data[4]
df

# COMMAND ----------

df_spark = spark.createDataFrame(df)
# Save the Spark DataFrame to a temporary table
df_spark.createOrReplaceTempView("temp_table")

# COMMAND ----------

query = """SELECT
  DISTINCT sto.created_at,
  tt.id as testkit_id,
  tt.sti_test_order_id,
  tt.test_kit_code,
  ec.brand,
  ec.customer_id,
  r.name AS region,
  d.reporting_age_bracket
FROM
  temp_table t
LEFT JOIN
  raw_admin.test_kits tt ON (
    CASE
      WHEN t.order_table = 'sti_test_orders' THEN t.id = tt.sti_test_order_id
      WHEN t.order_table = 'test_kits' THEN t.id = tt.id
    END
  )
JOIN
  raw_admin.sti_test_orders sto ON sto.id = tt.sti_test_order_id
JOIN
  raw_admin.episodes_of_care ec ON ec.id = sto.episode_of_care_id
JOIN
  warehouse.demographics d ON d.episode_of_care_id = sto.episode_of_care_id
JOIN
  warehouse.regions r ON r.id = ec.region_id;"""
  
result = spark.sql(query)
display(result)

# COMMAND ----------

# Retrieve fettle brand only
query2 = """SELECT
  DISTINCT sto.created_at,
  date_format(sto.created_at, 'E') AS weekday, 
  hour(sto.created_at) AS hours,
  tt.id,
  tt.sti_test_order_id,
  tcs.colour,
  ec.customer_id,
  d.reporting_age_bracket
FROM
  temp_table t
LEFT JOIN
  raw_admin.test_kits tt ON (
    CASE
      WHEN t.order_table = 'sti_test_orders' THEN t.id = tt.sti_test_order_id
      WHEN t.order_table = 'test_kits' THEN t.id = tt.id
    END
  )
LEFT JOIN
  raw_admin.sti_test_orders sto ON sto.id = tt.sti_test_order_id
LEFT JOIN
  raw_admin.episodes_of_care ec ON ec.id = sto.episode_of_care_id
LEFT JOIN
  warehouse.demographics d ON d.episode_of_care_id = sto.episode_of_care_id
LEFT JOIN
  default.testkitcode_colour_sample tcs ON tcs.test_kit_code = tt.test_kit_code
LEFT JOIN
  warehouse.regions r ON r.id = ec.region_id
WHERE ec.brand = 'fettle' 
ORDER BY sto.created_at;"""

result2 = spark.sql(query2)
display(result2)

# COMMAND ----------

# add date features
# repeat test orders (bad)
# repeat customers (good)


# COMMAND ----------

query4 ="""SELECT
  DISTINCT id,
  COUNT(id) OVER (PARTITION BY id) AS count_id,
  customer_id,
  COUNT(customer_id) OVER (PARTITION BY customer_id) AS count_cust
FROM
  table2;

"""

result4 = spark.sql(query4)
display(result4)
