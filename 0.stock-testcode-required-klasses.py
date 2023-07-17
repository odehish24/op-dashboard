# Databricks notebook source
# DBTITLE 1,Query to get all test_kit_code and their required_klasses
#Query to get all testkit_codes 8191
df = spark.sql("""
    SELECT test_kit_code, required_klasses
    FROM raw_admin.testkit_metadata 
    """)
display(df)

# COMMAND ----------

#remove the {} curly braces from the required_klasses
from pyspark.sql.functions import regexp_replace

df = df.withColumn('required_klasses', regexp_replace('required_klasses', '[{}]', ''))

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# create a function that extract test-sample-type from required_klasses
def extract_test_sample(row):
    keywords = ['Treponemal', 'RPR', 'Blood', 'Urine', 'Oral', 'Anal', 'Vaginal']
    found_keywords = []
    blood_keywords = ['Treponemal', 'RPR', 'Blood']
    blood_found = False

    for keyword in keywords:
        if keyword in row:
            if keyword in blood_keywords and not blood_found:
                found_keywords.append('Blood')
                blood_found = True
            elif keyword not in blood_keywords and keyword not in found_keywords:
                found_keywords.append(keyword)
                
    # Return unique keywords as a comma-separated string
    return ','.join(found_keywords)

# Define the UDF
extract_test_sample_udf = udf(extract_test_sample, StringType())

# Create the new column (test_sample) by applying the UDF
df = df.withColumn('test_sample', extract_test_sample_udf(df['required_klasses']))


# COMMAND ----------

display(df)
