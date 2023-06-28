# Databricks notebook source
# DBTITLE 1,Query to get all test_kit_code and their required_klasses
#Query to get all testkit_codes 8191
query_result = spark.sql("""
SELECT tkm.test_kit_code, tkm.color, tkm.required_klasses, 
    COUNT(*) AS total_test
    FROM raw_admin.test_kits tt
    RIGHT JOIN raw_admin.testkit_metadata tkm 
    ON tkm.test_kit_code = tt.test_kit_code
    GROUP BY tkm.test_kit_code, tkm.color, tkm.required_klasses   
    ORDER BY total_test DESC
    """)
display(df)

# COMMAND ----------

#remove the {} curly braces from the required_klasses
df['required_klasses'] = df['required_klasses'].str.replace('[{}]','', regex=True)
display(df)

# COMMAND ----------

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

# Create the new column (test_sample) by applying the function extract_test_sample 
df['test_sample'] = df['required_klasses'].apply(extract_test_sample)


# COMMAND ----------

# #Create a temp table (test_codes_colours) with spark_df 
# from pyspark.sql import SparkSession

# # Start a Spark Session
# spark = SparkSession.builder.getOrCreate()

# # Convert the pandas DataFrame to a PySpark DataFrame
# spark_df = spark.createDataFrame(df)

# # Create a temporary view
# spark_df.createOrReplaceTempView("test_codes_colours")

# COMMAND ----------

# # Drop temp view when not needed
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.getOrCreate()
# spark.catalog.dropTempView("table_df")
