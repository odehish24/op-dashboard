# Databricks notebook source
# Problem statement: How to measure the impact of discount on fettle repeat orders in 2023

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC #####Identify Repeat Purchases: For each customer, determine if there are any subsequent purchases after an initial purchase.
# MAGIC
# MAGIC Customer Segmentation into 2 groups on Discount Usage: 
# MAGIC - Customers who made their initial purchase with a discount.
# MAGIC - Customers who made their initial purchase without a discount.
# MAGIC
# MAGIC Repeat Purchase Analysis: For each segment, calculate the repeat purchase rate, i.e., the proportion of customers who made at least one more purchase after their initial purchase.
# MAGIC
# MAGIC

# COMMAND ----------

query = """WITH data AS (
    SELECT
        order_created_at,
        order_id,
        customer_sk,
        CASE
            WHEN discount_sk IS NULL THEN 'No'
            WHEN discount_sk = 'NULL' THEN 'No'
            ELSE 'Yes'
        END AS with_discount
    FROM 
        warehouse.sales_events
    WHERE 
        order_created_at BETWEEN '2023-01-01' AND '2024-01-01'
        AND brand_sk = 2
        AND product_type != 'STI Test Result Set'
),
FirstPurchases AS (
    SELECT 
        customer_sk,
        MIN(order_created_at) AS first_purchase_date,
        order_id,
        with_discount
    FROM 
        data
    GROUP BY 
        customer_sk, order_id, with_discount
),
RepeatPurchases AS (
    SELECT 
        fp.customer_sk,
        fp.with_discount,
        CASE 
            WHEN EXISTS (
                SELECT 1
                FROM data d
                WHERE d.customer_sk = fp.customer_sk 
                      AND d.order_id <> fp.order_id
                      AND d.order_created_at > fp.first_purchase_date
            ) THEN 1
            ELSE 0
        END AS made_repeat_purchase
    FROM 
        FirstPurchases fp
)
SELECT 
    with_discount,
    COUNT(*) AS total_customers,
    SUM(made_repeat_purchase) AS repeat_customers,
    COUNT(*) - SUM(made_repeat_purchase) AS non_repeat_customers
FROM 
    RepeatPurchases
GROUP BY 
    with_discount;
"""
data = spark.sql(query)
display(data)

# COMMAND ----------

# DBTITLE 1,#Using Chi-squared test
from scipy.stats import chi2_contingency

# Convert Spark DataFrame to Pandas DataFrame
data = data.toPandas()

# Prepare data for chi-squared test
# Adjust column names as per your pandas_df columns
observed_frequencies = pd.DataFrame({
    'repeat_customers': data['repeat_customers'],
    'non_repeat_customers': data['non_repeat_customers']
})

# Convert DataFrame to a numpy array as expected by chi2_contingency
observed = observed_frequencies.to_numpy()

# Perform chi-squared test
chi2, p, dof, expected = chi2_contingency(observed)

# Results
print("Chi-Squared:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)



# COMMAND ----------

# MAGIC %md
# MAGIC ####Test Result 
# MAGIC
# MAGIC Summary:
# MAGIC A chi-squared test was conducted to examine the relationship between the discount offering and repeat purchasing behaviour among two segments of customers.
# MAGIC
# MAGIC Chi-Squared Statistic: is 181.31. 
# MAGIC - This is a measure of how much the observed frequencies deviate from the expected frequencies. A higher value indicates a greater deviation than what would be expected by chance.
# MAGIC
# MAGIC P-Value: The p-value is approximately 2.51e-41 (very close to zero) and less than 0.05 (significant level).
# MAGIC - This indicates that the observed differences in repeat purchasing behavior between the discount and no-discount groups are extremely unlikely to have occurred by chance.
# MAGIC
# MAGIC Degrees of Freedom: The degrees of freedom for the test is 1, consistent with a basic two-category comparison.
# MAGIC - Degree freedom = 1 Degrees of freedom=(2−1)×(2−1)=1 2 rows: one for each group (discount, no discount) 2 columns: one for each outcome (repeat purchase, no repeat purchase)
# MAGIC
# MAGIC Expected Frequencies: The expected frequencies, if there were no association between the discount offering and repeat purchasing, would be:
# MAGIC  - 7,913.8 repeat customers and 33,847.2 non-repeat customers for the discount group.
# MAGIC  - 776.2 repeat customers and 3,319.8 non-repeat customers for the no-discount group.
# MAGIC
# MAGIC Interpretation:
# MAGIC The very low p-value suggests a strong statistical significance in the relationship between the offering of discounts and customer repeat purchasing behavior. Specifically, the deviation from the expected frequencies in our observed data implies that the presence or absence of a discount significantly influences whether customers make repeat purchases.
# MAGIC
# MAGIC Given the high chi-squared statistic and the extremely low p-value, we can confidently assert that discounts play a crucial role in driving repeat purchases among our customer base. This insight is critical for strategic planning, particularly in marketing and sales initiatives, to enhance customer retention and increase sales revenue.
# MAGIC
# MAGIC Conclusion:
# MAGIC The statistical evidence strongly supports the effectiveness of discount strategies in influencing customer purchasing behavior. This finding presents an opportunity to refine our marketing approach and maximize customer lifetime value through targeted discounting strategies.
