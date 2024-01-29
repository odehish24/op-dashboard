# Databricks notebook source
# MAGIC %md
# MAGIC - Progestogen only pill                      
# MAGIC - Emergency contraception        
# MAGIC - Combined oral contraception    
# MAGIC                 

# COMMAND ----------

# Problem statement: lack of understanding the uptake of contraception product in fettle brand
# customer purchasing habit analysis: Analyse the relationship between the price options and product-drugh length
# Goal - Improve product offering 

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# DBTITLE 1,Retrieve data needed
df = spark.sql('''
            SELECT se.order_created_at, se.sh24_uid, se.customer_sk, se.product_type, p.product_name, dc.drug_configuration_course_length_in_months, dc.drug_configuration_description, se.unit_price, 
            d.reporting_age_bracket
            FROM warehouse.sales_events se
            LEFT JOIN warehouse.products p ON p.sk = se.product_sk
            LEFT JOIN warehouse.drug_configurations dc ON dc.id = se.drug_configuration_sk
            LEFT JOIN warehouse.demographics d on d.id = se.demographic_sk
            WHERE se.product_type IN ('Progestogen only pill', 'Combined oral contraception', 'patch', 'Emergency contraception') 
            AND se.brand_sk = 2
               ''').toPandas()
df.tail()

# COMMAND ----------

# DBTITLE 1,Data prep
#rename long names
df = df.rename(columns={'drug_configuration_course_length_in_months': 'drug_length', 'drug_configuration_description': 'drug_descr', 'reporting_age_bracket': 'age_bracket'})

# Fill 'NULL' string values with 0 in the 'drug_length' column
df['drug_length'] = df['drug_length'].replace('NULL', 0)

# Convert the drug_length column to IntegerType
df['drug_length'] = df['drug_length'].astype(int)

# COMMAND ----------

# Create a function that label encode all non numerical colums
from sklearn.preprocessing import LabelEncoder

def label_encode_object_cols(df):
    object_cols = df.select_dtypes(exclude='int64').columns
    le = LabelEncoder()   
    # Label encode object columns
    for col in object_cols:
        df[col] = le.fit_transform(df[col])  
    return df

# Find if the product type are correlated, first, Pivot the product type 
pivot_df = df.pivot_table(index=df.order_created_at, columns='product_type', values='sh24_uid', aggfunc='count',fill_value=0)

# correlation between product type variables 
pivot_df = label_encode_object_cols(pivot_df)
corrcoef = pivot_df.corr()
sns.set(rc={'figure.figsize':(8,6)})
sns.heatmap(corrcoef, annot = True)

# COMMAND ----------

# Display the DataFrame
display(df)

# COMMAND ----------

# # Filter rows where the date is on or after April 2020
# df2 = df[df['date'] >= pd.Timestamp(year=2020, month=4, day=1)]
# display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Progestogen only pill (POP)

# COMMAND ----------

pop = df[df['product_type']=='Progestogen only pill']
pop1 = pop.drop(['product_type'],axis =1)
pop1 = label_encode_object_cols(pop1)
corrcoef = pop1.corr()
sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(corrcoef, annot = True)

# COMMAND ----------

display(pop)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Combined oral contraception (COC)

# COMMAND ----------

coc = df[df['product_type']=='Combined oral contraception']
coc1 = coc.drop(['product_type'],axis =1)
coc1 = label_encode_object_cols(coc1)
corrcoef = coc1.corr()
sns.heatmap(corrcoef, annot = True)

# COMMAND ----------

# COC
display(coc)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Emergency Contraception (EC)

# COMMAND ----------

#EC has same drug length of null
ec = df[df['product_type']=='Emergency contraception']
ec1 = ec.drop(['product_type', 'drug_length'],axis =1)
ec1 = label_encode_object_cols(ec1)
corrcoef = ec1.corr()
sns.heatmap(corrcoef, annot = True)

# COMMAND ----------

display(ec)

# COMMAND ----------

#the last 3 months

# COMMAND ----------

##all

# COMMAND ----------

data = spark.sql('''
            SELECT DISTINCT
              date_trunc('HOUR', se.order_created_at) AS order_date,
              coalesce(dt.kit_colour, se.product_type) as products
              FROM warehouse.sales_events se 
              LEFT JOIN warehouse.detailed_test_kits dt ON se.sh24_uid = dt.sh24_uid
              WHERE se.brand_sk = 2 
                AND se.item_dispatched_at >= '2023-01-01'
                AND se.product_type IN ('STI Test kit', 'Progestogen only pill', 'Combined oral contraception', 'Emergency contraception')
               ''').toPandas()
data.tail()

# COMMAND ----------

product_counts = data['products'].value_counts()
top_13_products = product_counts.head(13).index.tolist()
new_data = data[data['products'].isin(top_13_products)]

# COMMAND ----------

# Count the occurrences of each product
product_counts = data['products'].value_counts()

# Get the top 12 products
top_12_products = product_counts.head(12).index.tolist()

# Filter your dataset to include only these top 12 products
filtered_data = data[data['products'].isin(top_12_products)]

# Convert the 'products' column into dummy variables for correlation analysis
product_dummies = pd.get_dummies(filtered_data['products'])

# Compute the correlation matrix for these dummy variables
correlation_matrix = product_dummies.corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Top 12 Products')
plt.show()


# COMMAND ----------



# COMMAND ----------


