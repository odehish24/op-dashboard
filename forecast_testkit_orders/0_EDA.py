# Databricks notebook source
# MAGIC %md
# MAGIC ###Exploratory Data Analysis
# MAGIC - Data Cleaning
# MAGIC - Feature engineering
# MAGIC - Visualizations to understand the data

# COMMAND ----------

# DBTITLE 1,Import libraries needed
#import libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
%matplotlib inline
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore") 
plt.rcParams["figure.figsize"] = (20,5)

# COMMAND ----------

# DBTITLE 1,Retrieve preprocessed data 
#Retrieve data and convert to pandas df
df = spark.sql("""select * from prep_ord 
                    """).toPandas()
display(df)

# COMMAND ----------

df.shape

# COMMAND ----------

# DBTITLE 1,Remove null and duplicate rows
#drop null values
df.dropna()

#remove duplicated rows, about 200,000 duplicated orders
df.drop_duplicates(inplace=True)
df.shape

# COMMAND ----------

# DBTITLE 1,Understanding the columns in the data
#create a function to describe the columns selected

def describe_the_columns(df):
    column_stats = []

    for col_name in df.columns:
        col = df[col_name]
        
        count = col.count()
        unique = col.nunique()
        
        value_counts = col.value_counts().reset_index()
        value_counts.columns = [col_name, "count"]
        value_counts = value_counts.sort_values(by="count", ascending=False)
        
        highest = value_counts.iloc[0][col_name]
        highest_freq = value_counts.iloc[0]["count"]
        
        lowest = value_counts.iloc[-1][col_name]
        lowest_freq = value_counts.iloc[-1]["count"]
        
        min_value = col.dropna().min()
        max_value = col.dropna().max()
        
        column_stats.append((col_name, count, unique, highest, highest_freq, lowest, lowest_freq, min_value, max_value))
    
    column_description = pd.DataFrame(column_stats, columns=['column_name', 'count', 'unique', 'highest', 'highest_freq', 'lowest', 'lowest_freq', 'min', 'max'])
    
    return column_description


# COMMAND ----------

result = describe_the_columns(df)
result

# COMMAND ----------

# DBTITLE 1,Feature engineering
def create_date_features(df, datetime_column):
    # Convert the datetime_column to datetime type
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    
    # Extract date, time, and day of the week components
    df['Date'] = pd.to_datetime(df[datetime_column].dt.date)
    df['Time'] = df[datetime_column].dt.hour
    df['Weekday'] = df[datetime_column].dt.weekday
    df['Month'] = df[datetime_column].dt.month
    df['Quarter'] = df[datetime_column].dt.quarter
    df['WeekofYear'] = df[datetime_column].dt.week
    df['YearMon'] = df[datetime_column].dt.strftime('%Y-%b')
    df['Year'] = df[datetime_column].dt.year
    
    return df

# Assuming df is your pandas DataFrame and 'order_created_at' is your datetime column
create_date_features(df, 'order_created_at')

# COMMAND ----------

# MAGIC %md
# MAGIC ###Data Visualisation
# MAGIC - Labelencode object features
# MAGIC - Plot different charts

# COMMAND ----------

# DBTITLE 1,function that label encode all object features
#Create a function that label encode all non numerical columns for correlation analysis
from sklearn.preprocessing import LabelEncoder

def label_encode_object_cols(df):
    # Identify columns with object dtype
    object_cols = df.select_dtypes(exclude='int64').columns
    
    # Initialize LabelEncoder
    le = LabelEncoder()
    
    # Label encode object columns
    for col in object_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

# COMMAND ----------

# DBTITLE 1,Heapmap to display correlation between features
#create a label endoded df
df_label=df.copy()
label_encode_object_cols(df_label)

#heatmap showing correlation between the features/variables
plt.figure(figsize = (10,6))
corrcoef = df_label.corr()
sns.heatmap(corrcoef, annot = False)

# COMMAND ----------

# DBTITLE 1,Piechart to show proportion of fulfilment_method
orders =df['fulfilment_method'].replace(['click_and_collect', 'offline_kits'], 'others').value_counts()
labelx = ['postal_dispatch', 'others']

fig, ax = plt.subplots()
colors = ['#FFC107', '#03A9F4']
plt.title('Proportion of fulfilment_method')
ax.pie(orders.values, labels=labelx, colors = colors, autopct='%1.0f%%')
plt.show()

# COMMAND ----------

# DBTITLE 1,10 Least and 10 Top Test kit colours
df.colour.value_counts().tail(10).plot(kind='barh')
plt.xlabel("No of order")  
plt.ylabel("Test kit colour")
plt.title('10 least testkit colour') 
plt.show()

# Select only top 10 Test kit colours
df.colour.value_counts().head(10).plot(kind='bar')
plt.xlabel("No of order")  
plt.ylabel("Test kit colour")
plt.title('10 Top testkit colour') 
plt.show()

# COMMAND ----------

# DBTITLE 1,Total orders by Month and brand
# Orders by Month and brand
orders = df.groupby(['Month', 'brand'])['sh24_uid'].count().unstack()

Month_Order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

# Create the multiple bar chart
ax = orders.plot(kind='bar', figsize=(12, 6))
ax.set_xticklabels(Month_Order)  # Set the x-axis tick labels
ax.legend(loc='lower left', bbox_to_anchor=(1, -0.1), ncol=1)
ax.set_xlabel('Month')
ax.set_ylabel('Order count')
ax.set_title('Total Order by Month and brand')

plt.show()

# COMMAND ----------

# DBTITLE 1,Total Order by week of year and brand
orders = df.groupby(['WeekofYear', 'brand'])['sh24_uid'].count().unstack()

# Create the multiple bars
ax = orders.plot(kind='bar', stacked=True, figsize=(12, 6))
ax.legend(loc='lower left', bbox_to_anchor=(1, -0.1), ncol=1)
ax.set_xlabel('Week of year')
ax.set_ylabel('Order count')
ax.set_title('Total Order by week of year and brand')
ax.set_xticks(range(0, len(orders), 2))

plt.show()

# COMMAND ----------

# DBTITLE 1,SH24 order by Week and Quarter
sh['quarter'] = sh.index.quarter
sh['week'] = sh.index.dayofweek

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=sh.dropna(),
            x='week',
            y='sh24_uid',
            hue='quarter',
            ax=ax,
            linewidth=1)
ax.set_title('SH24 order by Week and Quarter')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Order')
ax.set_xticklabels(Weekday_Order)  # Set the x-axis tick labels
ax.legend(bbox_to_anchor=(1, 1))
plt.show()

# COMMAND ----------

# DBTITLE 1,Total Orders by Week and Top 10 test_kit colour
orders = df.groupby(['Weekday', 'colour'])['sh24_uid'].count().unstack()
top = orders.sum().nlargest(10).index
orders_df = orders[top]

Weekday_Order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Create the multiple lines
ax = orders_df.plot(kind='bar',stacked=True)
ax.set_xticklabels(Weekday_Order)
ax.legend(loc='lower left', bbox_to_anchor=(1, -0.1), ncol=1)
ax.set_xlabel('Week')
ax.set_ylabel('Order count')
ax.set_title('Total Orders by Week and test_kit colour')


# COMMAND ----------

# DBTITLE 1,Total Order per week over the Years
# Calculate the sum of meal prices for each year
orders = df.groupby(['WeekofYear', 'Year'])['sh24_uid'].count().unstack()

# Create the multiple lines
ax = orders.plot(kind='line', figsize=(16, 8))
ax.legend(loc='lower left', bbox_to_anchor=(1, -0.1), ncol=1)
ax.set_xlabel('Week of the year')
ax.set_ylabel('Order count')
ax.set_title('Total Order by week over the Years')
ax.set_xticks(range(0, len(orders), 2))

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Visualize trends of each brands over the years

# COMMAND ----------

sh = pd.DataFrame(df[df.brand=='SH:24'].groupby(['Date'])['sh24_uid'].count())
fe = pd.DataFrame(df[df.brand=='Fettle'].groupby(['Date'])['sh24_uid'].count())
ft = pd.DataFrame(df[df.brand=='Freetesting HIV'].groupby(['Date'])['sh24_uid'].count())
ir = pd.DataFrame(df[df.brand=='SH:24 Ireland'].groupby(['Date'])['sh24_uid'].count())
hc = pd.DataFrame(df[df.brand=='Hep C Ireland'].groupby(['Date'])['sh24_uid'].count())
ar = pd.DataFrame(df[df.brand=='Aras Romania'].groupby(['Date'])['sh24_uid'].count())

# Plotting for SH:24
sh.plot(title='Total Orders for SH:24')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.show()

# Plotting for Fettle
fe.plot(title='Total Orders for Fettle')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.show()

# Plotting for Freetesting HIV
ft.plot(title='Total Orders for Freetesting HIV')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.show()

# Plotting for SH:24 Ireland
ir.plot(title='Total Orders for SH:24 Ireland')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.show()

# Plotting for Hep C Ireland
hc.plot(title='Total Orders for Hep C Ireland')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.show()

# Plotting for Aras Romania
ar.plot(title='Total Orders for Aras Romania')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.show()

# COMMAND ----------

# DBTITLE 1,Total order by Time and brand
orders = df.groupby(['Time', 'brand'])['sh24_uid'].count().unstack()

# Create the multiple lines
ax = orders.plot(kind='line', figsize=(12, 5))
ax.legend(loc='lower left', bbox_to_anchor=(1, -0.1), ncol=1)
ax.set_xlabel('Time')
ax.set_ylabel('Order count')
ax.set_title('Total Order by Time and brand')
ax.set_xticks(range(0, len(orders), 1))

plt.show()

# COMMAND ----------

orders = df.groupby(['Weekday', 'brand'])['sh24_uid'].count().unstack()

# Create the multiple lines
ax = orders.plot(kind='line', figsize=(12, 5))
ax.legend(loc='lower left', bbox_to_anchor=(1, -0.1), ncol=1)
ax.set_xlabel('Week')
ax.set_ylabel('Order count')
ax.set_xticklabels(Weekday_Order)
ax.set_title('Total Order by Week and brand')
ax.set_xticks(range(0, len(orders), 1))

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ####Observation
# MAGIC - Trends for most brands are different over the years
# MAGIC - Freetesting HIV and Ireland has similar trends
# MAGIC - Build 3 predictive model for the brands

# COMMAND ----------



# COMMAND ----------


