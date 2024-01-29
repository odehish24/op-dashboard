Stock Usage Dashboard Project README

Table of Contents
  1.  Project Overview
  2.  Dependencies
  3.  Data Preparation
  4.  Dashboard Features
  5.  Contributors
   

Project Overview
  Objective: This project is to help operation team optimize inventory management by creating and updating a stock dashboard where they view how each consumables is used by each kit created
  Data Sources: 
  - Dabaricks > hive_metastore
                raw_admin.test_kits
                raw_admin.sti_test_orders
                raw_admin.episodes_of_care
                raw_admin.batches
                raw_admin.distribution_centres
                default.testkit_colour_sample

  Key Metrics:
  - kit consumables used daily
  - Total Quantity remaining for each consumables daily

Dependencies - Libraries/Packages:
  pyspark
  SparkFiles
  pyspark.sql
  from pyspark import SparkFiles
  from pyspark.sql.functions import sum, col, when, lit, regexp_replace
  from pyspark.sql import functions as F


Data Preparation
  Data Wrangling Steps:
  - Create a brand hookup table, this is to align the brand table sk identifier with testing_service in distribution_centres table (they are different) 
  - To get the brand for each kit created you need both brand attribute in episodes_of_care and distribution_centres table
  - To get the built_at, we use created_at for non_custom kit and packaged_at for custom_kit ie kit type = 1

Data Quality Checks:

Code Overview

Get Daily Kit Creation Data
This part of the code retrieves daily kit creation data using a SQL query that selects certain columns from several tables:
  - test_kits table
  - testkitcode_colour_sample table,
  - sti_test_orders table,
  - episodes_of_care table,
  - batches table,
  - distribution_centres table.


Data Cleaning
In this section, the kit_created1 table is filtered to remove rows where dispatched_at column is NULL or where brand_sk AND brand1 columns are NULL.

Add Updated Kit Creation results to a table
In this part of the code, the filtered kit_created1 DataFrame from the previous section is added to a table called kit_created using append mode.

Calculate Consumable Usage
This section queries data from two tables (kit_created and bill_of_materials) to calculate the consumable usage. Specifically, this code:
The code retrieves, filters and joins daily kit creation data with the bill of materials to calculate the consumable used. The resulting data is saved in tables called kit_created and consumable_used, respectively.

Dashboard Features
  - Total kit created for the current year
  - Filter by date


Contributors:
  - The operation and distribution team
  
