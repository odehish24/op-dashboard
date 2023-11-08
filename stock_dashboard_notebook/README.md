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


Data Preparation
  Data Wrangling Steps:
  - Create a brand hookup table, this is to align the brand table sk identifier with testing_service in distribution_centres table (they are different) 
  - To get the brand for each kit created you need both brand attribute in episodes_of_care and distribution_centres table
  - To get the built_at, we use created_at for non_custom kit and packaged_at for custom_kit ie kit type = 1

  Data Quality Checks:

Dashboard Features
  - Total kit created for the current year
  - Filter by date


Contributors:
  - Taran
  - Matt
  - Gaz
  - Odehi
