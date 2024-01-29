Kit by Consumables Predictions Project README

Table of Contents
  1.  Project Overview
  2.  Dependencies
  3.  Data Preparation
  4.  Model or Algorithm
  5.  Validation and Testing
  6.  Scalability and Performance
  7.  Future Work
  9.  Contributors  

Project Overview
  Objective: This project is is focus on STI test kits predictions and the required kit consumables to aid operation team for inventory management

Data Sources: 
  - warehouse.sales_event
  - warehouse.brands, lab
  - sample_testkitcode_colour

Dependencies - Libraries/Packages:
  pandas

Data Preparation
  Data Wrangling Steps:
  - Handle null values in date by filling it with the missing date and use 0 for the count_order
  Data Quality Checks:

Model or Algorithm
  Description: Xgboost and Prophet model
  Facebook prophet model

Validation and Testing
  There were three error metics used to evaluate the model
  Mean absolute error(MAE)
  Root mean squared error(RMSE)
  Mean absolute percentage error(MAPE)

Scalability and Performance

Future Work
Contributors
References