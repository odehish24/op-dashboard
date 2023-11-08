# Databricks notebook source
# MAGIC %md
# MAGIC ####Data preparation
# MAGIC - Retrieve data needed
# MAGIC - Save data as table

# COMMAND ----------

# DBTITLE 1,MODEL INITIAL DATA RETRIEVAL
# df = spark.sql("""with CTE AS (
#                             select sh24_uid, product_type, brand_sk, product_sk, order_created_at
#                                     from warehouse.sales_events
#                                     where product_type = 'STI Test kit'
#                                     and order_created_at < '2023-10-01T00:00:00.000+0000'
#                             )
#                             select c.sh24_uid, c.brand_sk, sto.test_kit_code, tcs.sample_sk, c.order_created_at
#                             from CTE c
#                             LEFT join raw_admin.sti_test_orders sto ON sto.sh24_uid = c.sh24_uid
#                             LEFT join testkit_colour_sample tcs on tcs.test_kit_code = sto.test_kit_code
#                """)
             
# df.write.mode("overwrite").saveAsTable("prep_sti_order")

# COMMAND ----------

# DBTITLE 1,Updating Model Data
#retrieve the data needed
df = spark.sql("""with CTE AS (
                            select sh24_uid, product_type, brand_sk, product_sk, order_created_at
                                    from warehouse.sales_events
                                    where product_type = 'STI Test kit'
                                    and order_created_at > (SELECT MAX(order_created_at) FROM prep_sti_order)
                            )
                            select c.sh24_uid, c.brand_sk, sto.test_kit_code, tcs.sample_sk, c.order_created_at
                            from CTE c
                            LEFT join raw_admin.sti_test_orders sto ON sto.sh24_uid = c.sh24_uid
                            LEFT join testkit_colour_sample tcs on tcs.test_kit_code = sto.test_kit_code
               """)
             
df.write.mode("append").saveAsTable("prep_sti_order")
