# Databricks notebook source
# MAGIC %md
# MAGIC ###### Include steps
# MAGIC - Load data
# MAGIC - Explorative data analysis
# MAGIC - Pre-processing and normalization 
# MAGIC - Save data

# COMMAND ----------

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# DBTITLE 1,Load data
customers = pd.read_csv('/dbfs//FileStore/shared_uploads/olist_customers_dataset.csv')
order_items = pd.read_csv('/dbfs//FileStore/shared_uploads/olist_order_items_dataset.csv')
order_payments = pd.read_csv('/dbfs//FileStore/shared_uploads/olist_order_payments_dataset.csv')
order_reviews = pd.read_csv('/dbfs//FileStore/shared_uploads/olist_order_reviews_dataset.csv')
orders = pd.read_csv('/dbfs//FileStore/shared_uploads/olist_orders_dataset.csv')
product = pd.read_csv('/dbfs//FileStore/shared_uploads/olist_products_dataset.csv')
sellers = pd.read_csv('/dbfs//FileStore/shared_uploads/olist_sellers_dataset.csv')
product_name = pd.read_csv('/dbfs//FileStore/shared_uploads/product_category_name_translation.csv')

# COMMAND ----------

# DBTITLE 1,EDA
# displaying data shape
dataset = {
    'customers': customers,
    'order_items': order_items,
    'order_payments': order_payments,
    'orders': orders,
    'order_reviews':order_reviews,
    'product': product,
    'product_name': product_name,
    'sellers': sellers
}

for x, y in dataset.items():
    print(f'{x}')
    print(y.info())
    print('Missing data:', y.isnull().any().any())
    if y.isnull().any().any():
      print("No. missing value: ")
      print(f'{y.isnull().sum()}\n')
    print("--------------------------------------------------------------")

# COMMAND ----------

# MAGIC %md ####### 1 customer can be in multiple states ==> drop state feature

# COMMAND ----------

# Check variable consistency
customers = customers[['customer_id',
                       'customer_unique_id',
                       'customer_state']]

customer = customers.groupby('customer_unique_id')['customer_state'].count().reset_index()
print("No customers living in one state: ", customer.loc[customer.customer_state==1,:]['customer_unique_id'].nunique())
print("No customers in total: ", customers['customer_unique_id'].nunique())

# COMMAND ----------

# DBTITLE 1,Merge data
# Select required parameters
customers = customers[['customer_id',
                       'customer_unique_id']]


orders = orders[['order_id',
                 'customer_id']]

order_items = order_items[['order_id',
                           'product_id',
                           'price']]

order_payments = order_payments[['order_id',
                                 'payment_value']]\
                              .groupby('order_id')['payment_value']\
                              .sum()\
                              .reset_index()

order_reviews = order_reviews[['order_id',
                               'review_score']]

products = product[['product_id',
                    'product_category_name']]

# merge into one dataset
dfList = [orders,order_items, order_payments, order_reviews]

df1 = reduce(lambda left, 
                      right: pd.merge(left, 
                                     right, 
                                             on=['order_id'],
                                             how='inner'),
                dfList )

df1.drop_duplicates(keep='first',inplace=True)

df2 = df1.merge(customers,
                how="inner",
                left_on="customer_id",
                right_on="customer_id")

df3 = df2.merge(product,
                how="inner",
                left_on="product_id",
                right_on="product_id")

df4 = df3.merge(product_name,
                how="inner",
                left_on="product_category_name",
                right_on="product_category_name")

olist = df4[['order_id', 'customer_id', 'product_id', 'payment_value','price',
             'review_score', 'customer_unique_id', 
             'product_category_name_english']]

olist.drop_duplicates(keep='first',
                      inplace=True)

# COMMAND ----------

# DBTITLE 1,Is data of acceptable quality?
# Normal distribution of log payment value
olist_ordervalue = olist.groupby('order_id')['payment_value'].sum().reset_index()
plt.hist(np.log(olist_ordervalue.payment_value.values))
print(kurtosis(np.log(olist_ordervalue.payment_value.values)))
print(skew(np.log(olist_ordervalue.payment_value.values)))

# COMMAND ----------

# MAGIC %md ##### Split data into 2 datasets: 1) customers with more than 1 order; customers ordered first time

# COMMAND ----------

num_order = olist.groupby('customer_unique_id')['order_id']\
                .count()\
                .reset_index()

repeatList = num_order.loc[num_order.order_id>=2,:]['customer_unique_id'].unique()
firsttimeList = num_order.loc[num_order.order_id==1,:]['customer_unique_id'].unique()

repeat_olist = olist[olist['customer_unique_id'].isin(repeatList)]

firsttime_olist =  olist[olist['customer_unique_id'].isin(firsttimeList)]

print(repeat_olist.customer_unique_id.nunique())
print(repeat_olist.product_id.nunique())
repeat_olist.head(3)

# COMMAND ----------

# MAGIC %md Feature Engineering

# COMMAND ----------

# DBTITLE 1,PRODUCT FEATURES
product_price = repeat_olist[['product_id',
                             'price']]

product_price = product_price.groupby('product_id')['price']\
                            .mean()\
                            .reset_index()

col         = 'price'
conditions  = [ product_price[col] > 120, 
               (product_price[col] <= 120) & (product_price[col]> 65),
               (product_price[col] <= 65) & (product_price[col]> 35),
                product_price[col] <= 35 ]

choices     = [ "very_high",
                "high", 
               'medium',
               'low' ]
    
product_price["price"] = np.select(conditions, 
                                  choices, 
                                  default=np.nan)

dummies_price = pd.get_dummies(product_price['price'])

product_price= pd.merge(
    left=product_price,
    right=dummies_price,
    left_index=True,
    right_index=True,
)

## Rating level
product_review = repeat_olist[['product_id',
                             'review_score']]
product_review.drop_duplicates(keep='first',inplace=True)
product_review = product_review.groupby('product_id')['review_score']\
                            .mean()\
                            .reset_index()

col         = 'review_score'
conditions  = [ product_review[col] > 2, 
                product_review[col] <= 2]

choices     = [ "like",
                "dislike" ]
    
product_review["review_score"] = np.select(conditions, 
                                  choices, 
                                  default=np.nan)

dummies_review = pd.get_dummies(product_review['review_score'])

product_review = pd.merge(
    left=product_review,
    right=dummies_review,
    left_index=True,
    right_index=True,
)
## product category
product_cat = repeat_olist[['product_id',
                             'product_category_name_english']]
product_cat.drop_duplicates(keep='first',inplace=True)

dummies_cat = pd.get_dummies(product_cat['product_category_name_english'])

product_cat = pd.merge(
    left=product_cat,
    right=dummies_cat,
    left_index=True,
    right_index=True,
)

product_features_list = [product_cat, product_review, product_price]

product_features = reduce(lambda left, 
                                  right: pd.merge(left, 
                                                  right, 
                                                  left_on = 'product_id', 
                                                  right_on='product_id',
                                                  how='inner'),
                            product_features_list )

product_features

# COMMAND ----------

# MAGIC %md ###### Customer - Feature Engineering

# COMMAND ----------

# DBTITLE 1,Check payment level
customer_pay = repeat_olist[['customer_unique_id',
                             'payment_value']]

customer_pay = customer_pay.groupby('customer_unique_id')['payment_value']\
                            .sum()\
                            .reset_index()
customer_pay.describe()

# COMMAND ----------

# MAGIC %md Create payment level category

# COMMAND ----------

# CUSTOMER FEATURE
customer_pay = repeat_olist[['customer_unique_id', 'payment_value']]

customer_pay = customer_pay.groupby('customer_unique_id')['payment_value']\
                            .sum()\
                            .reset_index()

col         = 'payment_value'
conditions  = [ customer_pay[col] > 510, 
               (customer_pay[col] <= 510) & (customer_pay[col]> 309),
               (customer_pay[col] <= 309) & (customer_pay[col]> 187),
                customer_pay[col] <= 187 ]

choices     = [ "very_high",
                "high", 
               'medium',
               'low' ]
    
customer_pay["pay_level"] = np.select(conditions, 
                                  choices, 
                                  default=np.nan)

dummies_pay = pd.get_dummies(customer_pay["pay_level"])
customer_features = pd.merge(
    left=customer_pay,
    right=dummies_pay,
    left_index=True,
    right_index=True,
    how = "inner"
)

customer_features 


# COMMAND ----------

# MAGIC %md ## Create customer-product interaction dataset

# COMMAND ----------

interaction = repeat_olist[['customer_unique_id','product_id','review_score']]
interaction.drop_duplicates(keep='first',inplace=True)

# Normalization 
col         = 'review_score'
conditions  = [ interaction[col] >= 3, 
                interaction[col] <= 2]

choices     = [ 1,0 ]
    
interaction["review_score"] = np.select(conditions, 
                                  choices, 
                                  default=np.nan)

interaction

# COMMAND ----------

# MAGIC %md Customer-Product Interaction transpose

# COMMAND ----------

interaction = pd.pivot_table(interaction,
                                       index='customer_unique_id',
                                       columns='product_id',
                                       values='review_score')

interaction = interaction.fillna(0)
interaction

# COMMAND ----------

# DBTITLE 1,Sparsity
unique_users = len(interaction.index)
unique_items = len(interaction.columns)
sparsity = 1 - (len(interaction) / (unique_users * unique_items))
print("Interaction matrix sparsity: {}%".format(round(100 * sparsity, 2)))

# COMMAND ----------

# DBTITLE 1,Save for later use
product_features.to_csv('/dbfs//FileStore/shared_uploads/phuong.le@man-es.com/in_item.csv', index=False)
customer_features.to_csv('/dbfs//FileStore/shared_uploads/phuong.le@man-es.com/in_customer.csv', index=False)
firsttime_olist.to_csv('/dbfs//FileStore/shared_uploads/phuong.le@man-es.com/in_firsttime.csv', index=False)
interaction.to_csv('/dbfs//FileStore/shared_uploads/phuong.le@man-es.com/in_interaction.csv')

# COMMAND ----------

repeat_olist.to_csv('/dbfs//FileStore/shared_uploads/phuong.le@man-es.com/in_repeatolist.csv', index=False)
