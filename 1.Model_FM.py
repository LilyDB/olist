# Databricks notebook source
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

!pip install lightFM

# COMMAND ----------

import pandas as pd
import numpy as np

import lightfm
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, auc_score
from scipy.sparse import csr_matrix

import itertools 

# COMMAND ----------

item = pd.read_csv('/dbfs//FileStore/shared_uploads/in_item.csv',sep=",")
customer = pd.read_csv('/dbfs//FileStore/shared_uploads/in_customer.csv',sep=",")
interactions = pd.read_csv('/dbfs//FileStore/shared_uploads/in_interaction.csv',sep=",")
repeatolist= pd.read_csv('/dbfs//FileStore/shared_uploads/in_repeatolist.csv',sep=",")

# COMMAND ----------

item_features = item[['product_id', 
        ## category features              
       'agro_industry_and_commerce', 'air_conditioning', 'art',
       'arts_and_craftmanship', 'audio', 'auto', 'baby', 'bed_bath_table',
       'books_general_interest', 'books_imported', 'books_technical',
       'christmas_supplies', 'cine_photo', 'computers',
       'computers_accessories', 'consoles_games',
       'construction_tools_construction', 'construction_tools_lights',
       'construction_tools_safety', 'cool_stuff', 'costruction_tools_garden',
       'costruction_tools_tools', 'diapers_and_hygiene', 'drinks',
       'dvds_blu_ray', 'electronics', 'fashio_female_clothing',
       'fashion_bags_accessories', 'fashion_childrens_clothes',
       'fashion_male_clothing', 'fashion_shoes', 'fashion_sport',
       'fashion_underwear_beach', 'fixed_telephony', 'flowers', 'food',
       'food_drink', 'furniture_bedroom', 'furniture_decor',
       'furniture_living_room', 'furniture_mattress_and_upholstery',
       'garden_tools', 'health_beauty', 'home_appliances', 'home_appliances_2',
       'home_comfort_2', 'home_confort', 'home_construction', 'housewares',
       'industry_commerce_and_business',
       'kitchen_dining_laundry_garden_furniture', 'la_cuisine',
       'luggage_accessories', 'market_place', 'music', 'musical_instruments',
       'office_furniture', 'party_supplies', 'perfumery', 'pet_shop',
       'signaling_and_security', 'small_appliances',
       'small_appliances_home_oven_and_coffee', 'sports_leisure', 'stationery',
       'tablets_printing_image', 'telephony', 'toys', 'watches_gifts', 
       ## like vs dislike
       'like','dislike',
       ## price level features
       'very_high', 'high', 'low', 'medium']]


customer_features = customer[['customer_unique_id', 
                              # payment level
                              'very_high', 'high', 'low', 'medium']]


# COMMAND ----------

## item features with price level only
item_features2 =item_features[['product_id', 'very_high', 'high', 'low', 'medium']]

## item features with like/dislike preference
item_features3 =item_features[['product_id','like','dislike']]

## item features with category only
item_features4 = item[['product_id', 
                      
       'agro_industry_and_commerce', 'air_conditioning', 'art',
       'arts_and_craftmanship', 'audio', 'auto', 'baby', 'bed_bath_table',
       'books_general_interest', 'books_imported', 'books_technical',
       'christmas_supplies', 'cine_photo', 'computers',
       'computers_accessories', 'consoles_games',
       'construction_tools_construction', 'construction_tools_lights',
       'construction_tools_safety', 'cool_stuff', 'costruction_tools_garden',
       'costruction_tools_tools', 'diapers_and_hygiene', 'drinks',
       'dvds_blu_ray', 'electronics', 'fashio_female_clothing',
       'fashion_bags_accessories', 'fashion_childrens_clothes',
       'fashion_male_clothing', 'fashion_shoes', 'fashion_sport',
       'fashion_underwear_beach', 'fixed_telephony', 'flowers', 'food',
       'food_drink', 'furniture_bedroom', 'furniture_decor',
       'furniture_living_room', 'furniture_mattress_and_upholstery',
       'garden_tools', 'health_beauty', 'home_appliances', 'home_appliances_2',
       'home_comfort_2', 'home_confort', 'home_construction', 'housewares',
       'industry_commerce_and_business',
       'kitchen_dining_laundry_garden_furniture', 'la_cuisine',
       'luggage_accessories', 'market_place', 'music', 'musical_instruments',
       'office_furniture', 'party_supplies', 'perfumery', 'pet_shop',
       'signaling_and_security', 'small_appliances',
       'small_appliances_home_oven_and_coffee', 'sports_leisure', 'stationery',
       'tablets_printing_image', 'telephony', 'toys', 'watches_gifts']]

# COMMAND ----------

# MAGIC %md ### Create matrices

# COMMAND ----------

# customer feature
user_features_csr = csr_matrix(customer_features.drop('customer_unique_id', axis=1).values)

# item features -all
item_features_csr = csr_matrix(item_features.drop('product_id', axis=1).values)
# item features - price only
item_features_csr2 = csr_matrix(item_features2.drop('product_id', axis=1).values)
# item features - preference
item_features_csr3 = csr_matrix(item_features3.drop('product_id', axis=1).values)
# item features - category only
item_features_csr4 = csr_matrix(item_features4.drop('product_id', axis=1).values)

# interaction
user_item_interaction_csr = csr_matrix(interactions.drop('customer_unique_id',axis=1).astype(np.float))

# Split train/test
train, test =random_train_test_split(user_item_interaction_csr,
                                     test_percentage=0.2, 
                                     random_state=None)



# COMMAND ----------

# MAGIC %md Hyperparameter tuning

# COMMAND ----------

def hyperparameters():
    """
    Yield possible hyperparameter choices.
    """
    while True:
        yield {
            "no_components": np.random.randint(16, 64),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp", "warp-kos"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-8),
            "user_alpha": np.random.exponential(1e-8),
            "max_sampled": np.random.randint(5, 15),
            "num_epochs": np.random.randint(5, 50),
        }

def random_search(test, train, user_features, item_features, num_samples=20):
    """
    Sample random hyperparameters, fit a LightFM model, and evaluate it
    on the test set.
    """

    for hyperparams in itertools.islice(hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)
        model.fit(train, 
                  user_features=user_features, 
                  item_features=item_features,
                  epochs=num_epochs, num_threads=1)

        score = auc_score(model = model, 
                        test_interactions = test,
                        train_interactions = train,
                        user_features = user_features,
                        item_features = item_features,
                        num_threads = 4, check_intersections=False).mean()

        hyperparams["num_epochs"] = num_epochs

        yield (score, hyperparams, model)

# COMMAND ----------


# MAGIC %md Interaction + User features

# COMMAND ----------

(score, hyperparams6, model) = max(random_search(test,
                                                train,
                                                user_features = user_features_csr,
                                                item_features = None), 
                                  key=lambda x: x[0])


print("Best score {} at {}".format(score, hyperparams6))

# COMMAND ----------

# MAGIC %md ####Training

# COMMAND ----------

# Train model
model_hyperparams6 = {key: value for key, value in hyperparams6.items() if key not in 'num_epochs'}

model1 = LightFM(**model_hyperparams6, random_state=3)


model1.fit(train,
          user_features=user_features_csr, 
          sample_weight=None, 
          epochs=46,     
          verbose=False)


auc_with_features = auc_score(model1, 
                              test_interactions = test,
                              train_interactions = train,
                              user_features=user_features_csr, 
                              check_intersections=False)

train_auc = auc_score(model1, train,  user_features=user_features_csr).mean()
test_auc = auc_score(model1, test,  user_features=user_features_csr).mean()

print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

# COMMAND ----------

# MAGIC %md Run for the whole dataset

# COMMAND ----------

# Train model
model_hyperparams6 = {key: value for key, value in hyperparams6.items() if key not in 'num_epochs'}

model2 = LightFM(**model_hyperparams6, random_state=3)


model2.fit(user_item_interaction_csr,
          sample_weight=None, 
          epochs=46,     
          verbose=False)

# COMMAND ----------

# MAGIC %md Make recommendations

# COMMAND ----------

def recommender(user_id, model, user_item_interaction,k=5):
  """
  Input: 
    user_id: user(customer) id or list of user(customer) ids
    model: lightFM model
    user_item_interaction: user-item data set before converted to sparse matrix
    k: Number of recommended products
  Output:
    User id
    Purchased product + category
    Recommended products + category
  
  """
  interactions = user_item_interaction
  interactions = interactions.set_index('customer_unique_id')
  user_ids = list(interactions.index)
  user_dict = {}
  counter = 0 
  for i in user_ids:
    user_dict[i] = counter
    counter += 1

   ###############################################################
  item_dict ={}
  df_item = repeatolist.groupby(['product_id', 
                                'product_category_name_english'])['price']\
                        .mean()\
                        .reset_index()

  df_item['price'] = df_item['price'].astype(int)
  
  df_item['product_category_name_english'] = df_item['product_id']+' / '\
                                            + df_item['product_category_name_english'] +', avg price: '\
                                            + df_item['price'].astype(str)

  for i in range(df_item.shape[0]):
    item_dict[(df_item.loc[i,'product_id'])] = df_item.loc[i,'product_category_name_english']
  
  ###############################################################
  user_x = user_dict[user_id]
  n_users, n_items = interactions.shape # no of users * no of items
  

  
  score= pd.Series(model2.predict(user_x, np.arange(n_items)))

  score.index = interactions.columns
  score= list(pd.Series(score.sort_values(ascending=False).index))

  purchased_items = list(
                      pd.Series(interactions.loc[user_id,:] \
                                [interactions.loc[user_id,:] > 0].index)\
                                 .sort_values(ascending=False)
                                                                  )

  score = [x for x in score if x not in purchased_items]
  no_return_items = score[0:k]
  purchased_items = list(pd.Series(purchased_items).apply(lambda x: item_dict[x]))
  score = list(pd.Series(no_return_items).apply(lambda x: item_dict[x]))

   ###############################################################
  print ("Customer: " + str(user_id))

  print("Purchased product(s):")
  counter = 1

  for i in purchased_items:
    print(str(counter) + ') ' + i)
    counter+=1

  print("\n Recommended products:")
  counter = 1
  for i in score:
    print(str(counter) + '- ' + i)
    counter+=1



# COMMAND ----------

# MAGIC %md Test recommender

# COMMAND ----------

user_id1 = '000fbf0473c10fc1ab6f8d2d286ce20c'
recommender(user_id1, model1, interactions, k=5)


# COMMAND ----------

user_id2 = '000bfa1d2f1a41876493be685390d6d3'
recommender(user_id2, model2,interactions, k=5)

# COMMAND ----------

u5 = '002ae492472e45ad6ebeb7a625409392'
recommender(u5, model1, interactions,k=5)
