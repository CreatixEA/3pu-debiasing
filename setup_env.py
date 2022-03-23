# In this file, we setup the experimental environment :
# All different process must be reproducible

import pandas as pd
import common
import pickle

import logging
log = logging.getLogger(__name__)

df = common.load_data()



# We define split indexes and save them
split_ratio = 0.2
validation_indexes = df.sample(frac=split_ratio).index
train_indexes = df.drop(validation_indexes).index

log.info(f"Dataset split {split_ratio} : {len(train_indexes)} + {len(validation_indexes)}")

with open('archived/validation_indexes.pkl', 'wb') as f:
    pickle.dump(validation_indexes, f)
    log.info("Validation indexes saved")

# We get all missing data value locations
df.isna().to_csv('archived/missing_values_matrix.csv')
log.info("Missing values saved")
