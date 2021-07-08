"""
Sam Voisin
February 2021
Separate and Segregate Validation Set
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


# read data
fp = os.path.abspath('./dataset/train.csv')
raw_df = pd.read_csv(fp)

model_set, valid_set = train_test_split(raw_df,
                                        test_size=0.2,
                                        random_state=1)

os.makedirs("./dataset/validation_data", exist_ok=True)
model_set.to_csv("./dataset/validation_data/validation_set.csv")
model_set.to_csv("./dataset/model_set.csv")


