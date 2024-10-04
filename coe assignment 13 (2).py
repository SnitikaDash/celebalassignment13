import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
dataset_path = r"C:\Users\sniti\OneDrive\Desktop\games_dataset.csv"


try:
    df = pd.read_csv(dataset_path, encoding='utf-8')   
except UnicodeDecodeError:
    df = pd.read_csv(dataset_path, encoding='latin1')   

 
print(df.head())

# 1. Handling Missing Values
 
missing_values = df.isnull().sum()
print(f"Missing values in each column:\n{missing_values}")

 
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

 
print(f"Missing values after imputation:\n{df.isnull().sum()}")

# 2. Transformation and Normalization
 
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 3. Encoding Categorical Features

print("Categorical columns:", cat_cols)


try:
    encoder = OneHotEncoder(sparse_output=False, drop='first')  
except TypeError:
    
    encoder = OneHotEncoder(sparse=False, drop='first')


encoded_cat_cols = encoder.fit_transform(df[cat_cols])

 
encoded_cat_df = pd.DataFrame(encoded_cat_cols, columns=encoder.get_feature_names_out(cat_cols))

 
print("Encoded DataFrame:")
print(encoded_cat_df)

 
encoded_cat_df = pd.DataFrame(encoded_cat_cols, columns=encoder.get_feature_names_out(cat_cols))


df = pd.concat([df[num_cols], encoded_cat_df], axis=1)


print(df.head())

# 4. Feature Engineering

print(df.head())
