#import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def preprocess(file_path,split_value,target_column_index,has_header):
    """this function is used for taking the dataset from .csv file making a test train split
    and x, y split (has header is either True or False)"""

    #reading the .csv
    if has_header:
        df = pd.read_csv(file_path, header=0, delimiter=",")  # First row as header
        print(f"Dataset shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
    else:
        df = pd.read_csv(file_path, header=None, delimiter=",")  # No header row
        print(f"Dataset shape: {df.shape}")

    #making target variable split
    x_ds = df.drop(df.columns[target_column_index], axis=1)  # All except target
    y_ds = df.iloc[:, target_column_index]  # Target column only

    #making test train split
    x_train, x_test, y_train, y_test = train_test_split(x_ds, y_ds, test_size=split_value, random_state=10, shuffle= True)
    print(f'shape of x training: {x_train.shape}')
    print(f'shape of x testing: {x_test.shape}')

    return x_train, x_test, y_train, y_test, x_ds, y_ds

        
#making target variable numarical(optional)

def implement( ):
    """this function contains knn implementation (modeling and compileing) on the dataset with 5 
    initiations and shows the results of the implementations. (it needs to take target variable varieties )"""
    KNeighborsClassifier().fit(x_train, y_train)



#running the code
x_train, x_test, y_train, y_test, x_ds, y_ds = preprocess("diabetes_kmeans.csv", 0.2, 8, True)
implement()

