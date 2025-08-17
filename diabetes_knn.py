#import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score


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

def implement(model):
    """this function contains knn implementation (modeling and fitting) on the dataset with 1 
    initiations and shows the results of the implementations. (it needs to take target variable varieties ).
    Note: This function doesn't have compile it is not appoprate to use this function for perceptron learning algs.
    because it doesn't have training vs test accuracy comparison and doesn't have epochs and .argmax(axis=1) """
    #initiante ml
    model.fit(x_train, y_train)
    #model evaluation
    y_pred = model.predict(x_test)

    print(f'Classification report: {classification_report(y_test, y_pred)}')
    print(f"Accuracy score is: {accuracy_score(y_test,y_pred, average='weighted')}")
    print(f"Percision score is: {precision_score(y_test,y_pred, average='weighted')}")
    print(f"Recall score is: {recall_score(y_test,y_pred)}")
    print(f'Confusion matrix: {confusion_matrix(y_test, y_pred)}')


#running the code
x_train, x_test, y_train, y_test, x_ds, y_ds = preprocess("diabetes_kmeans.csv", 0.2, 8, True)
implement(KNeighborsClassifier())

