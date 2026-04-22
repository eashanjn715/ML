import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelEncoder

try:
    df = pd.read_csv('iris.csv')
    X = df.drop('species', axis=1)
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    y_pred_linear = svm_linear.predict(X_test)
    print("--- Linear Kernel Results ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred_linear)}")
    
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    y_pred_rbf = svm_rbf.predict(X_test)
    print("\n--- RBF Kernel Results ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred_rbf)}")
except FileNotFoundError:
    pass
