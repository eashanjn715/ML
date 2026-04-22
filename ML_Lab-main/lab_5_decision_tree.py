import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.preprocessing import LabelEncoder

def build_classification_model(file_path, target_column):
    df = pd.read_csv(file_path)
    le = LabelEncoder()
    # Assume encoding for the sake of completion
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = DecisionTreeClassifier(criterion='entropy')
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    print(f"--- Results for {file_path} ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

try:
    build_classification_model('iris.csv', 'species')
    build_classification_model('drug.csv', 'Drug')
except FileNotFoundError as e:
    print(f"Error: {e}")

try:
    petrol_df = pd.read_csv('petrol_consumption.csv')
    X = petrol_df.drop('Petrol_Consumption', axis=1)
    y = petrol_df['Petrol_Consumption']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("--- Results for Petrol Consumption Regression ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

except FileNotFoundError:
    print("Error: petrol_consumption.csv not found.")
