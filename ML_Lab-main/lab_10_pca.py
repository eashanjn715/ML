import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

try:
    df = pd.read_csv('heart.csv')
    categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    def evaluate_models(X_tr, X_te, y_tr, y_te):
        models = {
            "SVM": SVC(),
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(random_state=42)
        }
        accuracies = {}
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            accuracies[name] = accuracy_score(y_te, y_pred)
        return accuracies
        
    results_initial = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    print("--- Accuracies Without PCA ---")
    for model, acc in results_initial.items():
        print(f"{model}: {acc:.4f}")
        
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"\nOriginal dimensions: {X_train_scaled.shape[1]}")
    print(f"Dimensions after PCA: {X_train_pca.shape[1]}")
    
    results_pca = evaluate_models(X_train_pca, X_test_pca, y_train, y_test)
    print("\n--- Accuracies With PCA ---")
    for model, acc in results_pca.items():
        print(f"{model}: {acc:.4f}")

except FileNotFoundError:
    pass
