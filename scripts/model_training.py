"""
Starter script: model_training.py
Run after exploring dataset in notebook.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_data(path="data/loan_default_analytics_dataset.csv"):
    return pd.read_csv(path)

def build_and_train(df):
    X = df.drop(columns=['Defaulted','CustomerID'], errors='ignore')
    y = df['Defaulted']
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    clf = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification Report:\\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))

    joblib.dump(clf, "model.joblib")
    print("Saved trained model to model.joblib")

if __name__ == "__main__":
    df = load_data()
    build_and_train(df)
