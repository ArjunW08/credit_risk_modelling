import logging
import os
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model_evaluation import train_xgboost_classifier, evaluate_classifier
import joblib
import pandas as pd

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="../../logs/train_xgboost.log",
    level=logging.DEBUG,
    force=True,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode="w",
)

def main():
    # Load the training data from the SQLite database
    conn = sqlite3.connect("../../data/credit_modelling.db")
    query = "SELECT * FROM credit_risk_load_data"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Separate features and target variable
    X = df.drop(columns=["Approved_Flag"])
    y = df["Approved_Flag"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    logging.info("Starting XGBoost training...")
    model = train_xgboost_classifier(X_train, y_train)
    logging.info("Training completed.")

    # Train and evaluate models
    random_search = train_xgboost_classifier(X_train, y_train)

    evaluate_classifier(
        random_search.best_estimator_,
        X_test,
        y_test,
        "XGBoost Classifier"
    )

    # Save the trained model to a file
    joblib.dump(model.best_estimator_, "../../models/predict_loan_possibility_model.pkl")
    logging.info("Model saved to models/predict_loan_possibility_model.pkl")


if __name__ == "__main__":
    main()