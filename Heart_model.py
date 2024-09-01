import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

if __name__ == "__main__":
    try:
        # Pass in environment variables and hyperparameters
        parser = argparse.ArgumentParser()
        parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
        parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

        args, _ = parser.parse_known_args()
        sm_model_dir = args.sm_model_dir
        training_dir = args.train

        # Read in data
        df = pd.read_csv(os.path.join(training_dir, "processed.cleveland.csv"))

        # Data Cleaning
        # Remove rows with missing values
        df_cleaned = df.dropna()
        
        # Split the data into features and target variable
        X = data_cleaned[['age', 'sex', 'cp', 'ca','exang', 'thal']]  
        # Target: 'target' column, which indicates presence or absence of heart disease
        y = data_cleaned['target']  

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Build and train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save the model
        joblib.dump(model, os.path.join(sm_model_dir, "model.joblib"))

        print("Model training and saving completed successfully.")

    except Exception as e:
        print(f"Error during training: {e}")
        raise
