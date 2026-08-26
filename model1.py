import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def prepare_data(df):
    """
    Prepares the data for modeling by separating features and target,
    and applying one-hot encoding to categorical features.
    """
    y = df['is_cancelled']
    X = df.drop(columns=['is_cancelled'])
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded, y

def split_data(X, y):
    """
    Splits the data into training and testing sets using stratification.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_model(X_train, y_train):
    """
    Trains a Decision Tree Classifier model.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance on the test set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def create_prediction_pipeline(filepath):
    """
    Orchestrates the full pipeline from loading data to returning a trained model.
    """
    print("Starting model training pipeline...")
    
    cleaned_df = pd.read_csv(filepath)
    print(f"Loaded {len(cleaned_df)} rows of cleaned data.")

    X, y = prepare_data(cleaned_df)
    print("Data prepared with one-hot encoding.")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")
    
    if y_train.nunique() < 2:
        print("Error: Training data does not contain both classes after splitting.")
        return None, None

    model = train_model(X_train, y_train)
    print("Model training complete.")

    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model evaluation complete. Accuracy: {accuracy:.4f}")
    
    return model, X.columns.tolist()

if __name__ == "__main__":
    CLEANED_DATA_FILE = 'clean_uber_rides_2024.csv'
    
    trained_model, model_columns = create_prediction_pipeline(CLEANED_DATA_FILE)
    
    if trained_model and model_columns:
        print("\n--- Prediction Example ---")
        sample_ride = {
            'Vehicle Type': 'Go Sedan',
            'Pickup Location': 'Khandsa',
            'Drop Location': 'Malviya Nagar',
            'Avg VTAT': 13.4,
            'Avg CTAT': 25.8,
            'Payment Method': 'UPI',
            'hour_of_day': 8
        }

        sample_df = pd.DataFrame([sample_ride])
        sample_encoded = pd.get_dummies(sample_df)
        
        sample_aligned = sample_encoded.reindex(columns=model_columns, fill_value=0)
        
        prediction = trained_model.predict(sample_aligned)
        prediction_proba = trained_model.predict_proba(sample_aligned)

        result = "Cancelled" if prediction[0] == 1 else "Completed"
        print(f"Sample ride data: {sample_ride}")
        print(f"Predicted outcome: {result}")
        print(f"Prediction probability (0=Completed, 1=Cancelled): {prediction_proba[0]}")
    else:
        print("Pipeline failed to return a trained model.")
