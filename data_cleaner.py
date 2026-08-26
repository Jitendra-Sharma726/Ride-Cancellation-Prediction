import pandas as pd
import numpy as np

def create_target_variable(df):
    cancellation_statuses = ['Cancelled by Customer', 'Cancelled by Driver', 'No Driver Found']
    df['is_cancelled'] = np.where(df['Booking Status'].isin(cancellation_statuses), 1, 
                         np.where(df['Booking Status'] == 'Completed', 0, -1))
    return df[df['is_cancelled'] != -1].copy()

def feature_engineering(df):
    df_copy = df.copy()
    df_copy['hour_of_day'] = pd.to_datetime(df_copy['Time'], format='%H:%M:%S').dt.hour
    return df_copy

def drop_unnecessary_columns(df):
    columns_to_drop = [
        'Booking Status', 'Cancelled Rides by Customer', 'Reason for cancelling by Customer',
        'Cancelled Rides by Driver', 'Driver Cancellation Reason', 'Incomplete Rides', 
        'Incomplete Rides Reason', 'Ride Distance', 'Booking Value', 'Driver Ratings', 
        'Customer Rating', 'Booking ID', 'Customer ID', 'Date', 'Time'
    ]
    cols_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=cols_to_drop_existing)

def clean_and_save_data(input_filepath, output_filepath):
    print("Starting data cleaning process...")
    raw_df = pd.read_csv(input_filepath)
    print(f"Loaded {len(raw_df)} rows from {input_filepath}")

    raw_df.replace('null', np.nan, inplace=True)
    print("Replaced 'null' strings with standard NaN values.")

    df_with_target = create_target_variable(raw_df)
    print(f"Step 1: Created target variable. Rows remaining: {len(df_with_target)}")

    df_featured = feature_engineering(df_with_target)
    print("Step 2: Engineered 'hour_of_day' feature.")

    print("Step 3: Imputing missing values for 'Avg VTAT' and 'Avg CTAT'.")
    df_imputed = df_featured.copy()
    
    # --- THIS IS THE FIX ---
    # Reassign the columns instead of using inplace=True
    df_imputed['Avg VTAT'] = df_imputed['Avg VTAT'].fillna(0)
    df_imputed['Avg CTAT'] = df_imputed['Avg CTAT'].fillna(0)
    # --- END OF FIX ---
    
    df_imputed['Avg VTAT'] = pd.to_numeric(df_imputed['Avg VTAT'])
    df_imputed['Avg CTAT'] = pd.to_numeric(df_imputed['Avg CTAT'])
    print(f"Imputation complete. Rows remaining: {len(df_imputed)}")

    df_cleaned = drop_unnecessary_columns(df_imputed)
    print("Step 4: Dropped unnecessary columns.")
    print(f"Final columns: {df_cleaned.columns.tolist()}")

    df_cleaned.to_csv(output_filepath, index=False)
    print(f"Data cleaning complete. Cleaned data saved to {output_filepath}")


if __name__ == "__main__":
    RAW_DATA_FILE = 'uber_rides_2024.csv'
    CLEANED_DATA_FILE = 'clean_uber_rides_2024.csv'
    clean_and_save_data(RAW_DATA_FILE, CLEANED_DATA_FILE)
