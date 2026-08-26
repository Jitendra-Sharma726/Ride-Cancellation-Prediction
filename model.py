import pandas as pd

def load_data(filepath):
    """
    Loads the dataset from a specified CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(filepath)

def get_basic_info(df):
    """
    Provides basic information about the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary containing the shape and column names.
    """
    return {
        'shape': df.shape,
        'columns': df.columns.tolist()
    }

def calculate_cancellation_rate(df):
    """
    Calculates the overall cancellation rate.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        float: The cancellation rate (a value between 0 and 1).
    """
    total_rides = len(df)
    cancellation_statuses = ['Cancelled by Customer', 'Cancelled by Driver']
    cancelled_rides = df[df['Booking Status'].isin(cancellation_statuses)]
    cancelled_count = len(cancelled_rides)
    
    return cancelled_count / total_rides

def find_vehicle_with_most_cancellations(df):
    """
    Finds the vehicle type with the highest number of cancellations.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        str: The name of the vehicle type with the most cancellations.
    """
    cancellation_statuses = ['Cancelled by Customer', 'Cancelled by Driver']
    cancelled_df = df[df['Booking Status'].isin(cancellation_statuses)]
    
    # Get the most frequent vehicle type from the filtered data
    top_vehicle = cancelled_df['Vehicle Type'].mode()[0]
    return top_vehicle

def find_top_customer_cancellation_reason(df):
    """
    Finds the most common reason provided by customers for cancelling.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        str: The most frequent cancellation reason.
    """
    # .mode()[0] gives the most frequent value in the series
    top_reason = df['Reason for cancelling by Customer'].mode()[0]
    return top_reason

def find_peak_cancellation_hour(df):
    """
    Finds the hour of the day with the most cancellations.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        int: The hour (0-23) with the highest number of cancellations.
    """
    cancellation_statuses = ['Cancelled by Customer', 'Cancelled by Driver']
    cancelled_df = df[df['Booking Status'].isin(cancellation_statuses)].copy()

    # Extract the hour from the 'Time' column by splitting the string and taking the first part
    cancelled_df['hour'] = cancelled_df['Time'].str.split(':').str[0].astype(int)
    
    peak_hour = cancelled_df['hour'].mode()[0]
    return int(peak_hour)


if __name__ == "__main__":
    DATA_FILEPATH = 'uber_rides_2024.csv' # Make sure your CSV file has this name

    # Task 1: Load Data
    uber_df = load_data(DATA_FILEPATH)
    print("--- Task 1: Data Loading ---")
    print(f"Data loaded successfully. First 5 rows:")
    print(uber_df.head())
    print("\n" + "="*50 + "\n")

    # Task 2: Basic Information
    info = get_basic_info(uber_df)
    print("--- Task 2: Basic Dataset Information ---")
    print(f"Dataset Shape: {info['shape']}")
    print(f"Dataset Columns: {info['columns']}")
    print("\n" + "="*50 + "\n")

    # Task 3: Calculate Cancellation Rate
    rate = calculate_cancellation_rate(uber_df)
    print("--- Task 3: Overall Cancellation Rate ---")
    print(f"The overall cancellation rate is: {rate:.2%}")
    print("\n" + "="*50 + "\n")

    # Task 4: Find Vehicle with Most Cancellations
    top_vehicle = find_vehicle_with_most_cancellations(uber_df)
    print("--- Task 4: Vehicle with Most Cancellations ---")
    print(f"The vehicle type with the most cancellations is: '{top_vehicle}'")
    print("\n" + "="*50 + "\n")
    
    # Task 5: Find Top Customer Cancellation Reason
    top_reason = find_top_customer_cancellation_reason(uber_df)
    print("--- Task 5: Top Customer Cancellation Reason ---")
    print(f"The most common reason for customer cancellations is: '{top_reason}'")
    print("\n" + "="*50 + "\n")

    # Task 6: Find Peak Cancellation Hour
    peak_hour = find_peak_cancellation_hour(uber_df)
    print("--- Task 6: Peak Cancellation Hour ---")
    print(f"The hour with the most cancellations is: {peak_hour}:00 - {peak_hour+1}:00")
    print("\n" + "="*50 + "\n")
