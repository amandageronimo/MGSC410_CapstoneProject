# warnings
import warnings
warnings.filterwarnings('ignore')

# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# Define columns to use
use_cols = [
    # Time-Related
    'FlightDate', 'DayOfWeek', 'Month', 'CRSDepTime', 'DepTimeBlk',
    # Flight-Specific
    'Distance', 'DistanceGroup', 'Marketing_Airline_Network',
    'Flight_Number_Marketing_Airline',
    # Location
    'Origin', 'Dest', 'OriginState', 'DestState',
    # Target Variables
    'ArrDel15', 'Cancelled', 'Diverted'
]

def process_chunk(df_chunk):
    """Process each chunk of data"""
    # Data cleaning
    numerical_columns = ['Distance', 'CRSDepTime']
    categorical_columns = ['DayOfWeek', 'Month', 'DepTimeBlk', 'DistanceGroup',
                          'Marketing_Airline_Network', 'Flight_Number_Marketing_Airline',
                          'Origin', 'Dest', 'OriginState', 'DestState']
    
    # Imputation
    for col in numerical_columns:
        df_chunk[col].fillna(df_chunk[col].median(), inplace=True)
    
    for col in categorical_columns:
        df_chunk[col].fillna(df_chunk[col].mode()[0], inplace=True)
    
    df_chunk['FlightDate'].fillna(method='ffill', inplace=True)
    
    target_columns = ['ArrDel15', 'Cancelled', 'Diverted']
    for col in target_columns:
        df_chunk[col].fillna(0, inplace=True)
    
    # Create target variable
    df_chunk['is_delayed'] = ((df_chunk['ArrDel15'] == 1) |
                             (df_chunk['Cancelled'] == 1) |
                             (df_chunk['Diverted'] == 1)).astype(int)
    
    # Convert FlightDate and extract features
    df_chunk['FlightDate'] = pd.to_datetime(df_chunk['FlightDate'])
    df_chunk['DayOfMonth'] = df_chunk['FlightDate'].dt.day
    df_chunk['Season'] = df_chunk['FlightDate'].dt.month.map({
        12:1, 1:1, 2:1,  # Winter
        3:2, 4:2, 5:2,   # Spring
        6:3, 7:3, 8:3,   # Summer
        9:4, 10:4, 11:4  # Fall
    })
    
    return df_chunk

def main(file_path, batch_size=100000, sample_frac=None):
    """Main function to process data and train model"""
    print(f"Starting processing at {datetime.now().strftime('%H:%M:%S')}")
    
    # Initialize label encoders
    label_encoders = {}
    categorical_cols = ['Origin', 'Dest', 'OriginState', 'DestState',
                       'DepTimeBlk', 'Marketing_Airline_Network']
    
    # Initialize lists for batches
    X_batches = []
    y_batches = []
    
    # Process data in batches
    chunk_count = 0
    for chunk in pd.read_csv(file_path, usecols=use_cols, chunksize=batch_size):
        chunk_count += 1
        print(f"Processing chunk {chunk_count} at {datetime.now().strftime('%H:%M:%S')}")
        
        # Sample data if specified
        if sample_frac:
            chunk = chunk.sample(frac=sample_frac)
        
        # Process the chunk
        chunk = process_chunk(chunk)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
                chunk[col] = label_encoders[col].fit_transform(chunk[col].astype(str))
            else:
                # Handle unseen labels
                new_labels = set(chunk[col].astype(str)) - set(label_encoders[col].classes_)
                if new_labels:
                    label_encoders[col].classes_ = np.concatenate([
                        label_encoders[col].classes_,
                        list(new_labels)
                    ])
                chunk[col] = label_encoders[col].transform(chunk[col].astype(str))
        
        # Prepare features
        X_batch = chunk.drop(['is_delayed', 'FlightDate', 'ArrDel15', 
                            'Cancelled', 'Diverted'], axis=1)
        y_batch = chunk['is_delayed']
        
        X_batches.append(X_batch)
        y_batches.append(y_batch)
        
        # Free memory
        del chunk
    
    print("Combining batches...")
    X = pd.concat(X_batches)
    y = pd.concat(y_batches)
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Free memory
    del X, y, X_batches, y_batches
    
    # Create and train the model
    print("Training model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    print("Generating predictions...")
    y_pred = rf_model.predict(X_test)
    
    # Print model performance
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    # Save model and related data in a single pickle file
    print("Saving model and encoders...")
    model_data = {
        'model': rf_model,
        'encoders': label_encoders,
        'feature_names': X_train.columns.tolist(),
        'feature_importance': feature_importance,
        'training_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'model_performance': {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        },
        'model_parameters': rf_model.get_params()
    }
    
    # Save as a single pickle file
    with open('flight_delay_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved as 'flight_delay_model.pkl'")
    
    return rf_model, label_encoders, feature_importance

if __name__ == "__main__":
    # File path
    file_path = "C:\\Users\\amand\\MGSC410\\Combined_Flights_2022.csv"
    
    # For testing with smaller sample (uncomment one of these options):
    model, encoders, importance = main(file_path, batch_size=100000, sample_frac=0.1)  # 10% of data
    # model, encoders, importance = main(file_path, batch_size=100000, sample_frac=0.5)  # 50% of data
    
    # For full training:
    # model, encoders, importance = main(file_path, batch_size=100000)
    
    # Verify saved files
    import os
    print([f for f in os.listdir() if f.endswith('.pkl')])