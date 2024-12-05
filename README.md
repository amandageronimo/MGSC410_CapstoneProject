# MGSC410_CapstoneProject
### Flight Delay Prediction System

## Overview
This project implements a machine learning system to predict flight delays using historical flight data. The system consists of two main components:
1. A training script that processes historical flight data and trains a Random Forest model
2. A Streamlit web application that provides an interactive interface for making flight delay predictions

## Features
- Processes large datasets in chunks to handle memory efficiently
- Predicts the likelihood of flight delays, cancellations, and diversions
- Interactive web interface for real-time predictions
- Comprehensive feature engineering including:
  - Seasonal patterns
  - Time-based features
  - Geographic information
  - Carrier-specific patterns

## Prerequisites
- Python 3.8 or higher
- Required Python packages:
  ```
  pandas
  numpy
  scikit-learn
  streamlit
  seaborn
  matplotlib
  ```
  
## Usage

### Training the Model
1. Prepare your flight data CSV file
2. Update the file path in `train_model.py`:
   ```python
   file_path = "path/to/your/flight_data.csv"
   ```
3. Run the training script:
   ```bash
   python train_model.py
   ```
   - Use `sample_frac` parameter to train on a subset of data
   - Adjust `batch_size` based on your system's memory

### Running the Web Application
1. Ensure the trained model file (`flight_delay_model.pkl`) is in the same directory
2. Start the Streamlit app:
   ```bash
   python -m streamlit run app.py
   ```
3. Access the application through your web browser

## Model Features
The prediction model uses the following features:
- Day of week
- Month
- Time of day
- Flight distance
- Airlines
- Origin and destination airports
- State information
- Seasonal patterns

## Web Interface Features
- Input form for flight details
- Real-time delay probability prediction
- Risk level assessment
- Recommendations based on prediction
- Visual indicators for prediction confidence

## Model Performance
The Random Forest model is trained with the following configurations:
- 100 estimators
- Maximum depth of 20
- Balanced class weights
- Stratified sampling

## Data Requirements
The training data should include the following columns:
- FlightDate
- DayOfWeek
- Month
- CRSDepTime
- DepTimeBlk
- Distance
- DistanceGroup
- Marketing_Airline_Network
- Flight_Number_Marketing_Airline
- Origin
- Dest
- OriginState
- DestState
- Target variables (ArrDel15, Cancelled, Diverted)

## Error Handling
- The application includes comprehensive error handling for:
  - Missing model files
  - Invalid input data
  - Unseen categorical values
  - Runtime exceptions

## Acknowledgments
- Data source: https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022
- Contributors: Amanda Geronimo & Alice Utsunomiya
