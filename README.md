# MGSC 410 Capstone Final Project
### Flight Status Prediction Tool

## Overview
This project consists of a machine learning model and web application that predicts flight delays for US domestic flights. The tool uses historical flight data to provide probability estimates for flight delays, along with risk assessments and recommendations for travelers.

## Features
- Real-time flight delay predictions
- Interactive visualizations of historical delay patterns
- Model performance insights and metrics
- Comprehensive help documentation
- User-friendly web interface

## Model Details
The prediction model is a Random Forest Classifier trained on US flight data with the following characteristics:
- Balanced training using SMOTE (Synthetic Minority Over-sampling Technique)
- Handles multiple delay types (15+ minute delays, cancellations, diversions)
- Includes seasonal and time-based patterns
- Accounts for route-specific factors

## Project Structure
```
├── flight_delay_model.pkl    # Trained model and encoders
├── Combined_Flights_2022.csv # Flight status data
├── Airlines.csv              # Airline code to name mappings
├── airport_state.csv         # Airport to state mappings
├── flight_delays.py          # Model training script
└── app.py                    # Streamlit web application
```

## Installation & Setup
1. Install required packages:
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

2. Place required data files in the project directory:
- flight_delay_model.pkl
- Combined_Flights_2022.csv
- Airlines.csv
- airport_state.csv

3. Run the Streamlit app:
```bash
python -m streamlit run app.py
```

## Data Requirements
Download dataset:  https://drive.google.com/file/d/1DuDCvu3aS5MIxwCG3IQDerBGCP5UAa6E/view?usp=sharing 

The model requires the following input features:
- Flight date and time
- Origin and destination airports
- Airline information
- Flight number
- Distance information

## Model Performance
- On-time Flight Prediction:
  - Precision: 0.860
  - Recall: 0.730
  - F1-Score: 0.790

- Delay Prediction:
  - Precision: 0.420
  - Recall: 0.620
  - F1-Score: 0.500

## Training Process
The model training script (`model_training.py`) includes:
- Batch processing for large datasets
- SMOTE for handling class imbalance
- Feature engineering
- Label encoding for categorical variables
- Model evaluation and performance metrics
- Automated saving of model pkl file

## Web Application Features
1. Prediction Page
   - Input flight details
   - Get delay probability
   - Risk level assessment
   - Recommendations

2. Model Insights
   - Performance metrics
   - Feature importance visualization
   - Classification report

3. Historical Analysis
   - Monthly delay patterns
   - Time-of-day trends
   - Interactive visualizations

## Technical Notes
- The model uses label encoding for categorical variables
- Handles missing values through imputation
- Includes error handling for edge cases
- Provides confidence scores with predictions

## Acknowledgments
- Data source: https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022
- Contributors: Amanda Geronimo & Alice Utsunomiya
