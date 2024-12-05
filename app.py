import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, time
import numpy as np

def load_model_and_encoders():
    """Load the trained model and label encoders from single pickle file"""
    try:
        with open('flight_delay_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        encoders = model_data['encoders']
        
        # Print debug info
        print("Model feature names:", model_data['feature_names'])
        print("Model training date:", model_data['training_date'])
        
        return model, encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def process_input_features(input_data, encoders, model_features):
    """Process and encode input features"""
    try:
        # Convert categorical inputs using label encoders
        for col in ['Origin', 'Dest', 'OriginState', 'DestState', 
                    'DepTimeBlk', 'Marketing_Airline_Network']:
            input_data[col] = encoders[col].transform([str(input_data[col])])[0]
        
        # Calculate season from month
        season_map = {
            12:1, 1:1, 2:1,  # Winter
            3:2, 4:2, 5:2,   # Spring
            6:3, 7:3, 8:3,   # Summer
            9:4, 10:4, 11:4  # Fall
        }
        input_data['Season'] = season_map[input_data['Month']]
        
        # Create DataFrame and reorder columns to match training data
        df = pd.DataFrame([input_data])
        df = df[model_features]  # Reorder columns to match training data
        
        return df
    except Exception as e:
        st.error(f"Error processing features: {str(e)}")
        st.info(f"Input data keys: {input_data.keys()}")
        st.info(f"Required features: {model_features}")
        return None

def main():
    st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="wide")
    
    # Load model and encoders
    model, encoders = load_model_and_encoders()
    if model is None or encoders is None:
        return
    
    # Application title and description
    st.title("✈️ Flight Delay Prediction Tool")
    st.markdown("""
    This tool predicts the likelihood of flight delays based on various flight characteristics.
    A delay is defined as:
    - Arrival delay of 15 minutes or more
    - Flight cancellation
    - Flight diversion
    """)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flight Details")
        
        # Date and time inputs
        flight_date = st.date_input("Flight Date", min_value=datetime.now().date())
        departure_time = st.time_input("Departure Time", value=time(12, 0))
        
        # Convert time to minutes since midnight
        crs_dep_time = departure_time.hour * 100 + departure_time.minute
        
        # Time block selection
        time_blocks = [
            '0600-0659', '0700-0759', '0800-0859', '0900-0959',
            '1000-1059', '1100-1159', '1200-1259', '1300-1359',
            '1400-1459', '1500-1559', '1600-1659', '1700-1759',
            '1800-1859', '1900-1959', '2000-2059', '2100-2159'
        ]
        dep_time_blk = st.selectbox("Departure Time Block", time_blocks)
        
        # Airline selection
        airlines = sorted(encoders['Marketing_Airline_Network'].classes_)
        airline = st.selectbox("Airline", airlines)
        
        flight_number = st.number_input("Flight Number", min_value=1, max_value=9999, value=1000)

    with col2:
        st.subheader("Route Information")
        
        # Origin and destination inputs
        origins = sorted(encoders['Origin'].classes_)
        destinations = sorted(encoders['Dest'].classes_)
        
        origin = st.selectbox("Origin Airport", origins)
        dest = st.selectbox("Destination Airport", destinations)
        
        # State information
        origin_states = sorted(encoders['OriginState'].classes_)
        dest_states = sorted(encoders['DestState'].classes_)
        
        origin_state = st.selectbox("Origin State", origin_states)
        dest_state = st.selectbox("Destination State", dest_states)
        
        # Distance inputs
        distance = st.number_input("Flight Distance (miles)", min_value=1, max_value=5000, value=500)
        distance_group = st.selectbox("Distance Group", range(1, 12))

    # Create prediction button
    if st.button("Predict Delay Probability", type="primary"):
        try:
            # Prepare input data
            input_data = {
                'DayOfWeek': flight_date.weekday() + 1,
                'Month': flight_date.month,
                'DayOfMonth': flight_date.day,
                'CRSDepTime': crs_dep_time,
                'DepTimeBlk': dep_time_blk,
                'Distance': distance,
                'DistanceGroup': distance_group,
                'Marketing_Airline_Network': airline,
                'Flight_Number_Marketing_Airline': flight_number,
                'Origin': origin,
                'Dest': dest,
                'OriginState': origin_state,
                'DestState': dest_state,
                'Season': None  # Will be calculated in process_input_features
            }
            
            # Get model feature names
            model_features = model.feature_names_in_
            
            # Process and encode features with correct order
            input_df = process_input_features(input_data, encoders, model_features)
            if input_df is None:
                return
            
            # Make prediction
            prediction_prob = model.predict_proba(input_df)[0][1]
            
            # Display results
            st.header("Prediction Results")
            
            # Create columns for displaying results
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric("Delay Probability", f"{prediction_prob:.1%}")
            
            with res_col2:
                risk_level = "Low" if prediction_prob < 0.3 else "Medium" if prediction_prob < 0.6 else "High"
                risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
                st.markdown(f"Risk Level: :{risk_color}[**{risk_level}**]")
            
            with res_col3:
                recommendation = (
                    "✅ Flight likely to be on time"
                    if prediction_prob < 0.3
                    else "⚠️ Consider having a backup plan"
                    if prediction_prob < 0.6
                    else "🚫 High risk of delay - consider alternatives"
                )
                st.markdown(f"**Recommendation:** {recommendation}")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please check your input values and try again.")
            st.info(f"Debug info - Feature names: {model_features}")

if __name__ == "__main__":
    main()