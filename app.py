import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, time
import numpy as np

def load_model_and_encoders():
    """Load the trained model and label encoders from a single pickle file."""
    try:
        with open('flight_delay_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        encoders = model_data['encoders']
        return model, encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def process_input_features(input_data, encoders, model_features):
    """Process and encode input features."""
    try:
        # Convert categorical inputs using label encoders
        for col in ['Origin', 'Dest', 'OriginState', 'DestState', 
                    'DepTimeBlk', 'Marketing_Airline_Network']:
            input_data[col] = encoders[col].transform([str(input_data[col])])[0]
        
        # Calculate season from month
        season_map = {
            12: 1, 1: 1, 2: 1,  # Winter
            3: 2, 4: 2, 5: 2,   # Spring
            6: 3, 7: 3, 8: 3,   # Summer
            9: 4, 10: 4, 11: 4  # Fall
        }
        input_data['Season'] = season_map[input_data['Month']]
        
        # Create DataFrame and reorder columns to match training data
        df = pd.DataFrame([input_data])
        df = df[model_features]  # Reorder columns to match training data
        
        return df
    except Exception as e:
        st.error(f"Error processing features: {str(e)}")
        return None

def get_time_block(departure_time):
    """Calculate the departure time block based on the input time."""
    time_blocks = {
        '0600-0659': (6, 7),
        '0700-0759': (7, 8),
        '0800-0859': (8, 9),
        '0900-0959': (9, 10),
        '1000-1059': (10, 11),
        '1100-1159': (11, 12),
        '1200-1259': (12, 13),
        '1300-1359': (13, 14),
        '1400-1459': (14, 15),
        '1500-1559': (15, 16),
        '1600-1659': (16, 17),
        '1700-1759': (17, 18),
        '1800-1859': (18, 19),
        '1900-1959': (19, 20),
        '2000-2059': (20, 21),
        '2100-2159': (21, 22),
    }
    for block, (start_hour, end_hour) in time_blocks.items():
        if start_hour <= departure_time.hour < end_hour:
            return block
    return "Unknown Time Block"

def main():
    st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="wide")
    
    # Load model and encoders
    model, encoders = load_model_and_encoders()
    if model is None or encoders is None:
        return
    
    # Load airline mappings
    airlines_data = pd.read_csv('Airlines.csv')
    airline_mapping = airlines_data.set_index('Code')['Description'].to_dict()
    airport_data = pd.read_csv('airport_state.csv')
    airport_to_state = airport_data.set_index('Airport')['State'].to_dict()
    
    st.title("✈️ Flight Delay Prediction Tool")
    st.markdown("""
    This tool predicts the likelihood of flight delays based on various flight characteristics.
    A delay is defined as:
    - Arrival delay of 15 minutes or more
    - Flight cancellation
    - Flight diversion
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flight Details")
        flight_date = st.date_input("Flight Date", min_value=datetime.now().date())
        departure_time = st.time_input("Departure Time", value=time(12, 0))
        selected_time_block = get_time_block(departure_time)
        st.text_input("Departure Time Block", value=selected_time_block, disabled=True)
        
        airlines = sorted(encoders['Marketing_Airline_Network'].classes_)
        airline_dict = {code: airline_mapping.get(code, code) for code in airlines}
        airline_names = [airline_dict[code] for code in airlines]
        selected_airline_name = st.selectbox("Airline", airline_names)
        selected_airline_code = {v: k for k, v in airline_dict.items()}.get(selected_airline_name)
        
        flight_number = st.number_input("Flight Number", min_value=1, max_value=9999, value=1000)

    with col2:
        st.subheader("Route Information")
        origins = sorted(encoders['Origin'].classes_)
        destinations = sorted(encoders['Dest'].classes_)
        
        origin = st.selectbox("Origin Airport", origins)
        dest = st.selectbox("Destination Airport", destinations)
        
        origin_state = airport_to_state.get(origin, "Unknown")
        dest_state = airport_to_state.get(dest, "Unknown")
        st.text_input("Origin State", value=origin_state, disabled=True)
        st.text_input("Destination State", value=dest_state, disabled=True)

        distance = st.number_input("Flight Distance (miles)", min_value=1, max_value=5000, value=500)

        # Auto-calculate distance group
        distance_group = 1 + (distance // 250)
        st.text_input("Distance Group", value=distance_group, disabled=True)

    if st.button("Predict Delay Probability", type="primary"):
        try:
            input_data = {
                'DayOfWeek': flight_date.weekday() + 1,
                'Month': flight_date.month,
                'DayOfMonth': flight_date.day,
                'CRSDepTime': departure_time.hour * 100 + departure_time.minute,
                'DepTimeBlk': selected_time_block,
                'Distance': distance,
                'DistanceGroup': distance_group,
                'Marketing_Airline_Network': selected_airline_code,
                'Flight_Number_Marketing_Airline': flight_number,
                'Origin': origin,
                'Dest': dest,
                'OriginState': origin_state,
                'DestState': dest_state,
                'Season': None
            }
            
            model_features = model.feature_names_in_
            input_df = process_input_features(input_data, encoders, model_features)
            if input_df is None:
                return
            
            prediction_prob = model.predict_proba(input_df)[0][1]
            
            st.header("Prediction Results")
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

if __name__ == "__main__":
    main()
