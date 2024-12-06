import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, time
import numpy as np
import plotly.express as px

def load_model_and_encoders():
    """Load the trained model and label encoders from a single pickle file."""
    try:
        with open('flight_delay_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        encoders = model_data['encoders']
        return model, encoders, model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

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
    return "0000-0559"  # Default time block for hours not covered above

def show_model_insights():
    """Display model performance metrics and feature importance."""
    model, encoders, model_data = load_model_and_encoders()
    if model_data is None:
        return
    
    st.subheader("Model Performance Insights")
    
    try:
        # Display classification report in a more readable format
        report = model_data['model_performance']['classification_report']
        
        # Create metrics dictionary with default values
        metrics = {
            'On-time': {
                'Precision': 0.0,
                'Recall': 0.0,
                'F1-Score': 0.0,
                'Support': 0
            },
            'Delayed': {
                'Precision': 0.0,
                'Recall': 0.0,
                'F1-Score': 0.0,
                'Support': 0
            }
        }
        
        # Parse the report string carefully
        lines = report.split('\n')
        for line in lines:
            if not line.strip():
                continue
            if any(x in line for x in ['accuracy', 'macro', 'weighted', 'avg']):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:  # Ensure we have all needed parts
                try:
                    class_name = 'On-time' if parts[0] == '0' else 'Delayed'
                    metrics[class_name] = {
                        'Precision': float(parts[1]),
                        'Recall': float(parts[2]),
                        'F1-Score': float(parts[3]),
                        'Support': int(parts[4])
                    }
                except (ValueError, IndexError):
                    continue
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        
        st.write("Model Performance Metrics:")
        st.dataframe(metrics_df.style.format({
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}',
            'Support': '{:,.0f}'
        }))
        
        # Feature importance plot
        st.write("Feature Importance:")
        if 'feature_importance' in model_data:
            importance_df = model_data['feature_importance'].head(10)
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Most Important Features'
            )
            st.plotly_chart(fig)
        else:
            st.warning("Feature importance data not available in the model file.")
            
    except Exception as e:
        st.error(f"Error processing model insights: {str(e)}")
        st.write("Please ensure the model file contains valid performance metrics.")

def show_historical_analysis():
    """Display historical delay patterns."""
    st.title("Historical Flight Analysis")
    
    try:
        st.subheader("Delay Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly delay patterns
            monthly_delays = pd.DataFrame({
                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'Delay_Rate': [0.25, 0.28, 0.22, 0.20, 0.18, 0.30, 
                              0.35, 0.32, 0.21, 0.19, 0.23, 0.29]
            })
            fig = px.line(monthly_delays, x='Month', y='Delay_Rate',
                         title='Monthly Delay Patterns')
            fig.update_layout(yaxis_title='Delay Rate', 
                             yaxis_tickformat='.0%')
            st.plotly_chart(fig)
        
        with col2:
            # Time block analysis
            time_blocks = [
                '0600-0659', '0700-0759', '0800-0859', '0900-0959',
                '1000-1059', '1100-1159', '1200-1259', '1300-1359',
                '1400-1459', '1500-1559', '1600-1659', '1700-1759',
                '1800-1859', '1900-1959', '2000-2059', '2100-2159'
            ]
            delay_rates = [
                0.18, 0.22, 0.25, 0.23, 0.20, 0.24, 0.28, 0.30,
                0.32, 0.35, 0.38, 0.36, 0.33, 0.30, 0.25, 0.22
            ]
            time_delays = pd.DataFrame({
                'Time Block': time_blocks,
                'Delay_Rate': delay_rates
            })
            fig = px.line(time_delays, x='Time Block', y='Delay_Rate',
                         title='Delay Patterns by Time Block')
            fig.update_layout(xaxis_tickangle=45,
                             yaxis_title='Delay Rate',
                             yaxis_tickformat='.0%')
            st.plotly_chart(fig)
    
    except Exception as e:
        st.error(f"Error displaying historical analysis: {str(e)}")

def show_help():
    """Display help and documentation."""
    st.title("Help & Documentation")
    
    st.markdown("""
    ## How to Use This Tool
    
    ### Prediction Page
    - Enter flight details including date, time, route, and airline
    - Get instant predictions with confidence intervals
    - Review risk levels and recommendations
    
    ### Model Insights
    - View model performance metrics
    - Explore feature importance
    - Understand what factors contribute most to delays
    
    ### Historical Analysis
    - Examine monthly delay patterns
    - Review time-of-day delay trends
    - Use patterns to plan optimal flight times
    
    ### Understanding the Results
    - **Probability**: Likelihood of delay (0-100%)
    - **Risk Level**: 
        - Low: <30% chance of delay
        - Medium: 30-60% chance of delay
        - High: >60% chance of delay
    - **Recommendations**: Suggested actions based on risk level
    
    ### Tips for Best Results
    1. Provide accurate flight information
    2. Consider seasonal patterns in your planning
    3. Use historical analysis to identify optimal travel times
    """)

def show_prediction_page():
    """Display the main prediction page."""
    model, encoders, model_data = load_model_and_encoders()
    if model is None or encoders is None:
        return
    
    st.title("✈️ Flight Delay Prediction Tool")
    
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        This tool predicts the likelihood of flight delays based on various flight characteristics.
        A delay is defined as:
        - Arrival delay of 15 minutes or more
        - Flight cancellation
        - Flight diversion
        """)
    
    try:
        # Load airline mappings
        airlines_data = pd.read_csv('Airlines.csv')
        airline_mapping = airlines_data.set_index('Code')['Description'].to_dict()
        airport_data = pd.read_csv('airport_state.csv')
        airport_to_state = airport_data.set_index('Airport')['State'].to_dict()
        
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
                    'Season': None  # Will be calculated in process_input_features
                }
                
                model_features = model.feature_names_in_
                input_df = process_input_features(input_data, encoders, model_features)
                if input_df is None:
                    return
                
                prediction_prob = model.predict_proba(input_df)[0][1]
                
                st.header("Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Delay Probability", f"{prediction_prob:.1%}")
                
                with col2:
                    risk_level = "Low" if prediction_prob < 0.3 else "Medium" if prediction_prob < 0.6 else "High"
                    risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
                    st.markdown(f"Risk Level: :{risk_color}[**{risk_level}**]")
                
                with col3:
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
                st.write("Please check your input values and try again.")
    
    except Exception as e:
        st.error(f"Error loading required data: {str(e)}")
        st.write("Please ensure all required files (Airlines.csv and airport_state.csv) are present.")

def main():
    st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="wide")
    
    # Add sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Model Insights", "Historical Analysis", "Help"])
    
    if page == "Prediction":
        show_prediction_page()
    elif page == "Model Insights":
        show_model_insights()
    elif page == "Historical Analysis":
        show_historical_analysis()
    else:
        show_help()

if __name__ == "__main__":
    main()
