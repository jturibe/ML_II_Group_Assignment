import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Load saved preprocessing objects and model
with open('weather_bangladesh/oh_encoder_weather_1.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)
with open('weather_bangladesh/scaler_weather_1.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('weather_bangladesh/models/xgb_weather_1.pkl', 'rb') as file:
    model = pickle.load(file)
with open('weather_bangladesh/col_order_weather_1.pkl', 'rb') as file:
    column_order = pickle.load(file)

compass_points = [
    'N', 'NNE', 'NE', 'ENE', 
    'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW',
    'W', 'WNW', 'NW', 'NNW'
]

# Function to preprocess user input
def preprocess_input(input_data, onehot_encoder, scaler, column_order):
    
    # Extract season
    input_data['Month'] = pd.to_datetime(input_data['Date']).dt.month

    # Map the month to the corresponding season in Bangladesh
    season_mapping = {1: 'Winter', 2: 'Spring', 3: 'Spring', 4: 'Summer', 5: 'Summer',
                      6: 'Monsoon', 7: 'Monsoon', 8: 'Autumn', 9: 'Autumn', 
                      10: 'Late Autumn', 11: 'Late Autumn', 12: 'Winter'}
    
    input_data['Season'] = input_data['Month'].map(season_mapping)
    
    # Since we have extracted the season, we can drop the 'Date' and 'Month' columns
    input_data.drop(['Date', 'Month'], axis=1, inplace=True)
    
    # One-hot encode categorical variables
    categorical_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'Season']
    encoded_features = onehot_encoder.transform(input_data[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out())

    # Drop the original categorical variables as they are now encoded
    input_data.drop(categorical_features, axis=1, inplace=True)
    
    # Combine the numerical features and the encoded categorical features
    combined_df = pd.concat([input_data, encoded_df], axis=1)
    
    # Align columns to match the training data
    processed_df = combined_df.reindex(columns=column_order)
	
    # Scale the combined DataFrame
    processed_df = scaler.transform(processed_df)
    processed_df = pd.DataFrame(processed_df, columns=column_order)
    
    return processed_df

# Streamlit form to capture user inputs
with st.form("user_input_form"):
    
    st.subheader("Weather Details")

    # Numerical Inputs
    date = st.date_input('Please input the date you want to check:', datetime(2023, 11, 7).date())
    avgtemp = st.number_input('Average Daily Temperature (°C)', value=25.0, step=0.5)
    evaporation = st.number_input('Evaporation during the day (millimeters)', value=25.0)
    sunshine = st.number_input('Hours of bright sunshine during the day', min_value=0.0, max_value=24.0, step=0.5)
    windgustspeed = st.number_input('Speed of strongest gust during the day (km per hour)', value=25.0)
    windspeed9am = st.number_input('Speed of the wind for the 10 minutes prior to 9 AM (km per hour)', value=25.0)
    windspeed3pm = st.number_input('Speed of the wind for the 10 minutes prior to 3 PM (km per hour)', value=25.0)
    humidity9am = st.number_input('Humidity of the wind at 9 AM (%)', value=25.0)
    humidity3pm = st.number_input('Humidity of the wind at 3 PM (%)', value=25.0)
    avgpressure = st.number_input('Average atmospheric pressure (hectopascals)', value=1000, step=1)
    cloud9am = st.number_input('Cloud-obscured portions of the sky at 9 AM (eighths)', min_value=0, max_value=8, step=1)
    cloud3pm = st.number_input('Cloud-obscured portions of the sky at 3 PM (eighths)', min_value=0, max_value=8, step=1)

    # Categorical Inputs
    WindGusDir = st.selectbox('Strongest Gust Direction', options=compass_points)
    WindDir9am = st.selectbox('Wind Direction at 9am', options=compass_points)
    WindDir3pm = st.selectbox('Wind Direction at 3pm', options=compass_points)

    submit_button = st.form_submit_button(label='Predict Rainfall')

if submit_button:
    # Create a DataFrame from the user inputs
    input_df = pd.DataFrame([[date, avgtemp, evaporation, sunshine, windgustspeed, windspeed9am, windspeed3pm, humidity9am, humidity3pm, avgpressure, cloud9am, cloud3pm, WindGusDir, WindDir9am, WindDir3pm]],
                            columns=['Date', 'AvgTemp', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'AvgPressure', 'Cloud9am', 'Cloud3pm', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
    
    # Preprocess input data
    processed_input_df = preprocess_input(input_df, onehot_encoder, scaler, column_order)
	
    # Predict whether it will rain or not
    prediction = model.predict(processed_input_df)
    
    # Output the prediction
    prediction_label = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f"The prediction for rain today is: {prediction_label}")
