import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Load saved preprocessing objects and model
with open('used_cars/oh_encoder_2.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)
with open('used_cars/s_scaler_2.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('used_cars/models/used_cars_ridge_2.pkl', 'rb') as file:
    model = pickle.load(file)
with open('used_cars/column_order_2.pkl', 'rb') as file:
    column_order = pickle.load(file)
with open('used_cars/unique_brands.pkl', 'rb') as file:
    unique_brands = pickle.load(file)
	
sorted_brands = sorted(unique_brands)
emission_classes = ['EURO ' + str(i) for i in range(1, 8)]


# Function to preprocess user input
def preprocess_input(input_data, onehot_encoder, scaler, column_order):
    # Compute car age
    current_year = datetime.now().year
    input_data['Car Age'] = current_year - input_data['Registration Year']
    input_data['Emission Class'] = input_data['Emission Class'].str.replace('EURO ','').astype(float)
    
    # One-hot encode categorical variables
    categorical_features = ['Fuel type', 'Body type', 'Gearbox', 'Brand Grouped']
    encoded_features = onehot_encoder.transform(input_data[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out())
    
    # Scale numerical features
    numerical_features = ['Mileage(miles)', 'Previous Owners', 'Engine', 'Doors', 'Seats', 'Emission Class', 'Car Age']
    scaled_features = scaler.transform(input_data[numerical_features])
    scaled_df = pd.DataFrame(scaled_features, columns=numerical_features)
    
    # Combine all features
    processed_df = pd.concat([scaled_df, encoded_df], axis=1)
	
	# Align columns
    processed_df = processed_df.reindex(columns=column_order, fill_value=0)

    return processed_df

st.title('Used Car Price Predictor')

# Streamlit form to capture user inputs
col00, col01 = st.columns([5,2])
with col00:
	with st.form("user_input_form"):
		st.subheader("Registration Details")
		col1, col2, col3, col4 = st.columns(4)
		with col1:
			car_brand = st.selectbox('Car Brand', options=sorted_brands)
		with col2:
			mileage = st.number_input('Mileage', min_value=0, max_value=1000000, step=5000)
		with col3:
			registration_year = st.number_input('Registration Year', min_value=2000, max_value=2023)
		with col4:
			previous_owners = st.number_input('Previous Owners', min_value=1, step=1)

		st.markdown("---")

		st.subheader("Engine Details")
		col5, col6, col7 = st.columns(3)
		with col5:
			fuel_type = st.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other'])
		with col6:
			gearbox = st.selectbox('Gearbox', options=['Manual', 'Automatic', 'Semi-automatic', 'Other'])
		with col7:
			engine = st.number_input('Engine Size (L)', min_value=0.8, step=0.05)
		emission_class = st.selectbox('Emission Class', options=emission_classes)

		st.markdown("---")

		st.subheader("Bodywork Details")
		col8, col9, col10 = st.columns(3)
		with col8:
			body_type = st.selectbox('Body Type', options=['Sedan', 'Hatchback', 'SUV', 'Coupe', 'Convertible', 'Other'])

		with col9:
			seats = st.number_input('Seats', min_value=2, max_value=9, step=1)
		with col10:
			doors = st.number_input('Doors', min_value=2, max_value=5, step=1)

		submit_button = st.form_submit_button(label='Predict Price')

with col01:
	if submit_button:
		# Create a DataFrame from the user inputs
		input_df = pd.DataFrame([[car_brand, mileage, previous_owners, engine, doors, seats, emission_class, fuel_type, body_type, registration_year, gearbox]],
								columns=['Brand Grouped','Mileage(miles)', 'Previous Owners', 'Engine', 'Doors', 'Seats', 'Emission Class', 'Fuel type', 'Body type', 'Registration Year', 'Gearbox'])

		# Preprocess input data
		processed_input_df = preprocess_input(input_df, onehot_encoder, scaler, column_order)

		prediction = model.predict(processed_input_df)

		prediction_capped = np.maximum(0,prediction)
		
		st.markdown(f"""
			<style>
				.success-box {{
					background-color: #D4EDDA;
					color: #155724;
					padding: 20px;
					border-radius: 5px;
					border-left: 6px solid #28A745;
					margin-top: 250px;
				}}
				.header-text {{
					text-align: center;
					font-size: 1.25rem;
				}}
				.prediction-result {{
					text-align: center;
					font-size: 2.5rem; /* Larger font size for emphasis */
				}}
			</style>
			<div class="success-box">
				<div class="header-text">
					The predicted price of the laptop is:
				</div>
				<div class="prediction-result">
					${prediction_capped[0]:,.1f}
				</div>
			</div>
			""", unsafe_allow_html=True)