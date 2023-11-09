import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load trained model, encoder, and scaler
with open('laptop_prices/models/rf_laptop_1.pkl', 'rb') as model_file:
	model = pickle.load(model_file)
with open('laptop_prices/oh_encoder_laptop_1.pkl', 'rb') as encoder_file:
	encoder = pickle.load(encoder_file)
with open('laptop_prices/s_scaler_laptop_1.pkl', 'rb') as scaler_file:
	scaler = pickle.load(scaler_file)
with open('laptop_prices/col_order_encoder_1.pkl','rb') as col_order_e:
	col_order_encoder = pickle.load(col_order_e)
with open('laptop_prices/col_order_scaler_1.pkl','rb') as col_order_s:
	col_order_scaler = pickle.load(col_order_s)

# Function to preprocess inputs
def preprocess_inputs(user_inputs, encoder, scaler, col_order_encoder, col_order_scaler):
	# Isolate categorical and numerical features from user_inputs based on the column orders
	categorical_features = {col: user_inputs[col] for col in col_order_encoder if col in user_inputs}
	numerical_features = {col: user_inputs[col] for col in col_order_scaler if col in user_inputs}

	# Convert dictionaries to DataFrames
	categorical_features_df = pd.DataFrame([categorical_features])
	numerical_features_df = pd.DataFrame([numerical_features])

	# Ensure the DataFrames have the same column order as during training
	categorical_features_df = categorical_features_df[col_order_encoder]
	numerical_features_df = numerical_features_df[col_order_scaler]

	# Encode categorical features using the encoder
	categorical_features_encoded = encoder.transform(categorical_features_df)

	# Scale numerical features using the scaler
	numerical_features_scaled = scaler.transform(numerical_features_df)

	# Combine categorical and numerical features into one array
	features_combined = np.hstack([categorical_features_encoded, numerical_features_scaled])

	return features_combined


st.title("Laptop Price Prediction")

with st.form("user_input_form"):
	# Mapping of user-friendly full names to model-expected abbreviated names
	cpu_mapping = {
		'Intel Core i3': 'core i3',
		'Intel Core i5': 'core i5',
		'Intel Core i7': 'core i7',
		'Intel Core i9': 'core i9',
		'Pentium': 'pentium',
		'Celeron': 'celeron',
		'AMD Ryzen 8': 'ryzen 8',
		'Other': 'other'
	}

	with st.container():
		st.subheader('Basic Specs')
		col1, col2, col3, col4 = st.columns(4)
		with col1:
			standardized_brand = st.selectbox('Select the brand:', ['Alienware', 'Apple', 'ASUS', 'Dell', 'HP', 'Lenovo', 'LG', 'Microsoft', 'MSI', 'ROKC', 'Samsung', 'Other'])
		with col2:
			standardized_os = st.selectbox('Select the OS:', ['Chrome OS', 'Mac OS', 'Windows', 'Other'])
		with col3:
			screen_size = st.number_input('Screen size (inches):', min_value=10.0, max_value=20.0, value=15.0, step=0.5)
		with col4:
			general_color = st.selectbox('Select the color:', ['Black', 'Gray/Silver', 'Other', 'Blue', 'White', 'Red'])
	st.markdown('---')
	with st.container():
		st.subheader('Technical Specs')
		col5, col6 = st.columns(2)
		with col5:
			standardized_cpu = st.selectbox('Select the CPU type:', options=list(cpu_mapping.keys()))
			cpu_speed = st.number_input('Enter the CPU speed in GHz:', min_value=1.0, max_value=5.0, value=2.5, step=0.1)
		with col6:
			graphics_category = st.selectbox('Select the graphics card type:', ['Integrated', 'Dedicated'])
			harddisk = st.number_input('Enter the hard disk size in GB:', min_value=128, max_value=2048, value=512, step=128)
	st.markdown('---')
	st.subheader('Customer Opinion')
	rating = st.slider('Enter the customer rating:', min_value=1.0, max_value=5.0, value=2.5, step=0.25)

	# Create a dictionary to hold the inputs
	user_inputs = {
		'general_color': general_color,
		'standardized_cpu': cpu_mapping[standardized_cpu],
		'standardized_os': standardized_os,
		'graphics_category': graphics_category,
		'standardized_brand': standardized_brand,
		'screen_size': screen_size,
		'harddisk': harddisk,
		'cpu_speed': cpu_speed,
		'rating': rating
	}
	
	submit_button = st.form_submit_button(label='Predict Price')

# Button to predict
if submit_button:
		# Preprocess inputs
	# Example usage:
	preprocessed_input = preprocess_inputs(
		user_inputs=user_inputs,
		encoder=encoder,
		scaler=scaler,
		col_order_encoder=col_order_encoder,
		col_order_scaler=col_order_scaler
	)

	# Make prediction
	prediction = model.predict(preprocessed_input)

	# Display prediction
	st.write(f'The predicted price of the laptop is: ${prediction[0]:.2f}')
