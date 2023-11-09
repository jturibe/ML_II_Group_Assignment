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

st.markdown("""
    <style>
    .centered {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100vh;
    }
    </style>
    """, unsafe_allow_html=True)
	
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


st.title("Laptop Price Predictor")

col00,col01 = st.columns([5,2])
with col00:
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
			col1, col2= st.columns(2)
			with col1:
				standardized_brand = st.selectbox('Select the brand:', ['Alienware', 'Apple', 'ASUS', 'Dell', 'HP', 'Lenovo', 'LG', 'Microsoft', 'MSI', 'ROKC', 'Samsung', 'Other'])
				standardized_os = st.selectbox('Select the OS:', ['Chrome OS', 'Mac OS', 'Windows', 'Other'])
			with col2:
				screen_size = st.number_input('Screen size (in):', min_value=10.0, max_value=20.0, value=15.0, step=0.5)
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


with col01:		
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
		
		



# Assuming 'submit_button' and 'prediction' are already defined and available

		# Custom CSS to style the header and prediction result with a success box style
		st.markdown(f"""
			<style>
				.success-box {{
					background-color: #D4EDDA;
					color: #155724;
					padding: 20px;
					border-radius: 5px;
					border-left: 6px solid #28A745;
					margin-top: 300px;
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
					${prediction[0]:.2f}
				</div>
			</div>
			""", unsafe_allow_html=True)


		
