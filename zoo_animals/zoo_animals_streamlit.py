import streamlit as st
import pandas as pd
import pickle

# Load your trained model
with open('zoo_animals/models/zoo_animals_rf_1.pkl', 'rb') as file:
	model = pickle.load(file)

# Mapping from numeric class to animal name
class_mapping = {
	1: 'Mammal 🐄',
	2: 'Bird 🦅',
	3: 'Reptile 🦎',
	4: 'Fish 🐠',
	5: 'Amphibian 🐸',
	6: 'Bug 🐞',
	7: 'Invertebrate 🐛'
}

st.title("Animal Classification App")

# Define the app
with st.form("user_input_form"):
	st.markdown("Please select any of the applicable characteristics, none of the options are mutually exclusive")
	
	with st.container():
		st.subheader('Physical appearance and basic traits')
		#Organize the toggles into two columns
		col1, col2 = st.columns(2)
		with col1:
			breathes = st.toggle('Does it breathe?', False)
			backbone = st.toggle('Does it have a backbone?', False)		
			toothed = st.toggle('Is it toothed?', False)
			hair = st.toggle('Does it have hair?', False)
			
		with col2:
			feathers = st.toggle('Does it have feathers?', False)
			fins = st.toggle('Does it have fins?', False)
			tail = st.toggle('Does it have a tail?', False)		
			catsize = st.toggle('Is it approximately the size of a cat?', False)
	
		legs = st.select_slider('How many legs does it have?', options=[0, 2, 4, 6, 8], value=0)  # Assuming legs can be from 0 to 8 and even-numbered
	
	st.markdown('---')
	
	with st.container():
		st.subheader('Reproductive capabilities')
		col3, col4 = st.columns(2)
		with col3:
			eggs = st.toggle('Does it lay eggs?', False)
		with col4:
			milk = st.toggle('Does it produce milk', False)

	st.markdown('---')
			
	with st.container():
		st.subheader("How does it live?")
		col5, col6 = st.columns(2)
		with col5:
			airborne = st.toggle('Is it airborne?', False)
		with col6:
			aquatic = st.toggle('Is it aquatic?', False)

	st.markdown('---')
			
	with st.container():
		st.subheader("Behavioral traits")
		col7, col8, col9 = st.columns(3)
		with col7:
			domestic = st.toggle('Is it domestic?', False)			
		with col8:
			predator = st.toggle('Is it a predator?', False)			
		with col9:
			venomous = st.toggle('Is it venomous?', False)

	submit_button = st.form_submit_button(label='Predict Animal Type')

# Button to make prediction
if submit_button:
	input_features = [hair, feathers, eggs, milk, airborne, aquatic, predator, toothed,
					  backbone, breathes, venomous, fins, legs, tail, domestic, catsize]
	input_data = pd.DataFrame([input_features], 
							  columns=['hair', 'feathers', 'eggs', 'milk', 'airborne', 
									   'aquatic', 'predator', 'toothed', 'backbone', 
									   'breathes', 'venomous', 'fins', 'legs', 'tail', 
									   'domestic', 'catsize'])
	prediction = model.predict(input_data)
	
	if prediction[0] == 1:
		message = "The animal is a MAMMAL 🐄"
		image_path = 'zoo_animals/mammal.jpg'
	elif prediction [0] == 2:
		message = "The animal is a BIRD 🦅"
		image_path = 'zoo_animals/bird.jpg'
	elif prediction [0] == 3:
		message = "The animal is a REPTILE 🦎"
		image_path = 'zoo_animals/reptile.jpg'
	elif prediction [0] == 4:
		message = 'The animal is a FISH 🐠'
		image_path = 'zoo_animals/fish.jpg'
	elif prediction [0] == 5:
		message = 'The animal is an AMPHIBIAN 🐸'
		image_path = 'zoo_animals/amphibian.jpg'
	elif prediction [0] == 6:
		message = 'The animal is a BUG 🐞'
		image_path = 'zoo_animals/bug.jpg'
	else:
		message = 'The animal is an INVERTEBRATE 🐛'
		image_path = 'zoo_animals/invertebrate.jpg'
		
	# Display the prediction message in a formatted box
	st.markdown(f"""
		<div style="
			color: black;
			border: 2px solid green;
			background-color: #D4EDDA;
			padding: 10px;
			border-radius: 5px;
			text-align: center;
			font-size: 2.0rem;
			margin: 10px 0;">
			{message}
		</div>
		""", unsafe_allow_html=True)

	# Display the image
	st.image(image_path)
