import streamlit as st
import pandas as pd
import pickle

# Load your trained model
with open('models/zoo_animals_rf_1.pkl', 'rb') as file:
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
    st.markdown("Please select any of the applicable characteristics")

    # All possible characteristics as they were in the training data
    all_characteristics = ['Hair', 'Feathers', 'Eggs', 'Milk', 'Airborne', 'Aquatic',
                           'Predator', 'Toothed', 'Backbone', 'Breathes', 'Venomous',
                           'Fins', 'Tail', 'Domestic', 'Catsize']

    # Characteristics selection with multiselect
    selected_characteristics = st.multiselect('Characteristics', options=all_characteristics)

    # Slider for number of legs
    legs = st.slider('Number of Legs', min_value=0, max_value=8, value=0, step=2)

    submit_button = st.form_submit_button(label='Predict Animal Type')

# Button to make prediction
if submit_button:
    # Initialize all characteristics to False
    input_features_dict = {char.lower(): False for char in all_characteristics}

    # Set selected characteristics to True
    for char in selected_characteristics:
        input_features_dict[char.lower()] = True

    # Update the 'legs' value
    input_features_dict['legs'] = legs

    # Order the features correctly
    ordered_features = [input_features_dict[feature.lower()] for feature in all_characteristics] + [input_features_dict['legs']]
    
    # Create DataFrame for prediction
    input_data = pd.DataFrame([ordered_features], columns=[char.lower() for char in all_characteristics] + ['legs'])
    
    prediction = model.predict(input_data)
    st.write(f'The predicted class for the animal is: {class_mapping[prediction[0]]}')
