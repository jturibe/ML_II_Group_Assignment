import streamlit as st

# Set page config to set the layout and initial title
st.set_page_config(page_title="Group 1's Home App", layout="wide")

# Title and introduction of your team
st.title("Welcome to Group 1's Home Page")
st.write("""
         This application is the central hub for our machine learning project.
         Our group is composed by Omar Altarakieh, Alan Corrales, Jose Manuel Cuenca, Regina de Alba, Javier Torres, and Moritz von Ditfurth.
         Here's an overview of the problems we've tackled and the algorithms we've employed:
         """)

# Define the CSS style for the info boxes
info_box_style = """
    <style>
        .info-box {
            border-radius: 5px;
            background-color: #f0f0f0;
            color: black;
            padding: 15px;
            margin: 10px 0;
        }
    </style>
    """

# Function to create a section with a header and text inside a styled box
def create_section(header, content):
    st.markdown(info_box_style, unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><h2 style="color:black;margin-bottom:10px;">{header}</h2><p>{content}</p></div>', unsafe_allow_html=True)

# Zoo Animal Classification explanation
create_section("🐘 Zoo Animal Classification",
               """
               For the zoo animal classification problem, we employed a Random Forest algorithm.
               This ensemble method is well-suited for categorical data and capable of handling the 
               various features associated with animals. Its ability to perform both classification and 
               regression tasks makes it a versatile tool for predicting the categories of animals 
               based on their characteristics.
               """)

# Laptop Price Prediction explanation
create_section("💻 Laptop Price Prediction",
               """
               The challenge of predicting laptop prices was addressed using a Random Forest model.
               Given the heterogeneity of laptop features and the nonlinear relationships between them, 
               Random Forest's ensemble approach helps in capturing complex interactions and providing 
               robust predictions. Its interpretability and performance on a range of metrics made it 
               an excellent choice for this task.
               """)


# Rain Prediction in Bangladesh explanation
create_section("🌦️ Rain Prediction in Bangladesh",
               """
               For predicting rainfall in Bangladesh on a specific day, we chose the XGBoost algorithm.
               XGBoost stands out for its performance and speed in training, particularly with large and 
               complex datasets. As we learnt in class, XGBoost is a powerful model, especially for a task 
               like this.
               """)

# Used Cars Price Prediction explanation
create_section("🚙 Used Cars Price Prediction",
               """
               To predict the prices of used cars, we turned to Ridge Regression. This technique 
               is particularly effective when dealing with multicollinearity among features, a common 
               scenario in used car datasets. By imposing a penalty on the size of coefficients, 
               Ridge Regression improves the model's generalization ability, thereby preventing 
               overfitting.
               """)


# Add a footer note
st.markdown('---')
st.write("Created by Group 1 - Omar Altarakieh | Alan Corrales | Jose Manuel Cuenca | Regina de Alba | Javier Torres | Moritz von Ditfurth")
