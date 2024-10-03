import pickle
import numpy as np
import streamlit as st

# Load the saved model and feature matrix
with open('linear.pkl', 'rb') as file:
    model = pickle.load(file)

with open('feature_matrix.pkl', 'rb') as file:
    X = pickle.load(file)

# Function to predict the price
def predict_price(location, sqft, bath, size):
    x = np.zeros(len(X.columns))
    x[0] = size
    x[1] = sqft
    x[2] = bath
    
    if location in X.columns:
        loc_index = X.columns.get_loc(location)
        x[loc_index] = 1
    else:
        return "Location not found."

    return model.predict([x])[0]

# Sidebar navigation for Contents
st.sidebar.title("Contents")
page = st.sidebar.radio("Go to", ["Project", "Model", "About"])

# Project Page
if page == "Project":
    st.title('House Price Prediction')

    st.write("""
    ## Welcome to the House Price Prediction App
    
    This application predicts house prices based on:
    - Location
    - Total Square Feet
    - Number of Bathrooms
    - Number of Bedrooms
    
    Enter the details below and get a price prediction.
    """)

    # Input fields for house price prediction
    location = st.selectbox('Location', X.columns[3:])
    sqft = st.number_input('Total Square Feet', min_value=300)
    bath = st.number_input('Number of Bathrooms', min_value=1)
    size = st.number_input('Number of Bedrooms (Size)', min_value=1)

    # Additional inputs (not used in prediction)
    lawn = st.checkbox('Lawn')
    garage = st.checkbox('Garage')
    year_built = st.number_input('Year Built', min_value=1900, max_value=2024, step=1)

    # Price prediction logic
    if st.button('Predict Price'):
        price = predict_price(location, sqft, bath, size)
        price_in_lacs = price
        st.write(f'Predicted Price: ₹{price_in_lacs:,.2f} Lacs')

# Model Explanation Page
elif page == "Model":
    st.title('Model Explanation: XGBoost')

    st.write("""
    ### XGBoost Regressor
    
    XGBoost (Extreme Gradient Boosting) is a powerful and scalable machine learning algorithm based on the gradient boosting framework. It has been optimized for speed and performance, making it highly effective for predictive modeling tasks.
    
    #### Why XGBoost?
    - **High Performance**: It’s known for delivering state-of-the-art results on structured datasets with high accuracy.
    - **Handles Outliers and Missing Data**: XGBoost can handle missing values and outliers more effectively compared to other models.
    - **Feature Importance**: Like Random Forest, XGBoost provides the ability to measure feature importance, helping to understand the influence of each feature on predictions.
    
    #### How It Works:
    1. XGBoost builds trees sequentially, and each new tree corrects errors made by the previous ones.
    2. It minimizes the loss function using gradient descent, making it more accurate with each iteration.
    3. The model tunes important parameters like learning rate, depth of trees, and number of boosting rounds for optimization.
    """)

# About Page
elif page == "About":
    st.title("About")

    st.write("""
    **Talha Azher**
    
    I am pursuing my degree in Computer Science and focus on ML, DL, Data Analysis, Web scraping, and Automation.
    
    ### Skills and Interests:
    - Data Science
    - Gen. AI
    - Web Scraping and Automation
    """)

# CSS styling for button hover animation
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
        transition-duration: 0.4s;
        cursor: pointer;
    }

    .stButton button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)
