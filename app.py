import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Segregate the dependent and independent variables
X = data.drop(columns=['Outcome'])
Y = data['Outcome']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)

# Streamlit App Title
st.title("Diabetes Prediction App")
st.write("Use the sidebar to enter your details and click 'Predict' to see if you are Diabetic or Not:")

# Sidebar for User Input
st.sidebar.header("Enter Your Details")

# Collect all user inputs within a form
with st.sidebar.form("prediction_form"):
    input_data = []
    for feature in X.columns:
        value = st.number_input(f"Enter value for {feature}", step=1.0)
        input_data.append(value)
    
    # Add a submit button inside the form
    submitted = st.form_submit_button("Predict")

# Prediction Logic after form submission
if submitted:
    # Convert user input to DataFrame with correct feature names
    user_input_df = pd.DataFrame([input_data], columns=X.columns)
    
    # Scale the input data
    user_input_scaled = scaler.transform(user_input_df)
    
    # Make a prediction
    prediction = model.predict(user_input_scaled)
    
    # Output the prediction result with background color
    if prediction[0] == 0:
        st.markdown(
            """
            <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb;">
                <h4 style="color: #155724;">Based on given input, the system predicted that the patient might NOT have Diabetes.</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb;">
                <h4 style="color: #721c24;">Based on given input, the system predicted that the patient might HAVE Diabetes. Consult your Doctor/Physician immediately!</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Calculate and display accuracy after prediction
    Y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, Y_pred)
    st.write("### Model Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}")