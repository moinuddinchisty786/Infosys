import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.mplot3d import Axes3D

# Load the model and scaler
def load_model():
    with open('streamlit_app/best_model.sav', 'rb') as file:
        loaded_model = pickle.load(file)
    model = loaded_model['model']  # Access the 'model' from the dictionary
    scaler = loaded_model['scaler']  # Access the 'scaler' from the dictionary
    metrics = loaded_model.get('metrics', None)  # Access metrics if saved
    return model, scaler, metrics

# Sidebar text input for features
def get_user_input():
    st.sidebar.title("Patient Feature Input")
    st.sidebar.write("Enter the values for the features below:")

    features = {}
    feature_names = ['texture_worst', 'compactness_se', 'concavity_worst', 'concave_points_mean', 'texture_mean', 'area_se', 'area_worst', 'perimeter_worst', 'radius_se', 'concave_points_worst', 'smoothness_worst', 'symmetry_worst', 'symmetry_se']
    
    for i, name in enumerate(feature_names, start=1):
        value = st.sidebar.text_input(f"{name}:", value="0.0")  # Default to 0.0
        try:
            features[f'feature_{i}'] = float(value)
        except ValueError:
            st.sidebar.warning(f"Please enter a valid number for {name}.")
            features[f'feature_{i}'] = 0.0

    input_data = np.array([list(features.values())])
    return input_data, features

# Display dataset information
def display_dataset_info():
    st.sidebar.title("Dataset Information")
    st.sidebar.write(""" 
    *Dataset*: Breast Cancer Wisconsin (Diagnostic) Dataset  
    *Source*: UCI Machine Learning Repository  
    *Total Records*: 569  
    *Features*: 30  
    *Classes*: Benign (Non-cancerous) and Malignant (Cancerous)  
    """)
    st.sidebar.markdown(""" 
    <style>
    .sidebar-info {
        background-color: #e0fbfc;
        padding: 10px;
        border-radius: 5px;
        font-family: Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Visualize dataset composition
def visualize_dataset_composition():
    st.markdown("### Trained Dataset Composition")
    benign_count = 357  # Example count
    malignant_count = 212  # Example count

    labels = ["Benign", "Malignant"]
    sizes = [benign_count, malignant_count]
    colors = ['#007f5f', '#d00000']
    explode = (0.1, 0)  # Explode the first slice for emphasis

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures a circular pie chart
    st.pyplot(fig)

# Medical-themed report display
def display_report(prediction, metrics):
    st.subheader("Prediction Report")
    st.markdown("""<style>
    .report-box {
        background-color: #f5f5f5;
        padding: 20px;
        border-left: 6px solid #0077b6;
        border-radius: 10px;
        font-family: 'Arial', sans-serif;
    }
    .result-positive {
        color: #007f5f;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .result-negative {
        color: #d00000;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>""", unsafe_allow_html=True)

    # Determine result
    result_class = "Benign (Non-cancerous)" if prediction[0] == 0 else "Malignant (Cancerous)"
    result_color = "result-positive" if prediction[0] == 0 else "result-negative"

    # Display the result
    st.markdown(f"""
    <div class="report-box">
        <h3 class="{result_color}">Prediction: {result_class}</h3>
        <p><strong>Accuracy:</strong> {metrics['accuracy'] * 100:.2f}%</p>
        <p><strong>Precision:</strong> {metrics['precision'] * 100:.2f}%</p>
        <p><strong>F1 Score:</strong> {metrics['f1_score'] * 100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

# Add a developer column or footer
# Add a stylish and creative developer section
# Developer information display
def display_developers():
    st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line
    st.markdown("### Developer Team")
    developer_info = [
        {"name": "Sayyed Nizar M", "email": "727723euai114@skcet.ac.in"},
        {"name": "Bala Sairam Goli", "email": "balasairamgoli4@gmail.com"},
        {"name": "Jayam V", "email": "jayamwcc@gmail.com"},
        {"name": "A.V.K Sai Surya", "email": "avksaisurya77@gmail.com"},
        {"name": "Shaik Moinuddin Chisty ", "email": "moinuddinchistyshaik@gmail.com"}
    ]
    
    for dev in developer_info:
        st.markdown(f"""
        <div style="margin-bottom: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 8px;">
            <h4 style="margin: 0; color: #023e8a;">{dev['name']}</h4>
            <p style="margin: 5px 0; color: #6c757d;">{dev['email']}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    model, scaler, metrics = load_model()

    st.markdown(""" 
    <style>
    body {
        background: linear-gradient(135deg, #edf2f4, #8ecae6);
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        font-size: 2.5rem;
        color: #023e8a;
        text-align: center;
        font-weight: bold;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #0077b6;
        text-align: center;
        margin-bottom: 20px;
    }
    footer {
        color: #6c757d;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">BREAST CANCER DETECTION USING ADABOOST CLASSIFIER PROJECT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Empowering early detection through advanced machine learning models.</p>', unsafe_allow_html=True)

    # Info about Breast Cancer and AdaBoost
    st.markdown(""" 
    ### About Breast Cancer
    Breast cancer is one of the most common types of cancer in women worldwide. Early detection and diagnosis are critical in improving survival rates. 
    """)
    st.image("streamlit_app/BCD.jpg", caption="Types of Breast Cancer Cells")

    st.markdown(""" 
    ### Why early detection of Breast Cancer is important?
    #### 💪Higher Survival Rates: 
    Early detection increases the chances of successful treatment and long-term survival.
    #### 🌿Less Aggressive Treatment:
    Timely diagnosis often allows for simpler and less invasive treatment options.
    #### 🌐Lower Risk of Metastasis: 
    Identifying cancer early reduces the likelihood of it spreading to other parts of the body.
    #### 💰Reduced Healthcare Costs: 
    Early treatment is typically more cost-effective than managing advanced-stage cancer.

    """)
    st.image("streamlit_app/rate.png", caption="Early detection saves lives.")

    st.markdown(""" 
    ### About AdaBoost Classifier
    AdaBoost (Adaptive Boosting) is a powerful ensemble learning algorithm that combines multiple weak learners to create a strong classifier, enhancing the accuracy of predictions.
    """)
    st.image("streamlit_app/model.png", caption="Early detection saves lives.")

    display_dataset_info()
    visualize_dataset_composition()

    input_array, features = get_user_input()

    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_array)

    if st.button('Predict', key="predict_button"):
        prediction = model.predict(scaled_input)
        display_report(prediction, metrics)  # Pass metrics to display the report

    display_developers()


if __name__ == "__main__":
    main()
