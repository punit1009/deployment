import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report  # Add these imports
from PIL import Image

# Load models
lr = joblib.load('models/lr.pkl')
dt = joblib.load('models/tree.pkl')
knn = joblib.load('models/KNN.pkl')
rf = joblib.load('models/rf.pkl')

# Load pre-fitted scaler (you should have saved your scaler during training)
scaler = joblib.load('scaler/scaler.pkl')  # Correct relative path
  # Correct relative path
 # Make sure you've saved the scaler during training

df = pd.read_csv('kidney-stone-dataset.csv')  # Replace with actual data path
features = ['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']
target = 'target'  # Assuming 'target' is the label column (0 = No, 1 = Yes)

# Preprocess the dataset (scale the features)
X = df[features]
y = df[target]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load CSS and icons
def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def load_icon(icon_name):
    st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)

def load_images(file_name):
    img = Image.open(file_name)
    return st.image(img, width=300)

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Create a visually appealing confusion matrix using seaborn
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Kidney Stone', 'Kidney Stone'],
                yticklabels=['No Kidney Stone', 'Kidney Stone'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    return plt

# Function to handle predictions
def make_prediction(model, input_data_scaled):
    prediction = model.predict(input_data_scaled)
    


    if prediction == 0:
        return "No Kidney Stone", 'img/nok.jpg'
    else:
        return "Kidney Stone Detected", 'img/yesk.png'
def main():
    st.set_page_config(page_title="Kidney Stone Prediction", layout="wide")
    
    st.title("🩺 Kidney Stone Detection System")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a page", 
        ["Prediction", "Model Performance", "About"])
    
    if app_mode == "Prediction":
        prediction_page()
    elif app_mode == "Model Performance":
        performance_page()
    else:
        about_page()

def prediction_page():
    st.subheader("Predict Kidney Stone Risk")

    # Input fields for the features
    col1, col2 = st.columns(2)
    
    with col1:
        gravity_input = st.number_input("Gravity", min_value=1.005, max_value=1.030, value=1.015, step=0.001)
        ph_input = st.number_input("pH", min_value=4.5, max_value=8.0, value=6.0, step=0.1)
        osmo_input = st.number_input("Osmolarity", min_value=50, max_value=1200, value=500, step=10)
    
    with col2:
        cond_input = st.number_input("Conductivity", min_value=1, max_value=34, value=5, step=1)
        urea_input = st.number_input("Urea", min_value=70, max_value=210, value=100, step=5)
        calc_input = st.number_input("Calcium", min_value=5, max_value=20, value=10, step=1)

    # Model selection
    model_select = st.selectbox("Select a model", 
        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Random Forest"])

    # Prepare input data
    input_data = np.array([gravity_input, ph_input, osmo_input, cond_input, urea_input, calc_input]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)

    # Choose the selected model
    model_dict = {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "K-Nearest Neighbors": knn,
        "Random Forest": rf
    }
    model = model_dict[model_select]

    # Prediction button
    if st.button("Predict Risk"):
        # Prediction
        prediction, img_file = make_prediction(model, input_data_scaled)

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Prediction**: {prediction}")
        with col2:
            st.image(img_file, width=300)

def performance_page():
    st.subheader("Model Performance Metrics")
    
    # Model selection for performance metrics
    model_select = st.selectbox("Select a model for detailed metrics", 
        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Random Forest"])

    # Choose the selected model
    model_dict = {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "K-Nearest Neighbors": knn,
        "Random Forest": rf
    }
    model = model_dict[model_select]

    # Predict on test set
    y_pred = model.predict(X_test)

    # Metrics calculation
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precision", f"{precision:.2f}")
    with col2:
        st.metric("Recall", f"{recall:.2f}")
    with col3:
        st.metric("F1 Score", f"{f1:.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig = plot_confusion_matrix(y_test, y_pred, model_select)
    st.pyplot(fig)

    # Classification Report
    st.subheader("Detailed Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

def about_page():
    st.subheader("About Kidney Stone Detection")
    st.write("""
    ### 🩺 Kidney Stone Prediction System
    
    This application uses machine learning models to predict the risk of kidney stones 
    based on various physiological parameters.

    #### Features Used:
    - Gravity
    - pH Level
    - Osmolarity
    - Conductivity
    - Urea
    - Calcium
    """)

if __name__ == '__main__':
    main()