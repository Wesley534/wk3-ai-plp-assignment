import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Configuration ---
st.set_page_config(page_title="Iris Species Predictor", layout="wide")

# --- Load Model and Encoder ---
@st.cache_resource
def load_model_components():
    """Loads the pickled model and label encoder with caching."""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, le
    except FileNotFoundError:
        st.error("Model files (model.pkl or label_encoder.pkl) not found. Please run 'python model.py' first.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None

model, le = load_model_components()
feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# --- Streamlit UI ---

st.title("🌺 Classical ML: Iris Species Prediction")
st.markdown("Use the Decision Tree Classifier to predict the species of an Iris flower based on its measurements.")

if model is not None and le is not None:
    # Sidebar for Input
    st.sidebar.header("Input Features")
    
    # Create input sliders for the four features
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.4, 0.1)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.4, 0.1)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.3, 0.1)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2, 0.1)

    # Compile the input data
    input_data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

    # --- Prediction Logic ---
    if st.sidebar.button('Predict Species'):
        # Make the prediction
        prediction_encoded = model.predict(input_data)
        
        # Decode the prediction
        prediction_label = le.inverse_transform(prediction_encoded)[0]

        st.subheader("Prediction Result")
        st.success(f"The predicted Iris species is: **{prediction_label}**")
        
        # Optional: Show the input data
        st.markdown("---")
        st.write("Input Data Used for Prediction:")
        input_df = pd.DataFrame(input_data, columns=feature_names)
        st.dataframe(input_df)

else:
    st.warning("Application startup failed. Please check the logs in the console/shell for errors.")