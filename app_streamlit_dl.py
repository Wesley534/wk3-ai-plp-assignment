import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Install Streamlit-drawable-canvas to let the user draw
# NOTE: You must have 'streamlit-drawable-canvas' in your requirements.txt
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Missing dependency: Run 'pip install streamlit-drawable-canvas' and restart.")
    st.stop()


# --- Configuration ---
st.set_page_config(page_title="MNIST CNN Predictor", layout="centered")

# --- Load Model ---
# Use st.cache_resource for heavy objects like Keras models
@st.cache_resource
def load_cnn_model():
    """Loads the saved Keras model."""
    model_filename = 'mnist_cnn_model.h5'
    try:
        model = tf.keras.models.load_model(model_filename)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure '{model_filename}' is present and compatible.")
        return None

model = load_cnn_model()

# --- Preprocessing Function ---
def preprocess_image(image_data):
    """Converts the drawing data into a model-ready NumPy array."""
    # Convert image data (RGBA) to grayscale (L)
    img = Image.frombytes("RGBA", (280, 280), image_data, 'raw', 'RGBA').convert('L')
    
    # Resize to 28x28 pixels
    img = img.resize((28, 28))
    
    # Convert to NumPy array
    img_array = np.array(img).astype('float32')
    
    # Invert colors (MNIST is white on black, drawing is black on white)
    img_array = 255 - img_array
    
    # Normalize and reshape: (1, 28, 28, 1)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

# --- Streamlit UI ---

st.title("✍️ CNN Deep Learning: MNIST Digit Classifier")
st.markdown("Draw a single digit (0-9) in the box below to get a real-time prediction from the trained CNN model.")

if model is not None:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="white",  # Background color
        stroke_width=15,     # Line thickness
        stroke_color="black",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Check if the user has drawn anything (i.e., if the canvas data is not all transparent/white)
        # Note: canvas_result.image_data is RGBA. The alpha channel is usually enough for a check.
        if np.any(canvas_result.image_data[:, :, 3] > 0):
            
            # --- Preprocessing and Prediction ---
            preprocessed_img = preprocess_image(canvas_result.image_data)

            # Make prediction
            predictions = model.predict(preprocessed_img)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100

            # --- Display Results ---
            st.subheader("Prediction")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(preprocessed_img.reshape(28, 28), caption="Input to CNN (28x28)", width=100)
            
            with col2:
                st.success(f"Predicted Digit: **{predicted_class}**")
                st.info(f"Confidence: **{confidence:.2f}%**")
                
            st.markdown("---")
            st.subheader("Confidence Score Breakdown")
            
            # Create a simple DataFrame for bar chart
            scores = pd.DataFrame({
                'Digit': np.arange(10),
                'Probability': predictions[0]
            })
            st.bar_chart(scores.set_index('Digit'))
        else:
             st.info("Start drawing a digit above!")
    
else:
    st.error("CNN Model failed to load. Please check the console/shell for errors and ensure you ran `python model_cnn.py`.")