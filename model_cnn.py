import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load and Preprocess the Data ---
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1.1 Reshape and Normalize
# CNN expects 4 dimensions: (samples, height, width, channels)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

# 1.2 Convert labels to one-hot encoding (10 classes: 0-9)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# --- 2. Build the CNN Model ---
def build_cnn_model():
    model = Sequential([
        # Convolutional Layer 1
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Convolutional Layer 2
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax') # Output layer for 10 classes
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_cnn_model()
model.summary()

# --- 3. Train the Model ---
print("\nStarting model training...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,  # Training for 10 epochs is usually sufficient for >95% accuracy
    verbose=1,
    validation_data=(x_test, y_test)
)

# --- 4. Evaluate and Save ---
print("\nEvaluating model...")
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

if acc > 0.95:
    print("Goal achieved: Test accuracy > 95%.")
else:
    print("Goal not achieved. Consider increasing epochs or adjusting architecture.")

# Save the trained model
model_filename = 'mnist_cnn_model.h5'
model.save(model_filename)
print(f"Model saved successfully as {model_filename}")

# --- 5. Visualize Predictions on 5 Sample Images ---
print("\nVisualizing 5 sample predictions...")
samples = x_test[:5]
true_labels = np.argmax(y_test[:5], axis=1)
predictions = np.argmax(model.predict(samples), axis=1)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axes[i].imshow(samples[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f"True: {true_labels[i]}\nPred: {predictions[i]}")
    axes[i].axis('off')
plt.tight_layout()
plt.show() # In a script, this opens a window; in Jupyter/Colab, it displays inline.