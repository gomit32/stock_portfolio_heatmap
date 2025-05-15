# Import necessary libraries
import os
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def fetch_and_preprocess_data(tickers, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            stock_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)

            if stock_data.empty:
                raise ValueError("Downloaded data is empty (likely due to rate limiting).")

            if 'Adj Close' in stock_data.columns:
                adj_close = stock_data['Adj Close']
            else:
                adj_close = stock_data['Close']

            returns = adj_close.pct_change().dropna()
            if returns.empty:
                raise ValueError("Returns data is empty.")

            correlation_matrix = returns.corr()
            return correlation_matrix

        except ValueError as e:
            print(f"[Attempt {attempt+1}/{max_retries}] {e}")
            time.sleep(5)
        except Exception as e:
            print(f"[Attempt {attempt+1}/{max_retries}] Unexpected error: {e}")
            time.sleep(5)

    print("All retries failed. Returning empty DataFrame.")
    return pd.DataFrame()
  # return empty DataFrame if all retries fail

# Function to plot and save the correlation heatmap
def plot_correlation_heatmap(correlation_matrix, image_filename):
    if correlation_matrix.isnull().values.any():
        print("Correlation matrix contains NaN values. Aborting heatmap generation.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap of Stocks")
    plt.savefig(image_filename)
    plt.close()

# Create a CNN model
def create_cnn_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess images
def preprocess_images(image_paths, labels, target_size=(224, 224)):
    images = []
    for image_path in image_paths:
        try:
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    return np.array(images), np.array(labels)

# Function to create a directory if it doesn't exist
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created!")
    else:
        print(f"Directory {directory_path} already exists!")

# Prepare dataset and train the CNN model
def prepare_and_train_model(tickers, start_date, end_date, image_folder_path):
    correlation_matrix = fetch_and_preprocess_data(tickers, start_date, end_date)

    if correlation_matrix.empty or correlation_matrix.isnull().values.any():
        print("No usable correlation matrix. Skipping training.")
        return None

    create_directory(image_folder_path)

    image_filename = f"{image_folder_path}/correlation_heatmap.png"
    plot_correlation_heatmap(correlation_matrix, image_filename)

    image_paths = [image_filename for _ in tickers]
    labels = [1 if i % 2 == 0 else 0 for i in range(len(tickers))]

    X, y = preprocess_images(image_paths, labels)
    if len(X) == 0:
        print("No images loaded successfully. Skipping training.")
        return None

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_cnn_model(input_shape=(224, 224, 3))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=16)

    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    return model

# Function to predict using heatmap images
def predict_using_heatmaps(model, image_paths, target_size=(224, 224)):
    X_pred = []
    for image_path in image_paths:
        try:
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            X_pred.append(img_array)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

    if not X_pred:
        print("No valid images for prediction.")
        return None

    X_pred = np.array(X_pred)
    predictions = model.predict(X_pred)
    return predictions

# Example usage
tickers = ["TCS.NS", "RELIANCE.BO", "INFY.NS", "HDFCBANK.NS"]
image_folder_path = '/content/heatmap_images'
model = prepare_and_train_model(tickers, "2020-01-01", "2021-01-01", image_folder_path)

if model:
    image_paths_for_prediction = [f"{image_folder_path}/correlation_heatmap.png" for _ in tickers]
    predictions = predict_using_heatmaps(model, image_paths_for_prediction)
    print(f"Predictions: {predictions}")
else:
    print("Skipping prediction due to earlier failure.")



import pandas as pd
import os

# Create dummy labels.csv for testing
os.makedirs('/content/heatmap_dataset', exist_ok=True)
dummy_data = {
    'file': ['heatmap_0.png', 'heatmap_1.png'],
    'label': [1, 0]
}
df = pd.DataFrame(dummy_data)
df.to_csv('/content/heatmap_dataset/labels.csv', index=False)
print("✅ Dummy labels.csv created.")



import matplotlib.pyplot as plt
import seaborn as sns

def generate_and_save_heatmap(matrix, filename):
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=False, cmap='coolwarm')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Example: create dummy correlation matrices
import numpy as np
for i in range(10):  # adjust count
    matrix = np.random.rand(5, 5)
    matrix = (matrix + matrix.T) / 2  # Make symmetric
    np.fill_diagonal(matrix, 1)
    generate_and_save_heatmap(matrix, f"/content/heatmap_dataset/heatmap_{i}.png")

import pandas as pd

df = pd.DataFrame({
    "file": [f"heatmap_{i}.png" for i in range(10)],
    "label": np.random.randint(0, 2, size=10)  # Replace with actual labels
})
df.to_csv("/content/heatmap_dataset/labels.csv", index=False)


import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Path setup
image_dir = '/content/heatmap_dataset'  # Ensure this directory exists and contains images
label_file = os.path.join(image_dir, 'labels.csv')  # Ensure this CSV exists and matches image names

# Check if required paths exist
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"❌ Directory not found: {image_dir}")
if not os.path.isfile(label_file):
    raise FileNotFoundError(f"❌ labels.csv not found at: {label_file}")

# Load labels
df = pd.read_csv(label_file)

# Image parameters
img_size = (128, 128)
X, y = [], []

# Loop through the labels and load images
missing_files = []
for _, row in df.iterrows():
    image_path = os.path.join(image_dir, row['file'])  # Full path of the image
    if os.path.isfile(image_path):
        img = load_img(image_path, target_size=img_size, color_mode='rgb')
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        X.append(img_array)
        y.append(row['label'])
    else:
        print(f"⚠️ Warning: Missing image {image_path}")
        missing_files.append(image_path)

# Handle the case where no images were loaded
if not X:
    raise ValueError("❌ No images loaded. Please ensure the dataset is present and labeled correctly.")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Train-validation split (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Output stats
print("✅ Images loaded and split. Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)

if missing_files:
    print(f"\n⚠️ {len(missing_files)} images were missing and skipped.")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Define the model
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=16,
    validation_data=(X_val, y_val)
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"✅ Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Plot the training history
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Predict a few samples
predictions = model.predict(X_val[:5])
for i, prediction in enumerate(predictions):
    print(f"Predicted: {prediction[0]:.2f}, Actual: {y_val[i]}")


# Save the model to a file
model.save('/content/portfolio_cnn_model.h5')
print("✅ Model saved successfully.")


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the saved model
model = load_model('/content/portfolio_cnn_model.h5')

# Load a sample image for prediction (you can change this file)
sample_image_path = '/content/heatmap_dataset/heatmap_0.png'

# Prepare the image
img = load_img(sample_image_path, target_size=(128, 128))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(img_array)[0][0]
print("🔍 Prediction score:", prediction)

# Interpret the result
if prediction > 0.5:
    print("✅ Model recommends: INVEST")
else:
    print("❌ Model recommends: DO NOT INVEST")


!pip install gradio

!pip install yfinance --upgrade --no-cache-dir

import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import gradio as gr
from datetime import datetime

# Load pre-trained model
model = load_model('/content/portfolio_cnn_model.h5')

# Function to fetch stock data and compute correlation matrix
def fetch_and_preprocess_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)

    # Check if 'Adj Close' exists, else use 'Close'
    if 'Adj Close' in data.columns:
        data = data['Adj Close']
    elif 'Close' in data.columns:
        data = data['Close']
    else:
        return None

    # If only one ticker, convert the data to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    returns = data.pct_change().dropna()
    return returns.corr()

# Function to plot and save the heatmap
def plot_correlation_heatmap(corr_matrix, image_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    plt.title("Correlation Heatmap of Selected Stocks", fontsize=16)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

# Function to fetch current stock prices
def fetch_current_prices(tickers):
    current_prices = {}
    for ticker in tickers:
        data = yf.Ticker(ticker)
        current_prices[ticker] = data.history(period="1d")['Close'][0]  # Get the latest close price
    return current_prices

# Main Gradio function
def generate_heatmap_and_predict(tickers, start_date, end_date):
    try:
        tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if len(tickers_list) < 2:
            return None, "❗ Enter at least two stock tickers."

        # Convert to string format if datetime
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")

        # Fetch correlation matrix
        corr_matrix = fetch_and_preprocess_data(tickers_list, start_date, end_date)

        # Check if data is available
        if corr_matrix is None or corr_matrix.empty:
            return None, "⚠️ Not enough data to compute correlation or 'Adj Close' and 'Close' are missing."

        # Save heatmap
        image_path = "/content/latest_heatmap.png"
        plot_correlation_heatmap(corr_matrix, image_path)

        # Fetch current stock prices
        current_prices = fetch_current_prices(tickers_list)

        # Load and preprocess for prediction
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        pred = model.predict(img_array)[0][0]

        # Determine recommendation
        recommendation = "✅ INVEST" if pred > 0.5 else "❌ DO NOT INVEST"

        # Prepare current prices string
        current_prices_str = "\n".join([f"{ticker}: ₹{price:.2f}" for ticker, price in current_prices.items()])

        return image_path, recommendation, current_prices_str

    except Exception as e:
        return None, f"Error: {str(e)}", ""
        print(f"Prediction: {pred}")


# Gradio Interface
iface = gr.Interface(
    fn=generate_heatmap_and_predict,
    inputs=[
        gr.Textbox(label="Enter Stock Tickers (comma-separated)", placeholder="e.g. INFY.NS, TCS.NS, RELIANCE.NS"),
        gr.Textbox(label="Start Date (YYYY-MM-DD)", placeholder="e.g. 2024-06-01"),
        gr.Textbox(label="End Date (YYYY-MM-DD)", placeholder="e.g. 2025-05-01")
    ],
    outputs=[
        gr.Image(label="📈 Correlation Heatmap"),
        gr.Textbox(label="💡 Investment Recommendation"),
        gr.Textbox(label="💰 Current Stock Prices")
    ],
    title="📊 Stock Portfolio Heatmap Analyzer",
    description="Enter stock tickers and date range to generate a correlation heatmap and get a CNN-based investment recommendation. Additionally, see the current stock prices.",
    theme="soft",
    live=False
)

# Launch the interface
iface.launch()


