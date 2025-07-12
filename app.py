from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.applications import MobileNetV3Small
from keras.preprocessing.image import load_img, img_to_array
import os
import cv2
from werkzeug.utils import secure_filename

# Parameters
IMG_SIZE = 224
NUM_CLASSES = 3
CLASS_NAMES = ['gelas', 'mangkok', 'piring']

# Load model
base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Load the weights for the model
model_path = os.path.join('model', 'mobilenetv3_3class_model-A1.h5')
model.load_weights(model_path)

# Flask App
app = Flask(__name__)  # Flask app instance
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Function: Classify image and detect multiple objects
def classify_and_draw_boxes(my_model, img_path):
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(img_batch)  # Preprocess

    # Prediction
    prediction = my_model.predict(img_preprocessed)

    # Find the class with highest probability
    result = np.argmax(prediction[0])
    class_name = CLASS_NAMES[result]
    
    # Get the coordinates of the bounding box (random example for now)
    h, w, _ = img.shape
    start_x, start_y = int(w * 0.2), int(h * 0.2)  # Example coordinates
    end_x, end_y = int(w * 0.8), int(h * 0.8)

    # Draw the bounding box
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # Add class name
    cv2.putText(img, class_name, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the image with bounding box
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_' + os.path.basename(img_path))
    cv2.imwrite(output_path, img)

    return output_path, class_name  # Return the path to the output image and predicted class

@app.route('/')
def home():
    return render_template('index.html')  # Home page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:  # Check if file is in request
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']  # Get file from request
    if file.filename == '':  # Check if no file selected
        return jsonify({'error': 'No selected file'}), 400

    # Validate if file is an image
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Secure the filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Save file path
        file.save(filepath)  # Save the file

        # Classify the uploaded image and draw bounding boxes
        output_image_path, class_name = classify_and_draw_boxes(model, filepath)

        # Count the number of each object detected in the image
        class_counts = {class_name: 1}  # Example counting
        result = {
            'classname': class_name,
            'image_url': output_image_path,
            'object_counts': class_counts
        }

        return jsonify(result)  # Return classification result as JSON with output image URL
    else:
        return jsonify({'error': 'Invalid file type. Only image files are allowed.'}), 400

# Allowable file types (images only)
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    # Configure Flask to listen on all network interfaces (LAN)
    app.run(host='0.0.0.0', port=5000, debug=True)  # This will allow access from LAN
