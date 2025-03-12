from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import cv2

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth for GPUs.")
    except RuntimeError as e:
        print("Error setting memory growth:", e)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path, target_size=(128, 128)):
    """
    Loads the image from disk, converts it to RGB, resizes it to target_size,
    and adds a batch dimension so it can be fed into the CNN model.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image from path: " + image_path)
    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to target size (e.g., 128x128)
    img = cv2.resize(img, target_size)
    # Expand dimensions to create a batch of one image
    img = np.expand_dims(img, axis=0)
    return img

# Load the trained model (ensure the path is correct)
MODEL_PATH = os.path.join(os.getcwd(), "saved_models", "vehicle_classifier_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            try:
                img = prepare_image(upload_path)
            except Exception as e:
                return f"Error processing image: {str(e)}"
            
            # Run inference on the entire image
            prediction = model.predict(img)
            label = "car" if prediction[0][0] >= 0.5 else "bus"
            confidence = prediction[0][0]
            
            return render_template('result2.html', filename=filename, label=label, confidence=confidence)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
