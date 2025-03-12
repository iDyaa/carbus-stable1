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

# Pre-defined polygon coordinates for the four regions.
# Each polygon is a list of four (x, y) tuples, in order: top-left, top-right, bottom-right, bottom-left.
polygons = [
    [(14, 15), (258, 12), (255, 175), (13, 174)],
    [(313, 25), (623, 22), (627, 202), (310, 240)],
    [(22, 245), (319, 281), (329, 448), (18, 439)],
    [(350, 256), (633, 264), (606, 466), (345, 460)]
]

# Desired output size for each rectified region (width, height)
OUTPUT_SIZE = (128, 128)

def rectify_region(image, src_pts, output_size=OUTPUT_SIZE):
    """
    Given an image and 4 source points (polygon),
    computes the perspective transform to rectify the region into a rectangle of output_size.
    """
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array([[0, 0],
                    [output_size[0]-1, 0],
                    [output_size[0]-1, output_size[1]-1],
                    [0, output_size[1]-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, output_size)
    return warped

def process_uploaded_image(image_path):
    """
    Loads the uploaded image, resizes it to 640x480, and for each pre-defined polygon,
    extracts the region and applies perspective correction.
    Returns a list of rectified region images.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image from path: " + image_path)
    
    # Resize the image to 640x480 (to match ROI calibration)
    img = cv2.resize(img, (128, 128))
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    rectified_regions = []
    for idx, poly in enumerate(polygons, start=1):
        rect_img = rectify_region(img, poly, output_size=OUTPUT_SIZE)
        rectified_regions.append(rect_img)
    return rectified_regions

# Load your trained model (ensure this path is correct)
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
                # Process the uploaded image: first resize to 640x480, then extract and rectify regions.
                regions = process_uploaded_image(upload_path)
            except Exception as e:
                return f"Error processing image: {str(e)}"
            
            results = []
            # For each rectified region, run inference
            for idx, region in enumerate(regions, start=1):
                input_img = np.expand_dims(region, axis=0)
                prediction = model.predict(input_img)
                label = "car" if prediction[0][0] >= 0.5 else "bus"
                confidence = prediction[0][0]
                results.append({"polygon": idx, "label": label, "confidence": confidence})
                
                # Optionally, save the rectified region for display
                region_filename = f"region_{idx}_{filename}"
                region_path = os.path.join(app.config['UPLOAD_FOLDER'], region_filename)
                cv2.imwrite(region_path, cv2.cvtColor(region, cv2.COLOR_RGB2BGR))
            
            region_files = [f"region_{i+1}_{filename}" for i in range(len(results))]
            return render_template('result.html', filename=filename, results=results, region_files=region_files)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
