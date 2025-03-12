import os
import numpy as np
import cv2
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth for GPUs.")
    except RuntimeError as e:
        print("Error setting memory growth:", e)


def prepare_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image from path: " + image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    # Do NOT normalize here; let the model's Rescaling layer handle it.
    img = np.expand_dims(img, axis=0)
    return img


# Load your trained model (update the path if necessary)
model_path = os.path.join(os.getcwd(), "saved_models", "vehicle_classifier_model.keras")
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)

# Define the directory containing test images
# Make sure you have a folder called 'test_images' with some images to test
test_images_dir = os.path.join(os.getcwd(), "test_images")

# Get a list of image files from the test_images folder
image_files = [
    os.path.join(test_images_dir, f)
    for f in os.listdir(test_images_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

if not image_files:
    print("No test images found in", test_images_dir)
else:
    for image_path in image_files:
        print("\nProcessing image:", image_path)
        try:
            # Preprocess the image
            img = prepare_image(image_path)
            # Run the model prediction
            prediction = model.predict(img)
            print("Raw prediction:", prediction)
            # For binary classification, threshold at 0.5
            label = "car" if prediction[0][0] >= 0.5 else "bus"
            print("Predicted label:", label)
        except Exception as e:
            print("Error processing image:", e)
