# Vehicle Classifier

Vehicle Classifier is a deep learning project that uses a Convolutional Neural Network (CNN) to classify images of vehicles as either **bus** or **car**. The project is built with TensorFlow/Keras and runs on Ubuntu under WSL2. It also features a simple Flask web application for interactive image classification.

---

## Project Structure
vehicle_classifier/ ├── data/ # Original dataset (or symlinks to /mnt/c/AI if preferred) │ ├── bus/ │ └── car/ ├── notebooks/ # (Optional) Jupyter notebooks for exploratory data analysis ├── saved_models/ # Directory to save trained models and weights │ └── vehicle_classifier_model.keras ├── src/ # Source code modules │ ├── init.py # Marks the directory as a Python package │ ├── preprocess.py # Functions for loading and preprocessing data │ ├── model.py # CNN architecture definition │ ├── train.py # Training routines and callbacks │ └── evaluate.py # (Optional) Evaluation functions ├── requirements.txt # List of project dependencies ├── main.py # Orchestrates training and evaluation: │ # - Loads and preprocesses data │ # - Builds the model │ # - Trains and evaluates the model ├── uploads/ # Directory for storing uploaded images (created automatically) ├── templates/ # HTML templates for the Flask web app │ ├── index.html # Main page with file uploader and button │ └── result.html # Page to display the image and classification result ├── app.py # Flask web application for image classification ├── .wslconfig # (Optional) WSL resource configuration (located in Windows user folder) └── README.md # Project documentation (this file)


---

## Features

- **Data Preprocessing:**  
  Loads images from the specified directory, splits them into training/validation sets, and performs on-the-fly preprocessing.

- **CNN Model:**  
  Implements a simple CNN architecture for binary classification (bus vs. car).

- **Training Pipeline:**  
  Includes training routines with callbacks (e.g., EarlyStopping) to prevent overfitting.

- **Model Evaluation:**  
  Evaluates the trained model on a validation set.

- **Web Application:**  
  A Flask app that enables users to upload images, run inference, and display classification results.

- **WSL2 Compatibility:**  
  Designed to run on Ubuntu under WSL2, with GPU support if available.

---

## Setup and Installation

### Prerequisites

- WSL2 with Ubuntu installed and running.
- Python 3 installed.
- A virtual environment is recommended for managing project dependencies.

### Installation Overview

1. Clone the repository to your local system.
2. Create and activate a Python virtual environment.
3. Install all required libraries as listed in the requirements file.
4. (Optionally) Configure WSL2 resource settings by creating a configuration file in your Windows user folder.

---

## Running the Project

### Training the Model

- **Dataset Preparation:**  
  Ensure your dataset is organized with subfolders "bus" and "car" and is placed within your WSL filesystem.

- **Training Process:**  
  Run the main training script which:
  - Loads and preprocesses the data.
  - Builds and trains the CNN model.
  - Evaluates the model.
  - Saves the trained model in the saved_models directory.

### Running the Web Application

- **Launching the Web App:**  
  Start the Flask web application to enable image uploads and classification.
- **Using the App:**  
  Open your browser and navigate to the local address provided by the app to upload an image and view its classification result.

---

## How It Works

- **Preprocessing:**  
  Uses TensorFlow’s image loading functions in the preprocessing module to dynamically load and split images into training and validation sets. The images are preprocessed on the fly, with no intermediate files saved.

- **Model Architecture:**  
  The CNN model normalizes images, extracts features via convolutional and pooling layers, and performs binary classification through dense layers with a sigmoid activation.

- **Training and Evaluation:**  
  The training pipeline utilizes callbacks to monitor and improve performance, with evaluation functions providing metrics on a validation set.

- **Inference Web App:**  
  The Flask application handles image uploads, applies consistent preprocessing, performs model inference, and displays the uploaded image alongside its predicted label and confidence score.

---

## Future Work

- Extend the model to classify additional vehicle types or conditions.
- Adapt the model for other tasks (e.g., detecting wilting microgreens).
- Enhance the web app with live camera feed integration and advanced analytics.
- Experiment with more advanced model architectures or transfer learning for improved accuracy.

---

## License

==

---

## Acknowledgements

- TensorFlow & Keras for the deep learning framework.
- Flask for the web application framework.
- The open-source community for valuable contributions and support.
