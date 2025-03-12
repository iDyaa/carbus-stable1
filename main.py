import os
import tensorflow as tf
from src.preprocess import get_datasets
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_model  # This module should define an evaluation function

def configure_gpu_memory():
    """Enable memory growth so TensorFlow allocates GPU memory on demand."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled memory growth for GPUs.")
        except RuntimeError as e:
            print("Error setting memory growth:", e)

def main():
    # Configure GPU memory growth
    configure_gpu_memory()
    
    # Set parameters for data loading and training
    data_dir = os.path.expanduser("~/vehicle_classifier/data")  # Adjust this path if needed
    img_size = (128, 128)
    batch_size = 32
    epochs = 10

    # Load datasets from the preprocessing module
    train_ds, val_ds = get_datasets(data_dir, img_size=img_size, batch_size=batch_size)
    print("Classes detected:", train_ds.class_names)

    # Create and compile the CNN model
    model = create_model(input_shape=(img_size[0], img_size[1], 3))
    model.summary()

    # Train the model using the training module
    history = train_model(model, train_ds, val_ds, epochs)
    
    # Ensure the directory for saved models exists
    save_dir = os.path.join(os.getcwd(), "saved_models")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the trained model
    model_path = os.path.join(save_dir, "vehicle_classifier_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate the model using the evaluation module
    eval_results = evaluate_model(model, val_ds)
    print("Evaluation results:", eval_results)

if __name__ == "__main__":
    main()
