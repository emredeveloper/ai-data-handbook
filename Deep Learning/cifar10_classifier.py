# Import necessary TensorFlow/Keras modules
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
from keras.utils import to_categorical

def main():
    # 1. Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # 2. Preprocess the data
    # Normalize pixel values of images to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    num_classes = 10
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # 3. Define a Convolutional Neural Network (CNN) model
    model = models.Sequential()

    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flattening Layer
    model.add(layers.Flatten())

    # Dense Layer
    model.add(layers.Dense(64, activation='relu'))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Print model summary
    model.summary()

    # 4. Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. Train the model
    epochs = 15 # Reasonable number of epochs
    batch_size = 64 # Common batch size

    print("\n--- Training Model ---")
    history = model.fit(x_train, y_train_cat,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test_cat))

    # 6. Evaluate the model
    print("\n--- Evaluating Model ---")
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Optionally, print training history details
    print("\n--- Training History ---")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {history.history['loss'][epoch]:.4f} - "
              f"Accuracy: {history.history['accuracy'][epoch]:.4f} - "
              f"Val_Loss: {history.history['val_loss'][epoch]:.4f} - "
              f"Val_Accuracy: {history.history['val_accuracy'][epoch]:.4f}")

if __name__ == '__main__':
    # Check if TensorFlow is available and if a GPU is detected
    print(f"TensorFlow Version: {tf.__version__}")
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"GPU detected: {gpu_devices}")
        # You can set memory growth to avoid OOM errors if needed
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled.")
        except RuntimeError as e:
            print(f"Could not set memory growth: {e}")
    else:
        print("No GPU detected, training on CPU.")
    
    main()
