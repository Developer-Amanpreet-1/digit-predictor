from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import BatchNormalization # type: ignore


MODEL_PATH = 'mnist_cnn.h5'
EPOCHS = 5  # You can increase this for better accuracy

def build_model(input_shape=(28,28,1), num_classes=10):
    model = Sequential([
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),

        Conv2D(32, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# Add channel dimension (required!)
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# One-hot labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Create model
model = build_model()
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

datagen.fit(x_train)

model.fit(datagen.flow(x_train, y_train, batch_size=128),
          validation_data=(x_test, y_test),
          epochs=EPOCHS)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)



def prepare_and_train(EPOCHS=30, MODEL_PATH="mnist_cnn.h5"):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = build_model()

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        callbacks=[lr_scheduler]
    )

    model.save(MODEL_PATH)
    return model


# Create Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Load or train model on startup (if model file not present)
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    print("Model file not found. Training a new model (this may take a few minutes)...")
    model = prepare_and_train()
    print("Training complete and model saved to", MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    pixels = data.get('pixels', None)
    if pixels is None:
        return jsonify({'error': 'no pixels provided'}), 400
    try:
        arr = np.array(pixels, dtype=np.float32)
        if arr.size != 28*28:
            return jsonify({'error': f'expected 784 pixels but got {arr.size}'}), 400
        arr = arr.reshape(1,28,28,1)
        # Ensure values are in [0,1]
        arr = np.clip(arr, 0.0, 1.0)
        preds = model.predict(arr)
        pred_class = int(np.argmax(preds[0]))
        confidences = preds[0].tolist()
        return jsonify({'prediction': pred_class, 'confidences': confidences})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
