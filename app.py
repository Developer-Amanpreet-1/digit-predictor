# app.py  -- optimized to use a quantized TensorFlow Lite model for low memory footprint.
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os
import sys
import tensorflow as tf

# --- Runtime tuning to reduce memory usage where possible ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # quiet logs
# Limit intra/inter op threads (reduce memory/cpu contention)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

MODEL_TFLITE = "mnist_cnn.tflite"
MODEL_KERAS = "mnist_cnn.h5"
EPOCHS_SMALL = 3  # very small (only used if no model exists); increase offline if you want accuracy

# Build a very small MNIST model (tiny number of params)
def build_small_model(input_shape=(28,28,1), num_classes=10):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    model = Sequential([
        Conv2D(8, (3,3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(16, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Convert a Keras model to quantized TFLite (int8) using a small representative set
def convert_to_tflite_int8(keras_model_path, tflite_path, representative_data):
    # load model
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Set optimization and representative dataset for post-training quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_gen():
        for i in representative_data:
            # i expected shape (28,28) or (28,28,1) normalized [0,1]
            arr = np.expand_dims(i, axis=0).astype(np.float32)
            if arr.ndim == 3:
                arr = np.expand_dims(arr, -1)
            yield [arr]

    converter.representative_dataset = rep_gen
    # Force full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    return tflite_path

# Create tiny Keras model, train briefly (only if no model exists)
def create_and_save_small_keras_model(keras_path, epochs=EPOCHS_SMALL):
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype('float32') / 255.0)
    x_test  = (x_test.astype('float32') / 255.0)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test, 10)

    model = build_small_model()
    # tiny quick training (this is optional — better to prepare model offline)
    model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_data=(x_test, y_test), verbose=2)
    model.save(keras_path)
    return keras_path, x_train[:200]  # return some samples for representative conversion

# Load or prepare TFLite model and return an interpreter ready for inference
def ensure_tflite_interpreter():
    # If tflite exists -> load
    if os.path.exists(MODEL_TFLITE):
        interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
        interpreter.allocate_tensors()
        return interpreter

    # If Keras exists -> convert to int8 tflite
    if os.path.exists(MODEL_KERAS):
        # load a small subset of mnist to use as representative dataset for quantization
        from tensorflow.keras.datasets import mnist
        (x_train, _), _ = mnist.load_data()
        x_rep = (x_train[:200].astype('float32') / 255.0)
        tpath = convert_to_tflite_int8(MODEL_KERAS, MODEL_TFLITE, x_rep)
        interpreter = tf.lite.Interpreter(model_path=tpath)
        interpreter.allocate_tensors()
        return interpreter

    # else: create a tiny keras model, save, convert
    print("No model found. Creating tiny model and converting to TFLite (this will take a moment).", file=sys.stderr)
    keras_path, rep = create_and_save_small_keras_model(MODEL_KERAS, epochs=EPOCHS_SMALL)
    tpath = convert_to_tflite_int8(keras_path, MODEL_TFLITE, rep)
    interpreter = tf.lite.Interpreter(model_path=tpath)
    interpreter.allocate_tensors()
    return interpreter

# Create Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Prepare interpreter at startup (this avoids keeping a big TF graph in memory)
interpreter = ensure_tflite_interpreter()

# Get input/output details for TFLite
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_with_tflite(pixels_1d):
    # Expect pixels in [0,1], length 784
    arr = np.array(pixels_1d, dtype=np.float32).reshape(1,28,28)
    # TFLite model expects uint8 if fully integer quantized; convert accordingly
    # Find input scale/zero_point
    in_detail = input_details[0]
    scale, zp = in_detail.get('quantization', (1.0, 0))
    if in_detail['dtype'] == np.uint8:
        # scale float [0,1] -> quantized uint8
        arr_q = (arr / scale + zp).astype(np.uint8)
        input_data = np.expand_dims(arr_q, -1)
    else:
        input_data = np.expand_dims(arr.astype(in_detail['dtype']), -1)

    interpreter.set_tensor(in_detail['index'], input_data)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]

    # If outputs are quantized, dequantize
    out_scale, out_zp = output_details[0].get('quantization', (1.0, 0))
    if output_details[0]['dtype'] == np.uint8:
        out = (out.astype(np.float32) - out_zp) * out_scale

    # softmax may be unnecessary if quantized model outputs logits; but treat as probabilities
    # ensure non-negative and normalized
    probs = np.array(out, dtype=np.float32)
    if probs.sum() <= 0:
        probs = np.exp(probs) / np.sum(np.exp(probs))
    else:
        probs = probs / probs.sum()

    pred_class = int(np.argmax(probs))
    return pred_class, probs.tolist()

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
        # clip and ensure normalized
        arr = np.clip(arr, 0.0, 1.0)
        pred_class, confidences = predict_with_tflite(arr)
        return jsonify({'prediction': pred_class, 'confidences': confidences})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    # Run in production-like mode
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
