Instructions to run the Digit Recognizer locally:

1. (Recommended) Create and activate a Python virtual environment:
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Flask app:
   python app.py

   - On first run the script will train a simple CNN on the MNIST dataset and save it to mnist_cnn.h5.
   - Training is controlled by the EPOCHS variable inside app.py (default 5). Increase for better accuracy.

4. Open your browser to:
   http://127.0.0.1:5000/

Notes:
- The frontend (cnn_digit_frontend_fixed.html) posts 28x28 normalized pixel data to /predict and expects a JSON response.
- If you already have a pre-trained Keras model file named 'mnist_cnn.h5' in the same folder, the app will load it instead of training.
