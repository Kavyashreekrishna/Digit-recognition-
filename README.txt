# MNIST Digit Recognition with Convolutional Neural Network

A deep learning project that implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset using TensorFlow and Keras.

## 📋 Overview

This project builds and trains a CNN model to classify handwritten digits (0-9) from the famous MNIST dataset. The model achieves high accuracy through multiple convolutional layers and is capable of recognizing digits with excellent performance.

## 🚀 Features

- **CNN Architecture**: Multi-layer convolutional neural network with pooling layers
- **High Accuracy**: Achieves >95% accuracy on test data
- **Visualization**: Training progress plots and prediction visualizations
- **Model Persistence**: Saves trained model for future use
- **Interactive Web App**: Streamlit-based drawing canvas for real-time digit recognition
- **Real-time Prediction**: Draw digits and get instant predictions with confidence scores
- **Easy to Use**: Simple, well-documented code structure

## 🛠️ Requirements

```
tensorflow>=2.10.0
matplotlib>=3.5.0
numpy>=1.21.0
streamlit>=1.28.0
streamlit-drawable-canvas>=0.9.0
opencv-python>=4.7.0
```

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition
```

2. Install required packages:
```bash
pip install tensorflow matplotlib numpy streamlit streamlit-drawable-canvas opencv-python
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

## 🏗️ Model Architecture

The CNN model consists of:

- **Input Layer**: 28×28×1 grayscale images
- **Conv2D Layer 1**: 32 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D Layer 1**: 2×2 pool size
- **Conv2D Layer 2**: 64 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D Layer 2**: 2×2 pool size
- **Conv2D Layer 3**: 64 filters, 3×3 kernel, ReLU activation
- **Flatten Layer**: Converts 2D feature maps to 1D
- **Dense Layer 1**: 64 neurons, ReLU activation
- **Output Layer**: 10 neurons, Softmax activation (for 10 digit classes)

## 🚀 Usage

### Training the Model

Run the training script to create and train the CNN model:

```bash
python mnist_cnn.py
```

The script will:
1. Load and preprocess the MNIST dataset
2. Build the CNN architecture
3. Train the model for 5 epochs
4. Evaluate model performance
5. Display training/validation curves
6. Show sample predictions with visualizations
7. Save the trained model as `mnist_cnn_model.h5`

### Running the Interactive Web App

After training the model (or if you already have `mnist.h5`), launch the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

This will open a web browser with an interactive canvas where you can:
- Draw digits using your mouse or touchscreen
- Click "Predict" to get real-time digit recognition
- See confidence scores for each prediction
- Draw multiple digits and get predictions for each

## 📊 Results

The model typically achieves:
- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~98-99%
- **Test Accuracy**: ~98-99%

### Sample Output
```
Test accuracy: 0.9923

Model saved as: mnist_cnn_model.h5
```

## 📈 Visualizations

The script generates two types of visualizations:

1. **Training Progress**: Accuracy and loss curves for both training and validation data
2. **Prediction Examples**: Grid showing actual vs predicted digits with color coding:
   - Green: Correct predictions
   - Red: Incorrect predictions

## 🖥️ Interactive Web Application

The Streamlit app provides a user-friendly interface for testing the trained model:

### Features
- **Drawing Canvas**: Interactive canvas with customizable brush settings
- **Real-time Processing**: Automatic image preprocessing and digit extraction
- **Multiple Digit Recognition**: Can recognize multiple digits drawn on the same canvas
- **Confidence Scores**: Shows prediction confidence as percentages
- **Visual Feedback**: Bounding boxes around detected digits with predictions

### How It Works
1. **Image Capture**: Canvas drawing is saved as an image
2. **Preprocessing**: Image undergoes grayscale conversion, Gaussian blur, and adaptive thresholding
3. **Contour Detection**: Finds individual digit boundaries using OpenCV
4. **Digit Extraction**: Each digit is cropped, resized to 18×18, and padded to 28×28
5. **Prediction**: Normalized digit is fed to the CNN model for classification
6. **Display**: Results shown with confidence scores and bounding boxes

### Usage Tips
- Draw digits clearly with sufficient spacing
- Use a thick stroke for better recognition
- The red brush color works well against the white background
- Click "Predict" after drawing to get results

The trained model is automatically saved as `mnist_cnn_model.h5` and can be loaded for future use:

```python
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Make predictions
predictions = model.predict(your_images)
```

**Note**: The Streamlit app expects the model to be saved as `mnist.h5`. If you've trained using the main script, rename `mnist_cnn_model.h5` to `mnist.h5` or update the model loading path in the Streamlit app.

## 🔧 Customization

### Modify Training Parameters
```python
# Change number of epochs
history = model.fit(train_images, train_labels, epochs=10,  # Default: 5
                    validation_data=(test_images, test_labels))

# Add different optimizer
model.compile(optimizer='sgd',  # Default: 'adam'
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Adjust Model Architecture
```python
# Add more convolutional layers
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

### Streamlit App Customization
```python
# Modify canvas settings
canvas_result = st_canvas(
    stroke_width=15,           # Make brush thicker (default: 10)
    stroke_color='blue',       # Change brush color (default: 'red')  
    height=200                 # Make canvas taller (default: 150)
)

# Adjust image processing parameters
blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # More blur (default: (5,5))
```

## 🗂️ File Structure

```
mnist-digit-recognition/
│
├── mnist_cnn.py              # Main training script
├── streamlit_app.py          # Interactive web application
├── mnist_cnn_model.h5        # Saved trained model (generated)
├── mnist.h5                  # Model file used by Streamlit app
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── examples/                 # Sample outputs and visualizations
    ├── training_curves.png
    ├── predictions_sample.png
    └── streamlit_demo.gif
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **TensorFlow/Keras**: Google's machine learning framework
- **Deep Learning Community**: For continuous inspiration and knowledge sharing

## 📚 References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras CNN Guide](https://keras.io/examples/vision/mnist_convnet/)

---

⭐ **Star this repository if you found it helpful!**