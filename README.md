Wildfire Detection Using Convolutional Neural Networks and VGG16

This project focuses on detecting wildfires using deep learning techniques, specifically Convolutional Neural Networks (CNN) and the VGG16 architecture. The model is designed to classify images of forests as either containing wildfires or not. The project aims to provide an automated system for early detection and prevention of forest fires, potentially aiding in fire monitoring and risk mitigation.

Table of Contents

Overview
Features
Getting Started
Prerequisites
Installation
Dataset
Model Training
Usage
Results and Evaluation
Visualization
Contributing
License
Acknowledgements
Overview

This project leverages the power of Convolutional Neural Networks (CNN) and the VGG16 model to classify images of forests to detect whether a wildfire is present or not. The dataset contains two classes of images: fire and nofire. The model has been trained using the dataset, and predictions can be made on new images.

Workflow:
Data Preprocessing: Images are processed and cleaned to remove any corrupted files.
Model Construction: The CNN model is built using convolutional layers followed by dense layers. In parallel, a VGG16 model is also used for comparison.
Training: Both models are trained using the dataset.
Evaluation: The models' performance is evaluated based on accuracy and loss.
Features

Deep Learning Models: Uses CNN and VGG16 models for image classification.
Image Preprocessing: Images are preprocessed to ensure consistency and eliminate corrupt files.
Data Augmentation: Image augmentation is applied to the training set to improve model generalization.
Prediction: The models are capable of predicting whether a given image contains a wildfire or not.
Visualization: Graphs to visualize training accuracy, loss, and model performance.
Model Comparison: Compares the accuracy of CNN and VGG16 for wildfire detection.
Getting Started

Prerequisites
Before running the project, make sure you have the following libraries installed:

Python 3.x
TensorFlow (2.x)
Keras
NumPy
Matplotlib
Seaborn
OpenCV
tqdm
PIL (Pillow)
You can install the required libraries using pip:

pip install tensorflow keras numpy matplotlib seaborn opencv-python tqdm pillow
Installation
Clone the repository to your local machine:

git clone https://github.com/your-username/wildfire-detection.git
cd wildfire-detection
Dataset
The dataset used for training consists of two main folders:

Training: Contains images labeled as fire and nofire.
Testing: Used for model evaluation with the same structure as the training dataset.
Ensure that the directory structure follows the format:

/Training
    /fire
    /nofire
/Testing
    /fire
    /nofire
You can upload your own dataset to these directories if needed.

Model Training

Preprocessing and Cleaning
Images are cleaned by removing any corrupted files in the training and testing datasets. This is done using the PIL library and tqdm for progress tracking.

Data Augmentation
The training dataset is augmented using ImageDataGenerator, which applies random transformations (e.g., horizontal flipping) to improve the model's robustness:

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.0, horizontal_flip=True)
CNN Model
A simple CNN architecture is used to train the model, which includes convolutional layers followed by max-pooling and dense layers. Dropout is also applied to prevent overfitting.

CNN.add(Conv2D(filters=32, kernel_size=3, input_shape=[64,64,3], activation='relu'))
VGG16 Model
VGG16 is used as a pre-trained model for feature extraction. We replace the final layers with a custom classification head.

base_model = VGG16(include_top=False, input_shape=(64,64,3), weights='imagenet')
Both models are compiled using the Adam optimizer and binary cross-entropy loss function.

Training the Models
Both models are trained for 20 epochs, and their performance is evaluated on the test set:

CNN.fit(x=train_data_set, validation_data=test_data_set, batch_size=32, epochs=20)
Usage

To make predictions with the trained models, use the following steps:

Load an image using the image.load_img() function.
Preprocess the image (resize, normalize).
Pass the image to the trained model for prediction.
Example of making a prediction with the CNN model:

from tensorflow.keras.preprocessing import image
img_path = '/path/to/image.jpg'
input_image = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(input_image) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = CNN.predict(img_array)[0][0]
The output is a probability score. If it is above 0.5, the image is classified as containing no fire (nofire), otherwise as a wildfire (fire).

Results and Evaluation

The models are evaluated based on their accuracy and loss during training. These metrics are plotted for both models over the course of training.

plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Test Loss', color='blue')
Visualization

Training and test accuracy, as well as loss, are visualized using Matplotlib:

plt.plot(history.history['accuracy'], label='Train Accuracy', color='red')
plt.plot(history.history['val_accuracy'], label='Test Accuracy', color='blue')
This helps to understand how well the models are performing and whether they are overfitting.

Contributing

We welcome contributions! If you'd like to improve the model, add features, or fix bugs, please fork the repository and submit a pull request.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements

TensorFlow/Keras: For providing the deep learning framework used for training the models.
VGG16 Model: For the pretrained model, which helps with faster and more efficient image classification.
Dataset: A collection of images representing wildfire and non-wildfire forest images, used for model training and evaluation.
