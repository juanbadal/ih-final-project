# Ironhack final project


### Facial Expression Classification Project
This project focuses on classifying facial expressions using the FER+ dataset. It utilizes various deep learning models, including ResNet50, DenseNet201, MobileNetV2, as well as custom architectures, to accurately predict the emotions depicted in facial images.

### Dataset
The FER+ dataset, which contains facial expressions annotated with eight emotion labels (anger, contempt, disgust, fear, happiness, sadness, surprise, and neutral), is used for training and evaluation. It provides a diverse collection of facial images, making it suitable for training robust emotion recognition models.

### Models
The following models are employed in this project:

- ResNet50
- DenseNet201
- MobileNetV2
- Custom architectures

These models are trained and fine-tuned on the FER+ dataset to capture intricate facial features and nuances associated with different emotions.

### Application
The project includes a user-friendly application built with Streamlit. Users can upload an image through the interface, and the application returns the predicted classes and probabilities for each emotion category. This feature allows for interactive exploration of the models' performance and facilitates practical usage in real-world scenarios.

### Usage
To run the application locally, follow these steps:

- Clone the repository to your local machine.
- Install the required dependencies listed in the requirements.txt file -> the environment was created with python 3.9 in order to be able to install TensorFlow 2.10 and leverage GPU usage in native Windows.
- cudnn and cudatoolkit have been installed through conda-forge with the following command: conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
- Run the Streamlit application by executing streamlit run fe_recognizer.py in the terminal.

### Conclusion
This facial expression classification project demonstrates the effectiveness of deep learning models in accurately recognizing emotions from facial images. By leveraging diverse architectures and the FER+ dataset, the project aims to contribute to the advancement of emotion recognition technology. The inclusion of a user-friendly application enhances accessibility and usability, showcasing the practical implications of the developed models.