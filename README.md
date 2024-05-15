# Plant Species Classification

This project focuses on classifying plant species using transfer learning with the VGG16 model. The project includes a Streamlit web application for predicting plant species from uploaded images.

## Introduction
The relationship between plants and the environment is complex and multifaceted. Plants are essential for regulating carbon emissions and climate change. This project contributes to plant conservation efforts by developing a machine learning model that can identify and classify plant species based on leaf images.

## Features
- **Image Classification**: Classify plant species using VGG16 with transfer learning.
- **Web Application**: Streamlit-based interface for uploading images and generating predictions.
- **High Accuracy**: Achieved an accuracy of 98.18% on the test dataset.
- **Interactive Visualization**: Plotting training and validation loss and accuracy.

## Dataset
The dataset comprises images of leaves from twelve economically and environmentally significant plant species. Each species has images of both healthy and diseased leaves. The dataset includes a total of 4503 images, with 2278 healthy and 2225 diseased leaves.I have used the healthy leaves for this project.

## Model Training
The model uses the VGG16 architecture with transfer learning. The pre-trained VGG16 model is fine-tuned on the leaf image dataset. Key steps include:
- Data augmentation with `ImageDataGenerator`.
- Training with categorical cross-entropy loss and the Adam optimizer.
- Evaluation of model performance with accuracy and loss metrics.

## Web Application
A Streamlit web application allows users to upload leaf images and get predictions on plant species. The app also provides additional information about the predicted species, including descriptions and relevant Wikipedia links.


## Results
The model achieved an accuracy of 98.18% on the test dataset. Precision, recall, and F1-scores are consistently high across all classes, indicating the model's reliability and effectiveness in classifying plant species.

![Model performance](https://github.com/LLkrishna/PLANT-SPECIES-DETECTION/assets/130060978/149156f2-bccb-40ae-91ec-582f0fb659dd)


## Future Work
- Expand the dataset to include more plant species.
- Fine-tune the model using newer architectures beyond VGG16.
- Integrate real-time data collection using drones or sensors or any other methods to yeild good predictions.

## License
This project is licensed under the MIT License.

